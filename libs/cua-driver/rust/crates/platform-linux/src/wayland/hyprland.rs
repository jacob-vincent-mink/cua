//! Hyprland-specific identity resolution for per-toplevel capture.
//!
//! Standard foreign-toplevel protocols do not expose process identity or an
//! opaque handle accepted by Hyprland's toplevel-export protocol. Hyprland's
//! user-owned IPC supplies that missing correlation. Capture is authorized only
//! when one mapped client owned by the requested PID matches the compositor
//! title and app-id observed on the Wayland connection.

use anyhow::{bail, Context, Result};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::process::{Command, Stdio};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

const FOCUS_TIMEOUT: Duration = Duration::from_millis(500);
const FOCUS_POLL_INTERVAL: Duration = Duration::from_millis(15);

#[derive(Clone, Debug, Default, Deserialize)]
struct Workspace {
    #[serde(default)]
    id: i64,
}

#[derive(Clone, Debug, Deserialize)]
struct Client {
    address: String,
    mapped: bool,
    #[serde(default)]
    hidden: bool,
    pid: i64,
    #[serde(default)]
    title: String,
    #[serde(default)]
    class: String,
    #[serde(default, rename = "initialClass")]
    initial_class: String,
    #[serde(default)]
    at: [i32; 2],
    #[serde(default)]
    size: [i32; 2],
    #[serde(default)]
    workspace: Workspace,
    #[serde(default, rename = "initialTitle")]
    initial_title: String,
    #[serde(default, rename = "stableId")]
    stable_id: String,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct ActiveWindow {
    #[serde(default)]
    address: String,
    #[serde(default)]
    pid: Option<u32>,
    #[serde(default)]
    class: String,
    #[serde(default, rename = "initialClass")]
    initial_class: String,
    #[serde(default, rename = "initialTitle")]
    initial_title: String,
    #[serde(default, rename = "stableId")]
    stable_id: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct WindowIdentity {
    address: u64,
    stable_id: String,
    pid: u32,
    initial_class: String,
    initial_title: String,
}

impl WindowIdentity {
    fn from_client(client: &Client) -> Option<Self> {
        Some(Self {
            address: parse_address(&client.address)?,
            stable_id: client.stable_id.clone(),
            pid: u32::try_from(client.pid).ok().filter(|pid| *pid != 0)?,
            initial_class: if client.initial_class.is_empty() {
                client.class.clone()
            } else {
                client.initial_class.clone()
            },
            initial_title: client.initial_title.clone(),
        })
    }
}

impl ActiveWindow {
    fn identity(&self) -> Option<WindowIdentity> {
        Some(WindowIdentity {
            address: parse_address(&self.address)?,
            stable_id: self.stable_id.clone(),
            pid: self.pid.filter(|pid| *pid != 0)?,
            initial_class: if self.initial_class.is_empty() {
                self.class.clone()
            } else {
                self.initial_class.clone()
            },
            initial_title: self.initial_title.clone(),
        })
    }
}

#[derive(Clone, Debug)]
struct FocusLease {
    target: WindowIdentity,
    prior: Option<WindowIdentity>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum FocusRestoreDecision {
    Restore(WindowIdentity),
    NoPriorFocus,
    ActiveFocusChanged,
    PriorWindowUnavailable,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum WindowIdentityTrust {
    Trusted(WindowIdentity),
    RebindPending(WindowIdentity),
    Missing(WindowIdentity),
    ConflictingSnapshot,
}

#[derive(Default)]
struct TrustedWindowIdentities {
    by_address: HashMap<u64, WindowIdentityTrust>,
}

impl TrustedWindowIdentities {
    fn observe_snapshot(&mut self, identities: &[WindowIdentity]) {
        let mut snapshot = HashMap::<u64, Option<WindowIdentity>>::new();
        for identity in identities {
            snapshot
                .entry(identity.address)
                .and_modify(|observed| {
                    if observed.as_ref() != Some(identity) {
                        *observed = None;
                    }
                })
                .or_insert_with(|| Some(identity.clone()));
        }

        self.by_address.retain(|address, state| {
            if snapshot.contains_key(address) {
                return true;
            }
            let last_identity = match state {
                WindowIdentityTrust::Trusted(identity)
                | WindowIdentityTrust::RebindPending(identity)
                | WindowIdentityTrust::Missing(identity) => Some(identity.clone()),
                WindowIdentityTrust::ConflictingSnapshot => None,
            };
            if let Some(identity) = last_identity {
                *state = WindowIdentityTrust::Missing(identity);
                true
            } else {
                false
            }
        });
        for (address, observed) in snapshot {
            let Some(identity) = observed else {
                self.by_address
                    .insert(address, WindowIdentityTrust::ConflictingSnapshot);
                continue;
            };
            let next = match self.by_address.get(&address) {
                None => WindowIdentityTrust::Trusted(identity),
                Some(WindowIdentityTrust::Trusted(current)) if current == &identity => {
                    WindowIdentityTrust::Trusted(identity)
                }
                Some(WindowIdentityTrust::RebindPending(current)) if current == &identity => {
                    WindowIdentityTrust::Trusted(identity)
                }
                Some(WindowIdentityTrust::Missing(current)) if current == &identity => {
                    WindowIdentityTrust::Trusted(identity)
                }
                Some(_) => WindowIdentityTrust::RebindPending(identity),
            };
            self.by_address.insert(address, next);
        }
    }

    fn trusted(&self, address: u64, pid: u32) -> Result<WindowIdentity> {
        let identity = match self.by_address.get(&address) {
            Some(WindowIdentityTrust::Trusted(identity)) => identity.clone(),
            Some(WindowIdentityTrust::RebindPending(_)) => {
                bail!(
                    "foreground_unavailable: Hyprland window address 0x{address:x} is pending identity rebind confirmation"
                )
            }
            Some(WindowIdentityTrust::Missing(_)) => {
                bail!(
                    "foreground_unavailable: Hyprland window address 0x{address:x} is not currently live"
                )
            }
            Some(WindowIdentityTrust::ConflictingSnapshot) => {
                bail!(
                    "foreground_unavailable: Hyprland reported conflicting identities for address 0x{address:x}"
                )
            }
            None => {
                bail!(
                    "foreground_unavailable: unknown Hyprland window address 0x{address:x}; call list_windows first"
                )
            }
        };
        anyhow::ensure!(
            identity.pid == pid,
            "foreground_unavailable: Hyprland window 0x{address:x} belongs to pid {}, not pid {pid}",
            identity.pid
        );
        Ok(identity)
    }
}

fn trusted_window_identities() -> &'static Mutex<TrustedWindowIdentities> {
    static IDENTITIES: OnceLock<Mutex<TrustedWindowIdentities>> = OnceLock::new();
    IDENTITIES.get_or_init(|| Mutex::new(TrustedWindowIdentities::default()))
}

#[derive(Clone, Debug, Default, Deserialize)]
struct Monitor {
    #[serde(default, rename = "activeWorkspace")]
    active_workspace: Workspace,
    #[serde(default)]
    x: i32,
    #[serde(default)]
    y: i32,
    #[serde(default)]
    width: u32,
    #[serde(default)]
    height: u32,
    #[serde(default)]
    scale: f64,
    #[serde(default)]
    transform: u8,
}

/// Logical compositor coordinate space accepted by Hyprland virtual pointers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OutputLayout {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

/// Compositor-owned Hyprland metadata for one mapped toplevel.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Window {
    pub address: u64,
    pub pid: u32,
    pub title: String,
    pub app_id: String,
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
    pub workspace: i64,
    pub visible: bool,
}

pub fn is_session() -> bool {
    std::env::var_os("HYPRLAND_INSTANCE_SIGNATURE").is_some()
        && std::env::var("XDG_CURRENT_DESKTOP")
            .ok()
            .is_some_and(|desktop| desktop.to_ascii_lowercase().contains("hyprland"))
}

fn windows_from_clients(clients: &[Client], active_workspaces: &HashSet<i64>) -> Vec<Window> {
    clients
        .iter()
        .filter(|client| client.mapped)
        .filter_map(|client| {
            let address = parse_address(&client.address)?;
            let pid = u32::try_from(client.pid).ok().filter(|pid| *pid != 0)?;
            Some(Window {
                address,
                pid,
                title: client.title.clone(),
                app_id: client.class.clone(),
                x: client.at[0],
                y: client.at[1],
                width: u32::try_from(client.size[0]).unwrap_or_default(),
                height: u32::try_from(client.size[1]).unwrap_or_default(),
                workspace: client.workspace.id,
                visible: !client.hidden && active_workspaces.contains(&client.workspace.id),
            })
        })
        .collect()
}

/// Read all mapped Hyprland clients with compositor-owned process, geometry,
/// workspace, and visibility metadata.
pub fn list_windows() -> Result<Vec<Window>> {
    if !is_session() {
        bail!("not a Hyprland session");
    }
    let monitors: Vec<Monitor> = hyprctl_json("monitors")?;
    let active_workspaces = monitors
        .into_iter()
        .map(|monitor| monitor.active_workspace.id)
        .filter(|workspace| *workspace != 0)
        .collect::<HashSet<_>>();
    let clients = clients()?;
    {
        let mut identities = trusted_window_identities()
            .lock()
            .map_err(|_| anyhow::anyhow!("Hyprland trusted-window registry is poisoned"))?;
        identities.observe_snapshot(&live_identities(&clients));
    }
    Ok(windows_from_clients(&clients, &active_workspaces))
}

/// Return the logical bounding rectangle of all Hyprland outputs. Hyprland's
/// monitor positions and client coordinates are logical, while monitor mode
/// dimensions are physical and must be divided by scale. Virtual-pointer
/// absolute coordinates are normalized across this complete layout, not one
/// arbitrarily selected `wl_output`.
pub fn output_layout() -> Result<OutputLayout> {
    if !is_session() {
        bail!("not a Hyprland session");
    }
    let monitors: Vec<Monitor> = hyprctl_json("monitors")?;
    layout_from_monitors(&monitors).context("Hyprland reported no valid monitor layout")
}

/// Map one physical output capture to Hyprland's logical coordinate size.
/// Returns `None` when dimensions do not identify exactly one output, so callers
/// never guess on mirrored or otherwise ambiguous layouts.
pub fn logical_output_size_for_capture(width: u32, height: u32) -> Option<(u32, u32)> {
    if !is_session() {
        return None;
    }
    let monitors: Vec<Monitor> = hyprctl_json("monitors").ok()?;
    logical_output_size_from_monitors(&monitors, width, height)
}

/// Position the real Hyprland seat cursor in compositor-logical coordinates.
/// Hyprland's compositor dispatcher is authoritative across mixed-scale and
/// multi-monitor layouts; button and axis events still use the standard
/// wlroots virtual-pointer protocol.
pub fn move_cursor(x: i32, y: i32) -> Result<()> {
    if !is_session() {
        bail!("not a Hyprland session");
    }
    let binary = hyprctl_binary();
    let output = Command::new(binary)
        .args(["dispatch", "movecursor", &x.to_string(), &y.to_string()])
        .output()
        .context("launch hyprctl dispatch movecursor")?;
    if !output.status.success() || !output.stdout.starts_with(b"ok") {
        bail!("hyprctl dispatch movecursor failed");
    }
    Ok(())
}

fn monitor_physical_and_logical_size(monitor: &Monitor) -> Option<((u32, u32), (u32, u32))> {
    if monitor.width == 0
        || monitor.height == 0
        || !monitor.scale.is_finite()
        || monitor.scale <= 0.0
    {
        return None;
    }
    let physical = if monitor.transform % 2 == 0 {
        (monitor.width, monitor.height)
    } else {
        (monitor.height, monitor.width)
    };
    let logical = (
        (f64::from(physical.0) / monitor.scale).round() as u32,
        (f64::from(physical.1) / monitor.scale).round() as u32,
    );
    (logical.0 > 0 && logical.1 > 0).then_some((physical, logical))
}

fn logical_output_size_from_monitors(
    monitors: &[Monitor],
    width: u32,
    height: u32,
) -> Option<(u32, u32)> {
    let matches = monitors
        .iter()
        .filter_map(monitor_physical_and_logical_size)
        .filter_map(|(physical, logical)| (physical == (width, height)).then_some(logical))
        .collect::<Vec<_>>();
    (matches.len() == 1).then_some(matches[0])
}

fn layout_from_monitors(monitors: &[Monitor]) -> Option<OutputLayout> {
    let rectangles = monitors.iter().filter_map(|monitor| {
        if monitor.width == 0
            || monitor.height == 0
            || !monitor.scale.is_finite()
            || monitor.scale <= 0.0
        {
            return None;
        }
        let (_, (logical_width, logical_height)) = monitor_physical_and_logical_size(monitor)?;
        let width = i64::from(logical_width);
        let height = i64::from(logical_height);
        (width > 0 && height > 0).then_some((
            i64::from(monitor.x),
            i64::from(monitor.y),
            i64::from(monitor.x) + width,
            i64::from(monitor.y) + height,
        ))
    });

    let (min_x, min_y, max_x, max_y) =
        rectangles.fold(None, |bounds: Option<(i64, i64, i64, i64)>, rect| {
            Some(match bounds {
                None => rect,
                Some((min_x, min_y, max_x, max_y)) => (
                    min_x.min(rect.0),
                    min_y.min(rect.1),
                    max_x.max(rect.2),
                    max_y.max(rect.3),
                ),
            })
        })?;
    Some(OutputLayout {
        x: i32::try_from(min_x).ok()?,
        y: i32::try_from(min_y).ok()?,
        width: u32::try_from(max_x - min_x).ok()?,
        height: u32::try_from(max_y - min_y).ok()?,
    })
}

pub fn window_for_address(address: u64) -> Option<Window> {
    list_windows()
        .ok()?
        .into_iter()
        .find(|window| window.address == address)
}

/// Correlate an accessibility observation to one compositor client. PID is
/// mandatory; title and app-id disambiguate sibling windows, and a sole
/// PID-owned client is the final safe fallback.
pub fn matching_window<'a>(
    windows: &'a [Window],
    pid: u32,
    title: &str,
    app_id: &str,
) -> Option<&'a Window> {
    let owned = windows
        .iter()
        .filter(|window| window.pid == pid)
        .collect::<Vec<_>>();
    let title_matches = owned
        .iter()
        .copied()
        .filter(|window| !title.is_empty() && window.title == title)
        .collect::<Vec<_>>();
    if let [window] = title_matches.as_slice() {
        return Some(*window);
    }
    let app_matches = owned
        .iter()
        .copied()
        .filter(|window| !app_id.is_empty() && window.app_id == app_id)
        .collect::<Vec<_>>();
    if let [window] = app_matches.as_slice() {
        return Some(*window);
    }
    match owned.as_slice() {
        [window] => Some(*window),
        _ => None,
    }
}

pub fn window_for_pid(pid: u32) -> Option<Window> {
    let matching = list_windows()
        .ok()?
        .into_iter()
        .filter(|window| window.pid == pid)
        .collect::<Vec<_>>();
    match matching.as_slice() {
        [window] => Some(window.clone()),
        _ => None,
    }
}

pub fn window_for_title(title: &str) -> Option<Window> {
    let matching = list_windows()
        .ok()?
        .into_iter()
        .filter(|window| !title.is_empty() && window.title == title)
        .collect::<Vec<_>>();
    match matching.as_slice() {
        [window] => Some(window.clone()),
        _ => None,
    }
}

pub fn window_for_app_id(app_id: &str) -> Option<Window> {
    let matching = list_windows()
        .ok()?
        .into_iter()
        .filter(|window| !app_id.is_empty() && window.app_id == app_id)
        .collect::<Vec<_>>();
    match matching.as_slice() {
        [window] => Some(window.clone()),
        _ => None,
    }
}

fn active_window() -> Result<ActiveWindow> {
    hyprctl_json("activewindow")
}

fn live_identities(clients: &[Client]) -> Vec<WindowIdentity> {
    clients
        .iter()
        .filter_map(WindowIdentity::from_client)
        .collect()
}

fn verified_prior_identity(
    active: Option<WindowIdentity>,
    target: &WindowIdentity,
    live: &[WindowIdentity],
) -> Option<WindowIdentity> {
    active.filter(|identity| identity != target && live.iter().any(|item| item == identity))
}

fn focus_restore_decision(
    lease: &FocusLease,
    active: Option<&WindowIdentity>,
    live: &[WindowIdentity],
) -> FocusRestoreDecision {
    if active != Some(&lease.target) {
        return FocusRestoreDecision::ActiveFocusChanged;
    }
    let Some(prior) = lease.prior.as_ref() else {
        return FocusRestoreDecision::NoPriorFocus;
    };
    match live
        .iter()
        .find(|identity| identity.address == prior.address)
    {
        Some(identity) if identity == prior => FocusRestoreDecision::Restore(prior.clone()),
        _ => FocusRestoreDecision::PriorWindowUnavailable,
    }
}

fn target_client(pid: u32, window_id: u64) -> Result<Client> {
    let expected = trusted_window_identities()
        .lock()
        .map_err(|_| anyhow::anyhow!("Hyprland trusted-window registry is poisoned"))?
        .trusted(window_id, pid)?;
    let clients = clients()?;
    let target = clients
        .into_iter()
        .find(|client| {
            WindowIdentity::from_client(client).as_ref() == Some(&expected) && client.mapped
        })
        .ok_or_else(|| {
            anyhow::anyhow!(
                "foreground_unavailable: stale or reused Hyprland address 0x{window_id:x}"
            )
        })?;
    let active_workspaces = hyprctl_json::<Vec<Monitor>>("monitors")?
        .into_iter()
        .map(|monitor| monitor.active_workspace.id)
        .filter(|workspace| *workspace != 0)
        .collect::<HashSet<_>>();
    anyhow::ensure!(
        !target.hidden && active_workspaces.contains(&target.workspace.id),
        "foreground_unavailable: Hyprland target 0x{window_id:x} is not visible on an active workspace"
    );
    Ok(target)
}

fn focus_identity(identity: &WindowIdentity) -> Result<()> {
    let current = live_identities(&clients()?)
        .into_iter()
        .find(|candidate| candidate.address == identity.address);
    anyhow::ensure!(
        current.as_ref() == Some(identity),
        "refusing to focus stale or reused Hyprland address 0x{:x}",
        identity.address
    );

    let selector = format!("address:0x{:x}", identity.address);
    let lua_selector = serde_json::to_string(&selector)?;
    let lua_dispatch = format!("hl.dsp.focus({{ window = {lua_selector} }})");
    let lua_output = Command::new(hyprctl_binary())
        .args(["dispatch", &lua_dispatch])
        .stdin(Stdio::null())
        .output()
        .context("launch Hyprland Lua focus dispatch")?;
    if !lua_output.status.success() || !lua_output.stdout.starts_with(b"ok") {
        let legacy_output = Command::new(hyprctl_binary())
            .args(["dispatch", "focuswindow", &selector])
            .stdin(Stdio::null())
            .output()
            .context("launch legacy Hyprland focuswindow dispatch")?;
        if !legacy_output.status.success() || !legacy_output.stdout.starts_with(b"ok") {
            bail!(
                "Hyprland focus dispatch failed (Lua: {}; legacy: {})",
                command_error(&lua_output),
                command_error(&legacy_output)
            );
        }
    }
    wait_for_focused_identity(identity)
}

fn command_error(output: &std::process::Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let detail = if stderr.trim().is_empty() {
        stdout.trim()
    } else {
        stderr.trim()
    };
    if detail.is_empty() {
        format!("exit status {}", output.status)
    } else {
        detail.to_owned()
    }
}

fn wait_for_focused_identity(identity: &WindowIdentity) -> Result<()> {
    let deadline = Instant::now() + FOCUS_TIMEOUT;
    loop {
        if let Ok(active) = active_window() {
            let active_identity = active.identity();
            if active_identity.as_ref() == Some(identity) {
                return Ok(());
            }
            if active_identity
                .as_ref()
                .is_some_and(|active| active.address == identity.address)
            {
                bail!(
                    "Hyprland address 0x{:x} was reused by a different window while focusing",
                    identity.address
                );
            }
        }
        if Instant::now() >= deadline {
            bail!(
                "Hyprland did not focus window 0x{:x} within 500ms",
                identity.address
            );
        }
        std::thread::sleep(FOCUS_POLL_INTERVAL);
    }
}

fn restore_temporary_focus(lease: &FocusLease) -> Result<FocusRestoreDecision> {
    let active = active_window()?.identity();
    let live = live_identities(&clients()?);
    let decision = focus_restore_decision(lease, active.as_ref(), &live);
    if let FocusRestoreDecision::Restore(prior) = &decision {
        // Revalidate the complete immutable identity immediately before
        // dispatch. Hyprland addresses are allocator-derived and may be reused.
        focus_identity(prior)?;
    }
    Ok(decision)
}

fn begin_temporary_focus(target: Client) -> Result<FocusLease> {
    let target = WindowIdentity::from_client(&target)
        .context("foreground_unavailable: Hyprland target has no valid identity")?;
    let live = live_identities(&clients()?);
    anyhow::ensure!(
        live.iter().any(|identity| identity == &target),
        "foreground_unavailable: Hyprland target changed identity before focus"
    );
    let active = active_window()?;
    let active_identity = active.identity();
    anyhow::ensure!(
        active.address.trim().is_empty() || active_identity.is_some(),
        "foreground_unavailable: Hyprland active window had no verifiable identity"
    );
    anyhow::ensure!(
        active_identity
            .as_ref()
            .is_none_or(|identity| identity == &target || live.iter().any(|item| item == identity)),
        "foreground_unavailable: Hyprland active window identity was not live"
    );
    let prior = verified_prior_identity(active_identity, &target, &live);
    let lease = FocusLease { target, prior };
    if let Err(focus_error) = focus_identity(&lease.target) {
        return match restore_temporary_focus(&lease) {
            Ok(outcome) => Err(focus_error.context(format!(
                "guarded rollback after failed Hyprland focus completed with {outcome:?}"
            ))),
            Err(rollback_error) => Err(focus_error.context(format!(
                "guarded rollback after failed Hyprland focus also failed: {rollback_error}"
            ))),
        };
    }
    Ok(lease)
}

/// Run one global-keyboard transaction under an exact Hyprland focus lease.
/// The prior window is restored only while the target still owns focus, so a
/// user focus takeover is never overwritten.
pub fn with_focused_window<T>(
    pid: u32,
    window_id: u64,
    body: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let lease = begin_temporary_focus(target_client(pid, window_id)?)?;
    let result = body();
    let restore = restore_temporary_focus(&lease);
    match (result, restore) {
        (Ok(value), Ok(_)) => Ok(value),
        (Err(error), Ok(_)) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Err(error), Err(restore_error)) => Err(error.context(format!(
            "the prior Hyprland focus also could not be restored: {restore_error}"
        ))),
    }
}

/// Resolve one exact Hyprland compositor address for a Wayland observation.
/// Ambiguous title/app-id matches fail closed rather than selecting a sibling.
pub fn resolve_capture_address(
    window_id: u64,
    target_pid: Option<u32>,
    title: &str,
    app_id: &str,
) -> Result<u64> {
    if !is_session() {
        bail!("not a Hyprland session");
    }
    let target_pid = target_pid.context("Hyprland capture requires a verified target PID")?;
    resolve_from_clients(&clients()?, window_id, target_pid, title, app_id)
}

fn resolve_from_clients(
    clients: &[Client],
    window_id: u64,
    target_pid: u32,
    title: &str,
    app_id: &str,
) -> Result<u64> {
    let mut owned: Vec<(u64, &Client)> = clients
        .iter()
        .filter(|client| client.mapped && !client.hidden && client.pid == i64::from(target_pid))
        .filter_map(|client| parse_address(&client.address).map(|address| (address, client)))
        .collect();

    if let Some((address, _)) = owned.iter().find(|(address, _)| *address == window_id) {
        return Ok(*address);
    }

    owned.retain(|(_, client)| {
        let title_matches = !title.is_empty() && client.title == title;
        let app_matches = !app_id.is_empty() && client.class == app_id;
        if !title.is_empty() && !app_id.is_empty() {
            title_matches && app_matches
        } else {
            title_matches || app_matches
        }
    });

    match owned.as_slice() {
        [(address, _)] => Ok(*address),
        [] => bail!("no mapped Hyprland client owned by PID {target_pid} matched title/app-id"),
        matches => bail!(
            "Hyprland capture identity is ambiguous: {} PID-owned clients matched title/app-id",
            matches.len()
        ),
    }
}

fn clients() -> Result<Vec<Client>> {
    hyprctl_json("clients")
}

fn hyprctl_binary() -> &'static str {
    if std::path::Path::new("/usr/bin/hyprctl").is_file() {
        "/usr/bin/hyprctl"
    } else {
        "hyprctl"
    }
}

fn hyprctl_json<T: DeserializeOwned>(query: &str) -> Result<T> {
    let output = Command::new(hyprctl_binary())
        .args(["-j", query])
        .output()
        .with_context(|| format!("launch hyprctl -j {query}"))?;
    if !output.status.success() || output.stdout.is_empty() {
        bail!("hyprctl -j {query} failed");
    }
    serde_json::from_slice(&output.stdout).with_context(|| format!("parse hyprctl {query} JSON"))
}

fn parse_address(address: &str) -> Option<u64> {
    u64::from_str_radix(address.trim_start_matches("0x"), 16).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn client(address: &str, pid: i64, title: &str, class: &str) -> Client {
        Client {
            address: address.to_owned(),
            mapped: true,
            hidden: false,
            pid,
            title: title.to_owned(),
            class: class.to_owned(),
            initial_class: class.to_owned(),
            at: [10, 20],
            size: [800, 600],
            workspace: Workspace { id: 1 },
            initial_title: title.to_owned(),
            stable_id: format!("stable-{address}"),
        }
    }

    fn identity(address: u64, stable_id: &str) -> WindowIdentity {
        WindowIdentity {
            address,
            stable_id: stable_id.to_owned(),
            pid: 42,
            initial_class: "fixture".to_owned(),
            initial_title: "Target".to_owned(),
        }
    }

    fn monitor(x: i32, y: i32, width: u32, height: u32, scale: f64) -> Monitor {
        Monitor {
            x,
            y,
            width,
            height,
            scale,
            ..Monitor::default()
        }
    }

    #[test]
    fn parses_full_hyprland_pointer_address() {
        assert_eq!(parse_address("0x55b5cd9af330"), Some(0x55b5cd9af330));
        assert_eq!(parse_address("invalid"), None);
    }

    #[test]
    fn identity_requires_pid_title_and_app_id() {
        let clients = [
            client("0x1111", 42, "Target", "fixture"),
            client("0x2222", 43, "Target", "fixture"),
            client("0x3333", 42, "Other", "fixture"),
        ];
        assert_eq!(
            resolve_from_clients(&clients, 0xff00, 42, "Target", "fixture").unwrap(),
            0x1111
        );
    }

    #[test]
    fn ambiguous_pid_owned_siblings_fail_closed() {
        let clients = [
            client("0x1111", 42, "Target", "fixture"),
            client("0x2222", 42, "Target", "fixture"),
        ];
        let error = resolve_from_clients(&clients, 0xff00, 42, "Target", "fixture")
            .expect_err("duplicate identities must not be guessed");
        assert!(error.to_string().contains("ambiguous"));
    }

    #[test]
    fn exact_hyprland_address_wins_within_verified_pid() {
        let clients = [
            client("0x1111", 42, "First", "fixture"),
            client("0x2222", 42, "Second", "fixture"),
        ];
        assert_eq!(
            resolve_from_clients(&clients, 0x2222, 42, "stale", "stale").unwrap(),
            0x2222
        );
    }

    #[test]
    fn compositor_metadata_preserves_geometry_and_workspace_visibility() {
        let mut visible = client("0x1111", 42, "Target", "fixture");
        visible.at = [-20, 30];
        visible.size = [940, 780];
        visible.workspace.id = 7;
        let mut hidden = client("0x2222", 43, "Hidden", "fixture");
        hidden.workspace.id = 8;

        let windows = windows_from_clients(&[visible, hidden], &HashSet::from([7]));
        assert_eq!(
            windows[0],
            Window {
                address: 0x1111,
                pid: 42,
                title: "Target".to_owned(),
                app_id: "fixture".to_owned(),
                x: -20,
                y: 30,
                width: 940,
                height: 780,
                workspace: 7,
                visible: true,
            }
        );
        assert!(!windows[1].visible);
    }

    #[test]
    fn accessibility_matching_uses_pid_then_unique_title() {
        let windows = windows_from_clients(
            &[
                client("0x1111", 42, "Main", "fixture"),
                client("0x2222", 42, "Child", "fixture"),
                client("0x3333", 43, "Main", "fixture"),
            ],
            &HashSet::from([1]),
        );
        assert_eq!(
            matching_window(&windows, 42, "Main", "fixture").map(|window| window.address),
            Some(0x1111)
        );
        assert!(matching_window(&windows, 42, "Unknown", "fixture").is_none());
        assert!(matching_window(&windows, 44, "Main", "fixture").is_none());
    }

    #[test]
    fn malformed_process_or_size_metadata_fails_closed() {
        let mut invalid_pid = client("0x1111", -1, "Bad", "fixture");
        invalid_pid.size = [800, 600];
        let mut invalid_size = client("0x2222", 42, "Target", "fixture");
        invalid_size.size = [-1, 600];

        let windows = windows_from_clients(&[invalid_pid, invalid_size], &HashSet::from([1]));
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].width, 0);
    }

    #[test]
    fn trusted_identity_registry_trusts_initial_snapshot() {
        let mut registry = TrustedWindowIdentities::default();
        registry.observe_snapshot(&[identity(0x1111, "first")]);
        assert_eq!(registry.trusted(0x1111, 42).unwrap().stable_id, "first");
    }

    #[test]
    fn trusted_identity_registry_guards_reuse_across_disappearance() {
        let mut registry = TrustedWindowIdentities::default();
        registry.observe_snapshot(&[identity(0x1111, "first")]);
        registry.observe_snapshot(&[]);
        let error = registry
            .trusted(0x1111, 42)
            .expect_err("an absent address must remain unavailable");
        assert!(error.to_string().contains("not currently live"));

        registry.observe_snapshot(&[identity(0x1111, "replacement")]);
        let error = registry
            .trusted(0x1111, 42)
            .expect_err("the first replacement after absence must remain untrusted");
        assert!(error
            .to_string()
            .contains("pending identity rebind confirmation"));

        registry.observe_snapshot(&[identity(0x1111, "replacement")]);
        assert_eq!(
            registry.trusted(0x1111, 42).unwrap().stable_id,
            "replacement"
        );
    }

    #[test]
    fn trusted_identity_registry_recovers_same_identity_after_omission() {
        let mut registry = TrustedWindowIdentities::default();
        registry.observe_snapshot(&[identity(0x1111, "first")]);
        registry.observe_snapshot(&[]);
        registry.observe_snapshot(&[identity(0x1111, "first")]);

        assert_eq!(registry.trusted(0x1111, 42).unwrap().stable_id, "first");
    }

    #[test]
    fn trusted_identity_registry_recovers_after_confirmed_rebind() {
        let mut registry = TrustedWindowIdentities::default();
        registry.observe_snapshot(&[identity(0x1111, "first")]);

        registry.observe_snapshot(&[identity(0x1111, "replacement")]);
        let error = registry
            .trusted(0x1111, 42)
            .expect_err("the first recycled-address observation must remain untrusted");
        assert!(error
            .to_string()
            .contains("pending identity rebind confirmation"));

        registry.observe_snapshot(&[identity(0x1111, "replacement")]);
        assert_eq!(
            registry.trusted(0x1111, 42).unwrap().stable_id,
            "replacement"
        );
    }

    #[test]
    fn trusted_identity_registry_restarts_grace_after_repeated_reuse() {
        let mut registry = TrustedWindowIdentities::default();
        registry.observe_snapshot(&[identity(0x1111, "first")]);
        registry.observe_snapshot(&[identity(0x1111, "second")]);
        registry.observe_snapshot(&[identity(0x1111, "third")]);

        let error = registry
            .trusted(0x1111, 42)
            .expect_err("a different identity must restart pending confirmation");
        assert!(error
            .to_string()
            .contains("pending identity rebind confirmation"));

        registry.observe_snapshot(&[identity(0x1111, "third")]);
        assert_eq!(registry.trusted(0x1111, 42).unwrap().stable_id, "third");

        registry.observe_snapshot(&[identity(0x1111, "fourth")]);
        assert!(registry.trusted(0x1111, 42).is_err());
    }

    #[test]
    fn temporary_focus_restore_skips_user_focus_takeover() {
        let target = identity(0x1111, "target");
        let prior = identity(0x2222, "prior");
        let user_target = identity(0x3333, "user");
        let lease = FocusLease {
            target,
            prior: Some(prior.clone()),
        };
        assert_eq!(
            focus_restore_decision(&lease, Some(&user_target), &[prior, user_target.clone()]),
            FocusRestoreDecision::ActiveFocusChanged
        );
    }

    #[test]
    fn temporary_focus_restore_refuses_reused_prior_address() {
        let target = identity(0x1111, "target");
        let prior = identity(0x2222, "prior");
        let replacement = identity(0x2222, "replacement");
        let lease = FocusLease {
            target: target.clone(),
            prior: Some(prior),
        };
        assert_eq!(
            focus_restore_decision(&lease, Some(&target), &[target.clone(), replacement]),
            FocusRestoreDecision::PriorWindowUnavailable
        );
    }

    #[test]
    fn temporary_focus_restore_requires_target_to_still_own_focus() {
        let target = identity(0x1111, "target");
        let prior = identity(0x2222, "prior");
        let lease = FocusLease {
            target: target.clone(),
            prior: Some(prior.clone()),
        };
        assert_eq!(
            focus_restore_decision(&lease, Some(&target), &[target.clone(), prior.clone()]),
            FocusRestoreDecision::Restore(prior)
        );
    }

    #[test]
    fn output_layout_uses_logical_scaled_bounds() {
        let monitors = [
            monitor(384, 288, 1920, 1080, 1.25),
            monitor(1920, 0, 3840, 2160, 1.5),
        ];
        assert_eq!(
            layout_from_monitors(&monitors),
            Some(OutputLayout {
                x: 384,
                y: 0,
                width: 4096,
                height: 1440,
            })
        );
    }

    #[test]
    fn output_layout_translates_negative_and_rotated_outputs() {
        let mut rotated = monitor(-1080, -200, 1920, 1080, 1.0);
        rotated.transform = 1;
        assert_eq!(
            layout_from_monitors(&[rotated, monitor(0, 0, 2560, 1440, 1.0)]),
            Some(OutputLayout {
                x: -1080,
                y: -200,
                width: 3640,
                height: 1920,
            })
        );
    }

    #[test]
    fn physical_capture_size_maps_to_one_logical_output() {
        let monitors = [
            monitor(384, 288, 1920, 1080, 1.25),
            monitor(1920, 0, 3840, 2160, 1.5),
        ];
        assert_eq!(
            logical_output_size_from_monitors(&monitors, 1920, 1080),
            Some((1536, 864))
        );
        assert_eq!(
            logical_output_size_from_monitors(&monitors, 3840, 2160),
            Some((2560, 1440))
        );
        assert_eq!(
            logical_output_size_from_monitors(
                &[
                    monitor(0, 0, 1920, 1080, 1.0),
                    monitor(1920, 0, 1920, 1080, 1.0)
                ],
                1920,
                1080,
            ),
            None
        );
    }
}
