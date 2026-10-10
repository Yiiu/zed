//! External colour textures (`PaintSurface::color_texture`) on D3D11: the
//! instance layout, retirement of the textures' producer leases after the GPU
//! has consumed them, and presentation observations for frame correlation.
//!
//! Retirement is the D3D11 counterpart of the Metal renderer's command-buffer
//! completion handler. After each present that sampled colour textures the
//! renderer signals a fence on the immediate context; a waiter thread blocks
//! on the fence's event and only then calls each texture's
//! consumer-completion observer and drops the frame's `Arc<ColorTexture>`s,
//! so a lease outlives the scene, the window and later frames until the GPU
//! is done with it, and is released even when no further frame is drawn.
//! Without fence support (no D3D11.4 runtime or driver), an event query
//! polled on later draws does the same, later.

use std::{
    collections::VecDeque,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::JoinHandle,
};

use gpui::{Bounds, ColorTexture, ScaledPixels, TexturePresentationObserver};
use windows::{
    Win32::{
        Foundation::{CloseHandle, HANDLE, HWND},
        Graphics::{
            Direct3D11::*,
            Dwm::{DWM_TIMING_INFO, DwmGetCompositionTimingInfo},
            Dxgi::*,
        },
        System::{
            Performance::{QueryPerformanceCounter, QueryPerformanceFrequency},
            Threading::{CreateEventW, WaitForSingleObject},
        },
    },
    core::Interface,
};

/// One external colour surface, as `color_surface_vertex` reads it (t1).
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct ColorSurfaceInstance {
    pub(crate) bounds: Bounds<ScaledPixels>,
    pub(crate) content_mask: Bounds<ScaledPixels>,
}

type Batch = (u64, Vec<Arc<ColorTexture>>);

#[derive(Default)]
struct Queue {
    batches: Mutex<VecDeque<Batch>>,
    ready: Condvar,
    stop: AtomicBool,
}

fn retire(textures: Vec<Arc<ColorTexture>>) {
    for texture in &textures {
        texture.notify_consumed();
    }
}

/// Holds sampled colour textures until the GPU has consumed them.
pub(crate) struct ColorRetirement {
    signal: Option<(ID3D11Fence, ID3D11DeviceContext4)>,
    next: u64,
    queue: Arc<Queue>,
    waiter: Option<JoinHandle<()>>,
    queries: VecDeque<(ID3D11Query, Vec<Arc<ColorTexture>>)>,
}

impl ColorRetirement {
    pub(crate) fn new(device: &ID3D11Device, context: &ID3D11DeviceContext) -> Self {
        let queue = Arc::new(Queue::default());
        let signal = Self::fence(device, context);
        let waiter = signal.as_ref().and_then(|(fence, _)| {
            let fence = fence.clone();
            let queue = queue.clone();
            std::thread::Builder::new()
                .name("gpui-color-retire".into())
                .spawn(move || wait_and_retire(fence, queue))
                .ok()
        });
        Self {
            signal: if waiter.is_some() { signal } else { None },
            next: 0,
            queue,
            waiter,
            queries: VecDeque::new(),
        }
    }

    fn fence(
        device: &ID3D11Device,
        context: &ID3D11DeviceContext,
    ) -> Option<(ID3D11Fence, ID3D11DeviceContext4)> {
        let device5: ID3D11Device5 = device.cast().ok()?;
        let context4: ID3D11DeviceContext4 = context.cast().ok()?;
        let mut fence: Option<ID3D11Fence> = None;
        // SAFETY: valid device; out pointer to a local.
        unsafe { device5.CreateFence(0, D3D11_FENCE_FLAG_NONE, &mut fence) }.ok()?;
        Some((fence?, context4))
    }

    /// Whether completion uses a fence and a waiter thread.
    pub(crate) fn uses_fence(&self) -> bool {
        self.signal.is_some()
    }

    /// Hands over the textures a just-submitted frame sampled. UI thread,
    /// immediate context.
    pub(crate) fn submit(
        &mut self,
        device: &ID3D11Device,
        context: &ID3D11DeviceContext,
        textures: Vec<Arc<ColorTexture>>,
    ) {
        if textures.is_empty() {
            return;
        }
        if let Some((fence, context4)) = &self.signal {
            self.next += 1;
            // SAFETY: immediate context of the fence's device, on the UI thread.
            if unsafe { context4.Signal(fence, self.next) }.is_ok() {
                if let Ok(mut batches) = self.queue.batches.lock() {
                    batches.push_back((self.next, textures));
                    self.queue.ready.notify_one();
                    return;
                }
            }
            // A failed signal (device removed) has nothing left to wait for.
            retire(textures);
            return;
        }
        let description = D3D11_QUERY_DESC {
            Query: D3D11_QUERY_EVENT,
            MiscFlags: 0,
        };
        let mut query = None;
        // SAFETY: valid device and context; the query is ended right away.
        let created = unsafe { device.CreateQuery(&description, Some(&mut query)) };
        match (created, query) {
            (Ok(()), Some(query)) => {
                unsafe { context.End(&query) };
                self.queries.push_back((query, textures));
            }
            // No way to observe completion: a full flush is the conservative
            // stand-in, after which the GPU work is at least submitted.
            _ => {
                // SAFETY: the immediate context, on the UI thread.
                unsafe { context.Flush() };
                retire(textures);
            }
        }
    }

    /// Fallback path only: retires textures whose event query completed.
    pub(crate) fn poll(&mut self, context: &ID3D11DeviceContext) {
        while let Some((query, _)) = self.queries.front() {
            let mut done = 0_u32;
            // SAFETY: an ended event query of this context; DONOTFLUSH never blocks.
            let result = unsafe {
                context.GetData(
                    query,
                    Some((&mut done as *mut u32).cast()),
                    std::mem::size_of::<u32>() as u32,
                    D3D11_ASYNC_GETDATA_DONOTFLUSH.0 as u32,
                )
            };
            if result.is_err() || done == 0 {
                return;
            }
            if let Some((_, textures)) = self.queries.pop_front() {
                retire(textures);
            }
        }
    }

    /// Device lost: the old device's work will not complete; release every
    /// lease without a completion notice.
    pub(crate) fn clear(&mut self) {
        if let Ok(mut batches) = self.queue.batches.lock() {
            batches.clear();
        }
        self.queries.clear();
    }
}

impl Drop for ColorRetirement {
    fn drop(&mut self) {
        self.queue.stop.store(true, Ordering::Release);
        self.queue.ready.notify_all();
        if let Some(waiter) = self.waiter.take() {
            // The waiter wakes at least every 100 ms.
            let _ = waiter.join();
        }
    }
}

fn wait_and_retire(fence: ID3D11Fence, queue: Arc<Queue>) {
    // SAFETY: an unnamed auto-reset event, closed below.
    let Ok(event) = (unsafe { CreateEventW(None, false, false, None) }) else {
        return;
    };
    let _close = CloseOnDrop(event);
    loop {
        let value = {
            let Ok(mut batches) = queue.batches.lock() else {
                return;
            };
            loop {
                if queue.stop.load(Ordering::Acquire) {
                    return;
                }
                if let Some((value, _)) = batches.front() {
                    break *value;
                }
                batches = match queue.ready.wait(batches) {
                    Ok(batches) => batches,
                    Err(_) => return,
                };
            }
        };
        // SAFETY: fence methods are free-threaded; the event outlives the wait.
        let completed = unsafe {
            fence.GetCompletedValue() >= value
                || (fence.SetEventOnCompletion(value, event).is_ok()
                    && WaitForSingleObject(event, 100).0 == 0
                    && fence.GetCompletedValue() >= value)
        };
        if !completed {
            continue;
        }
        let batch = queue.batches.lock().ok().and_then(|mut batches| {
            (batches.front().map(|(front, _)| *front) == Some(value))
                .then(|| batches.pop_front())
                .flatten()
        });
        if let Some((_, textures)) = batch {
            retire(textures);
        }
    }
}

pub(crate) struct CloseOnDrop(pub(crate) HANDLE);

impl Drop for CloseOnDrop {
    fn drop(&mut self) {
        // SAFETY: owned event handle.
        let _ = unsafe { CloseHandle(self.0) };
    }
}

/// Reports each present that sampled colour textures, with the time it was
/// shown (QPC seconds) and how that was learned, to the textures' observers.
///
/// Three sources, chosen per present by [`verdict`]:
/// - DXGI frame statistics give the exact present that was displayed, but
///   only while their `PresentCount` advances (a DirectComposition swap
///   chain composed by DWM returns zeros; indirect / remote displays may
///   return a count that stops);
/// - otherwise DWM's composition timing: on Windows 11 `qpcCompose` is the
///   *next* composition, so a present made before it is shown there, but
///   only while that time is plausible (not stale in the past, not far
///   ahead; remote and virtual displays have been seen to freeze it);
/// - otherwise an estimate: present time plus one refresh period.
///
/// Each present is reported exactly once, so frame-ID counting stays
/// honest; the source is recorded per texture so diagnostics can tell an
/// estimate from a measurement.
pub(crate) struct Presentations {
    pending: VecDeque<Pending>,
    frequency: f64,
    /// The last statistics count seen and when it last changed (QPC).
    statistics: Option<(u32, i64)>,
}

type Observers = Vec<(
    Arc<dyn TexturePresentationObserver>,
    Option<Arc<std::sync::atomic::AtomicU8>>,
)>;

struct Pending {
    count: u32,
    presented_qpc: i64,
    observers: Observers,
}

/// Unreported presents kept at most; older ones are dropped unreported.
const PENDING_PRESENTS: usize = 64;

/// A source that has not advanced for this long no longer counts (seconds).
const STALE_SECONDS: f64 = 0.25;

/// What the display timing says about one pending present.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Verdict {
    /// Shown at this QPC time, learned from this source.
    Shown(i64, gpui::PresentationSource),
    /// Never displayed: DXGI already reports a later present (DWM replaced
    /// it within one refresh). Not reported, so frame counts stay honest.
    Superseded,
    /// A live source will report it later.
    Wait,
}

/// One reading of the display timing (all times QPC ticks).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Timing {
    pub(crate) now: i64,
    /// `PresentCount` / `SyncQPCTime`, when non-zero and still advancing.
    pub(crate) statistics: Option<(u32, i64)>,
    /// DWM's `qpcCompose`, when the call succeeded.
    pub(crate) compose: Option<i64>,
    /// One refresh period.
    pub(crate) period: i64,
    /// [`STALE_SECONDS`] in ticks.
    pub(crate) stale: i64,
}

/// Decides how present `count` made at `presented` is reported.
pub(crate) fn verdict(count: u32, presented: i64, timing: Timing) -> Verdict {
    use gpui::PresentationSource::*;
    if let Some((shown, sync)) = timing.statistics {
        // The statistics name the one present on screen. An earlier present
        // that was never named was dropped by the compositor. (A reading
        // missed between two polls under-counts, the safe direction.)
        if count == shown {
            return Verdict::Shown(sync, Statistics);
        }
        if count < shown {
            return Verdict::Superseded;
        }
        if timing.now - presented < timing.stale {
            return Verdict::Wait;
        }
    }
    if let Some(compose) = timing.compose {
        // A prediction: at most half a period in the past, a few ahead.
        let plausible = compose >= timing.now - timing.period / 2
            && compose - timing.now <= 4 * timing.period;
        if plausible {
            if presented < compose {
                return Verdict::Shown(compose, Composition);
            }
            if timing.now - presented < timing.stale {
                return Verdict::Wait;
            }
        }
    }
    Verdict::Shown(presented + timing.period, Estimated)
}

impl Presentations {
    pub(crate) fn new() -> Self {
        let mut frequency = 0_i64;
        // SAFETY: out pointer to a local.
        let _ = unsafe { QueryPerformanceFrequency(&mut frequency) };
        Self {
            pending: VecDeque::new(),
            frequency: frequency.max(1) as f64,
            statistics: None,
        }
    }

    /// Records the present just made for the observers of its textures and
    /// reports it at once if no live source will.
    pub(crate) fn presented(&mut self, swap_chain: &IDXGISwapChain1, observers: Observers) {
        if observers.is_empty() {
            return;
        }
        let mut now = 0_i64;
        // SAFETY: out pointer to a local.
        let _ = unsafe { QueryPerformanceCounter(&mut now) };
        // SAFETY: the renderer's own swap chain, UI thread.
        let count = unsafe { swap_chain.GetLastPresentCount() }.unwrap_or(0);
        self.pending.push_back(Pending {
            count,
            presented_qpc: now,
            observers,
        });
        while self.pending.len() > PENDING_PRESENTS {
            self.pending.pop_front();
        }
        // Report now whatever no live source will report later: a window
        // that goes idle gets no further WM_PAINT polls.
        self.poll(swap_chain);
    }

    /// Reports every recorded present the display timing accounts for.
    /// Called each vsync tick (`WM_PAINT`), drawn or not.
    pub(crate) fn poll(&mut self, swap_chain: &IDXGISwapChain1) {
        if self.pending.is_empty() {
            return;
        }
        let mut now = 0_i64;
        // SAFETY: out pointer to a local.
        let _ = unsafe { QueryPerformanceCounter(&mut now) };
        let stale = (STALE_SECONDS * self.frequency) as i64;
        let mut statistics = DXGI_FRAME_STATISTICS::default();
        // SAFETY: the renderer's own swap chain, UI thread; out pointer to a local.
        let read = unsafe { swap_chain.GetFrameStatistics(&mut statistics) }.is_ok()
            && statistics.PresentCount != 0;
        if read {
            let changed = self
                .statistics
                .is_none_or(|(count, _)| count != statistics.PresentCount);
            if changed {
                self.statistics = Some((statistics.PresentCount, now));
            }
        }
        let advancing = read
            && self
                .statistics
                .is_some_and(|(_, changed_at)| now - changed_at < stale);
        let mut timing = DWM_TIMING_INFO {
            cbSize: std::mem::size_of::<DWM_TIMING_INFO>() as u32,
            ..Default::default()
        };
        // SAFETY: a null window asks for the desktop composition timing.
        let dwm = unsafe { DwmGetCompositionTimingInfo(HWND::default(), &mut timing) }.is_ok();
        // A refresh period between 2 and 100 ms, else 60 Hz.
        let refresh = timing.qpcRefreshPeriod as i64;
        let period = if dwm && refresh * 500 >= self.frequency as i64 && refresh * 10 <= self.frequency as i64 {
            refresh
        } else {
            (self.frequency / 60.0) as i64
        };
        let reading = Timing {
            now,
            statistics: advancing.then_some((statistics.PresentCount, statistics.SyncQPCTime)),
            compose: (dwm && timing.qpcCompose != 0).then_some(timing.qpcCompose as i64),
            period,
            stale,
        };
        while let Some(front) = self.pending.front() {
            let (shown, source) = match verdict(front.count, front.presented_qpc, reading) {
                Verdict::Shown(shown, source) => (shown, source),
                Verdict::Superseded => {
                    self.pending.pop_front();
                    continue;
                }
                Verdict::Wait => break,
            };
            let seconds = shown as f64 / self.frequency;
            if let Some(entry) = self.pending.pop_front() {
                for (observer, cell) in entry.observers {
                    if let Some(cell) = cell {
                        cell.store(source as u8, Ordering::Release);
                    }
                    observer.presented(seconds);
                }
            }
        }
    }

    pub(crate) fn clear(&mut self) {
        self.pending.clear();
        self.statistics = None;
    }
}

#[cfg(test)]
mod presentation_tests {
    use super::*;
    use gpui::PresentationSource::*;

    const MS: i64 = 10_000; // 10 MHz QPC: ticks per millisecond

    fn timing(now: i64) -> Timing {
        Timing {
            now,
            statistics: None,
            compose: None,
            period: 7 * MS,
            stale: 250 * MS,
        }
    }

    #[test]
    fn advancing_statistics_report_the_displayed_present() {
        let at = Timing {
            statistics: Some((10, 1_000 * MS)),
            ..timing(1_001 * MS)
        };
        assert_eq!(verdict(10, 990 * MS, at), Verdict::Shown(1_000 * MS, Statistics));
        assert_eq!(verdict(11, 995 * MS, at), Verdict::Wait);
    }

    #[test]
    fn a_present_the_statistics_skipped_was_dropped_not_shown() {
        // PresentMon saw ~19 % of composed presents dropped at 155 Hz; the
        // earlier rule counted them as shown.
        let at = Timing {
            statistics: Some((10, 1_000 * MS)),
            ..timing(1_001 * MS)
        };
        assert_eq!(verdict(9, 985 * MS, at), Verdict::Superseded);
    }

    #[test]
    fn a_plausible_composition_after_the_present_reports_it() {
        let at = Timing {
            compose: Some(1_004 * MS),
            ..timing(1_001 * MS)
        };
        assert_eq!(verdict(3, 1_000 * MS, at), Verdict::Shown(1_004 * MS, Composition));
    }

    #[test]
    fn a_present_after_the_next_composition_waits_for_the_following_one() {
        let at = Timing {
            compose: Some(1_004 * MS),
            ..timing(1_003 * MS)
        };
        assert_eq!(verdict(3, 1_004 * MS + 1, at), Verdict::Wait);
    }

    #[test]
    fn a_frozen_composition_in_the_past_falls_back_to_an_estimate() {
        // Seen on a remote / virtual display: qpcCompose stopped long ago.
        let at = Timing {
            compose: Some(500 * MS),
            ..timing(1_001 * MS)
        };
        assert_eq!(verdict(3, 1_000 * MS, at), Verdict::Shown(1_007 * MS, Estimated));
    }

    #[test]
    fn no_source_at_all_estimates_at_once() {
        assert_eq!(
            verdict(0, 1_000 * MS, timing(1_000 * MS)),
            Verdict::Shown(1_007 * MS, Estimated)
        );
    }

    #[test]
    fn a_waiting_present_is_estimated_once_its_source_goes_stale() {
        let at = Timing {
            statistics: Some((10, 700 * MS)),
            ..timing(1_000 * MS)
        };
        assert_eq!(verdict(11, 740 * MS, at), Verdict::Shown(747 * MS, Estimated));
    }

    #[test]
    fn a_far_future_composition_is_not_trusted() {
        let at = Timing {
            compose: Some(2_000 * MS),
            ..timing(1_000 * MS)
        };
        assert_eq!(verdict(3, 999 * MS, at), Verdict::Shown(1_006 * MS, Estimated));
    }
}


/// How the renderer presents (experimental, `GPUI_PRESENT_MODE`): today's
/// `Present(0, 0)` (`interval0`, default), sync interval 1 (`interval1`), or a
/// frame-latency waitable swap chain (maximum latency 1) waited on before each
/// frame, with sync interval 1 (`waitable`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PresentMode {
    Interval0,
    Interval1,
    Waitable,
}

pub(crate) fn present_mode() -> PresentMode {
    static MODE: std::sync::LazyLock<PresentMode> = std::sync::LazyLock::new(|| {
        match std::env::var("GPUI_PRESENT_MODE").as_deref() {
            Ok("interval1") => PresentMode::Interval1,
            Ok("waitable") => PresentMode::Waitable,
            _ => PresentMode::Interval0,
        }
    });
    *MODE
}

/// Opt-in present / display log (`GPUI_PRESENT_DIAGNOSTICS=<csv>`), appended
/// by every window: `p,<chain>,<present count>,<qpc>` for each present and
/// `d,<chain>,<present count>,<sync qpc>,<sync refresh>` for each present
/// DXGI statistics report as displayed. Offline analysis gives the share of
/// presents never displayed and the present-to-display latency.
pub(crate) struct PresentLog {
    out: Option<std::io::BufWriter<std::fs::File>>,
    last_displayed: u32,
    lines: usize,
}

impl PresentLog {
    pub(crate) fn new() -> Self {
        let out = std::env::var_os("GPUI_PRESENT_DIAGNOSTICS").and_then(|path| {
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .ok()
                .map(std::io::BufWriter::new)
        });
        Self {
            out,
            last_displayed: 0,
            lines: 0,
        }
    }

    fn write(&mut self, line: std::fmt::Arguments) {
        use std::io::Write;
        if let Some(out) = self.out.as_mut() {
            let _ = writeln!(out, "{line}");
            self.lines += 1;
            if self.lines % 256 == 0 {
                let _ = out.flush();
            }
        }
    }

    /// Records the present just made.
    pub(crate) fn presented(&mut self, swap_chain: &IDXGISwapChain1) {
        if self.out.is_none() {
            return;
        }
        self.sample(swap_chain);
        let mut now = 0_i64;
        // SAFETY: out pointer to a local.
        let _ = unsafe { QueryPerformanceCounter(&mut now) };
        // SAFETY: the renderer's own swap chain, UI thread.
        let count = unsafe { swap_chain.GetLastPresentCount() }.unwrap_or(0);
        let chain = swap_chain.as_raw() as usize;
        self.write(format_args!("p,{chain:x},{count},{now}"));
    }

    /// Records the displayed present DXGI now reports, once per change.
    pub(crate) fn sample(&mut self, swap_chain: &IDXGISwapChain1) {
        if self.out.is_none() {
            return;
        }
        let mut statistics = DXGI_FRAME_STATISTICS::default();
        // SAFETY: the renderer's own swap chain, UI thread; out pointer to a local.
        if unsafe { swap_chain.GetFrameStatistics(&mut statistics) }.is_ok()
            && statistics.PresentCount != 0
            && statistics.PresentCount != self.last_displayed
        {
            self.last_displayed = statistics.PresentCount;
            let chain = swap_chain.as_raw() as usize;
            self.write(format_args!(
                "d,{chain:x},{},{},{}",
                statistics.PresentCount, statistics.SyncQPCTime, statistics.SyncRefreshCount
            ));
        }
    }
}

impl Drop for PresentLog {
    fn drop(&mut self) {
        use std::io::Write;
        if let Some(out) = self.out.as_mut() {
            let _ = out.flush();
        }
    }
}
