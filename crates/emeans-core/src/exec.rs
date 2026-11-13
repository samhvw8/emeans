//! Helper threads and the clock, on targets with and without threads.
//!
//! Native targets run helper work (the next epoch's permutation, the epoch
//! statistics) on `std::thread`. WebAssembly (`wasm32-unknown-unknown`) has no
//! `std::thread::spawn` and no `std::time::Instant`: there the same work runs
//! inline, at the point where the native code would join, and the clock is
//! `performance.now()` (through `web-time`). The arithmetic is the same either
//! way, so results do not change, only when the work runs.

/// A monotonic clock: `std::time::Instant` natively, `web_time::Instant` on wasm.
#[cfg(not(target_family = "wasm"))]
pub use std::time::Instant;
#[cfg(target_family = "wasm")]
pub use web_time::Instant;

/// True when helper work runs on its own thread.
pub const THREADS: bool = cfg!(not(target_family = "wasm"));

/// Work started by [`spawn`]: a thread, or (no threads) the result already computed.
pub enum Task<T> {
    Thread(std::thread::JoinHandle<T>),
    Done(T),
}

/// Run `f` on a new thread, or inline where the target has no threads.
pub fn spawn<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> Task<T> {
    if THREADS {
        Task::Thread(std::thread::spawn(f))
    } else {
        Task::Done(f())
    }
}

impl<T> Task<T> {
    /// The result; `Err` when the thread panicked.
    pub fn join(self) -> std::thread::Result<T> {
        match self {
            Task::Thread(h) => h.join(),
            Task::Done(v) => Ok(v),
        }
    }
}

/// Work started by [`spawn_scoped`]: a scoped thread, or the result already computed.
pub enum ScopedTask<'s, T> {
    Thread(std::thread::ScopedJoinHandle<'s, T>),
    Done(T),
}

/// [`spawn`] inside a `std::thread::scope` (the scope itself runs on every target).
pub fn spawn_scoped<'s, T: Send + 's>(
    scope: &'s std::thread::Scope<'s, '_>,
    f: impl FnOnce() -> T + Send + 's,
) -> ScopedTask<'s, T> {
    if THREADS {
        ScopedTask::Thread(scope.spawn(f))
    } else {
        ScopedTask::Done(f())
    }
}

impl<T> ScopedTask<'_, T> {
    /// The result; `Err` when the thread panicked.
    pub fn join(self) -> std::thread::Result<T> {
        match self {
            ScopedTask::Thread(h) => h.join(),
            ScopedTask::Done(v) => Ok(v),
        }
    }
}
