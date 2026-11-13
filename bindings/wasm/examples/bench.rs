//! Native side of the wasm benchmark: train on `api::blobs` data, print one JSON
//! line (train time, peak RSS) and write the centroids for the hash check.
//!
//! `cargo run --release --example bench -- <n> <d> <k> <kernel> <seed> <centroids-out>`
//! `kernel`: fused, blocked, gemm (not on macOS) or fastest (`CpuKernel::fastest()`).
//! Threads: rayon's default (all cores) or `RAYON_NUM_THREADS`.

use anyhow::{Context, Result, bail};
use emeans::{Config, CpuAssigner, CpuKernel, Emeans};
use emeans_wasm::api;

/// Peak resident set size of this process in bytes (getrusage; libc is linked by std).
fn peak_rss() -> u64 {
    // struct rusage on 64-bit Linux and macOS: two timevals (32 bytes), then longs;
    // ru_maxrss is the first long. Kilobytes on Linux, bytes on macOS.
    let mut buf = [0i64; 18];
    unsafe extern "C" {
        fn getrusage(who: i32, usage: *mut i64) -> i32;
    }
    // SAFETY: buf is larger than struct rusage (144 bytes on these targets).
    let ok = unsafe { getrusage(0, buf.as_mut_ptr()) } == 0;
    let v = if ok { buf[4].max(0) as u64 } else { 0 };
    if cfg!(target_os = "macos") {
        v
    } else {
        v * 1024
    }
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let [n, d, k, kernel, seed, out] = &a[..] else {
        bail!("usage: bench <n> <d> <k> <kernel> <seed> <centroids-out>");
    };
    let (n, d, k, seed): (usize, usize, usize, u32) =
        (n.parse()?, d.parse()?, k.parse()?, seed.parse()?);
    let kernel = match kernel.as_str() {
        "fused" => CpuKernel::Fused,
        "blocked" => CpuKernel::Blocked,
        "gemm" => CpuKernel::Gemm,
        "fastest" => CpuKernel::fastest(),
        _ => bail!("unknown kernel {kernel}"),
    };
    let x = api::blobs(n, d, 512, seed)?;
    let mut cfg = Config::new(d, k);
    cfg.seed = seed;
    cfg.capture_assignments = false;
    let mut m = Emeans::with_assigner(cfg, Box::new(CpuAssigner::new(kernel)))?;
    let t = std::time::Instant::now();
    let trained = m.train(&x)?;
    let train_ms = t.elapsed().as_secs_f64() * 1e3;
    let bytes: Vec<u8> = trained
        .centroids
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    std::fs::write(out, bytes).with_context(|| format!("write {out}"))?;
    println!(
        "{}",
        serde_json::json!({
            "n": n, "d": d, "k": k, "kernel": kernel.name(),
            "threads": std::env::var("RAYON_NUM_THREADS").ok(),
            "train_ms": train_ms, "peak_rss_bytes": peak_rss(),
            "epochs": m.iteration_stats().len(),
        })
    );
    Ok(())
}
