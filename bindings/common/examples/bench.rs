//! Native Rust side of the binding benchmark (`bindings/bench/run.py`): the
//! facade called directly, timed the same way as the language benches.
//!
//! Usage: `bench <data.f32> <d> <k> <backend> <mode>`; prints one JSON line.
//! Mode `full` times train and assign; mode `call` times only the per-call
//! probe (the driver runs it with `RAYON_NUM_THREADS=1`).

use std::io::Read;
use std::time::Instant;

use anyhow::{Context, Result};
use emeans::{Config, Emeans, Metric, Rows, make_assigner};

/// FNV-1a 64 of the bytes of `v` (the same hash in every language bench).
fn fnv<T: Copy>(v: &[T]) -> String {
    // SAFETY: plain numeric data, read as bytes.
    let b =
        unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) };
    let mut h: u64 = 0xcbf29ce484222325;
    for &x in b {
        h ^= x as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let (path, d, k, backend, mode) = (
        &a[1],
        a[2].parse::<usize>()?,
        a[3].parse::<usize>()?,
        a[4].as_str(),
        a[5].as_str(),
    );
    // Read straight into the f32 buffer: one copy of the data, as in the other languages.
    let mut f = std::fs::File::open(path).with_context(|| path.clone())?;
    let mut x = vec![0f32; f.metadata()?.len() as usize / 4];
    // SAFETY: f32 has no invalid bit patterns; the file is little-endian f32.
    f.read_exact(unsafe {
        std::slice::from_raw_parts_mut(x.as_mut_ptr().cast::<u8>(), x.len() * 4)
    })?;
    let n = x.len() / d;
    if mode == "call" {
        println!(
            "{}",
            serde_json::json!({"lang": "rust", "call_us": call_us(&x[..d], &x[64 * d..128 * d], d)?})
        );
        return Ok(());
    }

    let mut cfg = Config::new(d, k);
    cfg.spherical = true;
    let t = Instant::now();
    let mut model = Emeans::with_assigner(cfg, make_assigner(backend.parse()?)?)?;
    let trained = model.train(&x)?;
    let train_s = t.elapsed().as_secs_f64();
    // Free the training engine (and its device copy of the data) before
    // assign, as the bindings do when `train` returns.
    let backend_name = model.assigner_name();
    drop(model);

    let t = Instant::now();
    let mut asg = make_assigner(backend.parse()?)?;
    let (labels, scores) = emeans::assign::assign_all(
        asg.as_mut(),
        Rows::Dense(&x),
        d,
        &trained.centroids,
        Metric::InnerProduct,
        emeans::recipe::DEFAULT_ASSIGNMENT_CHUNK,
    )?;
    let assign_s = t.elapsed().as_secs_f64();
    let objective = scores.iter().map(|&s| 2.0 - 2.0 * s as f64).sum::<f64>() / n as f64;

    println!(
        "{}",
        serde_json::json!({
            "lang": "rust", "backend": backend_name, "n": n, "d": d, "k": k,
            "train_s": train_s, "assign_s": assign_s, "objective": objective,
            "centroids_fnv": fnv(&trained.centroids), "labels_fnv": fnv(&labels),
        })
    );
    Ok(())
}

/// Mean time of one assign call (one row, 64 centroids, `cpu`), like the
/// bindings' probe: a new assigner per call, as `api::assign` makes one.
fn call_us(row: &[f32], cents: &[f32], d: usize) -> Result<f64> {
    let reps = 2000;
    let t = Instant::now();
    for _ in 0..reps {
        let mut cpu = make_assigner("cpu".parse()?)?;
        let r = emeans::assign::assign_all(
            cpu.as_mut(),
            Rows::Dense(row),
            d,
            cents,
            Metric::InnerProduct,
            emeans::recipe::DEFAULT_ASSIGNMENT_CHUNK,
        )?;
        std::hint::black_box(r);
    }
    Ok(t.elapsed().as_secs_f64() * 1e6 / reps as f64)
}
