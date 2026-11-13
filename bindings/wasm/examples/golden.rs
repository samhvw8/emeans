//! Native reference outputs for the wasm bit-equality test (`tests/golden.test.mjs`).
//!
//! Writes synthetic inputs, then, for every CPU kernel this machine builds
//! (`fused`, `blocked`, and `gemm` off macOS), the native results of the same
//! `api` calls the wasm build exports, as raw little-endian files under
//! `<dir>/<kernel>/`. `cases.json` holds the host, the kernels and the options
//! of every case. The Node test runs the wasm build on the same inputs and
//! options and compares the outputs byte for byte.
//!
//! `cargo run --release --example golden -- out/golden`

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use emeans::synth::{NestedSpec, Normal, nested, with_bursts};
use emeans_wasm::api;
use serde_json::{Value, json};

struct Out {
    dir: PathBuf,
}

impl Out {
    fn sub(&self, name: &str) -> Result<Out> {
        let dir = self.dir.join(name);
        fs::create_dir_all(&dir)?;
        Ok(Out { dir })
    }
    fn le<T: Copy, const N: usize>(&self, name: &str, v: &[T], f: fn(T) -> [u8; N]) -> Result<()> {
        let b: Vec<u8> = v.iter().flat_map(|&x| f(x)).collect();
        fs::write(self.dir.join(name), b).with_context(|| format!("write {name}"))
    }
    fn f32(&self, name: &str, v: &[f32]) -> Result<()> {
        self.le(name, v, f32::to_le_bytes)
    }
    fn u32(&self, name: &str, v: &[u32]) -> Result<()> {
        self.le(name, v, u32::to_le_bytes)
    }
    fn i32(&self, name: &str, v: &[i32]) -> Result<()> {
        self.le(name, v, i32::to_le_bytes)
    }
    fn f64(&self, name: &str, v: &[f64]) -> Result<()> {
        self.le(name, v, f64::to_le_bytes)
    }
    fn u8(&self, name: &str, v: &[bool]) -> Result<()> {
        fs::write(
            self.dir.join(name),
            v.iter().map(|&b| b as u8).collect::<Vec<_>>(),
        )
        .with_context(|| format!("write {name}"))
    }
    fn bytes(&self, name: &str, b: &[u8]) -> Result<()> {
        fs::write(self.dir.join(name), b).with_context(|| format!("write {name}"))
    }
}

/// A Gaussian mixture in `d` dimensions (for the L2 case).
fn mixture(n: usize, d: usize, centers: usize, spread: f32, seed: u32) -> Vec<f32> {
    let mut rng = Normal::new(seed);
    let c: Vec<f32> = (0..centers * d).map(|_| rng.sample() * 4.0).collect();
    let mut x = Vec::with_capacity(n * d);
    for _ in 0..n {
        let j = (rng.uniform() * centers as f64) as usize % centers;
        for t in 0..d {
            x.push(c[j * d + t] + spread * rng.sample());
        }
    }
    x
}

fn labels(out: &Out, p: &str, l: &emeans::Labels) -> Result<()> {
    out.u32(&format!("{p}.l1.u32"), &l.l1)?;
    out.u32(&format!("{p}.l2.u32"), &l.l2)?;
    out.f32(&format!("{p}.sim1.f32"), &l.sim1)?;
    out.f32(&format!("{p}.sim.f32"), &l.sim)?;
    out.u8(&format!("{p}.outlier.u8"), &l.outlier)?;
    out.u32(&format!("{p}.slot1.u32"), &l.slot1)?;
    out.u32(&format!("{p}.slot2.u32"), &l.slot2)
}

fn ms(t: std::time::Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

/// `opts` with `kernel` set.
fn with_kernel(opts: &Value, kernel: &str) -> Value {
    let mut o = opts.clone();
    o["kernel"] = json!(kernel);
    o
}

fn train_case(out: &Out, name: &str, x: &[f32], d: usize, k: usize, opts: &Value) -> Result<Value> {
    let t = std::time::Instant::now();
    let m = api::train(x, d, k, Some(opts))?;
    let train_ms = ms(t);
    out.f32(&format!("{name}.centroids.f32"), &m.centroids()?)?;
    if let Some(l) = &m.labels {
        out.i32(&format!("{name}.train_labels.i32"), l)?;
    }
    let (p, u) = m.model.ema_state().context("no EMA state")?;
    out.f32(&format!("{name}.positions.f32"), p)?;
    out.f32(&format!("{name}.usage.f32"), u)?;
    let t = std::time::Instant::now();
    let (lab, sc) = m.assign(x)?;
    let assign_ms = ms(t);
    out.u32(&format!("{name}.assign_labels.u32"), &lab)?;
    out.f32(&format!("{name}.assign_scores.f32"), &sc)?;
    out.bytes(&format!("{name}.model.bin"), &m.to_bytes()?)?;
    Ok(json!({"train_ms": train_ms, "assign_ms": assign_ms}))
}

fn main() -> Result<()> {
    let dir: PathBuf = std::env::args()
        .nth(1)
        .unwrap_or("out/golden".into())
        .into();
    fs::create_dir_all(&dir)?;
    let root = Out { dir: dir.clone() };
    let kernels: &[&str] = if cfg!(target_os = "macos") {
        &["fused", "blocked"]
    } else {
        &["fused", "blocked", "gemm"]
    };

    // Inputs.
    let (n, d, k) = (20_000, 32, 64);
    let x_l2 = mixture(n, d, 40, 1.0, 11);
    root.f32("l2.x.f32", &x_l2)?;
    let sph = nested(&NestedSpec {
        n: 12_000,
        d: 64,
        n_broad: 6,
        seed: 5,
        ..Default::default()
    });
    root.f32("sph.x.f32", &sph.x)?;
    let na = nested(&NestedSpec {
        n: 2400,
        d: 48,
        n_broad: 4,
        sub_range: (2, 3),
        min_sub: 60,
        noise_frac: 0.0,
        seed: 31,
        ..Default::default()
    });
    root.f32("auto.x.f32", &na.x)?;
    let base = nested(&NestedSpec {
        n: 3000,
        d: 64,
        seed: 17,
        ..Default::default()
    });
    let wb = with_bursts(&base.x, 64, 150, (2, 6), 0.08, 23);
    root.f32("dedup.x.f32", &wb.x)?;
    root.f64("dedup.created.f64", &wb.created_unix)?;

    let l2_opts = json!({"seed": 7, "nepochs": 3});
    let sph_opts =
        json!({"spherical": true, "nepochs": 2.5, "seed": 3, "maxPointsPerCentroid": 200});
    let n0 = 2000;
    let auto_opts = json!({"k1Range": [2, 8], "k2Max": 6, "restarts": 2, "seed": 9});
    let dedup_opts = json!({"k": 8, "simThreshold": 0.97, "maxGapSecs": 60.0});

    let mut results = serde_json::Map::new();
    for &kernel in kernels {
        let out = root.sub(kernel)?;
        let l2 = train_case(&out, "l2", &x_l2, d, k, &with_kernel(&l2_opts, kernel))?;
        let sp = train_case(
            &out,
            "sph",
            &sph.x,
            sph.d,
            48,
            &with_kernel(&sph_opts, kernel),
        )?;

        // Automatic k, then incremental adds through a byte round trip.
        let t = std::time::Instant::now();
        let (h, l) = api::auto(
            &na.x[..n0 * na.d],
            na.d,
            Some(&with_kernel(&auto_opts, kernel)),
        )?;
        let auto_ms = ms(t);
        labels(&out, "auto.fit", &l)?;
        let blob = h.to_bytes()?;
        let mut h = api::hierarchy_from_bytes(&blob, Some(&json!({"kernel": kernel})))?;
        let r = api::add(&mut h, &na.x, n0)?;
        labels(&out, "auto.add", &r.labels)?;

        // Near-duplicates with bursts and times.
        let side = api::DedupSide {
            created_unix: Some(wb.created_unix.clone()),
            ..Default::default()
        };
        let t = std::time::Instant::now();
        let dd = api::dedup(&wb.x, 64, Some(&with_kernel(&dedup_opts, kernel)), &side)?;
        let dedup_ms = ms(t);
        out.u32("dedup.group_of.u32", &dd.group_of)?;
        out.u32("dedup.representatives.u32", &dd.representatives)?;
        out.f32("dedup.nn_sim.f32", &dd.nn_sim)?;

        results.insert(
            kernel.into(),
            json!({
                "l2": l2, "sph": sp,
                "auto": {"auto_ms": auto_ms, "chosen_k": h.chosen_k(), "action": r.action},
                "dedup": {"dedup_ms": dedup_ms, "linked_pairs": dd.linked_pairs},
            }),
        );
    }

    let meta = json!({
        "host": format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS),
        "kernels": kernels,
        "cases": {
            "l2": {"x": "l2.x.f32", "d": d, "k": k, "options": l2_opts},
            "sph": {"x": "sph.x.f32", "d": sph.d, "k": 48, "options": sph_opts},
            "auto": {"x": "auto.x.f32", "d": na.d, "n0": n0, "options": auto_opts},
            "dedup": {"x": "dedup.x.f32", "created": "dedup.created.f64", "d": 64, "options": dedup_opts},
        },
        "native": results,
    });
    fs::write(dir.join("cases.json"), serde_json::to_vec_pretty(&meta)?)?;
    println!("golden: wrote {} ({})", dir.display(), kernels.join(", "));
    Ok(())
}
