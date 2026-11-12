//! Time one full assignment pass (all n rows, data bound on the device, as in
//! training) for every backend, and check labels against the Accelerate CPU
//! kernel. A mismatch counts as a tie when the two centroids' exact (f64)
//! scores differ by at most 1e-5 (relative for L2).
//!
//! cargo run --release -p emeans-bench --bin bench_assign -- --data s768_17k --ks 64,256,1024 \
//!     [--backends cpu-accelerate,cpu-blocked,metal-fused] (add cpu-candle-gemm, metal-unfused with --features baselines) [--reps 3] [--metric ip|l2]

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use emeans::rng::RandomState;
use emeans::{Backend, Metric, Rows, make_assigner};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn exact(x: &[f32], c: &[f32], metric: Metric) -> f64 {
    match metric {
        Metric::InnerProduct => x.iter().zip(c).map(|(a, b)| *a as f64 * *b as f64).sum(),
        Metric::L2 => x
            .iter()
            .zip(c)
            .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
            .sum(),
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../bench/data");
    let data = arg(&args, "--data").context("--data")?;
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join(format!("{data}.json")))?)?;
    let d = meta["d"].as_u64().context("d")? as usize;
    let x = emeans::io::read_f32_matrix(root.join(format!("{data}.f32")), d)?;
    let n = x.len() / d;
    let metric = match arg(&args, "--metric").as_deref() {
        Some("l2") => Metric::L2,
        _ => Metric::InnerProduct,
    };
    let reps: usize = arg(&args, "--reps").unwrap_or("3".into()).parse()?;
    let ks: Vec<usize> = arg(&args, "--ks")
        .unwrap_or("64,256,1024".into())
        .split(',')
        .map(|s| s.parse())
        .collect::<Result<_, _>>()?;
    let backends: Vec<String> = arg(&args, "--backends")
        .unwrap_or("cpu-accelerate,cpu-blocked,metal-fused".into())
        .split(',')
        .map(String::from)
        .collect();
    let ids: Vec<u32> = (0..n as u32).collect();
    let rows = Rows::Indexed {
        data: &x,
        ids: &ids,
    };
    let chunk = 65536;
    println!(
        "| n | d | k | backend | ms (median of {reps}) | GFLOP/s | label mismatches | non-tie mismatches |"
    );
    println!("|---|---|---|---|---|---|---|---|");
    for &k in &ks {
        let perm = RandomState::new(42).permutation(n);
        let mut c = vec![0f32; k * d];
        for (dst, &i) in c.chunks_mut(d).zip(&perm[..k]) {
            dst.copy_from_slice(&x[i as usize * d..(i as usize + 1) * d]);
        }
        let mut reference: Option<Vec<u32>> = None;
        for name in &backends {
            let backend: Backend = emeans_bench::backend(name)?;
            let mut a = make_assigner(backend)?;
            a.bind_data(&x, d)?;
            let run = |a: &mut Box<dyn emeans::Assigner>| -> Result<(Vec<u32>, f64)> {
                let t = Instant::now();
                let (l, _) = emeans::assign::assign_all(a.as_mut(), rows, d, &c, metric, chunk)?;
                Ok((l, t.elapsed().as_secs_f64() * 1e3))
            };
            let (labels, _) = run(&mut a)?; // warm-up (shader compile, uploads)
            let mut times = Vec::new();
            for _ in 0..reps {
                times.push(run(&mut a)?.1);
            }
            a.unbind_data();
            times.sort_by(f64::total_cmp);
            let ms = times[times.len() / 2];
            let gflops = 2.0 * n as f64 * k as f64 * d as f64 / (ms * 1e6);
            let (mut mism, mut nontie) = (0usize, 0usize);
            if let Some(r) = &reference {
                for i in 0..n {
                    if labels[i] != r[i] {
                        mism += 1;
                        let xi = &x[i * d..(i + 1) * d];
                        let a = exact(
                            xi,
                            &c[labels[i] as usize * d..(labels[i] as usize + 1) * d],
                            metric,
                        );
                        let b = exact(xi, &c[r[i] as usize * d..(r[i] as usize + 1) * d], metric);
                        let tol = match metric {
                            Metric::InnerProduct => 1e-5,
                            Metric::L2 => 1e-5 * b.abs().max(1.0),
                        };
                        if (a - b).abs() > tol {
                            nontie += 1;
                        }
                    }
                }
            } else {
                reference = Some(labels);
            }
            println!("| {n} | {d} | {k} | {name} | {ms:.1} | {gflops:.0} | {mism} | {nontie} |");
        }
    }
    Ok(())
}
