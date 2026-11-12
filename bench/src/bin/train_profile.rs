//! Time whole training runs on one backend, as `bench/compare` does, in one process.
//!
//! cargo run --release -p emeans-bench --bin train_profile -- --data bench/data/s768_200k.f32 --k 1024 --backend metal-fused --reps 3
//!
//! A warm-up engine trains one epoch on the first max(4k, 4096) rows; then a
//! fresh engine uploads the data (`upload`) and trains `--reps` times (`train`,
//! seeds 1..). With `EMEANS_METAL_PROFILE=batch|op` the Metal engine prints GPU
//! busy time and idle gaps when it is dropped (the timed engine's report comes last).

use std::time::Instant;

use anyhow::{Context, Result};
use emeans::{Config, Emeans};

fn arg(args: &[String], name: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
        .unwrap_or_else(|| default.to_string())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let data = arg(&args, "--data", "bench/data/s768_200k.f32");
    let d: usize = arg(&args, "--d", "768").parse()?;
    let k: usize = arg(&args, "--k", "1024").parse()?;
    let reps: u32 = arg(&args, "--reps", "3").parse()?;
    let backend = emeans_bench::backend(&arg(&args, "--backend", "metal-fused"))?;
    let x = std::sync::Arc::new(
        emeans::io::read_f32_matrix(std::path::Path::new(&data), d)
            .with_context(|| format!("read {data}"))?,
    );
    let n = x.len() / d;
    let mut cfg = Config::new(d, k);
    cfg.spherical = true;
    {
        let wr = n.min((4 * k).max(4096));
        let mut wc = cfg.clone();
        wc.nepochs = 1.0;
        let a = emeans::make_assigner(backend)?;
        Emeans::with_assigner(wc, a)?.train(&x[..wr * d])?;
    }
    eprintln!("--- timed engine ---");
    let mut a = emeans::make_assigner(backend)?;
    let t = Instant::now();
    a.bind_shared(x.clone(), d)?;
    println!("upload {:8.3} ms", t.elapsed().as_secs_f64() * 1e3);
    let shared = emeans::SharedAssigner::new(a);
    for seed in 1..=reps {
        let mut c = cfg.clone();
        c.seed = seed;
        let mut m = Emeans::with_assigner(c, Box::new(shared.clone()))?;
        let t = Instant::now();
        m.train(&x)?;
        let st = m.iteration_stats();
        println!(
            "train  {:8.3} ms seed {seed} steps {} obj {:.9e}",
            t.elapsed().as_secs_f64() * 1e3,
            m.recipe().unwrap().total_steps,
            st.last().map(|s| s.obj).unwrap_or(f64::NAN)
        );
    }
    Ok(())
}
