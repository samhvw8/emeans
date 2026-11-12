//! Train the Rust e-means on a bench/data file and print JSON in the schema of
//! bench/py/ref_emeans.py.
//!
//! cargo run --release -p emeans-bench --bin parity -- --data s768_20k --k 64 \
//!     --nepochs 5 --batch-law auto --seeds 1,2,3 --backend cpu

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use emeans::{Backend, BatchLaw, Config, Emeans, make_assigner};
use serde_json::{Value, json};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../bench/data");
    let data = arg(&args, "--data").context("--data")?;
    let meta: Value = serde_json::from_slice(&std::fs::read(root.join(format!("{data}.json")))?)?;
    let d = meta["d"].as_u64().context("d")? as usize;
    let spherical = match arg(&args, "--spherical").as_deref() {
        Some("yes") => true,
        Some("no") => false,
        _ => meta["spherical"].as_bool().unwrap_or(false),
    };
    let x = emeans::io::read_f32_matrix(root.join(format!("{data}.f32")), d)?;
    let k: usize = arg(&args, "--k").context("--k")?.parse()?;
    let nepochs: f64 = arg(&args, "--nepochs").unwrap_or("5".into()).parse()?;
    let law: BatchLaw = arg(&args, "--batch-law").unwrap_or("auto".into()).parse()?;
    let backend: Backend = emeans_bench::backend(&arg(&args, "--backend").unwrap_or("cpu".into()))?;
    let seeds: Vec<u32> = arg(&args, "--seeds")
        .unwrap_or("1234".into())
        .split(',')
        .map(|s| s.parse())
        .collect::<Result<_, _>>()?;
    let mut runs = Vec::new();
    for seed in seeds {
        let mut cfg = Config::new(d, k);
        cfg.nepochs = nepochs;
        cfg.batch_law = law;
        cfg.seed = seed;
        cfg.spherical = spherical;
        cfg.batch_size = arg(&args, "--batch-size").map(|s| s.parse()).transpose()?;
        cfg.alpha0 = arg(&args, "--alpha0").map(|s| s.parse()).transpose()?;
        cfg.max_points_per_centroid = arg(&args, "--max-points-per-centroid")
            .map(|s| s.parse())
            .transpose()?;
        let mut m = Emeans::with_assigner(cfg, make_assigner(backend)?)?;
        let t = Instant::now();
        m.train(&x)?;
        let secs = t.elapsed().as_secs_f64();
        let fin = m.assign(&x)?.mean_objective(spherical);
        let st = m.iteration_stats();
        runs.push(json!({
            "seed": seed,
            "seconds": secs,
            "final_objective": fin,
            "obj": st.iter().map(|s| s.obj).collect::<Vec<_>>(),
            "imbalance": st.iter().map(|s| s.imbalance_factor).collect::<Vec<_>>(),
            "nsplit": st.iter().map(|s| s.nsplit).collect::<Vec<_>>(),
            "n_splits": m.n_splits(),
            "recipe": m.recipe(),
            "backend": m.assigner_name(),
        }));
        if let Some(prefix) = arg(&args, "--dump-centroids") {
            emeans::io::write_f32(format!("{prefix}.{seed}.f32"), m.centroids().unwrap())?;
        }
    }
    println!(
        "{}",
        json!({"data": data, "k": k, "nepochs": nepochs, "batch_law": format!("{law:?}").to_lowercase(),
               "spherical": spherical, "runs": runs})
    );
    Ok(())
}
