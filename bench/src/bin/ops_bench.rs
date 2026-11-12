//! Time each layer-1 op on one backend, one training batch at a time.
//!
//! cargo run --release -p emeans-bench --bin ops_bench -- --backend metal --n 200000 --d 768 --k 1024 --batch 16384
//!
//! Synthetic unit rows (no data file). Each op runs `--reps` times after one
//! warm-up and is synced after every call, so the time includes one device
//! round trip; `loop` is a whole training run (`Emeans::train`) for reference.

use std::time::Instant;

use anyhow::Result;
use emeans::assign::Metric;
use emeans::core::{Device, EmaStep, Mat, Select};
use emeans::{Config, Emeans};

fn arg(args: &[String], name: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
        .unwrap_or_else(|| default.to_string())
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn bench<D: Device>(
    dev: &mut D,
    x: &[f32],
    d: usize,
    k: usize,
    b: usize,
    reps: usize,
) -> Result<()> {
    let n = x.len() / d;
    let xb = dev.bind(x)?;
    let perm: Vec<u32> = (0..n as u32).map(|i| (i * 7919 + 1) % n as u32).collect();
    let ids = dev.upload_u32(&perm)?;
    let c = dev.upload_f32(&x[..k * d])?;
    let mut unit = dev.zeros_f32(k * d)?;
    let mut lab = dev.zeros_u32(n)?;
    let mut sco = dev.zeros_f32(n)?;
    let mut sums = dev.zeros_f32(k * d)?;
    let mut counts = dev.zeros_u32(k)?;
    let mut pass = dev.zeros_f32(k * d)?;
    let mut pos = dev.upload_f32(&x[..k * d])?;
    let mut usage = dev.upload_f32(&vec![1.0; k])?;
    let probe = dev.zeros_u32(1)?;
    let mat = |xb| Mat {
        host: x,
        dev: xb,
        d,
    };
    let mut time = |name: &str, f: &mut dyn FnMut(&mut D) -> Result<()>| -> Result<()> {
        f(dev)?;
        let _ = dev.read_u32(&probe, 0..1)?;
        let mut t = Vec::new();
        for _ in 0..reps {
            let s = Instant::now();
            f(dev)?;
            let _ = dev.read_u32(&probe, 0..1)?;
            t.push(s.elapsed().as_secs_f64() * 1e3);
        }
        println!("{name:<14} {:8.3} ms", median(t));
        Ok(())
    };
    time("sync", &mut |_| Ok(()))?;
    time("normalize", &mut |dv| {
        dv.normalize_rows(&c, k, d, &mut unit)
    })?;
    time("top1", &mut |dv| {
        dv.top1(
            mat(&xb),
            Select {
                ids: Some(&ids),
                rows: 0..b,
            },
            &unit,
            None,
            k,
            Metric::InnerProduct,
            &mut lab,
            &mut sco,
        )
    })?;
    time("top1-dense", &mut |dv| {
        dv.top1(
            mat(&xb),
            Select {
                ids: None,
                rows: 0..b,
            },
            &unit,
            None,
            k,
            Metric::InnerProduct,
            &mut lab,
            &mut sco,
        )
    })?;
    time("scatter_add", &mut |dv| {
        dv.scatter_add(
            mat(&xb),
            Select {
                ids: Some(&ids),
                rows: 0..b,
            },
            &lab,
            k,
            &mut sums,
            &mut counts,
            Some(&mut pass),
        )
    })?;
    time("ema_update", &mut |dv| {
        dv.ema_update(
            &mut pos,
            &mut usage,
            &sums,
            &counts,
            k,
            d,
            EmaStep::new(0.3),
        )
    })?;
    time("read_usage", &mut |dv| {
        dv.read_f32(&usage, 0..k).map(|_| ())
    })?;
    time("read_pos", &mut |dv| {
        dv.read_f32(&pos, 0..k * d).map(|_| ())
    })?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let name = arg(&args, "--backend", "cpu");
    let n: usize = arg(&args, "--n", "200000").parse()?;
    let d: usize = arg(&args, "--d", "768").parse()?;
    let k: usize = arg(&args, "--k", "1024").parse()?;
    let b: usize = arg(&args, "--batch", "16384").parse()?;
    let reps: usize = arg(&args, "--reps", "5").parse()?;
    let x = emeans::core::contract::mixture(n, d, 500, 0.5, true, 5);
    // The ops bench needs the concrete device type, so it builds the engine itself.
    let mut a: Box<dyn emeans::Assigner> = match name.parse::<emeans::Backend>()? {
        emeans::Backend::Cpu(kernel) => {
            let mut dev = emeans::CpuAssigner::new(kernel);
            println!("{} n={n} d={d} k={k} batch={b}", dev.name());
            bench(&mut dev, &x, d, k, b, reps)?;
            Box::new(dev)
        }
        #[cfg(target_os = "macos")]
        emeans::Backend::MetalFused => {
            let mut dev = emeans::metal::MetalCore::new()?;
            println!("{} n={n} d={d} k={k} batch={b}", Device::name(&dev));
            bench(&mut dev, &x, d, k, b, reps)?;
            Box::new(dev)
        }
        #[cfg(feature = "cuda")]
        emeans::Backend::Cuda => {
            let mut dev = emeans::cuda::CudaCore::new()?;
            println!("{} n={n} d={d} k={k} batch={b}", Device::name(&dev));
            bench(&mut dev, &x, d, k, b, reps)?;
            Box::new(dev)
        }
        other => anyhow::bail!("ops_bench: no engine for {other:?} in this build"),
    };
    let mut cfg = Config::new(d, k);
    cfg.spherical = true;
    a.bind_data(&x, d)?;
    let mut m = Emeans::with_assigner(cfg, a)?;
    let s = Instant::now();
    m.train(&x)?;
    println!(
        "{:<14} {:8.3} ms ({} steps)",
        "loop",
        s.elapsed().as_secs_f64() * 1e3,
        m.recipe().unwrap().total_steps
    );
    Ok(())
}
