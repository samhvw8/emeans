//! GPU time of the Metal fused top-1 per tile (Metal timestamps, no host round trip).
//!
//! cargo run --release -p emeans-bench --bin top1_bench -- --rows 16384 --k 1024 --reps 7
//!
//! Synthetic unit rows; dense rows (no gather). Labels are compared with tile 0.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("this benchmark needs Metal (macOS)");
}

#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    mac::main()
}

#[cfg(target_os = "macos")]
mod mac {
    use anyhow::Result;
    use emeans::assign::Metric;
    use emeans::core::{Device, Mat, Select};
    use emeans::metal::{FUSED_TILES, MetalCore};

    fn arg(args: &[String], name: &str, default: &str) -> String {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
            .unwrap_or_else(|| default.to_string())
    }

    pub fn main() -> Result<()> {
        let args: Vec<String> = std::env::args().collect();
        let rows: usize = arg(&args, "--rows", "16384").parse()?;
        let d: usize = arg(&args, "--d", "768").parse()?;
        let reps: usize = arg(&args, "--reps", "7").parse()?;
        let ks: Vec<usize> = arg(&args, "--k", "64,256,1024")
            .split(',')
            .map(|s| s.parse())
            .collect::<Result<_, _>>()?;
        // Tile indexes of FUSED_TILES, or "auto" (the engine's choice per call).
        let tiles: Vec<Option<usize>> = match args.iter().position(|a| a == "--tiles") {
            Some(i) => args[i + 1]
                .split(',')
                .map(|s| {
                    if s == "auto" {
                        Ok(None)
                    } else {
                        s.parse().map(Some)
                    }
                })
                .collect::<Result<_, _>>()?,
            None => (0..FUSED_TILES.len()).map(Some).collect(),
        };
        // Centroid parts per dispatch; 0 = auto.
        let parts: Vec<usize> = arg(&args, "--parts", "0")
            .split(',')
            .map(|s| s.parse())
            .collect::<Result<_, _>>()?;
        let x = emeans::core::contract::mixture(rows, d, 500, 0.5, true, 5);
        for &k in &ks {
            let c = emeans::core::contract::mixture(k, d, 50, 0.5, true, 9);
            let mut base: Option<Vec<u32>> = None;
            for (&t, &pp) in tiles.iter().flat_map(|t| parts.iter().map(move |p| (t, p))) {
                let mut m = match t {
                    Some(t) => MetalCore::with_tile(t)?,
                    None => MetalCore::new()?,
                };
                m.set_top1_parts((pp > 0).then_some(pp));
                let xb = m.upload_f32(&x)?;
                let cb = m.upload_f32(&c)?;
                let mut lab = m.zeros_u32(rows)?;
                let mut sco = m.zeros_f32(rows)?;
                let mut v = Vec::new();
                for _ in 0..reps + 1 {
                    v.push(m.gpu_ms(|m| {
                        m.top1(
                            Mat {
                                host: &x,
                                dev: &xb,
                                d,
                            },
                            Select {
                                ids: None,
                                rows: 0..rows,
                            },
                            &cb,
                            None,
                            k,
                            Metric::InnerProduct,
                            &mut lab,
                            &mut sco,
                        )
                    })?);
                }
                v.remove(0);
                v.sort_by(f64::total_cmp);
                let l = m.read_u32(&lab, 0..rows)?;
                let diff = base
                    .as_ref()
                    .map(|b| b.iter().zip(&l).filter(|(a, b)| a != b).count());
                if base.is_none() {
                    base = Some(l);
                }
                let ms = v[v.len() / 2];
                println!(
                    "rows {rows:6} k {k:5} tile {:<22} parts {pp} {:8.3} ms {:7.0} GFLOP/s diff {:?}",
                    format!("{:?}", t.map(|t| FUSED_TILES[t])),
                    ms,
                    2.0 * (rows * k * d) as f64 / ms / 1e6,
                    diff
                );
            }
        }
        Ok(())
    }
}
