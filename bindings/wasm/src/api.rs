//! The binding API in plain Rust: options from JSON, results as plain structs.
//!
//! The wasm-bindgen layer (`lib.rs`) only converts JavaScript values to and
//! from these types. The native golden program (`examples/golden.rs`) calls the
//! same functions, so the browser and native runs take the same code path.
//!
//! Options are JSON objects. A key is camelCase or snake_case and names a field
//! of the Rust config (`Config`, `AutoConfig`, `DedupConfig`); missing keys keep
//! the Rust default, unknown keys fail. `kernel` picks the CPU kernel:
//! `gemm` (the default, [`DEFAULT_KERNEL`]), `fused` (the same bits as native
//! `cpu-fused` on every machine) or `blocked`.

use anyhow::{Context, Result, bail, ensure};
use emeans::{
    AddReport, AutoConfig, Backend, Config, CpuAssigner, CpuKernel, DedupConfig, DedupInput,
    DedupResult, Emeans, Hierarchy, Labels, auto_cluster, dedup as dedup_rows,
};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};

/// The kernel when the options name none: `gemm`, the fastest on wasm
/// (`CpuKernel::fastest()` there). Natively on macOS, where `gemm` is not
/// built, callers must name a kernel.
pub const DEFAULT_KERNEL: CpuKernel = CpuKernel::Gemm;

/// `fooBar1Baz` -> `foo_bar1_baz`; snake_case keys pass through.
fn snake(key: &str) -> String {
    let mut out = String::with_capacity(key.len() + 4);
    for c in key.chars() {
        if c.is_ascii_uppercase() {
            out.push('_');
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

/// Write the keys of `opts` over `base` (both JSON objects), recursing into
/// nested objects. A key that `base` lacks is an error.
fn overlay(base: &mut Map<String, Value>, opts: &Map<String, Value>, path: &str) -> Result<()> {
    for (k, v) in opts {
        let key = snake(k);
        let name = format!("{path}{k}");
        let slot = base
            .get_mut(&key)
            .with_context(|| format!("unknown option {name:?}"))?;
        match (slot, v) {
            (Value::Object(b), Value::Object(o)) => overlay(b, o, &format!("{name}."))?,
            (slot, v) => *slot = v.clone(),
        }
    }
    Ok(())
}

/// `defaults` with the keys of `opts` written over it.
fn with_options<T: Serialize + DeserializeOwned>(
    defaults: &T,
    opts: &Map<String, Value>,
) -> Result<T> {
    let Value::Object(mut base) = serde_json::to_value(defaults)? else {
        bail!("defaults are not an object");
    };
    overlay(&mut base, opts, "")?;
    serde_json::from_value(Value::Object(base)).context("invalid options")
}

/// The options object, or an empty one for `null` / missing; removes and
/// returns `kernel`.
fn split_kernel(opts: Option<&Value>) -> Result<(Map<String, Value>, CpuKernel)> {
    let mut m = match opts {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(m)) => m.clone(),
        Some(v) => bail!("options must be an object, got {v}"),
    };
    let kernel = match m.remove("kernel") {
        None | Some(Value::Null) => DEFAULT_KERNEL,
        Some(Value::String(s)) => match s.as_str() {
            "fused" => CpuKernel::Fused,
            "blocked" => CpuKernel::Blocked,
            "gemm" => CpuKernel::Gemm,
            _ => bail!("unknown kernel {s:?} (fused, blocked, gemm)"),
        },
        Some(v) => bail!("kernel must be a string, got {v}"),
    };
    Ok((m, kernel))
}

fn check_rows(x: &[f32], d: usize) -> Result<usize> {
    ensure!(d > 0, "d must be positive");
    ensure!(
        x.len().is_multiple_of(d),
        "x has {} values, not a multiple of d = {d}",
        x.len()
    );
    Ok(x.len() / d)
}

/// A trained e-means model and the labels of its last training pass.
pub struct Model {
    pub model: Emeans,
    pub kernel: CpuKernel,
    /// Last-pass label of every training row (-1 = not sampled); None when not captured.
    pub labels: Option<Vec<i32>>,
}

/// Train e-means with `k` centroids on `x` (`n x d`, row-major).
pub fn train(x: &[f32], d: usize, k: usize, opts: Option<&Value>) -> Result<Model> {
    check_rows(x, d)?;
    let (mut m, kernel) = split_kernel(opts)?;
    ensure!(
        !m.contains_key("d") && !m.contains_key("k"),
        "pass d and k as arguments, not options"
    );
    m.remove("verbose");
    let cfg = with_options(&Config::new(d, k), &m)?;
    let mut model = Emeans::with_assigner(cfg, Box::new(CpuAssigner::new(kernel)))?;
    let trained = model.train(x)?;
    Ok(Model {
        model,
        kernel,
        labels: trained.labels,
    })
}

impl Model {
    /// Read a model from `Model::to_bytes` (or a native model file).
    pub fn from_bytes(b: &[u8], opts: Option<&Value>) -> Result<Model> {
        let (m, kernel) = split_kernel(opts)?;
        ensure!(m.is_empty(), "fromBytes takes only the kernel option");
        let model = Emeans::from_bytes_with(b, Box::new(CpuAssigner::new(kernel)))?;
        Ok(Model {
            model,
            kernel,
            labels: None,
        })
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.model.to_bytes()
    }

    /// Nearest centroid of every row of `x`: (labels, scores).
    pub fn assign(&self, x: &[f32]) -> Result<(Vec<u32>, Vec<f32>)> {
        check_rows(x, self.model.config().d)?;
        let a = self.model.assign(x)?;
        Ok((a.labels, a.scores))
    }

    pub fn centroids(&self) -> Result<Vec<f32>> {
        Ok(self
            .model
            .centroids()
            .context("model is not trained")?
            .to_vec())
    }
}

/// Two-level clustering with automatic k.
pub fn auto(x: &[f32], d: usize, opts: Option<&Value>) -> Result<(Hierarchy, Labels)> {
    check_rows(x, d)?;
    let (mut m, kernel) = split_kernel(opts)?;
    m.remove("verbose");
    let mut cfg = with_options(&AutoConfig::default(), &m)?;
    cfg.backend = Backend::Cpu(kernel);
    auto_cluster(x, d, &cfg)
}

/// Read a hierarchy from `Hierarchy::to_bytes`.
pub fn hierarchy_from_bytes(b: &[u8], opts: Option<&Value>) -> Result<Hierarchy> {
    let (m, kernel) = split_kernel(opts)?;
    ensure!(m.is_empty(), "fromBytes takes only the kernel option");
    Hierarchy::from_bytes(b, Backend::Cpu(kernel))
}

/// Label rows `n_old..` of `x_all` and update the hierarchy by its policy.
pub fn add(h: &mut Hierarchy, x_all: &[f32], n_old: usize) -> Result<AddReport> {
    check_rows(x_all, h.d)?;
    h.add(x_all, n_old)
}

/// Per-row side information of [`dedup`].
#[derive(Default)]
pub struct DedupSide {
    pub created_unix: Option<Vec<f64>>,
    /// Burst id per row; negative = no burst.
    pub burst: Option<Vec<i32>>,
    pub priority: Option<Vec<f32>>,
}

/// Near-duplicate groups of the unit rows of `x`.
pub fn dedup(x: &[f32], d: usize, opts: Option<&Value>, side: &DedupSide) -> Result<DedupResult> {
    let n = check_rows(x, d)?;
    let (m, kernel) = split_kernel(opts)?;
    let cfg = with_options(&DedupConfig::default(), &m)?;
    let len_ok = |len: Option<usize>, what: &str| -> Result<()> {
        if let Some(l) = len {
            ensure!(l == n, "{what} has {l} entries for {n} rows");
        }
        Ok(())
    };
    len_ok(side.created_unix.as_ref().map(Vec::len), "createdUnix")?;
    len_ok(side.burst.as_ref().map(Vec::len), "burst")?;
    len_ok(side.priority.as_ref().map(Vec::len), "priority")?;
    let burst: Option<Vec<Option<u32>>> = side
        .burst
        .as_ref()
        .map(|b| b.iter().map(|&v| u32::try_from(v).ok()).collect());
    let input = DedupInput {
        created_unix: side.created_unix.as_deref(),
        burst: burst.as_deref(),
        priority: side.priority.as_deref(),
    };
    dedup_rows(x, d, &input, &cfg, Backend::Cpu(kernel))
}

/// Synthetic `n x d` rows around `centers` points, the same bits on every platform:
/// integer xorshift32 draws, centers at integers in [-8, 8], uniform noise in
/// [-0.5, 0.5) at a step of 2^-16. Every value is exact in `f32` (no libm, no
/// rounding), so native, Node and browser runs train on identical data.
pub fn blobs(n: usize, d: usize, centers: usize, seed: u32) -> Result<Vec<f32>> {
    ensure!(d > 0 && centers > 0, "d and centers must be positive");
    let len = n.checked_mul(d).context("n x d too large")?;
    let mut s = if seed == 0 { 0x9E37_79B9 } else { seed };
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    };
    let c: Vec<i32> = (0..centers * d).map(|_| (next() % 17) as i32 - 8).collect();
    let mut x = Vec::with_capacity(len);
    for _ in 0..n {
        let j = next() as usize % centers;
        for t in 0..d {
            let noise = (next() >> 16) as i32 - 32768;
            x.push((c[j * d + t] * 65536 + noise) as f32 / 65536.0);
        }
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FNV-1a 64 of the bits of `blobs(100, 8, 4, 1)` (also computed by a Python port).
    const BLOBS_100_8_4_1: u64 = 0x224f_0636_224a_f825;

    #[test]
    fn options_overlay_and_fail_loudly() {
        let o: Value = serde_json::json!({"nepochs": 2.5, "batchLaw": "quality", "seed": 9});
        let (m, kernel) = split_kernel(Some(&o)).unwrap();
        assert_eq!(kernel, DEFAULT_KERNEL);
        let c = with_options(&Config::new(4, 3), &m).unwrap();
        assert_eq!((c.nepochs, c.seed, c.d, c.k), (2.5, 9, 4, 3));
        assert_eq!(c.batch_law, emeans::BatchLaw::Quality);

        let o: Value = serde_json::json!({"k1Range": [2, 6], "incremental": {"updateEpochs": 2.0}});
        let a = with_options(&AutoConfig::default(), &split_kernel(Some(&o)).unwrap().0).unwrap();
        assert_eq!(a.k1_range, (2, 6));
        assert_eq!(a.incremental.update_epochs, 2.0);
        assert_eq!(a.k2_max, AutoConfig::default().k2_max);

        for bad in [
            serde_json::json!({"nepoch": 2}),
            serde_json::json!({"seed": "x"}),
            serde_json::json!({"kernel": "accelerate"}),
            serde_json::json!([1]),
        ] {
            let r =
                split_kernel(Some(&bad)).and_then(|(m, _)| with_options(&Config::new(4, 3), &m));
            assert!(r.is_err(), "{bad} should fail");
        }
        let x = vec![0f32; 7];
        assert!(train(&x, 2, 1, None).is_err(), "ragged x");
    }

    #[test]
    fn blobs_are_exact_and_fixed() {
        let x = blobs(100, 8, 4, 1).unwrap();
        assert_eq!(x.len(), 800);
        // Every value is a multiple of 2^-16 within 8.5 of zero: exact in f32.
        assert!(
            x.iter()
                .all(|v| (v * 65536.0).fract() == 0.0 && v.abs() <= 8.5)
        );
        // Pinned: a change of the generator changes every benchmark's data.
        let h = x.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, v| {
            (h ^ v.to_bits() as u64).wrapping_mul(0x100_0000_01b3)
        });
        assert_eq!(h, BLOBS_100_8_4_1);
    }
}
