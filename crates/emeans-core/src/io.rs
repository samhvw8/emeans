//! Model files and raw f32 matrices.
//!
//! Model file layout (little-endian):
//!
//! | Bytes | Content |
//! |---|---|
//! | 8 | magic `EMEANS\0\x01` |
//! | 8 | u64 header length h |
//! | h | JSON header: version, config, recipe, k, d, n_splits, iteration_stats, which arrays follow |
//! | k*d*4 | final centroids (if `has_centroids`) |
//! | k*d*4 | EMA positions (if `has_ema`) |
//! | k*4 | EMA masses (if `has_ema`) |
//!
//! Loading checks the exact file length, so a short or padded file fails.

use std::fs;
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

use crate::assign::{Assigner, CpuAssigner};
use crate::model::{Config, Emeans, EpochStats};
use crate::recipe::Recipe;

const MAGIC: &[u8; 8] = b"EMEANS\0\x01";
const VERSION: u32 = 1;

#[doc(hidden)]
pub struct Snapshot {
    pub config: Config,
    pub recipe: Option<Recipe>,
    pub n_splits: usize,
    pub iteration_stats: Vec<EpochStats>,
    pub centroids: Option<Vec<f32>>,
    pub positions: Vec<f32>,
    pub usage: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
struct Header {
    version: u32,
    k: usize,
    d: usize,
    config: Config,
    recipe: Option<Recipe>,
    n_splits: usize,
    iteration_stats: Vec<EpochStats>,
    has_centroids: bool,
    has_ema: bool,
}

fn put_f32(buf: &mut Vec<u8>, v: &[f32]) {
    buf.reserve(v.len() * 4);
    for x in v {
        buf.extend_from_slice(&x.to_le_bytes());
    }
}

fn get_f32(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

impl Emeans {
    /// Write the trained model (centroids, EMA state, recipe, config) to `path`.
    /// Writes to a temporary file first, then renames it over `path`.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let s = self.snapshot()?;
        let (k, d) = (s.config.k, s.config.d);
        let has_ema = !s.usage.is_empty();
        let header = Header {
            version: VERSION,
            k,
            d,
            config: s.config.clone(),
            recipe: s.recipe.clone(),
            n_splits: s.n_splits,
            iteration_stats: s.iteration_stats.clone(),
            has_centroids: s.centroids.is_some(),
            has_ema,
        };
        let json = serde_json::to_vec(&header)?;
        let mut buf = Vec::with_capacity(16 + json.len() + (2 * k * d + k) * 4);
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&(json.len() as u64).to_le_bytes());
        buf.extend_from_slice(&json);
        if let Some(c) = &s.centroids {
            put_f32(&mut buf, c);
        }
        if has_ema {
            put_f32(&mut buf, &s.positions);
            put_f32(&mut buf, &s.usage);
        }
        let path = path.as_ref();
        let tmp = path.with_extension("tmp");
        {
            let mut f =
                fs::File::create(&tmp).with_context(|| format!("create {}", tmp.display()))?;
            f.write_all(&buf)?;
            f.sync_all()?;
        }
        fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Read a model written by `save`, with the fastest CPU assigner.
    pub fn load(path: impl AsRef<Path>) -> Result<Emeans> {
        Self::load_with(path, Box::new(CpuAssigner::default()))
    }

    pub fn load_with(path: impl AsRef<Path>, assigner: Box<dyn Assigner>) -> Result<Emeans> {
        let path = path.as_ref();
        let b = fs::read(path).with_context(|| format!("read {}", path.display()))?;
        ensure!(
            b.len() >= 16 && &b[..8] == MAGIC,
            "{} is not an emeans model",
            path.display()
        );
        let h = u64::from_le_bytes(b[8..16].try_into().unwrap()) as usize;
        ensure!(b.len() >= 16 + h, "truncated header");
        let header: Header = serde_json::from_slice(&b[16..16 + h])?;
        if header.version != VERSION {
            bail!("unsupported model version {}", header.version);
        }
        let (k, d) = (header.k, header.d);
        ensure!(
            header.config.k == k && header.config.d == d,
            "header k/d disagree with config"
        );
        let kd = k * d;
        let want = 16
            + h
            + if header.has_centroids { kd * 4 } else { 0 }
            + if header.has_ema { (kd + k) * 4 } else { 0 };
        ensure!(
            b.len() == want,
            "model file has {} bytes, expected {}",
            b.len(),
            want
        );
        let mut off = 16 + h;
        let centroids = header.has_centroids.then(|| {
            let v = get_f32(&b[off..off + kd * 4]);
            off += kd * 4;
            v
        });
        let (positions, usage) = if header.has_ema {
            let p = get_f32(&b[off..off + kd * 4]);
            off += kd * 4;
            (p, get_f32(&b[off..off + k * 4]))
        } else {
            (Vec::new(), Vec::new())
        };
        Emeans::from_snapshot(
            Snapshot {
                config: header.config,
                recipe: header.recipe,
                n_splits: header.n_splits,
                iteration_stats: header.iteration_stats,
                centroids,
                positions,
                usage,
            },
            assigner,
        )
    }
}

/// Read a raw little-endian f32 matrix with `d` columns (the `embeddings.f32` contract).
pub fn read_f32_matrix(path: impl AsRef<Path>, d: usize) -> Result<Vec<f32>> {
    let path = path.as_ref();
    let b = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    ensure!(
        b.len().is_multiple_of(4 * d),
        "{}: {} bytes is not a whole number of {}-d f32 rows",
        path.display(),
        b.len(),
        d
    );
    Ok(get_f32(&b))
}

/// Write a raw little-endian f32 matrix.
pub fn write_f32(path: impl AsRef<Path>, v: &[f32]) -> Result<()> {
    let mut buf = Vec::with_capacity(v.len() * 4);
    put_f32(&mut buf, v);
    fs::write(path, buf)?;
    Ok(())
}
