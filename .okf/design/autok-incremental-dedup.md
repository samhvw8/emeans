---
type: Design
title: Automatic k, incremental updates and near-duplicate groups
description: How emeans-cluster picks k in two levels (top-down k sweeps with k-means++ seeding for the leaves, a re-split of large leaves with structure, then bottom-up average linkage of the leaves cut by silhouette for the broad groups), labels new rows, flags outliers with a robust per-leaf floor, grows new leaves from coherent outliers and updates the model with stable cluster ids, and groups near-duplicates with an exact kNN and an anti-chaining star cover around a stored anchor; defaults, rejected alternatives and synthetic validation.
resource: crates/emeans-cluster
tags: [emeans, auto-k, incremental, dedup, knn, metal, validation]
status: draft
generated:
  by: claude-code/claude-opus-5-5
  at: '2026-11-12T22:00:00Z'
sources:
  - id: schubert-elbow
    resource: https://arxiv.org/abs/2212.12189
    title: "Stop using the elbow criterion for k-means and how to choose the number of clusters instead (Schubert, 2023)"
  - id: arbelaitz
    resource: https://doi.org/10.1016/j.patcog.2012.07.021
    title: "An extensive comparative study of cluster validity indices (Arbelaitz et al., Pattern Recognition 46(1), 2013)"
  - id: kaczynska
    resource: https://www.esann.org/sites/default/files/proceedings/2020/ES2020-48.pdf
    title: "Comparison of cluster validity indices and decision rules for different degrees of cluster separation (Kaczynska, Marion, von Sachs, ESANN 2020)"
  - id: luxburg-stability
    resource: https://arxiv.org/abs/1007.1075
    title: "Clustering stability: an overview (von Luxburg, 2010)"
  - id: mourer-stadion
    resource: https://arxiv.org/abs/2006.08530
    title: "Selecting the number of clusters K with a stability trade-off: an internal validation criterion (Mourer et al., PAKDD 2023)"
  - id: kasarapu-mml
    resource: https://arxiv.org/abs/1502.07813
    title: "Minimum message length estimation of mixtures of multivariate Gaussian and von Mises-Fisher distributions (Kasarapu and Allison, 2015)"
  - id: hdbscan-params
    resource: https://hdbscan.readthedocs.io/en/latest/parameter_selection.html
    title: HDBSCAN documentation, Parameter Selection (min_cluster_size)
  - id: bertopic-tuning
    resource: https://maartengr.github.io/BERTopic/getting_started/parameter%20tuning/parametertuning.html
    title: BERTopic documentation, Parameter tuning (min_cluster_size / min_topic_size)
  - id: synth-runs
    resource: bench/src/bin/autok_synth.rs
    title: Validation runs of bench/src/bin/autok_synth.rs and knn_bench.rs on synthetic data, 2026-11-12 (this document's tables)
---

# Overview

Three algorithms in `crates/emeans-cluster`, built on the e-means API of `emeans-core` and validated
on synthetic data from `synth::nested`. Every threshold below is a parameter with a default; on real
embeddings, calibrate it (see [Calibrate on your data](#calibrate-on-your-data)).

| Module | Content |
|---|---|
| `knn` | Exact top-k inner-product search on the chosen backend: `knn_self`, `knn` (GPU engines through `Engine::run_topk`; the CPU path in `emeans_core::knn`) |
| `dedup` | `dedup`, `dedup_with_neighbors`, `DedupConfig`, `DedupInput`, `DedupResult` (`expand`, `gather`, `anchors`, `anchor`) |
| `auto` | `auto_cluster`, `AutoConfig`, `Criterion`, `Level1Method`, `Hierarchy`, `Labels`, `ChosenK`, `Diagnostics`, `KScore`, `GroupDecision`, `label_stability` |
| `incremental` | `Hierarchy::add`, `Hierarchy::add_photos`, `ema_update` (with the grow step), `full_retrain`, `save`, `load`; `IncrementalConfig`, `Action`, `AddReport`, `PhotoAddReport`, `match_centroids` |
| `metrics` | `ari`, `nmi`, `internal_scores` (cosine silhouette, Calinski-Harabasz, Davies-Bouldin, objective), `quantile` |
| `synth` | `nested` (two-level mixture on the sphere), `with_bursts` (injected near-duplicates) |
| `backend` | `Backend`, `CustomBackend`, `make_assigner` (see [crate layout](/decisions/crate-layout.md)) |

From `emeans-core` these use `gemm_nt` (CPU `x * y^T`) and `SharedAssigner` (one assigner shared by
many models, so a sweep compiles the GPU kernels once).

# API

```rust
use emeans::{Backend, CpuKernel, DedupConfig, DedupInput, DedupResult, AutoConfig, Hierarchy, Labels, AddReport, auto_cluster, dedup};

// 1. Near-duplicates, before clustering. All inputs optional.
let inp = DedupInput { created_unix: Some(&t), burst: Some(&burst_ids), priority: Some(&favorite) };
let dd: DedupResult = dedup(&x, 768, &inp, &DedupConfig::default(), Backend::cpu())?;
let reps: Vec<f32> = dd.gather(&x, 768);             // one row per group

// 2. Two-level clustering with automatic k.
let (mut h, labels): (Hierarchy, Labels) = auto_cluster(&reps, 768, &AutoConfig::default())?;
h.dedup = Some(DedupConfig::default());               // saved with the model
let per_row: Vec<(u32, u32)> = dd.expand(&(0..labels.len()).map(|i| labels.leaf(i)).collect::<Vec<_>>());
h.save("clusters/hier")?;

// 3. Later: new representatives appended to the same matrix.
let mut h = Hierarchy::load("clusters/hier", Backend::cpu())?;
let r: AddReport = h.add(&reps_all, h.n_seen())?;    // r.action, r.labels (all rows), r.new_outliers, r.new_leaves
```

`Labels` holds, per row, the stable level-1 id `l1`, the stable level-2 id `l2`, the cosine to the
level-1 centroid `sim1` and to the leaf centroid `sim`, the `outlier` flag, and the model slots.
`Hierarchy` exposes `chosen_k()`, `level1_centroids()`, `leaf_centroids(g)`, `groups[g].decision`
and `diagnostics` (every criterion value of every k tried, per level and group).

# Near-duplicate groups

| Step | Choice | Why |
|---|---|---|
| Neighbours | Exact kNN, `k = 10` | n = 16k gives 2.6e8 dot products (0.4 TFLOP): 0.25-0.4 s exact, so no approximate index |
| Link | `cos >= sim_threshold` (0.95); with `max_gap_secs` set, also `abs(dt) <= max_gap_secs` when both times are known | Similarity alone groups re-saved copies and edits with old dates; the time rule is opt-in |
| Bursts | Rows with the same burst id are one unit, always together. A burst joins another row's star only when every one of its rows passes the star rule | Cameras and photo libraries often know bursts already. With "any row passes" one burst row near the anchor pulled the whole burst in, and the diameter bound below did not hold |
| Components | Union-find over links | Cheap, order-independent (lower root wins) |
| Anti-chaining | Star cover per component: highest-link row becomes the group's **anchor** (`DedupResult::anchors`), every unassigned unit with `cos >= rep_threshold` to it joins; repeat | Bounds every member outside the anchor's own burst to one similarity hop from the anchor, so any two of them are within twice the threshold angle (cos >= 2t^2 - 1 = 0.805 at t = 0.95). The anchor's own burst is exempt: it stays together whatever its similarity. Complete linkage bounds the diameter tighter but is order-dependent and needs all pairwise similarities of a component; the star needs one dot product per member and center |
| Representative | Highest caller `priority` (for example favorites), then medoid (highest mean similarity in the group), then lowest row | Medoid is the most typical item; the caller can prefer its own choice. The representative is the row that is clustered; it is not the anchor, and membership is never tested against it |

`DedupResult` also returns `nn_sim` (each row's best neighbour similarity, for a histogram when
calibrating), `linked_pairs` and `split_components` (how often the star cover cut a chain).

## Top-k Metal kernel

`KNN_SOURCE` (`crates/emeans-metal/src/lib.rs`, `MetalKnn`) reuses the fused kernel's GEMM tiling
(64 query rows x 32 data rows, K step 32, `simdgroup_float8x8`, MLX fragment layout). After each tile
the 64 x 32 dot products go to threadgroup memory; two threads per query row scan half the tile each
and keep their best `k` (k <= 16, `KNN_KMAX`) in an unsorted list with the slot of the current worst,
so a rejected candidate costs one compare. The two lists merge at the end and the row is written
sorted (score, then lower id). Larger `k` falls back to the CPU.

Timing, synthetic unit vectors, n = 16,000, d = 768, k = 10, median of 5 runs, M1 Pro with load
average 9-28 from other jobs:[^synth-runs]

| Step | CPU (Accelerate GEMM + rayon top-k) | Metal top-k kernel | Metal incl. device setup and upload |
|---|---|---|---|
| All-pairs kNN | **0.23-0.28 s** (1.4-1.7 TFLOP/s) | 0.37 s (1.07 TFLOP/s) | 0.38-0.85 s |
| Whole `dedup`, 19,679 rows (16,000 + 3,679 injected burst copies) | **0.47 s** | | 0.62 s |

Both backends return the same neighbours (0 rank slots differ beyond a 1e-5 score tie). The
groups match the injected bursts exactly (ARI 1.0; 12,457 linked pairs). On the M1 Pro the AMX
units behind Accelerate beat the GPU kernel, so pass the CPU backend for dedup there; the Metal kernel
reaches about 70% of the fused assignment kernel's rate because of the extra tile store and scan.

# Automatic k

## Method chosen

Three stages. Each k of a sweep is the best of 4 e-means runs by objective, each seeded by
**k-means++** with the cosine distance (one seeding per restart serves the whole sweep: its first k
picks seed k).

1. **Leaves, top-down.** Level-1 sweep k = 2..40, argmax of the **cosine silhouette**. Inside each
   group of at least `min_split_size` (60) rows, a level-2 sweep k = 2..min(20, size / 20). A group is
   split only if the best silhouette of the sweep reaches `split_min_silhouette` (0.05); otherwise it
   stays one leaf. Among the k whose **Davies-Bouldin** index is within 5% of the best, take the one
   that puts the most rows into clusters of 20..200 rows (`min_cluster_size`..`max_cluster_size`),
   then the smaller k.
2. **Re-split** (`resplit_passes`, 2). Every leaf above `max_cluster_size` whose own sweep did not
   already find "no structure" gets a cheap screen: k = 2..4, best silhouette. When the screen
   reaches `split_min_silhouette`, the leaf gets a full level-2 sweep and is split by the same rule.
   Then every row moves to its nearest leaf mean (2 passes).
3. **Level 1, bottom-up** (`level1_method = BottomUp`). Average linkage on the leaves, with the
   exact mean row cosine `s_A . s_B / (|A| |B|)` (`s` = sum of a leaf's rows). Each cut k = 2..40 of
   the dendrogram gives a row partition; take the one with the best silhouette. The cut "no
   grouping" (every leaf its own group) is a candidate too, so data with no broad structure stays
   flat (`diagnostics.flat`). A cut whose largest group holds more than `level1_max_share` (0.5) of
   the rows is not a candidate (`diagnostics.level1_rejected`): on embeddings without broad
   structure, average linkage chains into one giant group plus a few small ones at every low cut,
   and the silhouette can prefer such a cut (`tests/hierarchy.rs`
   `level1_skips_cuts_with_one_giant_group`). `TopDown` keeps the stage-1 groups instead.

The silhouette is exact, not sampled. For the cosine distance `1 - x.y` the mean distance from row
`i` to cluster `C` is `1 - x_i . s_C / |C|` with `s_C` the sum of `C`'s rows, so all n silhouettes cost
one n x k product. All three indices match scikit-learn to 1e-5 (`crates/emeans-cluster/tests/metrics.rs`,
fixture from `bench/py/metrics_fixture.py`). A whole run (n = 16k, d = 768, CPU) takes 36-78 s on a
busy M1 Pro (load average 22-28): stage 1 is 28-77 s, the re-split 0.7-19 s, level 1 under 1 s.

## Level 1: bottom-up against top-down

The top-down level-1 sweep is unreliable on nested data in both directions. On data set 31 it
picked k1 = 21: at k = 8 e-means reaches a lower objective than the true broad partition (0.483
against 0.488) by splitting large broad groups and merging small ones, so its silhouette is 0.23
against 0.42 for the truth, and the sweep climbs to the sub-group scale. On data sets 9 and 10, and
on 5 of the 10 fresh data sets below, it merged broad groups. The leaves are good either way, so
grouping the leaves is the more robust way to form the broad layer: a leaf is almost always inside
one broad group, and average linkage of leaf means does not depend on e-means finding a
broad-scale optimum.

Fresh data sets 40-49 (never used for tuning), same leaves for both methods (stages 1 and 2 are
shared; only stage 3 differs). k1 counts every level-1 group: 9 = the 8 broad groups plus one
group of the 3% noise rows. ARI is on non-noise rows.[^synth-runs]

| Data set | Bottom-up k1 | Bottom-up ARI L1 | Top-down k1 | Top-down ARI L1 | ARI leaves | Leaves (true) | Re-splits | s bottom-up / top-down |
|---|---|---|---|---|---|---|---|---|
| 40 | 8 | 1.000 | 7 | 0.986 | 1.000 | 70 (69) | 0 | 44.4 / 41.9 |
| 41 | 9 | 1.000 | 8 | 1.000 | 1.000 | 55 (55) | 0 | 46.4 / 40.3 |
| 42 | 8 | 1.000 | 7 | 0.931 | 1.000 | 77 (75) | 3 | 37.9 / 50.8 |
| 43 | 9 | 1.000 | 7 | 0.975 | 0.999 | 61 (64) | 0 | 46.0 / 37.2 |
| 44 | 9 | 1.000 | 9 | 1.000 | 0.999 | 70 (71) | 1 | 43.1 / 36.3 |
| 45 | 9 | 1.000 | 7 | 0.967 | 1.000 | 59 (58) | 1 | 36.3 / 40.1 |
| 46 | 9 | 1.000 | 8 | 1.000 | 0.993 | 79 (64) | 2 | 44.5 / 37.9 |
| 47 | 9 | 1.000 | 10 | 0.881 | 1.000 | 49 (48) | 0 | 51.3 / 30.0 |
| 48 | 9 | 1.000 | 7 | 0.994 | 0.999 | 66 (67) | 1 | 43.3 / 42.3 |
| 49 | 8 | 1.000 | 8 | 1.000 | 1.000 | 60 (59) | 0 | 38.7 / 49.4 |
| Mean | | **1.000** | | 0.973 | 0.999 | | | 43.2 / 40.6 |

The data sets that failed before, with the final method (bottom-up, re-split on). "Before" for 9
and 10 is the earlier-version table below; for 31 and 32 it is the verification run that found the
problem:

| Data set | Before (top-down, no re-split): k1 / ARI L1 / ARI leaves | Now: k1 / ARI L1 / ARI leaves | Re-splits | s |
|---|---|---|---|---|
| 9 | 6 / 0.950 / 0.857 | 9 / 1.000 / 0.998 | 4 | 47.9 |
| 10 | 6 / 0.915 / 0.738 | 9 / 1.000 / 0.998 | 2 | 47.2 |
| 31 | 21 / 0.154 / 1.000 | 9 / 1.000 / 1.000 | 0 | 78.4 |
| 32 | 9 / not recorded / not recorded | 9 / 1.000 / 1.000 | 2 | 63.6 |

The re-split is what fixes the leaves of data sets 9 and 10: there the top-down pass merged broad
groups, level 2 found the broad split (k2 = 2) inside them, and the re-split screens those
oversized leaves and splits them into their sub groups. Without the k = 2..4 screen the re-split
pass cost 31 s on data set 31 and split nothing; with it, 0.7 s.

Flat data (no broad structure: 6 groups with one sub group each) stays flat: the "no grouping" cut
wins and every leaf is its own level-1 group (`tests/hierarchy.rs`, `flat_data_stays_flat`).

## Why

- No index wins everywhere, but silhouette, Davies-Bouldin and Calinski-Harabasz form the
  best-performing group in the largest comparison of validity indices,[^arbelaitz] and Schubert
  recommends variance-ratio (Calinski-Harabasz), BIC or the gap statistic over any elbow
  method.[^schubert-elbow] The decision rule matters as much as the index.[^kaczynska]
- Measured here (tables below): Calinski-Harabasz fails (k1 = 2-6, leaf ARI 0.23-0.49). Silhouette
  finds the broad groups best (mean level-1 ARI 0.979 over 9 data sets) but under-splits some
  groups at level 2 (mean leaf ARI 0.927). Davies-Bouldin recovers the leaves best (0.996) but its
  level-1 k is erratic (k1 = 19 on one data set). Silhouette at level 1 and Davies-Bouldin at level 2
  keeps the level-1 quality and lifts leaves to 0.953; it won leaves on 3/3 fresh data sets
  (13-15) that no choice was tuned on.
- The silhouette also gives the "no structure" floor (about 0 for one blob) for the split decision;
  Davies-Bouldin has no such floor.
- k-means++ instead of e-means' random rows: on unequal group sizes random seeding misses small
  groups; with it, one data set collapsed to a flat k1 = 24 (level-1 ARI 0.27).
- Argmax at level 1 instead of "smallest k within 5%": the looser rule merged broad groups
  (k1 = 5 on data set 8) that level 2 could not separate again.
- Sizes: density clustering tools expose the smallest cluster a user wants to see as the main knob
  (`min_cluster_size` in HDBSCAN, 10 by default in BERTopic, raised to cut
  micro-topics).[^hdbscan-params][^bertopic-tuning] Here it bounds the level-2 k and breaks near-ties,
  but never overrides the data criterion.

## Known limitations

- The 3% noise rows usually form their own level-1 group (k1 = 9 for 8 broad groups on 7 of 14
  runs above). In practice this is a "miscellaneous" group; it is not counted as an error.
- Over-splitting: on data set 46 one broad group of 7 sub groups ended as 20 leaves (79 leaves for
  64 true sub groups; leaf ARI still 0.993). The browsable-size preference at level 2 can choose many
  small clusters when Davies-Bouldin is flat.
- All validation is on synthetic data from one generator (`synth::nested`). Real embeddings are
  anisotropic and their broad structure may be weaker; check `diagnostics.level1` (silhouette of
  each cut) and `diagnostics.flat` on your data.

## Rejected

| Method | Why not |
|---|---|
| Elbow / Kneedle on the objective | Unreliable: on 25 blobs Kneedle picked 8-10, other elbow detectors similar[^schubert-elbow] |
| Stability (ARI between runs or subsamples) | Detects k too large but not too small, fails under symmetry;[^luxburg-stability] sampling-based variants picked k = 4-6 for two correlated Gaussians.[^mourer-stadion] Also costs extra runs per k |
| Gap statistic | Needs reference data sets per k; a uniform reference on the 768-d sphere is near-orthogonal to everything, so the reference is ill-defined for embeddings; fluctuates with seeds on many blobs[^schubert-elbow] |
| BIC / MML of a von Mises-Fisher mixture | Principled for directional data[^kasarapu-mml] but needs the concentration (Bessel ratio approximations at d = 768), and real embeddings are anisotropic, not vMF, so the penalty balance is untrustworthy |
| Calinski-Harabasz | Tested: k1 = 2-6 and leaf ARI 0.23-0.49 on data sets 7-9 (table below); in 768-d its value falls almost monotonically with k |
| Davies-Bouldin at level 1 | Tested: erratic k1 (19 for 8 broad groups on data set 10); used at level 2 instead |
| Top-down level 1 (the stage-1 groups) | Tested: merged broad groups on 5 of 10 fresh data sets (mean ARI L1 0.973) and went to the sub-group scale on data set 31 (k1 = 21, ARI L1 0.154); kept as `Level1Method::TopDown` |
| Average linkage without a size guard | Chains into one giant group on data without broad structure; `level1_max_share` (0.5) rejects such cuts |
| Gap criterion on the linkage dendrogram (largest drop in merge similarity) | Not tested as the chooser; computed as `diagnostics.gap_k` for inspection only. The silhouette of the cut already won on all 14 data sets |
| Silhouette at level 2 | Tested: picked k2 = 2 for one group with 7-12 true sub groups on data sets 11, 13, 14 and 15 (at n = 4,000 the 2-way split separated the noise rows from the rest) |

## Validation on synthetic data

`synth::nested`, d = 768, n = 16,000: 8 broad groups (lognormal sizes, sigma 0.7) of 3-12 sub
groups (at least 15 rows each), 3% noise rows that belong to no group. Mean cosine about 0.75 inside a
sub group, 0.55 inside a broad group, 0.22 across. ARI/NMI are on non-noise rows; "leaves" compares
the leaf ids with the true sub groups.

Results of the earlier top-down version (silhouette + Davies-Bouldin, k-means++, 4 restarts, no
re-split, top-down level 1), data sets 7-15. The current defaults are in
[Level 1: bottom-up against top-down](#level-1-bottom-up-against-top-down):[^synth-runs]

| Data set | True subs per broad group | k1 | Level-2 k (true) per level-1 group | Exact k2 | Mean abs k2 error | ARI L1 | NMI L1 | ARI leaves | NMI leaves | s |
|---|---|---|---|---|---|---|---|---|---|---|
| 7 | 11 8 6 4 10 12 4 12 | 8 | 4(4) 6(6) 11(11) 4(4) 12(12) 8(8) 11(12) 10(10) | 7/8 | 0.12 | 1.000 | 1.000 | 1.000 | 0.999 | 26.1 |
| 8 | 8 7 10 10 10 7 9 9 | 9 | 1n(1) 7(7) 10(10) 7(7) 9(10) 8(8) 8(9) 7(8) 10(10) | 6/9 | 0.33 | 0.983 | 0.978 | 0.999 | 0.998 | 29.7 |
| 9 | 6 4 9 6 10 9 9 10 | 6 | 9(10) 2(19) 8(9) 6(6) 9(9) 9(10) | 2/6 | 3.33 | 0.950 | 0.956 | 0.857 | 0.956 | 20.2 |
| 10 | 5 3 6 9 11 12 3 9 | 6 | 11(11) 6(6) 2(24) 3(3) 5(5) 9(9) | 5/6 | 3.67 | 0.915 | 0.935 | 0.738 | 0.918 | 22.1 |
| 11 | 8 7 8 10 5 12 10 11 | 8 | 11(12) 5(5) 8(8) 8(8) 9(10) 7(7) 10(10) 11(11) | 6/8 | 0.25 | 1.000 | 1.000 | 0.996 | 0.997 | 24.0 |
| 12 | 7 9 12 9 7 4 7 11 | 6 | 7(7) 12(12) 7(7) 11(11) 9(9) 18(20) | 5/6 | 0.33 | 0.960 | 0.962 | 0.998 | 0.998 | 22.2 |
| 13 | 12 5 9 11 7 4 3 12 | 8 | 7(7) 3(3) 5(5) 9(9) 11(12) 11(12) 11(11) 4(4) | 6/8 | 0.25 | 1.000 | 1.000 | 0.998 | 0.998 | 22.3 |
| 14 | 7 6 12 5 9 3 12 4 | 8 | 6(6) 9(9) 11(12) 5(5) 10(12) 4(4) 3(3) 8(7) | 5/8 | 0.50 | 1.000 | 1.000 | 0.996 | 0.996 | 22.6 |
| 15 | 8 10 8 12 4 10 10 4 | 8 | 4(4) 10(10) 9(10) 12(12) 8(8) 9(10) 8(8) 4(4) | 6/8 | 0.25 | 1.000 | 1.000 | 0.998 | 0.997 | 22.5 |

Level-2 cell: chosen k, a tag (`n` = no structure, so not split), and in parentheses the true sub
groups whose majority of rows lands in that level-1 group. True k1 is 8; a level-1 group can also be
one large sub group pulled out on its own (`1n(1)`), which is correct at the leaf level.

Criteria compared with the earlier top-down version (same seeding and rules; data sets 7-9 tuned on,
10-12 held out, 13-15 fresh):

| Data set | Silhouette k1 / ARI L1 / ARI leaves | Davies-Bouldin | Calinski-Harabasz | Silhouette + Davies-Bouldin |
|---|---|---|---|---|
| 7 | 8 / 1.000 / 0.992 | 7 / 0.988 / 0.997 | 5 / 0.826 / 0.488 | 8 / 1.000 / 1.000 |
| 8 | 9 / 0.983 / 0.984 | 9 / 0.983 / 0.999 | 2 / 0.563 / 0.234 | 9 / 0.983 / 0.999 |
| 9 | 6 / 0.950 / 0.923 | 9 / 0.867 / 1.000 | 6 / 0.950 / 0.476 | 6 / 0.950 / 0.857 |
| 10 | 6 / 0.915 / 0.825 | 19 / 0.461 / 0.999 | | 6 / 0.915 / 0.738 |
| 11 | 8 / 1.000 / 0.812 | 8 / 1.000 / 0.996 | | 8 / 1.000 / 0.996 |
| 12 | 6 / 0.960 / 0.920 | 5 / 0.828 / 0.978 | | 6 / 0.960 / 0.998 |
| 13 | 8 / 1.000 / 0.966 | 8 / 1.000 / 0.998 | | 8 / 1.000 / 0.998 |
| 14 | 8 / 1.000 / 0.972 | 8 / 1.000 / 0.996 | | 8 / 1.000 / 0.996 |
| 15 | 8 / 1.000 / 0.952 | 8 / 1.000 / 0.998 | | 8 / 1.000 / 0.998 |
| Mean (7-15) | abs(k1 - 8) 0.78 / 0.979 / 0.927 | 1.89 / 0.903 / 0.996 | | 0.78 / 0.979 / 0.953 |

Other settings, silhouette at both levels, data sets 7-9 (k1 / ARI L1 / ARI leaves):

| Setting | 7 | 8 | 9 |
|---|---|---|---|
| k-means++, level-1 argmax (chosen) | 8 / 1.000 / 0.992 | 9 / 0.983 / 0.984 | 6 / 0.950 / 0.923 |
| Random-row seeding, level-1 argmax | 7 / 0.976 / 0.992 | 24 / 0.273 / 1.000 | 7 / 0.982 / 0.941 |
| k-means++, level-1 smallest k within 5% | 7 / 0.988 / 0.938 | 5 / 0.920 / 0.879 | 6 / 0.950 / 0.923 |

With 2 restarts instead of 4 (random seeding, data set 7) leaf ARI was 0.803 instead of 0.992. On
the CPU a sweep at n = 4,000 took 8.6 s against 33 s with the Metal assigner (host loop at the
time), whose per-batch overhead dominated hundreds of small runs; `AutoConfig` therefore defaults
to the CPU backend. The true partitions score silhouette 0.415-0.436 at level 1 and 0.408-0.410 at
the leaves, so the criterion would pick them; the misses come from e-means not reaching them at that k.

### Incremental updates

Data set 40, bottom-up level 1, re-split on: train on the first 80% (12,800 rows, shuffled), add the
last 20% in 4 chunks of 800 with the default policy, then compare with a fresh `auto_cluster` on all
16,000 rows. The model is saved and loaded between training and the first add.[^synth-runs]

| Step | Rows | Action | New outliers | Pending | Old rows with unchanged leaf id | Objective | ARI leaves | s |
|---|---|---|---|---|---|---|---|---|
| Train 80% | 12,800 | `auto_cluster` (k1 = 8) | | | | 0.2859 | 0.998 | 37.1 |
| Add 12,800..13,600 | 13,600 | AssignOnly | 39 | 39 | 1.0000 | 0.2862 | 0.998 | 0.01 |
| Add 13,600..14,400 | 14,400 | EmaUpdate | 35 | 0 | 1.0000 | 0.2858 | 0.998 | 0.20 |
| Add 14,400..15,200 | 15,200 | AssignOnly | 35 | 35 | 1.0000 | 0.2861 | 0.998 | 0.01 |
| Add 15,200..16,000 | 16,000 | EmaUpdate | 22 | 0 | 1.0000 | 0.2860 | 0.998 | 0.39 |
| Full retrain 100% | 16,000 | `auto_cluster` (k1 = 8) | | | ids not matched | 0.2850 | 1.000 | 38.5 |

After four adds every old row keeps its leaf id, and the objective (mean `2 - 2 cos` to the leaf
centroid) is 0.37% above a full retrain, for 0.6 s of updates instead of 38.5 s. The updates trigger
on the 10% added-rows rule; new outliers ran at 3-5% of new rows under the quantile floor of that
run (2% by construction plus the 3% noise rows); the table predates the robust floor and the grow
step. The earlier top-down version on data set 7 kept 97.2% of ids at +1.7%.

# Incremental updates

`Hierarchy::add(x_all, n_old)` takes the whole matrix (old rows first, new rows appended) and
requires `n_old == n_seen()`, so a misaligned call fails.

| Step | Rule | Default |
|---|---|---|
| Label | Nearest leaf over all leaves (cosine); the level-1 group is that leaf's group | |
| Outlier | `sim` below the leaf's floor: the leaf's median member cosine at the last fit minus `outlier_mad_k` x 1.4826 x MAD (median absolute deviation). Leaves under 20 rows, or with zero spread, use the MAD pooled over all leaves; an empty leaf uses the global median. Or the absolute `outlier_min_sim` | 3.0 |
| Grow (first step of an EMA update) | Pending outliers that form a coherent group become a new leaf with a fresh id: see [Growing new leaves](#growing-new-leaves) | `grow_min_size` 20 rows; 0 = off |
| EMA update | Pending outliers >= `update_min_outliers`, or rows added since the last update >= `update_added_fraction` of the rows then: `update_epochs` of `train_resume` on all rows, level 1 then each level-2 model | 50 rows; 10%; 1 epoch |
| Full retrain | Rows added since the last full training >= `retrain_added_fraction` of its rows, or the bucket is full and outliers are >= `retrain_outlier_fraction` of rows added since the full training | 50%; 25% |
| Stable ids | Hungarian matching of new to old centroids by cosine; pairs below `match_min_sim` get a fresh id; ids are never reused. Level 2 matches inside matched level-1 groups | 0.8 |

## Outlier floor

The first version used the `outlier_quantile` (0.02) of the leaf's own member cosines as the floor,
then flagged the same rows against it. That flag was true by construction: 2% of every leaf, and no
row at all in a leaf under 26 rows (the nearest-rank 2% quantile of fewer than 26 values is the
minimum).

The floor is now `median - 3 x 1.4826 x MAD` of the leaf's member cosines (`auto::leaf_floors`). A
row is flagged when it is far below what is typical for its leaf, by the leaf's own robust spread:
a tight leaf with no stragglers flags no row, a leaf with a tail flags the tail, and a leaf of any
size can flag a row (small leaves borrow the pooled spread). On a normal distribution the rule
flags 0.13%; what it flags above that is the lower tail the data has. The same floor tests
training rows and new rows, so the `outlier` flag of `Labels` and the pending bucket mean the same
thing. A model saved before this change keeps its stored quantile floors until its next update.

Tests: `auto::tests::floors_flag_stragglers_not_a_fixed_share` and `tests/hierarchy.rs`
`outlier_flag_carries_information` (clean training rows under 1% flagged, held-out rows of the same
groups under 2%, 50 rows of an unseen group all flagged).

## Growing new leaves

`ema_update` once kept k fixed and then cleared the pending bucket, so a new event was absorbed into
old leaves until a full retrain. It now starts with a grow step (`Hierarchy::grow`):

1. Candidates: pending rows that are still outliers.
2. Thresholds from the model: `cohesion` = the 10% quantile of the existing leaves' median member
   cosines; `floor` = `cohesion` minus `outlier_mad_k` pooled robust deviations.
3. The candidate with the most candidates at `cos >= floor^2` seeds a group. Four rounds of: mean
   of the members, then members = free candidates within `floor` of that mean and nearer to it
   than to their current leaf.
4. The group becomes a leaf when it has `grow_min_size` rows and its median member cosine reaches
   `cohesion` (as tight as the leaves already there). Repeat on the remaining candidates; stop
   after 3 failed seeds.
5. Placement: the leaf joins the level-1 group (of two or more leaves) whose centroid it is as
   close to as that group's own leaves are; otherwise it is a new level-1 group. Slots are
   appended, so every existing slot keeps its id; the new leaf takes a fresh id
   (`AddReport::new_leaves`).

The e-means update then runs with the new leaves in place. Test `update_grows_a_leaf_for_a_new_group`:
train on 2,400 rows, add 340 rows of known groups plus 60 rows of a new group; the update forms one
new leaf with a new id, at least 95% of the new rows are in it, at most 3 old rows move to it, old
rows keep their leaf ids (> 97%), the result survives save and load, and 30 scattered outliers
form no leaf. Validated on synthetic data only.

## New items and near-duplicate groups

`Hierarchy::add_photos(items, n_old, &mut dd, input, backend)` is the item-level entry when the
hierarchy clusters dedup representatives (the name comes from the first user, a photo tool; it
works on any rows). Each new item:

1. joins an existing dedup group when it shares a burst id with one of the group's items, or when
   one of its `k` neighbours among all items is an old item with `cos >= sim_threshold` (and the
   time rule), and the item is within `rep_threshold` of that group's **anchor** and passes the
   time rule to it (the star rule of `dedup`, against the same row); the closest such anchor wins.
   New items of one burst are one unit: it joins only when every one of its items fits;
2. otherwise is grouped with the other remaining new items by `dedup`, and each such group appends
   one representative.

Batch dedup builds each group around its anchor (the highest-link row) and then picks the
representative by priority and medoid. An earlier `add_photos` tested new items against the
representative, so an item next to the anchor could be refused and an item next to a prioritized
row on the rim could be accepted, both against what a fresh dedup does. Both paths now test the
anchor. A `dedup.json` written before anchors were stored has none; `add_photos` then uses the
representatives as anchors (the old behaviour for those groups) and stores them.

Old groups, anchors and representatives never change, so the hierarchy rows stay append-only, and
the new representatives go through `add`. A fresh dedup can still differ where a new item would
become the anchor of old items or would link two old groups; that needs old groups to change.
`PhotoAddReport` gives each item's leaf and outlier flag plus `joined_existing` and `new_groups`.
Tests (`crates/emeans-cluster/tests/hierarchy.rs`, `dedup.rs`): `add_photos_joins_near_duplicate_groups`
(20 near copies of old items join their sources, 10 new pairs stay together, 30 fresh items form 30
groups, a second call with the same offsets fails), `add_photos_follows_burst_ids`,
`add_photos_agrees_with_a_fresh_dedup` (a prioritized row on the rim is the representative; an item
near the anchor joins, an item near the prioritized row but outside the anchor's threshold does
not, a new burst with one unrelated item stays together outside the group; `group_of`, sizes,
anchors and representatives equal a fresh dedup) and `burst_joins_a_star_only_when_every_row_fits`.

## Save and load

`save(dir)` writes `hierarchy.json` (config, dedup parameters, ids, floors, decisions, counters,
diagnostics) and `level1.emeans` plus `group-<slot>.emeans`, each with its EMA state, into a temporary
directory that then replaces `dir`. `load` checks that models, ids and groups agree.

# Calibrate on your data

| Parameter | Default | How to calibrate |
|---|---|---|
| `DedupConfig::sim_threshold` | 0.95 | Histogram of `nn_sim`; inspect pairs around the threshold |
| `DedupConfig::max_gap_secs` | None | Check whether pairs above the threshold far apart in time are true duplicates; a gap of minutes separates retakes from similar scenes |
| `DedupConfig::k` | 10 | Raise if large bursts split into several groups |
| `AutoConfig::k1_range` | 2..40 | Bounds the stage-1 sweep and the bottom-up cuts; widen if the silhouette still rises at 40 |
| `AutoConfig::split_min_silhouette` | 0.05 | Look at `NoStructure` groups and their best silhouette in `diagnostics` |
| `AutoConfig::min_cluster_size` / `max_cluster_size` / `min_split_size` | 20 / 200 / 60 | What the user wants to browse |
| `AutoConfig::criterion1` / `criterion2` | silhouette / Davies-Bouldin | `criterion1` also picks the bottom-up cut; look at `diagnostics.level1` |
| `AutoConfig::level1_method` | `BottomUp` | `TopDown` only to compare |
| `AutoConfig::level1_max_share` | 0.5 | 1.0 turns the guard off; look at `diagnostics.level1_rejected` |
| `AutoConfig::resplit_passes` | 2 | 0 turns the re-split off; look at leaves above `max_cluster_size` |
| `AutoConfig::level1_tolerance` / `level2_tolerance` | 0 / 0.05 | How flat the real criterion curves are (`diagnostics`) |
| `AutoConfig::restarts` | 4 | Fewer if the curves are smooth |
| `IncrementalConfig::outlier_mad_k` / `outlier_min_sim` | 3.0 / None | Look at the flagged rows; a lower k flags more |
| `IncrementalConfig::grow_min_size` | 20 | The smallest new event that should get its own leaf before a full retrain |
| `IncrementalConfig::match_min_sim` | 0.8 | Cosine between sibling leaf centroids in your data |
| `IncrementalConfig` update and retrain fractions | 50 / 10% / 50% / 25% | Data growth rate |

# Reproduce

```bash
cargo test -p emeans-cluster
cargo run --release -p emeans-bench --bin autok_synth -- --seeds 40,41,42,43,44,45,46,47,48,49 --no-incremental --level1 bottomup
cargo run --release -p emeans-bench --bin autok_synth -- --seeds 40,41,42,43,44,45,46,47,48,49 --no-incremental --level1 topdown
cargo run --release -p emeans-bench --bin autok_synth -- --seeds 9,10,31,32 --no-incremental --verbose
cargo run --release -p emeans-bench --bin autok_synth -- --seeds 7,8,9,10,11,12,13,14,15 --criteria silhouette+db,silhouette,db --no-incremental --level1 topdown --resplit 0
cargo run --release -p emeans-bench --bin autok_synth -- --seeds 7,8,9 --criteria ch --no-incremental
cargo run --release -p emeans-bench --bin autok_synth -- --seeds 7,8,9 --criteria silhouette --init random --no-incremental
cargo run --release -p emeans-bench --bin autok_synth -- --seeds 40         # incremental table
cargo run --release -p emeans-bench --bin knn_bench -- 16000 768 10 5
uv run --python 3.12 bench/py/metrics_fixture.py   # regenerates crates/emeans-cluster/tests/fixtures/metrics*.{json,f32}
```

Related: [e-means algorithm](/design/emeans-algorithm.md), [core](/design/core.md).

[^schubert-elbow]: Schubert 2023, Table 1 and section 5 (Conclusion).
[^arbelaitz]: Arbelaitz et al. 2013, abstract: Silhouette, Davies-Bouldin, Calinski-Harabasz, generalized Dunn, COP and SDbw form the best group.
[^kaczynska]: Kaczynska et al. 2020, section 4: alternative decision rules improve every index.
[^luxburg-stability]: von Luxburg 2010, chapter 2, Conclusion 5 and Lemma 1.
[^mourer-stadion]: Mourer et al. 2023, sections 3 and 7.3.
[^kasarapu-mml]: Kasarapu and Allison 2015, abstract.
[^hdbscan-params]: HDBSCAN docs, "Selecting min_cluster_size".
[^bertopic-tuning]: BERTopic docs, "min_cluster_size".
[^synth-runs]: Runs on 2026-11-12, M1 Pro 16 GB, CPU backend, load average 17-28 from other jobs, synthetic data only.
