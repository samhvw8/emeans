---
type: Summary
title: What emeans improves over the original faiss e-means
description: "One page: what the Rust port keeps from faiss contrib/e_means (exact recipe and random stream), and what it adds: Metal and fast exact CUDA/CPU backends behind an open backend trait, a device-resident training loop, automatic k, near-duplicate groups, incremental updates with stable ids and a grow step, and a robust outlier flag; with the measured numbers on synthetic data and the limits."
tags: [summary, faiss, e-means, incremental, auto-k, dedup, cuda, metal, cpu]
status: draft
generated:
  by: claude-code/claude-opus-5-5
  at: '2026-11-12T22:00:00Z'
sources:
  - id: upstream
    resource: reference/faiss_e_means/UPSTREAM_COMMIT
    title: "faiss contrib/e_means at commit 2b648f06 (the original)"
---

# The original

faiss `contrib/e_means` (Python, MIT, commit 2b648f06) trains k-means with an exponential moving
average over mini-batches: batch size and step size from scaling laws, cosine step decay, revival of
dead clusters by splitting, an exact closing mean, and a spherical mode.[^upstream] It runs on NumPy
(assignment through `faiss.knn`) and on torch, with CUDA kernels (native, Triton, CuTe) that convert
FP32 input to BF16 by default. The package also has quantized (int8/fp8), IVF, sharded and distributed
variants, which emeans has not ported. faiss on the Mac GPU (MPS) fails: it keeps statistics in
float64, which MPS does not support.

# What emeans keeps

| Property | Evidence |
|---|---|
| The same recipe (batch laws, step schedule, revival, closing mean, spherical mode) | Objective within seed noise of faiss on every case; parity table worst gap 0.43% ([e-means algorithm](/design/emeans-algorithm.md)) |
| The same random draws: Rust replays NumPy's MT19937 stream | ARI 1.000 against faiss NumPy at k = 64 on 100k and 200k rows, where other ports land at the seed-to-seed floor |

# What emeans adds

## 1. Backends that run everywhere and agree bit for bit

One op contract (`core::Device`: top-1, top-k, scatter-add, EMA update, split, norms) implemented by
CPU, Metal and CUDA and checked against an f64 reference, including a production-size case and
near-tie cases ([core](/design/core.md), [adding a backend](/recipes/adding-an-emeans-backend.md)).
Same seed, same objective on every backend: Metal equals the CPU bit for bit, CUDA within 6e-11.
The set is open: any crate that implements `Device` gets the training loop and top-k through the
`Engine` trait, and plugs in as `Backend::Custom` ([crate layout](/decisions/crate-layout.md)).

| 200k x 768 (synthetic), k = 1024, train time | faiss (best of NumPy / torch / Triton) | emeans |
|---|---|---|
| Mac M1 Pro, CPU | 1.80 s (torch) | about 1.0–1.2 s (Accelerate) |
| Mac M1 Pro, GPU | does not run (MPS lacks float64) | **0.74 s** (Metal) |
| 2x Xeon E5-2683 v4, CPU | 3.73 s (torch / oneDNN) | **1.83 s** (AVX2 fused kernel, no new dependency) |
| RTX 3060, exact FP32 | 0.31 s (Triton FP32) | **0.16 s** |
| RTX 3060, BF16 (lossy, opt-in in emeans, default in faiss) | 0.15 s (Triton BF16) | **0.10 s** |

How: a device-resident loop (data, centroids and statistics stay on the GPU; the host joins only at
revival checks and epoch ends); fused kernels that never write the batch x k score matrix; on CUDA,
an exact FP32 result through an FP16 tensor-core screen plus an exact re-check of the candidate
centroids (bit-equal to the plain FP32 kernel, proven by test); on Metal, a no-copy upload that is
sound because the engine owns the data (`bind_shared` with an `Arc`). Numbers:
[original vs core](/results/original-vs-core.md), [CPU](/results/cpu-kernels.md),
[Metal](/results/metal-kernels.md), [CUDA](/results/cuda-kernels.md).

## 2. Automatic k in two levels

The original needs k. emeans sweeps k with k-means++ seeding and picks leaves by Davies-Bouldin with a
browsable-size preference, re-splits leaves that are too large and have structure, and forms level 1
by average linkage of the leaves cut by silhouette, with a guard against one giant group. On data
without broad structure it stays flat instead of inventing groups
([auto-k](/design/autok-incremental-dedup.md)). On fresh synthetic nested data, level-1 ARI 1.000
and leaf ARI 0.993-1.000 on 10 of 10 sets.

## 3. Near-duplicate groups

Bursts and retakes are grouped before clustering by an exact kNN (CPU, Metal or CUDA) and an
anti-chaining star cover around a stored anchor, optionally limited by a time gap. Adding items
later uses the same anchor rule, so `add` agrees with a fresh dedup (tested). On synthetic data with
injected bursts the groups match the bursts exactly (ARI 1.0).

## 4. Incremental training with stable ids

e-means is online by nature (EMA over mini-batches; `train_resume` continues a saved model). emeans
turns that into a maintained model (`Hierarchy::add`, `add_photos`):

| Step | When | What |
|---|---|---|
| Assign | every add | New rows go to the nearest leaf; ids never change; rows far from every leaf wait as pending |
| Grow | at each update | Pending rows that form a coherent group of at least 20 become a **new leaf** with a fresh id |
| EMA update | 50 pending outliers or 10% added rows | One short e-means pass from the current state |
| Full retrain | about 50% growth | Re-cluster, then Hungarian matching carries ids over (cosine ≥ 0.8) |

Measured (synthetic, 80% trained + 4 adds of 5%): **100% of old rows kept their leaf id**, objective
0.37% above a full retrain, **0.6 s of updates instead of 38.5 s**. The grow step test: a new group of
60 rows becomes one new leaf with at least 95% of its rows, old ids over 97% stable.

## 5. An outlier flag that carries information

A row is an outlier when its cosine to the leaf is below `median - 3 x 1.4826 x MAD` of that leaf
(pooled spread for leaves under 20 rows), not the leaf's bottom 2% by construction. A tight leaf
flags nothing, a leaf with a tail flags the tail, and small leaves can flag at all (tested: clean
training rows under 1% flagged, rows of an unseen group all flagged).

# Limits and open items

- Not ported: faiss's quantized, IVF, sharded and distributed variants.
- Auto-k, incremental updates and the grow step are validated on synthetic data only; their
  thresholds (50, 10%, 20 rows, cosine 0.8) are defaults to calibrate on real embeddings.
- BF16 paths change some labels (809 of 200k at k = 1024 on the bench data); they stay opt-in.
- Language bindings exist for Python, Node.js, Go, C and WebAssembly, each byte-identical to Rust ([bindings](/design/bindings.md), [web](/design/web.md)); none is published to a package registry yet, so they build from source.

[^upstream]: The original package as fetched for the benchmarks; its NumPy files are byte-identical to `reference/faiss_e_means`.
