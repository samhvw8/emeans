# Results

All measured on synthetic data (bench/py/gen_data.py or generated unit rows).

* [What emeans improves over faiss](improvements-over-faiss.md) - one-page summary: what is kept, what is added, measured numbers and limits.
* [Core engine](core-engine.md) - first device-loop pass on Metal and CUDA vs CPU and faiss, parity reruns, cross-backend exactness.
* [CPU kernels](cpu-kernels.md) - the fused AVX2 top-1 on x86, Accelerate tuning on the Mac, small-op threading.
* [Metal kernels](metal-kernels.md) - GPU-timestamp profile, top-1 tile sweep and the second Metal pass on the M1 Pro.
* [CUDA kernels](cuda-kernels.md) - exact FP32 through an FP16 screen, pipelined BF16, segmented scatter on the RTX 3060.
* [Original vs core](original-vs-core.md) - fair interleaved run of faiss against the rebuilt Rust core.
* [Host-loop baseline](compare-mac-ubuntu.md) - the first comparison, before the device loop; the baseline later passes are measured against.
* [Bindings vs native Rust](bindings-bench.md) - Python, Node and Go against the facade at 200k x 768: train, assign, call cost, memory, byte equality (Linux box and Mac).
* [WebAssembly vs native](wasm-bench.md) - the wasm build in Node and headless Chrome against native Rust at d = 768, n up to 200k, k 256 and 1024: train time, memory, centroid bits (box and Mac).
