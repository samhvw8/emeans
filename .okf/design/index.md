# Design

* [e-means core](core.md) - the Device op contract, the device-resident training loop and the Engine trait every backend runs on.
* [e-means algorithm](emeans-algorithm.md) - the Rust port of faiss NumpyEmeans: API, deviations, kernels, parity and timing.
* [Automatic k, incremental updates and near-duplicate groups](autok-incremental-dedup.md) - the emeans-cluster algorithms, their defaults, rejected alternatives and synthetic validation.
* [Language bindings](bindings.md) - Python, Node.js and Go over one plain Rust layer and a C ABI: JSON config overrides, zero-copy inputs, byte-for-byte tests against the facade.
* [Web version](web.md) - the CPU backend on WebAssembly with a JavaScript/TypeScript API for the browser and Node.js: same bits as native `cpu-fused`, size, timings, and the WebGPU and threads research.
* [C API](c-api.md) - crates/emeans-ffi as a public C library: functions, result mechanism, errors, ownership, ABI stability, dist tree with pkg-config and CMake.
