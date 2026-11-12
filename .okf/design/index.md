# Design

* [e-means core](core.md) - the Device op contract, the device-resident training loop and the Engine trait every backend runs on.
* [e-means algorithm](emeans-algorithm.md) - the Rust port of faiss NumpyEmeans: API, deviations, kernels, parity and timing.
* [Automatic k, incremental updates and near-duplicate groups](autok-incremental-dedup.md) - the emeans-cluster algorithms, their defaults, rejected alternatives and synthetic validation.
