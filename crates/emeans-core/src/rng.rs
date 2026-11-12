//! MT19937 with numpy's legacy `RandomState` algorithms.
//!
//! The faiss reference draws its initial centroids, epoch permutations and
//! split donors from `np.random.RandomState(seed)`. Reproducing those streams
//! bit for bit lets the Rust port start from the same centroids and visit rows
//! in the same order, so parity checks compare the algorithms, not the seeds.

const N: usize = 624;
const M: usize = 397;

/// numpy `RandomState(seed)` (MT19937, `mt19937_seed`).
#[derive(Clone)]
pub struct RandomState {
    mt: [u32; N],
    pos: usize,
}

impl RandomState {
    pub fn new(seed: u32) -> Self {
        let mut mt = [0u32; N];
        let mut s = seed;
        for (i, slot) in mt.iter_mut().enumerate() {
            *slot = s;
            s = 1_812_433_253u32
                .wrapping_mul(s ^ (s >> 30))
                .wrapping_add(i as u32 + 1);
        }
        Self { mt, pos: N }
    }

    fn generate(&mut self) {
        const UPPER: u32 = 0x8000_0000;
        const LOWER: u32 = 0x7fff_ffff;
        const MATRIX: u32 = 0x9908_b0df;
        for i in 0..N {
            let y = (self.mt[i] & UPPER) | (self.mt[(i + 1) % N] & LOWER);
            let mut v = self.mt[(i + M) % N] ^ (y >> 1);
            if y & 1 == 1 {
                v ^= MATRIX;
            }
            self.mt[i] = v;
        }
        self.pos = 0;
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.pos == N {
            self.generate();
        }
        let mut y = self.mt[self.pos];
        self.pos += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    fn next_u64(&mut self) -> u64 {
        ((self.next_u32() as u64) << 32) | self.next_u32() as u64
    }

    /// numpy `random_sample()`: 53-bit double in [0, 1).
    pub fn random_sample(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as f64;
        let b = (self.next_u32() >> 6) as f64;
        (a * 67_108_864.0 + b) / 9_007_199_254_740_992.0
    }

    /// numpy legacy `random_interval(max)`: uniform in [0, max] by masked rejection.
    pub fn random_interval(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        let mut mask = max;
        for s in [1, 2, 4, 8, 16, 32] {
            mask |= mask >> s;
        }
        if max <= 0xffff_ffff {
            loop {
                let v = (self.next_u32() as u64) & mask;
                if v <= max {
                    return v;
                }
            }
        }
        loop {
            let v = self.next_u64() & mask;
            if v <= max {
                return v;
            }
        }
    }

    /// numpy `RandomState.permutation(n)` as u32 row ids.
    pub fn permutation(&mut self, n: usize) -> Vec<u32> {
        let mut out: Vec<u32> = (0..n as u32).collect();
        for i in (1..n).rev() {
            let j = self.random_interval(i as u64) as usize;
            out.swap(i, j);
        }
        out
    }
}
