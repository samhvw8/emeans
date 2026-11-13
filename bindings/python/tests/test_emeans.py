"""Python binding tests: results equal the Rust facade byte for byte.

Needs the fixture written by the Rust side (`mise run bind:py:test` does both):
    cargo run --release -p emeans-bindings-common --example write_fixture -- DIR
    EMEANS_FIXTURE=DIR python -m unittest discover -s tests
"""

import array
import json
import os
import tempfile
import unittest
from pathlib import Path

import emeans
from emeans import _emeans

try:
    import numpy as np
except ImportError:  # NumPy is optional; its tests skip without it.
    np = None

FIXTURE = Path(os.environ.get("EMEANS_FIXTURE", ""))


def read(name: str) -> bytes:
    return (FIXTURE / name).read_bytes()


@unittest.skipUnless((FIXTURE / "fixture.json").is_file(), "set EMEANS_FIXTURE")
class SameAsRust(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta = json.loads(read("fixture.json"))
        cls.d = cls.meta["d"]
        cls.x = read("x.f32")  # bytes: the zero-copy raw path

    def same(self, name: str, arr: emeans.Array):
        self.assertEqual(bytes(memoryview(arr)), read(name), f"{name} differs from Rust")

    def same_labels(self, prefix: str, l: emeans.Labels):
        files = {"l1": "u32", "l2": "u32", "sim1": "f32", "sim": "f32", "outlier": "u8"}
        for field, ext in files.items():
            self.same(f"{prefix}.{field}.{ext}", getattr(l, field))

    def test_train_and_assign(self):
        m = self.meta
        t = emeans.train(self.x, m["k"], d=self.d, **m["train"]["config"])
        self.same("train.centroids.f32", t.centroids)
        self.same("train.labels.i32", t.labels)
        self.assertEqual(t.objective, m["train"]["objective"])
        self.assertIsNone(t.exact_objective)
        te = emeans.train(self.x, m["k"], d=self.d, exact_objective=True, **m["train"]["config"])
        self.same("train.centroids.f32", te.centroids)
        self.assertEqual(te.exact_objective, m["train"]["exact_objective"])
        self.assertEqual([s.obj for s in t.stats], m["train"]["obj"])
        self.assertEqual(t.centroids.shape, [m["k"], self.d])
        self.assertEqual(t.centroids.dtype, "float32")
        # assign takes d from the 2-D centroids buffer.
        a = emeans.assign(self.x, t.centroids, metric=m["assign"]["metric"])
        self.same("assign.labels.u32", a.labels)
        self.same("assign.scores.f32", a.scores)

    def test_auto_save_load_add(self):
        m = self.meta
        n0 = m["n0"]
        mv = memoryview(self.x).cast("f")  # a float32 memoryview
        h, labels = emeans.auto_cluster(mv[: n0 * self.d], d=self.d, **m["auto"]["config"])
        self.same_labels("auto", labels)
        self.assertEqual(h.chosen_k, (m["auto"]["k1"], m["auto"]["k2"]))
        self.assertEqual(h.n_seen, n0)
        with tempfile.TemporaryDirectory() as tmp:
            h.save(Path(tmp) / "model")
            h2 = emeans.Hierarchy.load(Path(tmp) / "model", backend="cpu")
        r2 = h2.add(mv)
        self.same_labels("add", r2.labels)
        r = h.add(mv)
        self.same_labels("add", r.labels)
        self.assertEqual(r.action, m["add"]["action"])
        self.assertEqual(h.n_seen, m["n"])
        with self.assertRaises(emeans.EmeansError):
            h.add(mv)  # rows already seen: nothing new

    def test_dedup(self):
        m = self.meta
        created = array.array("d", read("created.f64"))
        priority = array.array("f", read("priority.f32"))
        g = emeans.dedup(self.x, d=self.d, created=created, priority=priority, **m["dedup"]["config"])
        for f in ("group_of", "representatives", "sizes"):
            self.same(f"dedup.{f}.u32", getattr(g, f))
        self.same("dedup.nn_sim.f32", g.nn_sim)
        self.assertEqual(g.n_groups, m["dedup"]["n_groups"])
        self.assertEqual(g.linked_pairs, m["dedup"]["linked_pairs"])

    @unittest.skipIf(np is None, "NumPy not installed")
    def test_numpy_in_and_out(self):
        m = self.meta
        x = np.frombuffer(self.x, dtype=np.float32).reshape(-1, self.d)
        t = emeans.train(x, m["k"], **m["train"]["config"])  # d from the shape
        self.same("train.centroids.f32", t.centroids)
        c = t.centroids.numpy()
        self.assertEqual((c.shape, c.dtype), ((m["k"], self.d), np.float32))
        self.assertFalse(c.flags.writeable)
        self.assertTrue(np.shares_memory(c, np.asarray(t.centroids)))  # no copy out
        self.assertEqual(t.labels.numpy().dtype, np.int32)


class Misuse(unittest.TestCase):
    def test_errors(self):
        x = array.array("f", [0.0] * 40)
        with self.assertRaisesRegex(emeans.EmeansError, "sead"):
            emeans.train(x, 2, d=4, sead=1)
        with self.assertRaisesRegex(emeans.EmeansError, "multiple of d"):
            emeans.train(x, 2, d=3)
        with self.assertRaisesRegex(ValueError, "pass d"):
            emeans.train(x, 2)
        with self.assertRaisesRegex(emeans.EmeansError, "unknown backend"):
            emeans.train(x, 2, d=4, backend="tpu")
        with self.assertRaisesRegex(TypeError, "float32"):
            emeans.train(array.array("d", [0.0] * 40), 2, d=4)
        with self.assertRaisesRegex(TypeError, "buffer protocol"):
            emeans.train([0.0] * 40, 2, d=4)
        with self.assertRaisesRegex(emeans.EmeansError, "metric"):
            emeans.assign(x, x[:8], d=4, metric="manhattan")
        with self.assertRaisesRegex(emeans.EmeansError, "unknown config kind"):
            emeans.default_config("nope")
        self.assertTrue(issubclass(emeans.EmeansError, ValueError))

    @unittest.skipIf(np is None, "NumPy not installed")
    def test_numpy_misuse(self):
        x = np.zeros((10, 4), dtype=np.float32)
        with self.assertRaisesRegex(BufferError, "contiguous"):
            emeans.train(x[:, :2], 2)
        with self.assertRaisesRegex(ValueError, "columns"):
            emeans.train(x, 2, d=5)
        with self.assertRaisesRegex(TypeError, "float32"):
            emeans.train(x.astype(np.float64), 2)
        t = emeans.train(np.random.default_rng(0).random((10, 4), dtype=np.float32), 2)
        with self.assertRaises(ValueError):
            t.centroids.numpy()[0, 0] = 1.0  # read-only


class ZeroCopy(unittest.TestCase):
    def test_inputs_are_read_in_place(self):
        a = array.array("f", range(64))
        self.assertEqual(_emeans._buffer_address(a), a.buffer_info()[0])
        mv = memoryview(a)
        self.assertEqual(_emeans._buffer_address(mv), a.buffer_info()[0])
        ba = bytearray(a.tobytes())
        addr = _emeans._buffer_address(ba)
        self.assertEqual(addr, _emeans._buffer_address(memoryview(ba)))
        b = bytes(ba)
        self.assertEqual(_emeans._buffer_address(b), _emeans._buffer_address(memoryview(b)))

    @unittest.skipIf(np is None, "NumPy not installed")
    def test_numpy_input_is_read_in_place(self):
        x = np.arange(64, dtype=np.float32).reshape(16, 4)
        self.assertEqual(_emeans._buffer_address(x), x.ctypes.data)

    def test_outputs_are_buffers(self):
        x = array.array("f", [float(i % 7) for i in range(80)])
        a = emeans.assign(x, x[:8], d=4)
        self.assertEqual(memoryview(a.labels).format, "I")
        self.assertEqual(len(a.labels), 20)
        self.assertEqual(a.labels.tolist()[:2], [0, 1])
        self.assertEqual(array.array("f", bytes(memoryview(a.scores)))[0], 0.0)


class Info(unittest.TestCase):
    def test_defaults_and_backends(self):
        cfg = emeans.default_config("auto")
        self.assertEqual(cfg["backend"], "cpu")
        self.assertEqual(cfg["k1_range"], [2, 40])
        self.assertIn("incremental", cfg)
        self.assertNotIn("d", emeans.default_config("train"))
        self.assertEqual(emeans.backends()[0], "cpu")


class StubsMatch(unittest.TestCase):
    """The .pyi stubs list what the module has, and the config TypedDicts the library's keys."""

    def test_stubs(self):
        import ast

        stub_file = Path(emeans.__file__).with_name("_emeans.pyi")
        body = {n.name: n for n in ast.parse(stub_file.read_text()).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}

        def members(node):
            return {x.name if isinstance(x, ast.FunctionDef) else x.target.id
                    for x in node.body if isinstance(x, (ast.FunctionDef, ast.AnnAssign))}

        public = {k for k in dir(_emeans) if not k.startswith("__")}
        self.assertEqual(public - set(body), set())
        for cls in ("Array", "EpochStats", "TrainResult", "Assignment", "Labels", "AddReport", "DedupResult", "Hierarchy"):
            runtime = {k for k in dir(getattr(_emeans, cls)) if not k.startswith("_")}
            self.assertEqual(runtime, members(body[cls]) - {"__len__", "__buffer__"}, cls)
        for kind, td in (("train", "TrainConfig"), ("auto", "AutoConfig"), ("dedup", "DedupConfig")):
            self.assertEqual(set(emeans.default_config(kind)), members(body[td]), td)
        incremental = emeans.default_config("auto")["incremental"]
        self.assertEqual(set(incremental), members(body["IncrementalConfig"]))


@unittest.skipUnless(os.environ.get("EMEANS_TEST_METAL") == "1", "set EMEANS_TEST_METAL=1 (holds the GPU)")
class Metal(unittest.TestCase):
    def test_metal_trains_to_the_cpu_objective(self):
        x = read("x.f32")
        meta = json.loads(read("fixture.json"))
        t = emeans.train(x, meta["k"], d=meta["d"], backend="metal", **meta["train"]["config"])
        self.assertEqual(t.backend, "metal-fused")
        cpu = meta["train"]["objective"]  # both moving objectives: same epochs, same rows
        self.assertLess(abs(t.objective - cpu) / cpu, 1e-4)


if __name__ == "__main__":
    unittest.main()
