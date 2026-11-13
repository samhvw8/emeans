package emeans

// Go binding tests: results equal the Rust facade byte for byte.
//
// Needs the fixture written by the Rust side (`mise run bind:go:test` does both):
//   cargo run --release -p emeans-bindings-common --example write_fixture -- DIR
//   EMEANS_FIXTURE=DIR go test ./...

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"
	"unsafe"
)

type fixture struct {
	dir  string
	N    int `json:"n"`
	D    int `json:"d"`
	K    int `json:"k"`
	N0   int `json:"n0"`
	Meta struct {
		Train struct {
			Config    json.RawMessage `json:"config"`
			Objective float64         `json:"objective"`
			Exact     float64         `json:"exact_objective"`
			Obj       []float64       `json:"obj"`
		} `json:"train"`
		Assign struct {
			Metric string `json:"metric"`
		} `json:"assign"`
		Auto struct {
			Config json.RawMessage `json:"config"`
			K1     int             `json:"k1"`
			K2     []int           `json:"k2"`
		} `json:"auto"`
		Add struct {
			Action string `json:"action"`
		} `json:"add"`
		Dedup struct {
			Config      json.RawMessage `json:"config"`
			NGroups     int             `json:"n_groups"`
			LinkedPairs int             `json:"linked_pairs"`
		} `json:"dedup"`
	}
}

func load(t *testing.T) *fixture {
	t.Helper()
	dir := os.Getenv("EMEANS_FIXTURE")
	b, err := os.ReadFile(filepath.Join(dir, "fixture.json"))
	if dir == "" || err != nil {
		t.Skip("set EMEANS_FIXTURE to a fixture directory")
	}
	f := &fixture{dir: dir}
	if err := json.Unmarshal(b, f); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &f.Meta); err != nil {
		t.Fatal(err)
	}
	return f
}

func (f *fixture) read(t *testing.T, name string) []byte {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(f.dir, name))
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func readAs[T any](t *testing.T, f *fixture, name string) []T {
	b := f.read(t, name)
	var zero T
	out := make([]T, len(b)/int(unsafe.Sizeof(zero)))
	if err := binary.Read(bytes.NewReader(b), binary.LittleEndian, out); err != nil {
		t.Fatal(err)
	}
	return out
}

func le(t *testing.T, v any) []byte {
	var buf bytes.Buffer
	if err := binary.Write(&buf, binary.LittleEndian, v); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func same(t *testing.T, f *fixture, name string, v any) {
	t.Helper()
	if !bytes.Equal(le(t, v), f.read(t, name)) {
		t.Errorf("%s differs from Rust", name)
	}
}

func sameLabels(t *testing.T, f *fixture, prefix string, l *Labels) {
	t.Helper()
	same(t, f, prefix+".l1.u32", l.L1)
	same(t, f, prefix+".l2.u32", l.L2)
	same(t, f, prefix+".sim1.f32", l.Sim1)
	same(t, f, prefix+".sim.f32", l.Sim)
	out := make([]uint8, len(l.Outlier))
	for i, o := range l.Outlier {
		if o {
			out[i] = 1
		}
	}
	same(t, f, prefix+".outlier.u8", out)
}

// overrides applies a fixture's JSON config onto a default Go config.
func overrides[T any](t *testing.T, base T, raw json.RawMessage) *T {
	t.Helper()
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&base); err != nil {
		t.Fatal(err)
	}
	return &base
}

func TestTrainAndAssignEqualRust(t *testing.T) {
	f := load(t)
	x := readAs[float32](t, f, "x.f32")
	cfg := overrides(t, DefaultTrainConfig(), f.Meta.Train.Config)
	tr, err := Train(x, f.D, f.K, cfg)
	if err != nil {
		t.Fatal(err)
	}
	same(t, f, "train.centroids.f32", tr.Centroids)
	same(t, f, "train.labels.i32", tr.Labels)
	if tr.Objective != f.Meta.Train.Objective || tr.ExactObjective != nil {
		t.Errorf("objective %v (exact %v), Rust %v", tr.Objective, tr.ExactObjective, f.Meta.Train.Objective)
	}
	cfg.ExactObjective = true
	te, err := Train(x, f.D, f.K, cfg)
	if err != nil {
		t.Fatal(err)
	}
	same(t, f, "train.centroids.f32", te.Centroids)
	if te.ExactObjective == nil || *te.ExactObjective != f.Meta.Train.Exact {
		t.Errorf("exact objective %v, Rust %v", te.ExactObjective, f.Meta.Train.Exact)
	}
	cfg.ExactObjective = false
	for i, s := range tr.Stats {
		if s.Obj != f.Meta.Train.Obj[i] {
			t.Errorf("epoch %d objective %v, Rust %v", i, s.Obj, f.Meta.Train.Obj[i])
		}
	}
	a, err := Assign(x, tr.Centroids, f.D, Metric(f.Meta.Assign.Metric), "")
	if err != nil {
		t.Fatal(err)
	}
	same(t, f, "assign.labels.u32", a.Labels)
	same(t, f, "assign.scores.f32", a.Scores)
}

func TestAutoSaveLoadAddEqualRust(t *testing.T) {
	f := load(t)
	x := readAs[float32](t, f, "x.f32")
	cfg := overrides(t, DefaultAutoConfig(), f.Meta.Auto.Config)
	h, l, err := AutoCluster(x[:f.N0*f.D], f.D, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()
	sameLabels(t, f, "auto", l)
	info, err := h.Info()
	if err != nil {
		t.Fatal(err)
	}
	if info.K1 != f.Meta.Auto.K1 || info.NSeen != f.N0 || info.D != f.D {
		t.Errorf("info %+v", info)
	}

	path := filepath.Join(t.TempDir(), "model")
	if err := h.Save(path); err != nil {
		t.Fatal(err)
	}
	h2, err := LoadHierarchy(path, "cpu")
	if err != nil {
		t.Fatal(err)
	}
	r2, err := h2.Add(x)
	if err != nil {
		t.Fatal(err)
	}
	sameLabels(t, f, "add", r2.Labels)
	h2.Close()

	r, err := h.Add(x)
	if err != nil {
		t.Fatal(err)
	}
	sameLabels(t, f, "add", r.Labels)
	if r.Action != f.Meta.Add.Action {
		t.Errorf("action %q, Rust %q", r.Action, f.Meta.Add.Action)
	}
	if _, err := h.Add(x); !errors.Is(err, ErrFailed) {
		t.Errorf("adding seen rows again: %v", err)
	}
}

func TestDedupEqualsRust(t *testing.T) {
	f := load(t)
	x := readAs[float32](t, f, "x.f32")
	cfg := overrides(t, DefaultDedupConfig(), f.Meta.Dedup.Config)
	side := DedupSide{
		Created:  readAs[float64](t, f, "created.f64"),
		Priority: readAs[float32](t, f, "priority.f32"),
	}
	g, err := Dedup(x, f.D, cfg, side)
	if err != nil {
		t.Fatal(err)
	}
	same(t, f, "dedup.group_of.u32", g.GroupOf)
	same(t, f, "dedup.representatives.u32", g.Representatives)
	same(t, f, "dedup.sizes.u32", g.Sizes)
	same(t, f, "dedup.nn_sim.f32", g.NNSim)
	if g.NGroups != f.Meta.Dedup.NGroups || g.LinkedPairs != f.Meta.Dedup.LinkedPairs {
		t.Errorf("groups %d pairs %d", g.NGroups, g.LinkedPairs)
	}
}

func TestMisuse(t *testing.T) {
	x := make([]float32, 40)
	var e *Error

	_, err := Train(x, 3, 2, nil)
	if !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("bad shape: %v", err)
	}
	cfg := DefaultTrainConfig()
	cfg.Backend = "tpu"
	_, err = Train(x, 4, 2, &cfg)
	if !errors.Is(err, ErrFailed) || !errors.As(err, &e) || !bytes.Contains([]byte(e.Msg), []byte("unknown backend")) {
		t.Errorf("bad backend: %v", err)
	}
	if _, err := Assign(x, x[:8], 4, "manhattan", ""); !errors.Is(err, ErrFailed) {
		t.Errorf("bad metric: %v", err)
	}
	if _, err := Dedup(x, 4, nil, DedupSide{Priority: []float32{1}}); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("short side array: %v", err)
	}

	h, _, err := AutoCluster(unitRows(200, 4), 4, &AutoConfig{})
	if err == nil || h != nil {
		t.Errorf("a zero AutoConfig must be refused (k1_range 0..0): %v", err)
	}
	h, _, err = AutoCluster(unitRows(200, 4), 4, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := h.Close(); err != nil {
		t.Fatal(err)
	}
	if err := h.Close(); err != nil {
		t.Errorf("second Close: %v", err)
	}
	if _, err := h.Add(x); !errors.Is(err, ErrClosed) {
		t.Errorf("use after Close: %v", err)
	}
}

func TestDefaultsMatchTheLibrary(t *testing.T) {
	// Each default decodes with DisallowUnknownFields (no library key is
	// missing here) and goes back without an unknown-key error (no Go key is
	// unknown there).
	tc := DefaultTrainConfig()
	if tc.Backend != "cpu" || tc.NEpochs != 5 || !tc.CaptureAssignments {
		t.Errorf("train defaults %+v", tc)
	}
	ac := DefaultAutoConfig()
	if ac.K1Range != [2]int{2, 40} || ac.Incremental.UpdateMinOutliers != 50 {
		t.Errorf("auto defaults %+v", ac)
	}
	if _, err := Train(unitRows(50, 4), 4, 3, &tc); err != nil {
		t.Errorf("train with the Go defaults: %v", err)
	}
	dc := DefaultDedupConfig()
	if _, err := Dedup(unitRows(50, 4), 4, &dc, DedupSide{}); err != nil {
		t.Errorf("dedup with the Go defaults: %v", err)
	}
	b, err := Backends()
	if err != nil || b[0] != "cpu" {
		t.Errorf("backends %v %v", b, err)
	}
}

func TestInputsAreNotCopied(t *testing.T) {
	x := make([]float32, 16)
	if uintptr(unsafe.Pointer(f32ptr(x))) != uintptr(unsafe.Pointer(&x[0])) {
		t.Error("f32ptr must pass the slice's own memory")
	}
	if f32ptr(nil) != nil {
		t.Error("an empty slice passes NULL")
	}
}

func TestMetalTrainsToTheCPUObjective(t *testing.T) {
	if os.Getenv("EMEANS_TEST_METAL") != "1" {
		t.Skip("set EMEANS_TEST_METAL=1 (holds the GPU)")
	}
	f := load(t)
	x := readAs[float32](t, f, "x.f32")
	cfg := overrides(t, DefaultTrainConfig(), f.Meta.Train.Config)
	cfg.Backend = "metal"
	tr, err := Train(x, f.D, f.K, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if tr.Backend != "metal-fused" || math.Abs(tr.Objective-f.Meta.Train.Objective)/f.Meta.Train.Objective > 1e-4 {
		t.Errorf("metal %s objective %v, cpu %v", tr.Backend, tr.Objective, f.Meta.Train.Objective)
	}
}

// unitRows is a small deterministic data set of unit rows.
func unitRows(n, d int) []float32 {
	x := make([]float32, n*d)
	for i := range n {
		var s float64
		for j := range d {
			v := math.Sin(float64(i*d+j)*0.7) + float64(i%5)
			x[i*d+j] = float32(v)
			s += v * v
		}
		for j := range d {
			x[i*d+j] /= float32(math.Sqrt(s))
		}
	}
	return x
}
