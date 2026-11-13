// Package emeans is the Go binding of emeans: e-means clustering
// (mini-batch k-means with scaling laws) on the CPU, Apple GPUs (Metal) and
// NVIDIA GPUs (CUDA), with automatic k, incremental updates and
// near-duplicate groups.
//
// It wraps the emeans C ABI (crates/emeans-ffi) with cgo. Build the static
// library first: `mise run bind:go:lib` in the repository root writes
// lib/<GOOS>_<GOARCH>/libemeans_ffi.a next to this file.
//
// Data is row-major float32 (n x d). Input slices are read in place, without
// a copy, during the call: do not change them from another goroutine until
// the call returns. Output slices are Go memory, copied once from the result.
package emeans

/*
#cgo CFLAGS: -I${SRCDIR}/../../crates/emeans-ffi/include
#cgo darwin,arm64 LDFLAGS: ${SRCDIR}/lib/darwin_arm64/libemeans_ffi.a -framework Metal -framework CoreFoundation -framework Foundation -framework Accelerate -lobjc
#cgo linux,amd64 LDFLAGS: ${SRCDIR}/lib/linux_amd64/libemeans_ffi.a -lm -ldl -lpthread
#cgo linux,arm64 LDFLAGS: ${SRCDIR}/lib/linux_arm64/libemeans_ffi.a -lm -ldl -lpthread
#include <stdlib.h>
#include "emeans.h"
*/
import "C"

import (
	"encoding/json"
	"errors"
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

// Sentinel errors; test with errors.Is. Every returned error is an *Error.
var (
	// ErrInvalidArgument: a bad length, string or JSON text.
	ErrInvalidArgument = errors.New("emeans: invalid argument")
	// ErrFailed: the library refused or failed the call (shape, config,
	// backend, I/O). The message says why.
	ErrFailed = errors.New("emeans: call failed")
	// ErrPanic: a bug in the library (a Rust panic, caught at the boundary).
	ErrPanic = errors.New("emeans: internal panic")
	// ErrClosed: the Hierarchy was closed.
	ErrClosed = errors.New("emeans: hierarchy is closed")
)

// Error is an error from the library: a status code and its message.
type Error struct {
	Code int
	Msg  string
}

func (e *Error) Error() string { return "emeans: " + e.Msg }

// Is matches the sentinel error of the code.
func (e *Error) Is(target error) bool {
	switch e.Code {
	case C.EMEANS_INVALID_ARGUMENT:
		return target == ErrInvalidArgument
	case C.EMEANS_ERROR:
		return target == ErrFailed
	case C.EMEANS_PANIC:
		return target == ErrPanic
	}
	return false
}

// call runs f on one locked OS thread, so the thread-local last error read
// after it belongs to this call.
func call(f func() C.int32_t) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	code := f()
	if code == C.EMEANS_OK {
		return nil
	}
	msg := "unknown error"
	if p := C.emeans_last_error(); p != nil {
		msg = C.GoString(p)
	}
	return &Error{Code: int(code), Msg: msg}
}

// f32ptr is the address of s's first element (nil for an empty slice): the
// library reads the caller's memory directly.
func f32ptr(s []float32) *C.float {
	if len(s) == 0 {
		return nil
	}
	return (*C.float)(unsafe.Pointer(unsafe.SliceData(s)))
}

func rows(data []float32, d int, what string) (C.size_t, error) {
	if d <= 0 {
		return 0, &Error{Code: C.EMEANS_INVALID_ARGUMENT, Msg: "d must be positive"}
	}
	if len(data)%d != 0 {
		return 0, &Error{Code: C.EMEANS_INVALID_ARGUMENT,
			Msg: fmt.Sprintf("%s has %d values, not a multiple of d = %d", what, len(data), d)}
	}
	return C.size_t(len(data) / d), nil
}

// configJSON marshals a config, or returns nil (the defaults) for nil.
func configJSON(cfg any) (*C.char, error) {
	if cfg == nil {
		return nil, nil
	}
	b, err := json.Marshal(cfg)
	if err != nil {
		return nil, fmt.Errorf("emeans: config: %w", err)
	}
	return C.CString(string(b)), nil
}

func freeStr(p *C.char) {
	if p != nil {
		C.free(unsafe.Pointer(p))
	}
}

// ------------------------------------------------------------------ results

type result struct{ p *C.EmeansResult }

func (r result) free() { C.emeans_result_free(r.p) }

func (r result) meta(v any) error {
	return json.Unmarshal([]byte(C.GoString(C.emeans_result_json(r.p))), v)
}

func (r result) array(name string, dtype C.int32_t) (unsafe.Pointer, int, error) {
	cname := C.CString(name)
	defer freeStr(cname)
	var a C.EmeansArray
	if err := call(func() C.int32_t { return C.emeans_result_array(r.p, cname, &a) }); err != nil {
		return nil, 0, err
	}
	if a.dtype != dtype {
		return nil, 0, &Error{Code: C.EMEANS_PANIC, Msg: fmt.Sprintf("array %s has dtype %d, want %d", name, a.dtype, dtype)}
	}
	return a.data, int(a.len), nil
}

func copyOut[T any](p unsafe.Pointer, n int) []T {
	out := make([]T, n)
	if n > 0 {
		copy(out, unsafe.Slice((*T)(p), n))
	}
	return out
}

func (r result) f32(name string) ([]float32, error) {
	p, n, err := r.array(name, C.EMEANS_F32)
	return copyOut[float32](p, n), err
}

func (r result) u32(name string) ([]uint32, error) {
	p, n, err := r.array(name, C.EMEANS_U32)
	return copyOut[uint32](p, n), err
}

func (r result) i32(name string) ([]int32, error) {
	p, n, err := r.array(name, C.EMEANS_I32)
	return copyOut[int32](p, n), err
}

func (r result) u8(name string) ([]uint8, error) {
	p, n, err := r.array(name, C.EMEANS_U8)
	return copyOut[uint8](p, n), err
}

// EpochStats are the statistics of one training epoch.
type EpochStats struct {
	Obj             float64 `json:"obj"`
	Time            float64 `json:"time"`
	ImbalanceFactor float64 `json:"imbalance_factor"`
	NSplit          int     `json:"nsplit"`
}

// TrainResult is the output of Train.
type TrainResult struct {
	Centroids []float32 // K x D, row-major
	K, D      int
	// Objective is free: the last epoch's moving objective as a mean (that
	// epoch's obj, the sum of the squared distance or 2 - 2 cos from each row
	// it processed to its centroid when the row's batch was assigned, divided
	// by those rows). A little above ExactObjective.
	Objective float64
	// ExactObjective is the mean over all rows against the final centroids
	// (one extra assign pass); nil unless TrainConfig.ExactObjective is set.
	ExactObjective *float64
	Stats          []EpochStats
	// Labels is the last-pass label per row (-1 outside the training
	// subset); nil when CaptureAssignments is off.
	Labels  []int32
	Backend string // the backend that ran
}

// Assignment is the output of Assign.
type Assignment struct {
	Labels []uint32  // nearest centroid per row
	Scores []float32 // squared distance (L2) or inner product per row
}

// Labels are the labels of rows under a Hierarchy.
type Labels struct {
	L1      []uint32  // stable level-1 id
	L2      []uint32  // stable level-2 id inside the level-1 group
	Sim1    []float32 // cosine to the level-1 centroid
	Sim     []float32 // cosine to the leaf centroid
	Outlier []bool    // below the leaf's outlier floor
}

func (r result) labels() (*Labels, error) {
	var l Labels
	var err error
	var out []uint8
	for _, step := range []func() error{
		func() error { l.L1, err = r.u32("l1"); return err },
		func() error { l.L2, err = r.u32("l2"); return err },
		func() error { l.Sim1, err = r.f32("sim1"); return err },
		func() error { l.Sim, err = r.f32("sim"); return err },
		func() error { out, err = r.u8("outlier"); return err },
	} {
		if err := step(); err != nil {
			return nil, err
		}
	}
	l.Outlier = make([]bool, len(out))
	for i, v := range out {
		l.Outlier[i] = v != 0
	}
	return &l, nil
}

// --------------------------------------------------------------------- calls

// Train trains e-means with k centroids on data (n x d). A nil cfg uses
// DefaultTrainConfig().
func Train(data []float32, d, k int, cfg *TrainConfig) (*TrainResult, error) {
	n, err := rows(data, d, "data")
	if err != nil {
		return nil, err
	}
	cj, err := configJSON(cfgOrNil(cfg))
	if err != nil {
		return nil, err
	}
	defer freeStr(cj)
	var r result
	err = call(func() C.int32_t {
		return C.emeans_train(f32ptr(data), n, C.size_t(d), C.size_t(k), cj, &r.p)
	})
	runtime.KeepAlive(data)
	if err != nil {
		return nil, err
	}
	defer r.free()
	out := &TrainResult{}
	var meta struct {
		K, D           int
		Objective      float64
		ExactObjective *float64 `json:"exact_objective"`
		Stats          []EpochStats
		Backend        string
	}
	if err := r.meta(&meta); err != nil {
		return nil, err
	}
	out.K, out.D, out.Objective, out.Stats, out.Backend = meta.K, meta.D, meta.Objective, meta.Stats, meta.Backend
	out.ExactObjective = meta.ExactObjective
	if out.Centroids, err = r.f32("centroids"); err != nil {
		return nil, err
	}
	if cfg == nil || cfg.CaptureAssignments {
		if out.Labels, err = r.i32("labels"); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// Metric of Assign.
type Metric string

const (
	L2           Metric = "l2"            // squared Euclidean distance, smallest wins
	InnerProduct Metric = "inner_product" // inner product (cosine for unit rows), largest wins
)

// Assign finds the nearest of centroids (k x d) for every row of data (n x d)
// on backend ("" = "cpu").
func Assign(data, centroids []float32, d int, metric Metric, backend string) (*Assignment, error) {
	n, err := rows(data, d, "data")
	if err != nil {
		return nil, err
	}
	k, err := rows(centroids, d, "centroids")
	if err != nil {
		return nil, err
	}
	cm := C.CString(string(metric))
	defer freeStr(cm)
	var cb *C.char
	if backend != "" {
		cb = C.CString(backend)
		defer freeStr(cb)
	}
	var r result
	err = call(func() C.int32_t {
		return C.emeans_assign(f32ptr(data), n, C.size_t(d), f32ptr(centroids), k, cm, cb, &r.p)
	})
	runtime.KeepAlive(data)
	runtime.KeepAlive(centroids)
	if err != nil {
		return nil, err
	}
	defer r.free()
	a := &Assignment{}
	if a.Labels, err = r.u32("labels"); err != nil {
		return nil, err
	}
	if a.Scores, err = r.f32("scores"); err != nil {
		return nil, err
	}
	return a, nil
}

// DedupSide is optional per-row side information for Dedup; each slice is
// nil or has one value per row.
type DedupSide struct {
	Created  []float64 // creation time (Unix seconds); NaN = unknown
	Burst    []int64   // burst id; negative = none. Rows of one burst share a group
	Priority []float32 // representative preference; higher wins
}

// DedupResult is the output of Dedup.
type DedupResult struct {
	GroupOf         []uint32  // group of every row (groups numbered by their lowest row)
	Representatives []uint32  // representative row of every group
	Anchors         []uint32  // anchor (star center) row of every group
	Sizes           []uint32  // members per group
	NNSim           []float32 // cosine of every row to its nearest other row
	NGroups         int
	LinkedPairs     int
	SplitComponents int
}

func sidePtr[T, P any](s []T, n C.size_t, what string) (*P, error) {
	if s == nil {
		return nil, nil
	}
	if len(s) != int(n) {
		return nil, &Error{Code: C.EMEANS_INVALID_ARGUMENT, Msg: fmt.Sprintf("%s has %d values for %d rows", what, len(s), n)}
	}
	if len(s) == 0 {
		return nil, nil
	}
	return (*P)(unsafe.Pointer(unsafe.SliceData(s))), nil
}

// Dedup finds near-duplicate groups of the unit rows of data (n x d). A nil
// cfg uses DefaultDedupConfig().
func Dedup(data []float32, d int, cfg *DedupConfig, side DedupSide) (*DedupResult, error) {
	n, err := rows(data, d, "data")
	if err != nil {
		return nil, err
	}
	created, err := sidePtr[float64, C.double](side.Created, n, "Created")
	if err != nil {
		return nil, err
	}
	burst, err := sidePtr[int64, C.int64_t](side.Burst, n, "Burst")
	if err != nil {
		return nil, err
	}
	priority, err := sidePtr[float32, C.float](side.Priority, n, "Priority")
	if err != nil {
		return nil, err
	}
	cj, err := configJSON(cfgOrNil(cfg))
	if err != nil {
		return nil, err
	}
	defer freeStr(cj)
	var r result
	err = call(func() C.int32_t {
		return C.emeans_dedup(f32ptr(data), n, C.size_t(d), cj, created, burst, priority, &r.p)
	})
	runtime.KeepAlive(data)
	runtime.KeepAlive(side)
	if err != nil {
		return nil, err
	}
	defer r.free()
	g := &DedupResult{}
	var meta struct {
		NGroups         int `json:"n_groups"`
		LinkedPairs     int `json:"linked_pairs"`
		SplitComponents int `json:"split_components"`
	}
	if err := r.meta(&meta); err != nil {
		return nil, err
	}
	g.NGroups, g.LinkedPairs, g.SplitComponents = meta.NGroups, meta.LinkedPairs, meta.SplitComponents
	for _, f := range []struct {
		dst  *[]uint32
		name string
	}{{&g.GroupOf, "group_of"}, {&g.Representatives, "representatives"}, {&g.Anchors, "anchors"}, {&g.Sizes, "sizes"}} {
		if *f.dst, err = r.u32(f.name); err != nil {
			return nil, err
		}
	}
	if g.NNSim, err = r.f32("nn_sim"); err != nil {
		return nil, err
	}
	return g, nil
}

// Backends lists the backends this build can use on this machine ("cpu" first).
func Backends() ([]string, error) {
	var p *C.char
	if err := call(func() C.int32_t { return C.emeans_backends(&p) }); err != nil {
		return nil, err
	}
	defer C.emeans_string_free(p)
	var out []string
	return out, json.Unmarshal([]byte(C.GoString(p)), &out)
}

// Version is the library version.
func Version() string { return C.GoString(C.emeans_version()) }

// ----------------------------------------------------------------- Hierarchy

// Hierarchy is a two-level clustering that labels new rows and updates
// itself. Its methods are safe for concurrent use (they run one at a time).
// Close it when done; a cleanup frees it if it becomes unreachable first.
type Hierarchy struct {
	mu      sync.Mutex
	p       *C.EmeansHierarchy
	cleanup runtime.Cleanup
}

func newHierarchy(p *C.EmeansHierarchy) *Hierarchy {
	h := &Hierarchy{p: p}
	h.cleanup = runtime.AddCleanup(h, func(p *C.EmeansHierarchy) { C.emeans_hierarchy_free(p) }, p)
	return h
}

// HierarchyInfo describes a Hierarchy.
type HierarchyInfo struct {
	D     int   `json:"d"`
	NSeen int   `json:"n_seen"` // rows trained on or added
	K1    int   `json:"k1"`
	K2    []int `json:"k2"` // level-2 k per level-1 group (1 = not split, 0 = empty)
}

// AutoCluster clusters data (n x d, unit rows) in two levels with automatic
// k. A nil cfg uses DefaultAutoConfig().
func AutoCluster(data []float32, d int, cfg *AutoConfig) (*Hierarchy, *Labels, error) {
	n, err := rows(data, d, "data")
	if err != nil {
		return nil, nil, err
	}
	cj, err := configJSON(cfgOrNil(cfg))
	if err != nil {
		return nil, nil, err
	}
	defer freeStr(cj)
	var hp *C.EmeansHierarchy
	var r result
	err = call(func() C.int32_t {
		return C.emeans_auto_cluster(f32ptr(data), n, C.size_t(d), cj, &hp, &r.p)
	})
	runtime.KeepAlive(data)
	if err != nil {
		return nil, nil, err
	}
	defer r.free()
	h := newHierarchy(hp)
	l, err := r.labels()
	if err != nil {
		h.Close()
		return nil, nil, err
	}
	return h, l, nil
}

// LoadHierarchy reads a hierarchy written by Save onto backend ("" = "cpu").
func LoadHierarchy(path, backend string) (*Hierarchy, error) {
	cp := C.CString(path)
	defer freeStr(cp)
	var cb *C.char
	if backend != "" {
		cb = C.CString(backend)
		defer freeStr(cb)
	}
	var hp *C.EmeansHierarchy
	if err := call(func() C.int32_t { return C.emeans_hierarchy_load(cp, cb, &hp) }); err != nil {
		return nil, err
	}
	return newHierarchy(hp), nil
}

// AddReport is the output of Hierarchy.Add.
type AddReport struct {
	Action          string  `json:"action"` // "assign_only", "ema_update" or "full_retrain"
	Labels          *Labels `json:"-"`      // all rows after the action (old rows can move)
	NewOutliers     int     `json:"new_outliers"`
	Pending         int     `json:"pending"`
	OutlierFraction float64 `json:"outlier_fraction"`
	// NewLeaves are the stable (l1, l2) ids of leaves formed from pending outliers.
	NewLeaves [][2]uint32 `json:"new_leaves"`
}

// Add labels the rows of dataAll after the first NSeen and updates the model
// when its policy says so. Pass every row the model has seen (in order), then
// the new rows.
func (h *Hierarchy) Add(dataAll []float32) (*AddReport, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.p == nil {
		return nil, ErrClosed
	}
	info, err := h.info()
	if err != nil {
		return nil, err
	}
	n, err := rows(dataAll, info.D, "dataAll")
	if err != nil {
		return nil, err
	}
	var r result
	err = call(func() C.int32_t { return C.emeans_hierarchy_add(h.p, f32ptr(dataAll), n, &r.p) })
	runtime.KeepAlive(dataAll)
	if err != nil {
		return nil, err
	}
	defer r.free()
	rep := &AddReport{}
	if err := r.meta(rep); err != nil {
		return nil, err
	}
	if rep.Labels, err = r.labels(); err != nil {
		return nil, err
	}
	return rep, nil
}

// Save writes the hierarchy to directory path (replaced as a whole).
func (h *Hierarchy) Save(path string) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.p == nil {
		return ErrClosed
	}
	cp := C.CString(path)
	defer freeStr(cp)
	return call(func() C.int32_t { return C.emeans_hierarchy_save(h.p, cp) })
}

// Info describes the hierarchy.
func (h *Hierarchy) Info() (HierarchyInfo, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.p == nil {
		return HierarchyInfo{}, ErrClosed
	}
	return h.info()
}

func (h *Hierarchy) info() (HierarchyInfo, error) {
	var p *C.char
	var info HierarchyInfo
	if err := call(func() C.int32_t { return C.emeans_hierarchy_info(h.p, &p) }); err != nil {
		return info, err
	}
	defer C.emeans_string_free(p)
	return info, json.Unmarshal([]byte(C.GoString(p)), &info)
}

// Close frees the hierarchy. Calling it again does nothing.
func (h *Hierarchy) Close() error {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.p != nil {
		h.cleanup.Stop()
		C.emeans_hierarchy_free(h.p)
		h.p = nil
	}
	return nil
}
