// Command bench is the Go side of the binding benchmark
// (bindings/bench/run.py).
//
// Usage: go run ./examples/bench <data.f32> <d> <k> <backend> <mode>; prints
// one JSON line. Mode "full" times train and assign; "call" only the per-call
// probe. The data is read into a []float32 first: timings cover only the
// emeans calls.
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strconv"
	"time"
	"unsafe"

	emeans "github.com/samhvw8/emeans/bindings/go"
)

// fnv is FNV-1a 64 of the bytes (the same hash in every language bench).
func fnv[T any](v []T) string {
	var zero T
	b := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(v))), len(v)*int(unsafe.Sizeof(zero)))
	h := uint64(0xcbf29ce484222325)
	for _, x := range b {
		h ^= uint64(x)
		h *= 0x100000001b3
	}
	return fmt.Sprintf("%016x", h)
}

func main() {
	path := os.Args[1]
	d, _ := strconv.Atoi(os.Args[2])
	k, _ := strconv.Atoi(os.Args[3])
	backend, mode := os.Args[4], os.Args[5]
	raw, err := os.ReadFile(path)
	if err != nil {
		log.Fatal(err)
	}
	x := unsafe.Slice((*float32)(unsafe.Pointer(unsafe.SliceData(raw))), len(raw)/4)
	n := len(x) / d

	if mode == "call" {
		row, cents := x[:d], x[64*d:128*d]
		const reps = 2000
		t := time.Now()
		for range reps {
			if _, err := emeans.Assign(row, cents, d, emeans.InnerProduct, ""); err != nil {
				log.Fatal(err)
			}
		}
		out, _ := json.Marshal(map[string]any{"lang": "go", "call_us": float64(time.Since(t).Nanoseconds()) / 1e3 / reps})
		fmt.Println(string(out))
		return
	}

	cfg := emeans.DefaultTrainConfig()
	cfg.Spherical, cfg.Backend = true, backend
	t := time.Now()
	tr, err := emeans.Train(x, d, k, &cfg)
	if err != nil {
		log.Fatal(err)
	}
	trainS := time.Since(t).Seconds()

	t = time.Now()
	a, err := emeans.Assign(x, tr.Centroids, d, emeans.InnerProduct, backend)
	if err != nil {
		log.Fatal(err)
	}
	assignS := time.Since(t).Seconds()
	var sum float64
	for _, s := range a.Scores {
		sum += 2 - 2*float64(s)
	}

	out, _ := json.Marshal(map[string]any{
		"lang": "go", "backend": tr.Backend, "n": n, "d": d, "k": k,
		"train_s": trainS, "assign_s": assignS, "objective": sum / float64(n),
		"centroids_fnv": fnv(tr.Centroids), "labels_fnv": fnv(a.Labels),
	})
	fmt.Println(string(out))
}
