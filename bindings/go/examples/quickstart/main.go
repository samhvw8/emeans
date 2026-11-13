// Command quickstart shows emeans from Go: train, assign, AutoCluster, add
// new rows, save and load.
//
// Run: mise run example:go   (or: go run ./examples/quickstart [backend])
// The data is synthetic (16 blobs of unit rows).
package main

import (
	"fmt"
	"log"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"time"

	emeans "github.com/samhvw8/emeans/bindings/go"
)

func timed[T any](label string, f func() (T, error)) T {
	t := time.Now()
	r, err := f()
	if err != nil {
		log.Fatalf("%s: %v", label, err)
	}
	fmt.Printf("%-28s %8.1f ms\n", label, float64(time.Since(t).Microseconds())/1000)
	return r
}

func main() {
	backend := "cpu"
	if len(os.Args) > 1 {
		backend = os.Args[1]
	}
	const n, d, blobs = 20_000, 64, 16
	rng := rand.New(rand.NewPCG(7, 7))
	centers := make([][]float64, blobs)
	for i := range centers {
		centers[i] = make([]float64, d)
		for j := range d {
			centers[i][j] = rng.NormFloat64()
		}
	}
	x := make([]float32, n*d)
	row := make([]float64, d)
	for i := range n {
		c := centers[rng.IntN(blobs)]
		var s float64
		for j := range d {
			row[j] = c[j] + 0.35*rng.NormFloat64()
			s += row[j] * row[j]
		}
		for j := range d {
			x[i*d+j] = float32(row[j] / math.Sqrt(s))
		}
	}
	avail, _ := emeans.Backends()
	fmt.Printf("emeans %s: %d x %d rows, backend %s (available: %v)\n", emeans.Version(), n, d, backend, avail)

	cfg := emeans.DefaultTrainConfig()
	cfg.Spherical, cfg.Seed, cfg.Backend = true, 7, backend
	t := timed("train k=32", func() (*emeans.TrainResult, error) { return emeans.Train(x, d, 32, &cfg) })
	fmt.Printf("  objective %.4f after %d epochs on %s\n", t.Objective, len(t.Stats), t.Backend)
	a := timed("assign", func() (*emeans.Assignment, error) {
		return emeans.Assign(x, t.Centroids, d, emeans.InnerProduct, backend)
	})
	fmt.Printf("  first labels %v\n", a.Labels[:10])

	const nOld = 16_000
	acfg := emeans.DefaultAutoConfig()
	acfg.Seed, acfg.K1Range, acfg.Restarts = 7, [2]int{2, 24}, 2 // a smaller k sweep than the default (2..40, 4 restarts)
	type auto struct {
		h *emeans.Hierarchy
		l *emeans.Labels
	}
	r0 := timed("AutoCluster (16k rows)", func() (auto, error) {
		h, l, err := emeans.AutoCluster(x[:nOld*d], d, &acfg)
		return auto{h, l}, err
	})
	info, _ := r0.h.Info()
	leaves := 0
	for _, k := range info.K2 {
		leaves += k
	}
	fmt.Printf("  level 1: %d groups, %d leaves; first l1 labels %v\n", info.K1, leaves, r0.l.L1[:10])

	dir, err := os.MkdirTemp("", "emeans-")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(dir)
	path := filepath.Join(dir, "model")
	timed("save", func() (struct{}, error) { return struct{}{}, r0.h.Save(path) })
	r0.h.Close()
	h := timed("load", func() (*emeans.Hierarchy, error) { return emeans.LoadHierarchy(path, "") })
	defer h.Close()

	r := timed("add 4k new rows", func() (*emeans.AddReport, error) { return h.Add(x) }) // all rows seen, then the new ones
	info, _ = h.Info()
	fmt.Printf("  action %s, %d new outliers, model has seen %d rows\n", r.Action, r.NewOutliers, info.NSeen)
	fmt.Printf("  labels of the new rows: %v ...\n", r.Labels.L1[nOld:nOld+10])
}
