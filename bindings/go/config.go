package emeans

/*
#include <stdlib.h>
#include "emeans.h"
*/
import "C"

import (
	"bytes"
	"encoding/json"
	"fmt"
	"unsafe"
)

// The config structs mirror the library's JSON configs field for field. Get
// one from DefaultTrainConfig, DefaultAutoConfig or DefaultDedupConfig (the
// library's own defaults), change fields, and pass a pointer; nil means the
// defaults. A field the library does not know is an error, and so is a
// library field these structs lack (tests check both directions).

// TrainConfig configures Train.
type TrainConfig struct {
	Backend              string   `json:"backend"`         // "cpu", "metal", "cuda", ...; see Backends
	ExactObjective       bool     `json:"exact_objective"` // also compute TrainResult.ExactObjective (one more assign pass)
	NEpochs              float64  `json:"nepochs"`
	BatchSize            *int     `json:"batch_size"` // nil: the batch law
	Alpha0               *float64 `json:"alpha0"`     // nil: the step law
	BatchLaw             string   `json:"batch_law"`  // "auto", "throughput" or "quality"
	Seed                 uint32   `json:"seed"`
	Spherical            bool     `json:"spherical"` // unit rows, cosine, centroids on the unit sphere
	UnusedThreshold      *float64 `json:"unused_threshold"`
	MaxPointsPerCentroid *int     `json:"max_points_per_centroid"`
	AssignmentChunkSize  *int     `json:"assignment_chunk_size"`
	CaptureAssignments   bool     `json:"capture_assignments"`
	Verbose              bool     `json:"verbose"`
}

// IncrementalConfig is the update policy of Hierarchy.Add.
type IncrementalConfig struct {
	OutlierMadK            float64  `json:"outlier_mad_k"`
	OutlierMinSim          *float32 `json:"outlier_min_sim"`
	GrowMinSize            int      `json:"grow_min_size"`
	UpdateMinOutliers      int      `json:"update_min_outliers"`
	UpdateAddedFraction    float64  `json:"update_added_fraction"`
	UpdateEpochs           float64  `json:"update_epochs"`
	RetrainAddedFraction   float64  `json:"retrain_added_fraction"`
	RetrainOutlierFraction float64  `json:"retrain_outlier_fraction"`
	MatchMinSim            float32  `json:"match_min_sim"`
}

// AutoConfig configures AutoCluster.
type AutoConfig struct {
	Backend            string            `json:"backend"`
	K1Range            [2]int            `json:"k1_range"`
	K2Max              int               `json:"k2_max"`
	Criterion1         string            `json:"criterion1"` // "silhouette", "calinski_harabasz", "davies_bouldin"
	Criterion2         string            `json:"criterion2"`
	Level1Tolerance    float64           `json:"level1_tolerance"`
	Level2Tolerance    float64           `json:"level2_tolerance"`
	MinClusterSize     int               `json:"min_cluster_size"`
	MaxClusterSize     int               `json:"max_cluster_size"`
	MinSplitSize       int               `json:"min_split_size"`
	SplitMinSilhouette float64           `json:"split_min_silhouette"`
	Restarts           int               `json:"restarts"`
	NEpochs            float64           `json:"nepochs"`
	Init               string            `json:"init"`           // "k_means_plus_plus" or "random_rows"
	Level1Method       string            `json:"level1_method"`  // "bottom_up" or "top_down"
	ResplitPasses      int               `json:"resplit_passes"` // 0 = off
	Level1MaxShare     float64           `json:"level1_max_share"`
	Seed               uint32            `json:"seed"`
	Incremental        IncrementalConfig `json:"incremental"`
	Verbose            bool              `json:"verbose"`
}

// DedupConfig configures Dedup.
type DedupConfig struct {
	Backend      string   `json:"backend"`
	K            int      `json:"k"`             // neighbours searched per row
	SimThreshold float32  `json:"sim_threshold"` // cosine at or above which rows are near-duplicates
	RepThreshold *float32 `json:"rep_threshold"` // nil: SimThreshold
	MaxGapSecs   *float64 `json:"max_gap_secs"`  // nil: no time rule
}

// defaults decodes the library's default config of kind into v, refusing
// keys v lacks.
func defaults(kind string, v any) error {
	ck := C.CString(kind)
	defer C.free(unsafe.Pointer(ck))
	var p *C.char
	if err := call(func() C.int32_t { return C.emeans_default_config(ck, &p) }); err != nil {
		return err
	}
	defer C.emeans_string_free(p)
	dec := json.NewDecoder(bytes.NewReader([]byte(C.GoString(p))))
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		return fmt.Errorf("emeans: the Go %s config is out of date with the library: %w", kind, err)
	}
	return nil
}

func mustDefaults[T any](kind string) T {
	var v T
	if err := defaults(kind, &v); err != nil {
		panic(err) // a binding bug: the structs no longer match the library
	}
	return v
}

// DefaultTrainConfig returns the library's defaults for Train.
func DefaultTrainConfig() TrainConfig { return mustDefaults[TrainConfig]("train") }

// DefaultAutoConfig returns the library's defaults for AutoCluster.
func DefaultAutoConfig() AutoConfig { return mustDefaults[AutoConfig]("auto") }

// DefaultDedupConfig returns the library's defaults for Dedup.
func DefaultDedupConfig() DedupConfig { return mustDefaults[DedupConfig]("dedup") }

// cfgOrNil turns a nil config pointer into an untyped nil (the defaults).
func cfgOrNil[T any](c *T) any {
	if c == nil {
		return nil
	}
	return c
}
