/* emeans C quickstart: train a model, assign, save and load it as bytes,
 * cluster with automatic k, add new rows, and handle an error.
 *
 * Build and run (after `mise run c:dist`), from the repo root:
 *   cc examples/c/quickstart.c $(PKG_CONFIG_PATH=dist/lib/pkgconfig pkg-config --cflags --libs emeans) \
 *      -Wl,-rpath,$PWD/dist/lib -o quickstart && ./quickstart
 * or with CMake: see examples/c/CMakeLists.txt. `mise run c:test` does both.
 * Exits non-zero if any check fails.
 */
#define _POSIX_C_SOURCE 199309L /* clock_gettime under -std=c11 on Linux */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "emeans.h"

/* Abort with emeans' own message when a call fails. */
#define CHECK(call)                                                               \
  do {                                                                            \
    int32_t s_ = (call);                                                          \
    if (s_ != EMEANS_OK) {                                                        \
      fprintf(stderr, "%s:%d: %s failed (%d): %s\n", __FILE__, __LINE__, #call, \
              s_, emeans_last_error());                                           \
      exit(1);                                                                    \
    }                                                                             \
  } while (0)

#define EXPECT(cond)                                                    \
  do {                                                                  \
    if (!(cond)) {                                                      \
      fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
      exit(1);                                                          \
    }                                                                   \
  } while (0)

static double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

/* A small deterministic generator (xorshift64*) and Box-Muller normals. */
static uint64_t rng = 7;
static double uniform(void) {
  rng ^= rng >> 12, rng ^= rng << 25, rng ^= rng >> 27;
  return ((rng * 2685821657736338717ULL) >> 11) * (1.0 / 9007199254740992.0);
}
static double gauss(void) {
  return sqrt(-2.0 * log(1.0 - uniform())) * cos(6.283185307179586 * uniform());
}

/* n unit rows of d floats near one of `blobs` random centers. */
static float *unit_rows(size_t n, size_t d, size_t blobs) {
  double *centers = malloc(blobs * d * sizeof *centers);
  float *x = malloc(n * d * sizeof *x);
  for (size_t i = 0; i < blobs * d; i++) centers[i] = gauss();
  for (size_t i = 0; i < n; i++) {
    const double *c = centers + (size_t)(uniform() * blobs) * d;
    double s = 0;
    for (size_t j = 0; j < d; j++) {
      double v = c[j] + 0.35 * gauss();
      x[i * d + j] = (float)v;
      s += v * v;
    }
    for (size_t j = 0; j < d; j++) x[i * d + j] /= (float)sqrt(s);
  }
  free(centers);
  return x;
}

static EmeansArray array(const EmeansResult *r, const char *name) {
  EmeansArray a;
  CHECK(emeans_result_array(r, name, &a));
  return a;
}

int main(int argc, char **argv) {
  const char *backend = argc > 1 ? argv[1] : "cpu";
  const size_t n = 20000, d = 64, n_old = 16000;
  char cfg[256];
  EXPECT(emeans_abi_version() == EMEANS_ABI_VERSION);
  char *backends;
  CHECK(emeans_backends(&backends));
  printf("emeans %s (ABI %u): %zu x %zu rows, backend %s (available: %s)\n", emeans_version(),
         emeans_abi_version(), n, d, backend, backends);
  emeans_string_free(backends);
  float *x = unit_rows(n, d, 16);

  /* 1. Train a model: k = 32, cosine (spherical), seed 7. */
  snprintf(cfg, sizeof cfg, "{\"spherical\": true, \"seed\": 7, \"backend\": \"%s\"}", backend);
  EmeansModel *model;
  EmeansResult *trained;
  double t = now_ms();
  CHECK(emeans_model_train(x, n, d, 32, cfg, &model, &trained));
  printf("%-28s %8.1f ms  %s\n", "train k=32", now_ms() - t, emeans_result_json(trained));

  /* 2. Assign with the model (its metric and backend). */
  EmeansResult *a1;
  t = now_ms();
  CHECK(emeans_model_assign(model, x, n, &a1));
  EmeansArray labels = array(a1, "labels");
  const uint32_t *l = labels.data;
  printf("%-28s %8.1f ms  first labels %u %u %u %u %u\n", "assign", now_ms() - t, l[0], l[1],
         l[2], l[3], l[4]);

  /* 3. The model as bytes (for a database), then back: the same labels. */
  EmeansResult *blob;
  CHECK(emeans_model_to_bytes(model, &blob));
  EmeansArray bytes = array(blob, "bytes");
  EmeansModel *model2;
  CHECK(emeans_model_from_bytes(bytes.data, bytes.len, NULL, &model2));
  EmeansResult *a2;
  CHECK(emeans_model_assign(model2, x, n, &a2));
  EXPECT(memcmp(array(a2, "labels").data, labels.data, n * sizeof(uint32_t)) == 0);
  printf("%-28s %8zu bytes, same labels after the round trip\n", "model to/from bytes", bytes.len);

  /* 4. Automatic k in two levels on the first 16k rows. */
  EmeansHierarchy *h;
  EmeansResult *auto_labels;
  t = now_ms();
  CHECK(emeans_auto_cluster(x, n_old, d, "{\"seed\": 7, \"k1_range\": [2, 24], \"restarts\": 2}",
                            &h, &auto_labels));
  printf("%-28s %8.1f ms  %s\n", "auto_cluster (16k rows)", now_ms() - t,
         emeans_result_json(auto_labels));

  /* 5. Through bytes, then add the other 4k rows (pass all rows seen, then the new ones). */
  EmeansResult *hblob;
  CHECK(emeans_hierarchy_to_bytes(h, &hblob));
  EmeansArray hb = array(hblob, "bytes");
  EmeansHierarchy *h2;
  CHECK(emeans_hierarchy_from_bytes(hb.data, hb.len, NULL, &h2));
  EmeansResult *added, *added2;
  t = now_ms();
  CHECK(emeans_hierarchy_add(h, x, n, &added));
  printf("%-28s %8.1f ms  %s\n", "add 4k new rows", now_ms() - t, emeans_result_json(added));
  CHECK(emeans_hierarchy_add(h2, x, n, &added2));
  EXPECT(memcmp(array(added, "l1").data, array(added2, "l1").data, n * sizeof(uint32_t)) == 0);
  const uint32_t *l1 = array(added, "l1").data;
  printf("  level-1 labels of the new rows: %u %u %u %u %u ...\n", l1[n_old], l1[n_old + 1],
         l1[n_old + 2], l1[n_old + 3], l1[n_old + 4]);
  char *info;
  CHECK(emeans_hierarchy_info(h, &info));
  printf("  hierarchy %s\n", info);
  emeans_string_free(info);

  /* 6. Errors: a status code plus a thread-local message; nothing to free. */
  EmeansResult *none = NULL;
  int32_t s = emeans_train(x, n, d, 8, "{\"sead\": 1}", &none);
  EXPECT(s == EMEANS_ERROR && none == NULL);
  printf("%-28s status %d: %s\n", "a misspelled config key", s, emeans_last_error());
  s = emeans_model_assign(model, NULL, 10, &none);
  EXPECT(s == EMEANS_INVALID_ARGUMENT);
  printf("%-28s status %d: %s\n", "a NULL data pointer", s, emeans_last_error());

  emeans_result_free(trained), emeans_result_free(a1), emeans_result_free(a2);
  emeans_result_free(blob), emeans_result_free(hblob), emeans_result_free(auto_labels);
  emeans_result_free(added), emeans_result_free(added2);
  emeans_model_free(model), emeans_model_free(model2);
  emeans_hierarchy_free(h), emeans_hierarchy_free(h2);
  free(x);
  printf("ok\n");
  return 0;
}
