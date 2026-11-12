//! Internal and external metrics against scikit-learn (bench/py/metrics_fixture.py).

use emeans_cluster::metrics::{ari, internal_scores, nmi};
use emeans_core::io::read_f32_matrix;

fn close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * b.abs().max(1e-12)
}

#[test]
fn metrics_match_scikit_learn() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let j: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("metrics.json")).unwrap()).unwrap();
    for c in j["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let d = c["d"].as_u64().unwrap() as usize;
        let k = c["k"].as_u64().unwrap() as usize;
        let x = read_f32_matrix(dir.join(format!("metrics_{name}.f32")), d).unwrap();
        let lab = |key: &str| -> Vec<u32> {
            c[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect()
        };
        let (labels, other) = (lab("labels"), lab("other"));
        let s = internal_scores(&x, d, &labels, k).unwrap();
        let f = |key: &str| c[key].as_f64().unwrap();
        assert!(
            close(s.silhouette, f("silhouette"), 1e-5),
            "{name} silhouette {} vs {}",
            s.silhouette,
            f("silhouette")
        );
        assert!(
            close(s.calinski_harabasz, f("calinski_harabasz"), 1e-5),
            "{name} CH {} vs {}",
            s.calinski_harabasz,
            f("calinski_harabasz")
        );
        assert!(
            close(s.davies_bouldin, f("davies_bouldin"), 1e-5),
            "{name} DB {} vs {}",
            s.davies_bouldin,
            f("davies_bouldin")
        );
        assert!(close(ari(&labels, &other), f("ari"), 1e-12), "{name} ARI");
        assert!(close(nmi(&labels, &other), f("nmi"), 1e-12), "{name} NMI");
    }
}
