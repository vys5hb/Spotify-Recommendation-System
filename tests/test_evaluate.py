"""Unit tests for the metric math in scripts/twotower/evaluate.py.

The metric definitions must match run_baselines.compute_metrics exactly, so these
check playlist_metrics against hand-computed values.
"""
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from twotower.evaluate import playlist_metrics, load_baseline_metrics  # noqa: E402


def _approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_metrics_partial_hits():
    # Ranked track indices (rank 1 first); two of them are targets.
    ranked = [10, 20, 30, 40, 50]
    targets = {20, 50}
    m = playlist_metrics(ranked, targets, target_count=2, k_values=[3, 5])

    # K=3: only 20 hits (rank 2). recall 1/2, precision 1/3.
    assert _approx(m[3]["recall"], 0.5)
    assert _approx(m[3]["precision"], 1 / 3)
    dcg3 = 1 / math.log2(3)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)   # min(target_count=2, K)=2
    assert _approx(m[3]["ndcg"], dcg3 / idcg)

    # K=5: 20 (rank 2) and 50 (rank 5) hit. recall 2/2, precision 2/5.
    assert _approx(m[5]["recall"], 1.0)
    assert _approx(m[5]["precision"], 0.4)
    dcg5 = 1 / math.log2(3) + 1 / math.log2(6)
    assert _approx(m[5]["ndcg"], dcg5 / idcg)


def test_metrics_perfect_ranking():
    # Both targets are the top two -> recall 1 and NDCG 1 (DCG == IDCG).
    m = playlist_metrics([5, 6, 7], {5, 6}, target_count=2, k_values=[3])
    assert _approx(m[3]["recall"], 1.0)
    assert _approx(m[3]["precision"], 2 / 3)
    assert _approx(m[3]["ndcg"], 1.0)


def test_metrics_no_hits():
    m = playlist_metrics([1, 2, 3], {99}, target_count=1, k_values=[3])
    assert m[3]["recall"] == 0.0
    assert m[3]["precision"] == 0.0
    assert m[3]["ndcg"] == 0.0


def test_ndcg_zero_when_no_targets_reachable():
    # target_count larger than K just caps IDCG at K terms; still valid.
    m = playlist_metrics([5, 6, 7, 8], {5}, target_count=1, k_values=[2])
    # one target at rank 1: dcg = 1/log2(2) = 1, idcg = 1 -> ndcg 1.
    assert _approx(m[2]["ndcg"], 1.0)
    assert _approx(m[2]["recall"], 1.0)
    assert _approx(m[2]["precision"], 0.5)


def test_load_baseline_metrics_missing(tmp_path):
    assert load_baseline_metrics(tmp_path / "does_not_exist.json") is None
