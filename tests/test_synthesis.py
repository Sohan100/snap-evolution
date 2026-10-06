import math

from snapevo import synthesis


def test_kn_success_branch_is_a_z_rotation_near_theta(tmp_path, monkeypatch):
    monkeypatch.setenv("SNAPEVO_RUNS", str(tmp_path))
    r = synthesis.synthesize(0.7, 1e-4)
    assert math.isclose(r["expected_t"], 17.082, abs_tol=1e-3)          # the package README's example
    off = abs(r["theta_s"] - 0.7)
    assert off <= 0.9e-4 * (1 + 1e-9)
    assert math.isclose(2 * math.sin(off / 2), r["d0"], rel_tol=1e-6)
    assert r["d_branch"] <= r["d0"] + r["d1"] + 1e-12
    again = synthesis.synthesize(0.7 + 2 * math.pi, 1e-4)                # cached, other branch
    assert math.isclose(again["theta_s"] - 2 * math.pi, r["theta_s"], abs_tol=1e-12)
    assert (tmp_path / "kn_cache.json").exists()
