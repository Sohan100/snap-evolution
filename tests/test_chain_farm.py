import subprocess
import sys
import threading

import numpy as np
import pytest

import snapevo.chain_snap as cs
from snapevo.cells import cell_for
from snapevo.chain_farm import FarmPool, LocalFarm, SocketFarm, use_farm
from snapevo.chain_snap import ChainProblem, Library, compile_chain
from snapevo.circuits import product_vectors, random_product_states
from snapevo.hamiltonian import XXZ, evolve
from snapevo.mps_train import mps_from_dense
from snapevo.train_chain import trotter_z


def _problem(model, N=8, t=0.7, n=4, seed=5):
    states = random_product_states(N, n, seed)
    targets = [mps_from_dense(v) for v in evolve(model.sparse(N, "open"), states, t)]
    return product_vectors(N, n, seed), targets


def _gpu():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:                                             # noqa: BLE001
        return False


@pytest.fixture(autouse=True)
def _no_factory():
    yield
    cs.POOL_FACTORY = None


def _hessian_and_grads(P, z, nb):
    c, g = P.objective(z)
    H = P.hessian(z, list(range(min(nb, 6))))
    return c, g, H


def test_farm_pool_on_local_cpu_workers_is_the_chain_pool_number_for_number():
    model = XXZ(0.25, 1.6)
    cell = cell_for(model)
    N, t, D = 8, 0.7, 4
    vecs, targets = _problem(model, N, t)
    z = trotter_z(model, cell, t, D) + 0.03
    nb = D * cell.n_angles
    with ChainProblem(model, cell, t, D, N, vecs, targets, chi_max=16, cutoff=1e-10, groups=4, replicas=2) as P:
        ref = _hessian_and_grads(P, z, nb)
    farm = LocalFarm(cpus=6)
    try:
        use_farm(farm, replicas=1, groups=4)
        with ChainProblem(model, cell, t, D, N, vecs, targets, chi_max=16, cutoff=1e-10, groups=4, replicas=2) as P:
            assert isinstance(P.pool, FarmPool) and P.pool.size == 4
            got = _hessian_and_grads(P, z, nb)
            assert len(farm._free) == 2                          # 4 leased, 2 free
            # environments (the compile's curves) on the farm pool
            env = P.envs_at(z, [1, 2])
            cv = env.curves([cell.n_angles + 1])
            env.release()
        assert sorted(farm._free) == list(range(6))              # returned on close
    finally:
        farm.close()
    assert got[0] == ref[0] and np.array_equal(got[1], ref[1]) and np.array_equal(got[2], ref[2])
    assert cv[cell.n_angles + 1].residual < 1e-12


def test_socket_farm_with_worker_processes_runs_a_compile(tmp_path):
    model = XXZ(0.25, 0.0)
    cell = cell_for(model)
    N, t, D = 6, 0.5, 3
    vecs, targets = _problem(model, N, t, n=3)
    z = trotter_z(model, cell, t, D)
    addr = tmp_path / "farm.json"
    procs = [subprocess.Popen([sys.executable, "-m", "snapevo.chain_farm", "worker", "--address-file", str(addr),
                               "--wait", "120"]) for _ in range(3)]
    try:
        farm = SocketFarm(addr, expected=3, wait=120, log=lambda _s: None)
        try:
            with ChainProblem(model, cell, t, D, N, vecs, targets, chi_max=10 ** 6, cutoff=0.0, groups=3,
                              local=True) as P:
                ref = P.objective(z)
            use_farm(farm)
            with ChainProblem(model, cell, t, D, N, vecs, targets, chi_max=10 ** 6, cutoff=0.0, groups=3) as P:
                assert P.pool.K == 1 and P.pool.G == 3
                got = P.objective(z)
                budget = 3 * max(P.c_fn(z), 1e-6)
                res = compile_chain(z, cell, D, cs.ChainSettings(budget=budget, max_round_trials=4, reopt="newton"),
                                    P.objective, P.c_fn, P.hessian, P.envs_at, Library(22.0))
                assert res["C_train"] <= budget * (1 + 1e-9)
        finally:
            farm.close()
    finally:
        for p in procs:
            p.wait(timeout=60)
    assert got[0] == ref[0] and np.array_equal(got[1], ref[1])
    assert all(p.returncode == 0 for p in procs)


def test_a_farm_lease_waits_for_free_workers():
    model = XXZ(0.25, 0.0)
    cell = cell_for(model)
    vecs, targets = _problem(model, 6, 0.5, n=2)
    farm = LocalFarm(cpus=2)
    try:
        a = FarmPool(farm, cell, vecs, targets, 16, 1e-10, groups=2, replicas=1)
        got = []
        th = threading.Thread(target=lambda: got.append(FarmPool(farm, cell, vecs, targets, 16, 1e-10, groups=2)))
        th.start()
        th.join(timeout=2)
        assert th.is_alive() and not got                         # blocked: no free workers
        a.close()
        th.join(timeout=60)
        assert got and got[0].size == 2
        got[0].close()
    finally:
        farm.close()


@pytest.mark.skipif(not _gpu(), reason="no GPU here")
def test_gpu_workers_give_the_cpu_numbers_up_to_rounding():
    model = XXZ(0.25, 1.6)
    cell = cell_for(model)
    N, t, D = 10, 1.0, 5
    vecs, targets = _problem(model, N, t, n=3)
    z = trotter_z(model, cell, t, D) + 0.02
    nb = D * cell.n_angles
    with ChainProblem(model, cell, t, D, N, vecs, targets, chi_max=24, cutoff=1e-10, groups=3) as P:
        c0, g0 = P.objective(z)
        H0 = P.hessian(z, [0, 1, nb])
        env = P.envs_at(z, [1, 3], sweep=True)
        cv0 = env.curves([nb + 2, cell.n_angles + 2])
        env.set_angle(cell.n_angles + 1, 0.3)
        env.advance(1)
        cv0b = env.curve(3 * cell.n_angles)
        env.release()
    farm = LocalFarm(devices=[0], per_device=3)
    try:
        use_farm(farm)
        with ChainProblem(model, cell, t, D, N, vecs, targets, chi_max=24, cutoff=1e-10, groups=3) as P:
            c1, g1 = P.objective(z)
            H1 = P.hessian(z, [0, 1, nb])
            env = P.envs_at(z, [1, 3], sweep=True)
            cv1 = env.curves([nb + 2, cell.n_angles + 2])
            env.set_angle(cell.n_angles + 1, 0.3)
            env.advance(1)
            cv1b = env.curve(3 * cell.n_angles)
            env.release()
    finally:
        farm.close()
    assert c1 == pytest.approx(c0, rel=1e-9, abs=1e-14)
    assert np.allclose(g1, g0, rtol=0, atol=1e-9 * np.abs(g0).max())
    assert np.allclose(H1, H0, rtol=0, atol=1e-5 * np.abs(H0).max())
    for i in cv0:
        assert np.allclose(cv1[i].coeffs, cv0[i].coeffs, rtol=0, atol=1e-10)
    assert np.allclose(cv1b.coeffs, cv0b.coeffs, rtol=0, atol=1e-10)


def test_tasks_of_a_lost_worker_run_on_the_others(tmp_path):
    import time
    model = XXZ(0.25, 1.6)
    cell = cell_for(model)
    N, t, D = 6, 0.5, 3
    vecs, targets = _problem(model, N, t, n=2)
    z = trotter_z(model, cell, t, D) + 0.01
    addr = tmp_path / "farm.json"
    procs = [subprocess.Popen([sys.executable, "-m", "snapevo.chain_farm", "worker", "--address-file", str(addr),
                               "--wait", "120"]) for _ in range(4)]
    try:
        farm = SocketFarm(addr, expected=4, wait=120, log=lambda _s: None)
        try:
            with ChainProblem(model, cell, t, D, N, vecs, targets, chi_max=10 ** 6, cutoff=0.0, groups=2,
                              local=True) as P:
                ref = P.hessian(z, [0, 1, 2])
            use_farm(farm, replicas=2, groups=2)
            with ChainProblem(model, cell, t, D, N, vecs, targets, chi_max=10 ** 6, cutoff=0.0, groups=2) as P:
                victim = P.pool.workers[0][1]
                pid = victim.hello["pid"]
                next(p for p in procs if p.pid == pid).kill()
                t0 = time.time()
                while victim._dead is None and time.time() - t0 < 30:
                    time.sleep(0.1)
                got = P.hessian(z, [0, 1, 2])
                assert P.pool._gone[0][1]
            with pytest.raises(RuntimeError, match="alive"):          # the farm has 3 live workers now
                FarmPool(farm, cell, vecs, targets, 16, 1e-10, groups=2, replicas=2)
        finally:
            farm.close()
    finally:
        for p in procs:
            p.wait(timeout=60)
    assert np.array_equal(got, ref)
