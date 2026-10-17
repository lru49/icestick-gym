#!/usr/bin/env python3
"""Train an FPGA policy for one Gym task with CMA-ES (runs in GitHub Actions).

The fitness of every candidate is computed with the bit-exact FPGA model
(host/mlp_model.py), so the policy that comes out behaves identically on the
iCEstick. Writes to --out:
  policy.json   quantised weights + config for the FPGA
  eval.json     held-out evaluation (100 seeds) with the bit-exact model
  trace.jsonl   per-step (x_int, outputs, argmax) of every evaluation episode,
                replayed through the real FPGA by host/replay.py
"""
import argparse, json, os, sys, time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "host"))
import envs  # noqa: E402
import mlp_model as M  # noqa: E402

EVAL_SEEDS = range(10000, 10100)
VAL_SEEDS = range(5000, 5030)

_env = {}


def _get_env(task):
    if task not in _env:
        _env[task] = envs.make(task)
    return _env[task]


def evaluate(args):
    task, N, H, K, theta, seeds = args
    pol = M.from_vector(theta, N, H, K)
    env = _get_env(task)
    fn = lambda x: (lambda o, a: (o, a))(*M.forward(pol, x))
    return float(np.mean([envs.run_episode(task, env, fn, s)[0] for s in seeds]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--max-gens", type=int, default=2000)
    ap.add_argument("--pop", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--target", type=float, default=None, help="stop once validation >= target")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import cma
    spec = envs.TASKS[args.task]
    N, K, H = len(spec["obs_gain"]), spec["K"], args.H
    n = M.n_params(N, H, K)
    opts = {"seed": args.seed, "verbose": -9, "maxiter": args.max_gens}
    if args.pop:
        opts["popsize"] = args.pop
    es = cma.CMAEvolutionStrategy(np.zeros(n), args.sigma, opts)
    rng = np.random.default_rng(args.seed)
    target = spec["solved"] if args.target is None else args.target
    print(f"{args.task}: N={N} H={H} K={K} params={n} pop={es.popsize} cpus={os.cpu_count()}", flush=True)

    best = (-np.inf, None)
    t0, gen = time.time(), 0
    log = []
    with Pool(os.cpu_count()) as pool:
        while not es.stop() and time.time() - t0 < args.minutes * 60:
            seeds = [int(s) for s in rng.integers(0, 2**31, args.episodes)]
            cands = es.ask()
            fit = pool.map(evaluate, [(args.task, N, H, K, c, seeds) for c in cands])
            es.tell(cands, [-f for f in fit])
            gen += 1
            if gen % 10 == 0 or gen == 1:
                # Validate the distribution mean and the generation's best candidate
                tops = [es.mean, cands[int(np.argmax(fit))]]
                vals = pool.map(evaluate, [(args.task, N, H, K, c, list(VAL_SEEDS)) for c in tops])
                for v, c in zip(vals, tops):
                    if v > best[0]:
                        best = (v, np.array(c))
                log.append({"gen": gen, "t": round(time.time() - t0), "pop_best": max(fit),
                            "pop_mean": float(np.mean(fit)), "val": max(vals), "best_val": best[0]})
                print(json.dumps(log[-1]), flush=True)
                if best[0] >= target + abs(target) * 0.05:   # small margin over the threshold
                    break

    pol = M.from_vector(best[1], N, H, K)
    M.check_shape(pol)
    os.makedirs(args.out, exist_ok=True)
    env = envs.make(args.task)
    fn = lambda x: M.forward(pol, x)
    rets, steps = [], 0
    with open(os.path.join(args.out, "trace.jsonl"), "w") as f:
        for s in EVAL_SEEDS:
            r, k, tr = envs.run_episode(args.task, env, fn, s, record=True)
            rets.append(r); steps += k
            f.write(json.dumps({"seed": s, "return": r, "steps": tr}) + "\n")
    ev = {"task": args.task, "H": H, "params": n, "gens": gen, "train_s": round(time.time() - t0),
          "mean": float(np.mean(rets)), "std": float(np.std(rets)), "min": float(np.min(rets)),
          "max": float(np.max(rets)), "episodes": len(rets), "steps": steps,
          "solved_threshold": spec["solved"], "solved": bool(np.mean(rets) >= spec["solved"]),
          "best_val": best[0], "log": log}
    with open(os.path.join(args.out, "policy.json"), "w") as f:
        json.dump({"task": args.task, **M.to_json(pol)}, f)
    with open(os.path.join(args.out, "eval.json"), "w") as f:
        json.dump(ev, f, indent=1)
    print(f"EVAL {args.task} H={H}: mean {ev['mean']:.1f} ± {ev['std']:.1f} over 100 episodes "
          f"(threshold {spec['solved']}, {'SOLVED' if ev['solved'] else 'not solved'})")


if __name__ == "__main__":
    main()
