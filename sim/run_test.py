#!/usr/bin/env python3
"""Randomized self-checking simulation of rtl/mlp.v against host/mlp_model.py.

Each seed loads a random network (random N/H/K/shifts, weights that hit the
saturation and ReLU cases), reads part of it back, then streams observations
with up to 4 'x' commands outstanding and checks every reply byte. Partial
commands (dropped by the parser timeout) are mixed in.
"""
import argparse, os, random, subprocess, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "host"))
import mlp_model as M  # noqa: E402


def rand_policy(rng):
    N = rng.randint(1, M.MAX_N)
    H = rng.choice([0, rng.randint(1, M.MAX_H), rng.randint(1, 8)])
    K = rng.randint(1, M.MAX_K)
    while M.n_params(N, H, K) > M.WMEM:
        H = rng.randint(0, H)
    big = rng.random() < 0.3
    lim = 32767 if big else rng.choice([300, 2000])
    rw = lambda *shape: np.array([[rng.randint(-lim, lim) for _ in range(shape[1])] for _ in range(shape[0])],
                                 dtype=np.int64)
    pol = {"N": N, "H": H, "K": K, "S1": rng.randint(0, 15), "S2": rng.randint(0, 15),
           "W1": rw(H, N + 1) if H else np.zeros((0, N + 1), np.int64), "W2": rw(K, (H or N) + 1)}
    if rng.random() < 0.3:
        pol["W2"][:, :] = pol["W2"][0]          # ties: argmax must pick the first
    M.check_shape(pol)
    return pol


def word(v):
    return [(v >> 8) & 255, v & 255]


def gen(rng, pol, n_obs):
    stim, exp = [], []
    def send(bs): stim.extend(("B", b) for b in bs)
    def wait(): stim.append(("W", len(exp)))
    for a, v in M.config_words(pol) + list(enumerate(M.pack(pol))):
        send([ord("w")] + word(a) + word(v)); exp.append(ord("k"))
    wait()
    image = M.pack(pol)
    for a in rng.sample(range(len(image)), min(6, len(image))):
        send([ord("r")] + word(a)); exp.extend(word(image[a])); wait()
    for a, v in ((M.CFG_BASE + 0, pol["N"]), (M.CFG_BASE + 1, pol["H"]), (M.REG_ID, M.ID_VALUE)):
        send([ord("r")] + word(a)); exp.extend(word(v)); wait()
    if rng.random() < 0.5:                      # partial command, dropped after ~4 ms
        send([ord("x")] + [rng.randint(0, 255) for _ in range(rng.randint(0, 2 * pol["N"] - 1))])
        stim.append(("D", 5 * 2000 + 100))
    # observations: mostly realistic, some extremes
    X = [[rng.choice([rng.randint(-2000, 2000), rng.randint(-32768, 32767), -32768, 32767])
          for _ in range(pol["N"])] for _ in range(n_obs)]
    o, am = M.forward(pol, np.array(X))
    ends = []
    window = rng.choice([1, 2, 4])
    for k, x in enumerate(X):
        if k >= window:
            stim.append(("W", ends[k - window]))
        send([ord("x")] + sum((word(v) for v in x), []))
        exp.append(int(am[k]))
        for v in o[k]:
            exp.extend(word(int(v)))
        ends.append(len(exp))
    wait()
    return stim, exp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--obs", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "build", "sim"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    vvp = os.path.join(args.out, "tb.vvp")
    rtl = os.path.join(HERE, "..", "rtl")
    subprocess.run(["iverilog", "-g2012", "-o", vvp, os.path.join(HERE, "tb.v"),
                    os.path.join(rtl, "mlp.v"), os.path.join(rtl, "uart.v")], check=True)
    total = 0
    for seed in range(1, args.seeds + 1):
        rng = random.Random(seed)
        pol = rand_policy(rng)
        stim, exp = gen(rng, pol, args.obs)
        with open(os.path.join(args.out, "stim.txt"), "w") as f:
            f.writelines("%s %d\n" % s for s in stim)
        res = subprocess.run(["vvp", "-n", vvp], cwd=args.out, capture_output=True, text=True)
        if "DONE" not in res.stdout:
            print("seed %d: sim failed: %s" % (seed, res.stdout[-500:])); sys.exit(1)
        got = [int(l) for l in open(os.path.join(args.out, "rx.txt"))]
        if got != exp:
            k = next((i for i, (a, b) in enumerate(zip(got, exp)) if a != b), min(len(got), len(exp)))
            print("seed %d: FAIL N=%d H=%d K=%d S=%d/%d: byte %d got %s exp %s (len %d vs %d)" % (
                seed, pol["N"], pol["H"], pol["K"], pol["S1"], pol["S2"], k,
                got[k:k + 4], exp[k:k + 4], len(got), len(exp)))
            sys.exit(1)
        total += args.obs
        print("seed %d: ok  N=%d H=%d K=%d S=%d/%d" % (seed, pol["N"], pol["H"], pol["K"], pol["S1"], pol["S2"]))
    print("PASS: %d inferences bit-exact" % total)


if __name__ == "__main__":
    main()
