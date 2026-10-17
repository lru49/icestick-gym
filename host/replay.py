#!/usr/bin/env python3
"""Replay CI evaluation traces through the real FPGA.

The Gym episodes ran in GitHub Actions with the bit-exact model. Here every
recorded observation is streamed through the iCEstick and its outputs and
argmax must equal the recorded ones. If all match, the FPGA would have chosen
the same action at every step, so those episodes (and their returns) are the
FPGA's own closed-loop results.

  replay.py build/policies/*/        (directories holding policy.json, eval.json, trace.jsonl)
"""
import argparse, json, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlp_model as M  # noqa: E402
from fpga import Fpga, PORT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--port", default=PORT)
    ap.add_argument("--out", default="results/replay.json")
    args = ap.parse_args()
    dev = Fpga(args.port)
    if dev.read(M.REG_ID) != M.ID_VALUE:
        sys.exit("FPGA ID mismatch: is the policy-engine bitstream loaded?")
    results = []
    for d in args.dirs:
        pol = M.from_json(json.load(open(os.path.join(d, "policy.json"))))
        ev = json.load(open(os.path.join(d, "eval.json")))
        eps = [json.loads(l) for l in open(os.path.join(d, "trace.jsonl"))]
        dev.load(pol)
        steps = [s for e in eps for s in e["steps"]]
        t0 = time.time()
        got = dev.stream([s[0] for s in steps])
        dt = time.time() - t0
        bad = sum(1 for (o, a), s in zip(got, steps) if o != s[1] or a != s[2])
        bad_eps = sum(1 for e, i in zip(eps, np.cumsum([0] + [len(e["steps"]) for e in eps]))
                      if any(got[i + k][0] != s[1] or got[i + k][1] != s[2] for k, s in enumerate(e["steps"])))
        cyc = dev.read(M.REG_CYCLES)
        r = {"task": ev["task"], "H": pol["H"], "params": ev["params"], "episodes": len(eps),
             "steps": len(steps), "mismatched_steps": bad, "mismatched_episodes": bad_eps,
             "mean_return": ev["mean"], "std": ev["std"], "threshold": ev["solved_threshold"],
             "solved": ev["solved"], "inference_cycles": cyc, "inference_us": round(cyc / 12, 1),
             "replay_steps_per_s": round(len(steps) / dt)}
        results.append(r)
        print("%-26s H=%-2d %3d eps %6d steps  FPGA mismatches: %d  return %.1f ± %.1f (threshold %s) %s  %d cyc/inf" % (
            r["task"], r["H"], r["episodes"], r["steps"], bad, r["mean_return"], r["std"], r["threshold"],
            "SOLVED" if r["solved"] else "-", cyc), flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=1)
    print("wrote", args.out)
    return 0 if all(r["mismatched_steps"] == 0 for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
