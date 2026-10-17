#!/usr/bin/env python3
"""Hardware self-test: random networks + observations, FPGA vs bit-exact model."""
import os, random, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim"))
import mlp_model as M  # noqa: E402
from fpga import Fpga  # noqa: E402
from run_test import rand_policy  # noqa: E402

dev = Fpga()
assert dev.read(M.REG_ID) == M.ID_VALUE, "bad ID"
bad = total = 0
for seed in range(1, 21):
    rng = random.Random(seed)
    pol = rand_policy(rng)
    dev.load(pol)
    X = np.array([[rng.randint(-32768, 32767) if rng.random() < 0.2 else rng.randint(-3000, 3000)
                   for _ in range(pol["N"])] for _ in range(500)])
    o, a = M.forward(pol, X)
    got = dev.stream(X)
    bad += sum(1 for k, (go, ga) in enumerate(got) if go != list(o[k]) or ga != a[k])
    total += len(X)
print("HWTEST %s: %d / %d inferences mismatched" % ("PASS" if bad == 0 else "FAIL", bad, total))
sys.exit(1 if bad else 0)
