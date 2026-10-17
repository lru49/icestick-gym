"""Bit-exact model of rtl/mlp.v and helpers to pack a policy into its memory.

A policy is a dict: {"N", "H", "K", "S1", "S2", "W1": HxN+1 ints, "W2": Kx(H or N)+1 ints}
(W1 is absent / empty when H == 0). All weights are signed 16-bit integers;
the last column of each matrix multiplies the constant input ONE = 256.
"""
import numpy as np

ONE = 256
ID_VALUE = 0x4D4C
CFG_BASE = 0x200
REG_CYCLES = 0x2FE
REG_ID = 0x2FF
MAX_N, MAX_H, MAX_K, WMEM = 14, 32, 4, 512


def sat16(a):
    return np.clip(a, -32768, 32767)


def layer(W, inp, s, relu):
    """W: (rows, n+1) int64, inp: (..., n) int64 -> (..., rows) int64."""
    full = np.concatenate([inp, np.full(inp.shape[:-1] + (1,), ONE, dtype=np.int64)], axis=-1)
    acc = full @ W.T                 # exact: |acc| < 2^35 fits int64
    v = sat16(acc >> s)              # numpy >> on int64 is an arithmetic (floor) shift
    return np.maximum(v, 0) if relu else v


def forward(pol, x):
    """x: (..., N) ints -> (outputs (..., K), argmax (...)). Batched over leading dims."""
    x = np.asarray(x, dtype=np.int64)
    W2 = np.asarray(pol["W2"], dtype=np.int64)
    if pol["H"]:
        h = layer(np.asarray(pol["W1"], dtype=np.int64), x, pol["S1"], True)
    else:
        h = x
    o = layer(W2, h, pol["S2"], False)
    return o, np.argmax(o, axis=-1)


def check_shape(pol):
    N, H, K = pol["N"], pol["H"], pol["K"]
    assert 1 <= N <= MAX_N and 0 <= H <= MAX_H and 1 <= K <= MAX_K
    if H:
        assert np.shape(pol["W1"]) == (H, N + 1)
    assert np.shape(pol["W2"]) == (K, (H or N) + 1)
    assert len(pack(pol)) <= WMEM


def pack(pol):
    """Weight-memory image: layer-1 rows then layer-2 rows, as uint16 words."""
    rows = (list(np.asarray(pol["W1"]).reshape(-1)) if pol["H"] else []) + \
        list(np.asarray(pol["W2"]).reshape(-1))
    return [int(v) & 0xFFFF for v in rows]


def config_words(pol):
    return [(CFG_BASE + 0, pol["N"]), (CFG_BASE + 1, pol["H"]), (CFG_BASE + 2, pol["K"]),
            (CFG_BASE + 3, pol["S1"]), (CFG_BASE + 4, pol["S2"])]


def from_vector(theta, N, H, K, scale=256, S1=8, S2=8):
    """Float parameter vector (as optimised by CMA-ES) -> quantised policy."""
    q = np.clip(np.round(np.asarray(theta) * scale), -32768, 32767).astype(np.int64)
    n1 = H * (N + 1)
    pol = {"N": N, "H": H, "K": K, "S1": S1, "S2": S2,
           "W1": q[:n1].reshape(H, N + 1) if H else np.zeros((0, N + 1), np.int64),
           "W2": q[n1:].reshape(K, (H or N) + 1)}
    return pol


def n_params(N, H, K):
    return H * (N + 1) + K * ((H or N) + 1)


def to_json(pol):
    return {k: (np.asarray(v).tolist() if k in ("W1", "W2") else int(v)) for k, v in pol.items()}


def from_json(d):
    pol = dict(d)
    pol["W1"] = np.asarray(d.get("W1", []), dtype=np.int64).reshape(d["H"], d["N"] + 1) if d["H"] \
        else np.zeros((0, d["N"] + 1), np.int64)
    pol["W2"] = np.asarray(d["W2"], dtype=np.int64)
    return pol
