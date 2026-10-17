"""Gym task specs shared by training (CI), trace generation and the FPGA host.

Each task fixes how observations are quantised for the FPGA (obs_gain:
x_int = round(obs * obs_gain * 256), saturated to int16, like a fixed ADC
scaling) and how the FPGA's outputs become an action.
"""
import numpy as np

TASKS = {
    "CartPole-v1":               dict(obs_gain=[1, 1, 4, 1], K=2, kind="argmax", solved=475),
    "Acrobot-v1":                dict(obs_gain=[1, 1, 1, 1, 0.25, 0.25], K=3, kind="argmax", solved=-100),
    "MountainCar-v0":            dict(obs_gain=[1, 20], K=3, kind="argmax", solved=-110),
    "MountainCarContinuous-v0":  dict(obs_gain=[1, 20], K=1, kind="cont", act_scale=1.0, solved=90),
    "Pendulum-v1":               dict(obs_gain=[1, 1, 0.25], K=1, kind="cont", act_scale=2.0, solved=-200),
    "LunarLander-v3":            dict(obs_gain=[1, 1, 1, 1, 1, 1, 1, 1], K=4, kind="argmax", solved=200),
}


def make(name, **kw):
    import gymnasium as gym
    return gym.make(name, **kw)


def quantize_obs(name, obs):
    g = np.asarray(TASKS[name]["obs_gain"], dtype=np.float64)
    return np.clip(np.round(np.asarray(obs, dtype=np.float64) * g * 256), -32768, 32767).astype(np.int64)


def to_action(name, outputs, amax):
    t = TASKS[name]
    if t["kind"] == "argmax":
        return int(amax)
    a = np.clip(np.asarray(outputs, dtype=np.float64) / 256.0 * t["act_scale"],
                -t["act_scale"], t["act_scale"])
    return a.astype(np.float32)


def run_episode(name, env, policy_fn, seed, record=False):
    """policy_fn(x_int) -> (outputs, amax). Returns (return, steps, trace)."""
    obs, _ = env.reset(seed=seed)
    total, steps, trace = 0.0, 0, []
    while True:
        x = quantize_obs(name, obs)
        o, a = policy_fn(x)
        if record:
            trace.append((x.tolist(), [int(v) for v in o], int(a)))
        obs, r, term, trunc, _ = env.step(to_action(name, o, a))
        total += r
        steps += 1
        if term or trunc:
            return total, steps, trace
