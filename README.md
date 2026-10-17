# Gym controllers on a Lattice iCEstick

The iCE40HX1K runs a fixed-point policy engine: a 2-layer MLP, or a single
linear layer, i.e. linear state feedback. The weights are loaded over the
UART, so one bitstream serves every task. The policies are trained by CMA-ES
**in GitHub Actions**, against a bit-exact model of the FPGA arithmetic.
Evaluation also runs in Actions.

## How the work is split

| where | what |
|---|---|
| GitHub Actions `bitstream` | RTL simulation (random networks, pipelined streams, against the golden model), yosys + nextpnr + icepack |
| GitHub Actions `train` | one job per (task, policy size): CMA-ES on `host/mlp_model.py`, then 100 held-out episodes logged step by step to `trace.jsonl` |
| this machine | flash; hardware selftest; replay every logged observation through the FPGA (`host/replay.py`) |

**Why the replay proves closed-loop behaviour:** each Gym step's action is a
pure function of the FPGA's output for that step's observation. If the FPGA
returns the recorded outputs at every step, it would have produced the same
actions, and so the identical trajectories and returns.

## Engine (`rtl/mlp.v`)

- **Datapath:** int16 inputs, weights and activations. The 37-bit accumulator
  is arithmetic-shifted by a per-layer S and saturated to 16 bits.
  Layer 1 uses ReLU.
- **Sizes:** N ≤ 14 inputs, H ≤ 32 hidden units (H = 0 gives a linear
  policy), K ≤ 4 outputs. The FPGA also returns the argmax of the outputs.
- **Multiplier:** one shared 16-cycle shift-add multiplier. The HX1K has no
  DSPs.
- **Memory:** weights in a 512×16 block RAM; activations in four banks.
- **Link:** 1 Mbaud UART with up to 4 pipelined inference requests.
- **Area:** 1097 / 1280 LCs (85%), predicted Fmax 76.5 MHz (the clock is
  12 MHz). Placement needed `synth_ice40 -abc9 -dffe_min_ce_use 4`: the default
  flow left too many flip-flop control sets to place.
- Observations are quantised as `round(obs * gain * 256)`. Discrete actions
  are the argmax; continuous actions are `out / 256 * scale` (`host/envs.py`).

## Results (FPGA-verified)

Each task is trained twice: a linear controller (H = 0) and a small MLP.
The returns are means over 100 held-out seeds. Every step of every episode
was replayed on the iCEstick.

| task | policy | params | mean return (100 eps) | threshold | solved | FPGA steps replayed | mismatches |
|---|---|---|---|---|---|---|---|
| Acrobot-v1 | linear | 21 | -94.7 ± 28.7 | -100 | yes | 9569 | 0 |
| Acrobot-v1 | MLP 16 hidden | 163 | -95.5 ± 43.2 | -100 | yes | 9649 | 0 |
| CartPole-v1 | linear | 10 | 500.0 ± 0.0 | 475 | yes | 50000 | 0 |
| CartPole-v1 | MLP 8 hidden | 58 | 497.2 ± 12.6 | 475 | yes | 49724 | 0 |
| LunarLander-v3 | linear | 36 | 100.6 ± 176.8 | 200 | no | 17391 | 0 |
| LunarLander-v3 | MLP 16 hidden | 212 | 264.4 ± 33.0 | 200 | yes | 27946 | 0 |
| MountainCar-v0 | linear | 9 | -110.3 ± 9.2 | -110 | no | 11035 | 0 |
| MountainCar-v0 | MLP 16 hidden | 99 | -104.7 ± 5.3 | -110 | yes | 10473 | 0 |
| MountainCarContinuous-v0 | linear | 3 | 97.9 ± 0.6 | 90 | yes | 38262 | 0 |
| MountainCarContinuous-v0 | MLP 8 hidden | 33 | 97.1 ± 9.9 | 90 | yes | 41932 | 0 |
| Pendulum-v1 | linear | 4 | -812.3 ± 212.8 | -200 | no | 20000 | 0 |
| Pendulum-v1 | MLP 16 hidden | 81 | -179.0 ± 107.0 | -200 | yes | 20000 | 0 |

- **Thresholds:** CartPole, MountainCar, MountainCarContinuous and
  LunarLander use the official Gym "solved" thresholds. Acrobot (-100) and
  Pendulum (-200) have no official threshold; those values are common
  references.
- **Linear controllers** solve CartPole, Acrobot and MountainCarContinuous.
  On MountainCar-v0 the linear controller misses by 0.3 (-110.3 against -110).
- **Pendulum** needs the MLP: a linear controller cannot do the nonlinear
  swing-up.
- **Throughput:** about 2.4-3.9k inferences/s over the UART.

## Use

```
make test-sim                       # local RTL sim (also runs in CI)
make fetch-bit flash test-hw        # CI bitstream -> iCEstick (selected by USB node), random bit-exact test
gh workflow run train.yml           # train all tasks in Actions
make fetch-policies RUN=<run id> replay
```
