# Phase Y: optimizer-schedule diagnostic (Mode 1, 5 000 steps)

**Outcome:**
- **CASE B + CASE E.** The schedule helps the hard-constrained PINN (Y1) but does not make it pass the oscillation gate: it is over-damped after about 2 cycles.
- The same schedule substantially improves the supervised control (Y3), while the physics-trained runs remain non-oscillatory. That points to an interaction between optimization and physics-constrained training, not a generic optimizer problem.
- The paper baseline is **not** rescued (Y4), so CASE D does not apply.

STOPPED for approval.

## Configuration: parent → child (diff printed at creation; this is the only change)

| Child | Parent | Diff |
|---|---|---|
| Y1 | X2 (hard tanh², D4 convention) | `optim.lr 1e-4 → 1e-3`, `optim.schedule constant → exp_decay` |
| Y2 | X1 (hard tanh², paper convention) | same |
| Y3 | X4 (supervised, D4 convention; **diagnostic**) | same |
| Y4 | C0 (paper Fourier + NTK) | same (+ `threads 4 → 1`, the Phase-A execution setting already used by A1's C0) |

LR schedule (reference code `tf.train.exponential_decay(1e-3, step, 1000, 0.9, staircase=False)`): **lr(step) = 1e-3 · 0.9^(step/1000)**, which is exactly ×0.9 at every 1 000-step boundary. Verified values: 1e-3, 9e-4, 8.1e-4, 5.9e-4 at steps 0, 1k, 2k, 5k.

Everything else (architecture, conventions, seed 1234, mini-batch 32, 640 points per epoch, 5 000 steps, IC/BC, ansatz, NTK, evaluation) is identical to the parent.

*Label note:* the approval text described Y2 as "D4 convention", but defined Y2 = X1 + schedule, which is the **paper** convention. The run follows the definition.

## Results (`tables/phaseY_oscillation.csv`; figures `phaseY_midspan_traces.png` (displacement + velocity) and `phaseY_loss_history.png`)

| | Y1 (X2 + sched) | Y2 (X1 + sched) | Y3 supervised (diag.) | Y4 (C0 + sched) | *X2* | *X1* | *X4* | *C0 @5k* |
|---|---|---|---|---|---|---|---|---|
| L2_paper / L2_exact | **0.774 / 0.774** | 1.47 / 1.47 | **0.120 / 0.120** | 2.75 / 2.75 | 0.909 | 1.50 | 0.316 | 3.26 |
| best-so-far L2_exact | 0.772 | 1.47 | 0.120 | 1.49 | 0.907 | 1.50 | 0.146 | — |
| late-window L2_exact (t ≥ 0.5 s) | 1.06 | 4.73 | 0.39 | 10.7 | — | — | — | 13.0 (@20k) |
| fitted ω / ratio (exact 129.32) | global fit fails* | fit fails* | **129.4 / 1.000** | 0 / 0.000 | 112 / 0.87 | fails | 1.000 | 0 |
| fitted decay [1/s] (exact 3.54) | 18.4* | 249* | **3.56** | −0.04 | 31.1 | — | 3.64 | — |
| amplitude ratio | 0.47 | 0.19 | 0.97 | 0.15 | 0.38 | 0.19 | 1.03 | 0.06 (@20k) |
| max \|u_t\| ratio | 0.76 | 0.55 | 0.98 | 0.02 | 0.71 | 0.54 | 0.97 | 0.01 (@20k) |
| IC error, u (× A₀) | 4.8e-6 | 4.8e-6 | 0.037 | 0.31 | 4.8e-6 | 4.8e-6 | 0.19 | — |
| BC error | 1.4e-5 | 1.4e-5 | 0.90 | 0.30 | 1.4e-5 | 1.4e-5 | 1.21 | — |
| relative PDE residual | **0.178** | 2.36 | n/a (data-only) | 71.5 | 0.215 | 3.64 | n/a | — |
| training wall-clock | 300 s | 298 s | 24 s | 373 s (NTK 56 s) | 296 s | 297 s | 23 s | — |
| PDE evaluations | 1.6e5 | 1.6e5 | 0 | 1.6e5 | 1.6e5 | 1.6e5 | 0 | 1.6e5 |
| parameters / training peak RSS | 241 601 / 0.82 GB | 241 601 / 0.82 GB | 241 601 / 0.80 GB | 241 601 / 0.82 GB | | | | |
| **Oscillation verdict** | **NO: correct period for ≈2 cycles, then over-damped by ≈0.15 s** | NO: constant after 0.01 s | YES (diagnostic) | NO: static | NO | NO | YES (diag.) | NO |

\* The global damped-cosine fit is not meaningful for a trace that oscillates and then collapses to zero. The verdict for Y1 is therefore taken from the traces: the first ≈2 troughs and peaks in displacement **and velocity** align with the exact solution (period ≈ 0.049 s), after which the amplitude collapses. Per the approved rule ("correct first trough but over-damped within ~0.1 s is not success"), Y1 fails the gate.

## Interpretation

- **Y1 (hard tanh², D4 convention):** the schedule helps (L2 0.909 → 0.774, PDE residual 0.215 → 0.178, oscillation sustained ≈0.15 s instead of ≈0.1 s) but is **insufficient**.
- **Y2 (hard tanh², paper convention):** **no change**. The constant-output collapse of the inner network persists under the paper convention (biases N(0,1), 2π on physical inputs).
- **Y3 (supervised control):** the schedule **substantially** improves pure function fitting (0.316 → 0.120; frequency, decay, amplitude and velocity all correct). The optimizer change is beneficial in itself.
- **Y4 (paper Fourier + NTK):** **not rescued.** It stays static (amplitude 0.15, |u_t| 0.02), and the NTK weights still reach 1e12–6e14. The static-attractor failure of the soft-constrained NTK baseline is not an optimizer-schedule artefact.
- **CASE E:** the schedule fixes the supervised problem much more than the physics-trained one. The remaining failure is specific to **physics-constrained training**.

**Mechanism (measured; `profiles/phaseY_residual_time_profile.txt`).** u ≡ 0 satisfies the homogeneous PDE and both BCs exactly. With the IC enforced only at t = 0, the network learns the first cycles and then relaxes to the trivial solution, with residual left only in the transition region. In Y1, **60 % of the squared PDE residual lies in t < 0.1 s, which is 10 % of the domain** (X2: 68 %). This is the "propagation failure" pattern of PINNs.

In Phase E the residual was diffuse and RAD had nothing to target. Here it is concentrated, so a residual-driven sampler now has a well-defined target.

## Recommended next experiment (requires approval)

Primary single change: **Y1 + RAD**, with exactly the already-benchmarked settings (k = 1, c = 1, 5 000 candidates every 500 steps).

Hypothesis: concentrating collocation points in the high-residual transition region counters the propagation failure. This is a hypothesis; residual-based sampling is reported in the literature as a remedy for this failure mode, and the citation is to be verified before any presentation claim.

The other three free cores can each run one independent single-change arm relative to its parent, in the same 5k-step, about 6 min batch:

| ID | Change | Purpose |
|---|---|---|
| Z1 | Y1 + RAD | primary hypothesis |
| Z2 | Y1, seed 1235 | is Y1's improvement over X2 (0.774 vs 0.909) larger than seed noise? |
| Z3 | Y4 + RAD (= paper Fourier + NTK + RAD, the C1′ arm) | fairness: does RAD help the paper baseline equally? |
| Z4 | Y1 with mini-batch 128 (same steps, 4× PDE evaluations, reported as such) | the alternative lever, more residual points per step |

Decision:
- If Z1 passes the oscillation gate, it is the Mode-1 candidate for one confirmation run (with your approval).
- If none passes, the propagation-failure finding is the validated result for the presentation.

**Timeline:** it is ≈22:12 UTC. Z batch plus analysis ≈ 12 min.
