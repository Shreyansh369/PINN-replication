"""Initial (untrained) loss magnitudes and NTK weights for unspecified-by-paper choices. No training."""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from beampinn.config import ExperimentConfig
from beampinn.training.trainer import resolve_problem, setup_torch
from beampinn.models.networks import build_model
from beampinn.sampling.samplers import PaperEpochSampler
from beampinn.losses.residuals import term_residuals, reduce_loss
from beampinn.losses.weighting import ntk_traces, ntk_weights
from beampinn.evaluation.metrics import validation_metrics
variants = {
 "C0 (2pi, physical, bias N(0,1))": {},
 "C0 bias zeros":                    {"bias_init": "zeros"},
 "C0-rc (no 2pi, standardised)":     {"two_pi": False, "input_norm": "standardize"},
 "C0-rc bias zeros":                 {"two_pi": False, "input_norm": "standardize", "bias_init": "zeros"},
 "2pi, unit-normalised inputs":      {"input_norm": "unit"},
}
print(f"{'variant':34s} {'L_u':>9s} {'L_ut':>9s} {'L_ux':>9s} {'L_f':>9s} | {'lam_u':>8s} {'lam_ut':>8s} {'lam_ux':>8s} | L2_init  |u|rms/A0")
for name, kw in variants.items():
    vals = []
    for seed in (1234, 1235, 1236):
        cfg = ExperimentConfig(seed=seed, threads=4); [setattr(cfg.model, k, v) for k, v in kw.items()]
        setup_torch(cfg); bm, refs, c2, g, u0 = resolve_problem(cfg); m = build_model(cfg, bm)
        s = PaperEpochSampler(bm, u0, cfg.sampler, "paper", seed)
        L = {}
        for _ in range(20):                                   # one epoch of mini-batches, no step
            r = term_residuals(m, s.next_batch(), c2, g)
            for k, v in r.items(): L.setdefault(k, []).append(float(reduce_loss(v).detach()))
        lam = ntk_weights(ntk_traces(r, list(m.parameters()), 32)[0])
        v = validation_metrics(m, refs, bm.L, bm.t_end)
        x = torch.rand(2000, 1, dtype=torch.float32) * bm.L; t = torch.rand(2000, 1) * bm.t_end
        urms = float(m(x, t).detach().pow(2).mean().sqrt()) / bm.A0
        vals.append([np.mean(L[k]) for k in ("u", "ut", "ux", "f")] + [lam["u"], lam["ut"], lam["ux"], v["L2_exact"], urms])
    a = np.exp(np.mean(np.log(np.array(vals)), 0))            # geometric mean over 3 seeds
    print(f"{name:34s} " + " ".join(f"{x:9.2e}" for x in a[:4]) + " | " + " ".join(f"{x:8.1e}" for x in a[4:7]) + f" | {a[7]:6.2f}  {a[8]:6.2f}")
print("paper Fig. 5(b), damped FE, first logged point (epoch 0): L_r ~1e8-1e9, L_ut ~1e1-1e2, L_u ~1e-2..1e-1, L_ux ~1e-2 (read from plot, +-1 decade)")
