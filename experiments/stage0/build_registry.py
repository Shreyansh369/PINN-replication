"""Write paper_benchmark_registry.csv. Every paper_* value is transcribed from the PDF
(table/equation cited in paper_source); 'NS' = not stated in the paper."""
import csv
from pathlib import Path

COLS = ["benchmark_id", "beam_type", "boundary_conditions", "initial_conditions", "damping",
        "mode", "E", "rho", "L", "A", "I", "domain", "time_window", "normalization",
        "paper_architecture", "paper_fourier", "paper_optimizer", "paper_learning_rate",
        "paper_batch_size", "paper_minibatch_size", "paper_epochs", "paper_error", "paper_RMSE",
        "paper_frequency_error", "paper_source", "notes"]
STEEL = dict(E="2.0e11", rho="7845.0", A="9.0e-4", I="6.75e-8")
FF_T4 = "Fourier+NTK PINN; Table 4 test 12: 6 hidden x 200 tanh (Section 4 text says 4 x 200)"
FF_FOURIER = "M_x=1 (sigma_x=1); M_t=2 (sigma_t=1,10); feature count NS; Eq.38-39 use 2*pi*B"
R = []
def row(**k):
    d = {c: "NS" for c in COLS}; d.update(k); R.append(d)

row(benchmark_id="FE-D-M1", beam_type="fixed-fixed", boundary_conditions="u=u_x=0 at x=0,L (Eq.49, Table 1)",
    initial_conditions="u(x,0)=first-mode projection of static deflection Eq.27 (amplitude ~0.08 m from Fig.5c; F NS); u_t(x,0)=0",
    damping="b=50 N s/m; PDE coeff gamma=b/rhoA=7.08 (Eq.49); Table 3 zeta=5.79e-2 inconsistent (standard zeta=0.0274)",
    mode="1 (single-mode reference Eq.28, beta1*l=4.7300)", L="2.75", domain="x in [0,2.75] m", time_window="t in [0,1] s (20.59 cycles)",
    normalization="NS (reference code standardises inputs)", paper_architecture=FF_T4, paper_fourier=FF_FOURIER + "; Table 4 #12: sigma_t=(10,1), sigma_x=1",
    paper_optimizer="Adam; NTK weights Eq.37 (update interval NS)", paper_learning_rate="1e-4 (Table 4 #12; schedule NS)",
    paper_batch_size="640 per term (N_u=N_ut=N_ux=N_f, Eq.48)", paper_minibatch_size="32", paper_epochs="45000",
    paper_error="4.64e-4", paper_RMSE="1.36e-3 (vs Abaqus FEA at mid-span, m)", paper_frequency_error="NS (Wn=20.594 Hz; computed 20.5889)",
    paper_source="Sec.5.1.1, Eq.49, Fig.5, Table 3, Table 4 #12, Table 5, Fig.6",
    notes="PROPOSED CANONICAL. Same number appears in Table 4 #12, Table 5 and Fig.6 (mb=32).", **STEEL)
for name, err in [("FCNN", "1.00"), ("vanilla-PINN", "1.11"), ("PINN+NTK", "8.81e-1"), ("PINN+NTK+Fourier", "4.64e-4")]:
    row(benchmark_id=f"FE-D-M1/T5-{name}", beam_type="fixed-fixed", paper_architecture=name, paper_error=err,
        paper_source="Table 5", notes="Method ladder on FE-D-M1; per-method hyperparameters NS")
for mb, err in [(640, "7.32e-1"), (320, "6.18e-1"), (128, "6.57e-2"), (64, "1.46e-3"), (32, "4.64e-4")]:
    row(benchmark_id=f"FE-D-M1/F6-mb{mb}", beam_type="fixed-fixed", paper_minibatch_size=str(mb), paper_batch_size="640",
        paper_error=err, paper_source="Fig.6, Sec.5.1.1", notes="Mini-batch sweep on FE-D-M1 (epochs presumably 45000)")
for no, d, w, st, sx, lr, err in [(1,3,200,"10,1","1","1e-4","1.23e-3"), (2,4,200,"10,1","1","1e-4","5.68e-4"),
        (3,5,200,"10,1","1","1e-4","4.80e-4"), (4,6,100,"10,1","1","1e-4","1.45e-3"), (5,6,150,"10,1","1","1e-4","5.04e-4"),
        (6,6,200,"1","1","1e-4","5.97e-1"), (7,6,200,"10","1","1e-4","4.52e-4"), (8,6,200,"20","1","1e-4","5.57e-4"),
        (9,6,200,"1","10","1e-4","9.43e-1"), (10,6,200,"10,1","1","1e-2","9.78e-1"), (11,6,200,"10,1","1","1e-3","2.01e-1"),
        (12,6,200,"10,1","1","1e-4","4.64e-4"), (13,6,200,"10,1","1","1e-5","2.67e-1"), (14,7,200,"10,1","1","1e-4","5.19e-4"),
        (15,6,250,"10,1","1","1e-4","4.59e-4"), (16,6,200,"20,1","1","1e-4","4.17e-4"), (17,6,200,"10,1,1","1","1e-4","4.96e-4"),
        (18,6,200,"10,1","1,1","1e-4","4.97e-4")]:
    row(benchmark_id=f"FE-?-M1/T4-{no}", beam_type="fixed-fixed", paper_architecture=f"{d} x {w}",
        paper_fourier=f"sigma_t=({st}); sigma_x=({sx})", paper_learning_rate=lr, paper_batch_size="640",
        paper_minibatch_size="32", paper_epochs="45000", paper_error=err, paper_source="Table 4",
        notes="Caption says fixed-end forward problem; damping not stated (#12 value equals FE-D-M1)")
row(benchmark_id="FE-U-M1", beam_type="fixed-fixed", boundary_conditions="u=u_x=0 at x=0,L (Eq.46)",
    initial_conditions="as FE-D-M1", damping="none (b=0)", mode="1", L="2.75", domain="x in [0,2.75] m",
    time_window="t in [0,1] s (20.59 cycles)", paper_architecture=FF_T4, paper_fourier=FF_FOURIER, paper_optimizer="Adam + NTK",
    paper_learning_rate="1e-4", paper_batch_size="640", paper_minibatch_size="32", paper_epochs="45000",
    paper_error="2.70e-3", paper_RMSE="9.71e-3 (vs FEA, mid-span)", paper_source="Sec.5.1.1, Eq.46-48, Fig.4",
    notes="Paper: undamped harder than damped (damped L2 is early-time weighted)", **STEEL)
row(benchmark_id="SS-U-M1", beam_type="simply-supported", boundary_conditions="u=u_xx=0 at x=0,L (prose, Table 1); Eq.A.4 prints u_xx=u_xxx=0 (typo: free-free)",
    initial_conditions="first-mode projection of Eq.A.2 (amplitude ~0.065 m, Fig.A.19c); u_t=0", damping="none",
    mode="1 (Eq.A.3, beta1*l=3.1416)", L="2.75", domain="x in [0,2.75] m", time_window="t in [0,1] s (9.08 cycles)",
    paper_architecture="'same as Section 5.1.1'", paper_fourier=FF_FOURIER, paper_optimizer="Adam + NTK",
    paper_learning_rate="1e-4 (inferred)", paper_batch_size="960 per term (Eq.A.6)", paper_minibatch_size="NS (32 if 'same as 5.1.1')",
    paper_epochs="30000", paper_error="2.3e-3", paper_RMSE="8.82e-4 (vs FEA)", paper_source="Appendix A.2, Table A.10, Fig.A.19",
    notes="Legacy repo implements this case (P_paper_window: 0.967 at 4000 it). Target coincides with c^2-rounding floor 2.27e-3", **STEEL)
row(benchmark_id="SS-D-M1", beam_type="simply-supported", boundary_conditions="u=u_xx=0 (Eq.A.7 prints u_xx=u_xxx=0)",
    initial_conditions="as SS-U-M1", damping="b=50 (gamma 7.08, Eq.A.7)", mode="1", L="2.75", domain="x in [0,2.75] m",
    time_window="t in [0,1] s", paper_architecture="as SS-U-M1", paper_batch_size="960", paper_epochs="30000",
    paper_error="4.07e-2", paper_RMSE="6.12e-4 (vs FEA)", paper_source="Appendix A.2, Fig.A.20", **STEEL)
row(benchmark_id="CF-U-M1", beam_type="cantilever (fixed-free)", boundary_conditions="u=u_x=0 at x=0; u_xx=u_xxx=0 at x=L (Eq.B.5)",
    initial_conditions="first-mode projection of Eq.B.3; u_t=0", damping="none", mode="1 (beta1*l=1.8751)", L="4.0",
    domain="x in [0,4] m", time_window="t in [0,1] s (1.53 cycles)", paper_architecture="as Section 4", paper_batch_size="640 (Eq.B.7)",
    paper_epochs="70000", paper_error="3.07e-5", paper_RMSE="2.74e-3 (vs FEA, free tip)", paper_source="Appendix B.2, Table B.11, Fig.B.22", **STEEL)
row(benchmark_id="CF-D-M1", beam_type="cantilever (fixed-free)", boundary_conditions="as CF-U-M1 (Eq.B.8)",
    initial_conditions="as CF-U-M1", damping="b=5 N s/m (gamma=0.708, Eq.B.8)", mode="1", L="4.0", domain="x in [0,4] m",
    time_window="t in [0,5] s (7.6 cycles)", paper_batch_size="640", paper_epochs="70000", paper_error="7.20e-4",
    paper_RMSE="1.64e-3 (vs FEA)", paper_source="Appendix B.2, Fig.B.23", **STEEL)
row(benchmark_id="CF-D-INV", beam_type="cantilever (fixed-free)", boundary_conditions="as CF-D-M1 ('periodic' per Sec.5.1.2 text)",
    initial_conditions="as CF-D-M1", damping="identify b (true 5.0); gamma=exp(eps), eps0=-0.2", mode="1", L="4.0",
    time_window="t in [0,5] s (Fig.7b)", paper_architecture="4 x 200; 1 temporal + 2 spatial Fourier mappings",
    paper_batch_size="3200 (N_u=N_f)", paper_minibatch_size="128", paper_epochs="250000", paper_error="6.35e-2 (field L2); b error 1.41% (5.07)",
    paper_source="Sec.5.1.2, Eq.50-54, Table 6, Fig.7-8", notes="Inverse problem: uses interior analytical data", **STEEL)
row(benchmark_id="EXP-CF-PLEXI", beam_type="cantilever, plexiglass", boundary_conditions="Eq.55", damping="gamma=2.16; impulse 0.142 N",
    E="3.5e9", rho="1185.0", L="0.65", A="3.36e-4", I="1.9757e-9", time_window="t in [0,1] s", paper_epochs="30000 (sensor) / 20000 (synthetic)",
    paper_error="1.44e-1 (sensor velocity) / 1.24e-1 (synthetic)", paper_frequency_error="6.00 Hz PINN vs 5.99 Hz sensor (Table 8)",
    paper_source="Sec.5.2, Table 7-8", notes="Needs measured data that the paper does not release ('available on request')")
row(benchmark_id="WAVE1D-REF", beam_type="not a beam (1-D wave eq., Wang et al. 2022)", paper_architecture="PINN+RAD / gPINN+RAR / PINN+NTK / PINN+NTK+FF",
    paper_error="9.00e-2 / 7.31e-3 / 1.73e-3 / 6.22e-4", paper_source="Table 2", notes="Out of scope; context only")
p = Path(__file__).resolve().parents[2] / "paper_benchmark_registry.csv"
with p.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(R)
print(f"wrote {p} ({len(R)} rows)")
