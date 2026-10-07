"""Internal consistency check of the final deck against the frozen artifacts. Read-only.

    python presentation_build/check_consistency.py

1. Every number shown on a slide is recomputed from its frozen source, formatted as on the slide,
   and must appear in that slide's text.
2. Language and scope checks: no validated-optimized-PINN claim, no Mode-2 result, railway only as
   future work, the paper's 4.64e-4 is labelled as the published value.
"""
import csv
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RO = ROOT / "results_optimization"
DECK = ROOT / "final_scientific_conference_presentation.pptx"


def slide_text(n, notes=False):
    z = zipfile.ZipFile(DECK)
    part = f"ppt/notesSlides/notesSlide{n}.xml" if notes else f"ppt/slides/slide{n}.xml"
    txt = " ".join(re.findall(r"<a:t>([^<]*)</a:t>", z.read(part).decode("utf8")))
    return txt.replace("&amp;", "&")


def rows(name, key="name"):
    return {r[key]: r for r in csv.DictReader(open(RO / "tables" / f"{name}.csv"))}


def ck(name, run):
    return {int(r["step"]): r for r in csv.DictReader(open(RO / "tables" / f"{name}.csv")) if r["run"] == run}


A = rows("phaseA_summary", "method")
E = rows("phaseE_oscillation"); X = rows("phaseX_oscillation"); Y = rows("phaseY_oscillation"); Z = rows("phaseZ_oscillation")
XC = rows("phaseX_conditioning")
Y20 = ck("phaseY1_20K_checkpoints", "Y1_20K"); Z20 = ck("phaseZ4_20K_checkpoints", "Z4_20K")
REF = [r for r in csv.DictReader(open(RO / "tables" / "stage01_reference_comparison.csv"))
       if r["grid"] == "201x2001" and r["pred"] == "exact" and r["ref"] == "paper"][0]
c0_hist = list(csv.DictReader(open(RO / "logs" / "C0_paper__s1234__c41ccb1cdf" / "history.csv")))
c0_last = [r for r in c0_hist if r.get("lam_ut")][-1]
prof = (RO / "profiles" / "phaseE_ansatz_conditioning.txt").read_text()
nstar = re.search(r"g=\(t/T\)\^2\s+N\* range \[\s*(-?[\d.]+),\s*(-?[\d.]+)\]", prof)
nstar_t = re.search(r"tanh\^2\(w t\), w=omega_1 from PDE coeffs\s+N\* range \[\s*(-?[\d.]+),\s*(-?[\d.]+)\]", prof)

f2, f3 = (lambda v: f"{float(v):.2f}"), (lambda v: f"{float(v):.3f}")
f1 = lambda v: f"{float(v):.1f}"
def e1(v):
    m, e = f"{float(v):.1e}".split("e")
    return f"{m}e{int(e)}"

# (slide, text as shown, value recomputed from source, source)
CHECKS = [
    (4, "3.26", f2(A["C0_paper"]["L2_exact"]), "tables/phaseA_summary.csv C0_paper L2_exact"),
    (4, "3.26", f2(A["C0_paper"]["L2_paper"]), "tables/phaseA_summary.csv C0_paper L2_paper"),
    (4, "8.5e14", e1(c0_last["lam_ut"]), "logs/C0_paper__s1234__c41ccb1cdf/history.csv lam_ut (last)"),
    (4, "9.9e14", e1(c0_last["lam_ux"]), "logs/C0_paper__s1234__c41ccb1cdf/history.csv lam_ux (last)"),
    (5, "−8 369", "−" + f"{abs(float(nstar.group(1))):,.0f}".replace(",", " "), "profiles/phaseE_ansatz_conditioning.txt"),
    (5, "−1.9", "−" + f"{abs(float(nstar_t.group(1))):.1f}", "profiles/phaseE_ansatz_conditioning.txt"),
    (5, "−0.16", "−" + f"{abs(float(nstar_t.group(2))):.2f}", "profiles/phaseE_ansatz_conditioning.txt"),
    (5, "7.2e3", e1(XC["E4_hard_fourier"]["gradnorm_pde_early_t<0.02"]), "tables/phaseX_conditioning.csv"),
    (5, "1.5e8", e1(XC["E4_hard_fourier"]["gradnorm_pde_late_t>0.5"]), "tables/phaseX_conditioning.csv"),
    (5, "8.4e5", e1(XC["X2_hard_tanh2_rc"]["gradnorm_pde_early_t<0.02"]), "tables/phaseX_conditioning.csv"),
    (5, "9.8e4", e1(XC["X2_hard_tanh2_rc"]["gradnorm_pde_late_t>0.5"]), "tables/phaseX_conditioning.csv"),
    (5, "0.463", f3(X["X3_supervised_paperconv"]["L2_exact"]), "tables/phaseX_oscillation.csv"),
    (5, "0.316", f3(X["X4_supervised_rcconv"]["L2_exact"]), "tables/phaseX_oscillation.csv"),
    (5, "129.25", f2(X["X3_supervised_paperconv"]["fit_w"]), "tables/phaseX_oscillation.csv"),
    (5, "129.37", f2(X["X4_supervised_rcconv"]["fit_w"]), "tables/phaseX_oscillation.csv"),
    (5, "129.32", f2(X["X4_supervised_rcconv"]["w_exact"]), "tables/phaseX_oscillation.csv w_exact"),
    (6, "2.83", f2(E["E4_hard_fourier"]["L2_exact"]), "tables/phaseE_oscillation.csv"),
    (6, "0.909", f3(X["X2_hard_tanh2_rc"]["L2_exact"]), "tables/phaseX_oscillation.csv"),
    (6, "112.2", f1(X["X2_hard_tanh2_rc"]["fit_w"]), "tables/phaseX_oscillation.csv"),
    (6, "0.774", f3(Y["Y1_X2_lrsched"]["L2_exact"]), "tables/phaseY_oscillation.csv"),
    (6, "0.718", f3(Z["Z1_Y1_rad"]["L2_exact"]), "tables/phaseZ_oscillation.csv"),
    (6, "0.526", f3(Z["Z4_Y1_mb128"]["L2_exact"]), "tables/phaseZ_oscillation.csv"),
    (6, "0.263", f3(Z20[20000]["L2_exact"]), "tables/phaseZ4_20K_checkpoints.csv"),
    (6, "1.2 cycles", f1(Y20[5000]["persistence_cycles"]) + " cycles", "tables/phaseY1_20K_checkpoints.csv (5k = Y1)"),
    (6, "3.2 cycles", f1(Z20[5000]["persistence_cycles"]) + " cycles", "tables/phaseZ4_20K_checkpoints.csv (5k = Z4)"),
    (6, "8.1 cycles", f1(Z20[20000]["persistence_cycles"]) + " cycles", "tables/phaseZ4_20K_checkpoints.csv"),
    (6, "129.2 rad/s", f1(Z20[20000]["fit_w"]) + " rad/s", "tables/phaseZ4_20K_checkpoints.csv fit_w"),
    (6, "5.40", f2(Z20[20000]["fit_decay"]), "tables/phaseZ4_20K_checkpoints.csv fit_decay"),
    (6, "0.393 s", f3(Z20[20000]["collapse_time_s"]) + " s", "tables/phaseZ4_20K_checkpoints.csv"),
    (7, "0.057 s", f3(Y20[5000]["collapse_time_s"]) + " s", "tables/phaseY1_20K_checkpoints.csv"),
    (7, "0.151 s", f3(Y20[20000]["collapse_time_s"]) + " s", "tables/phaseY1_20K_checkpoints.csv"),
    (7, "0.153 s", f3(Z20[5000]["collapse_time_s"]) + " s", "tables/phaseZ4_20K_checkpoints.csv"),
    (7, "0.393 s", f3(Z20[20000]["collapse_time_s"]) + " s", "tables/phaseZ4_20K_checkpoints.csv"),
    (7, "160k", f"{float(Y20[5000]['pde_evaluations_cum']) / 1e3:.0f}k", "tables/phaseY1_20K_checkpoints.csv"),
    (7, "640k", f"{float(Y20[20000]['pde_evaluations_cum']) / 1e3:.0f}k", "tables/phaseY1_20K_checkpoints.csv"),
    (7, "2.56M", f"{float(Z20[20000]['pde_evaluations_cum']) / 1e6:.2f}M", "tables/phaseZ4_20K_checkpoints.csv"),
    (7, "0.550", f3(Y20[20000]["L2_exact"]), "tables/phaseY1_20K_checkpoints.csv"),
    (7, "0.526", f3(Z20[5000]["L2_exact"]), "tables/phaseZ4_20K_checkpoints.csv"),
    (7, "3.1", f1(Y20[20000]["persistence_cycles"]), "tables/phaseY1_20K_checkpoints.csv"),
    (7, "1 078 s", f"{float(Y20[20000]['train_seconds_cum']):,.0f} s".replace(",", " "), "tables/phaseY1_20K_checkpoints.csv"),
    (7, "+0.047 s", f"+{float(Z20[20000]['collapse_time_s']) - float(Z20[15000]['collapse_time_s']):.3f} s", "tables/phaseZ4_20K_checkpoints.csv"),
    (7, "+0.096 s", f"+{float(Z20[10000]['collapse_time_s']) - float(Z20[5000]['collapse_time_s']):.3f} s", "tables/phaseZ4_20K_checkpoints.csv"),
    (3, "4.386e-4", f"{float(REF['L2']):.3e}".replace("e-04", "e-4"), "tables/stage01_reference_comparison.csv 201x2001 exact vs paper"),
    (3, "94.5 %", f"{100 * float(REF['floor_over_target']):.1f} %", "tables/stage01_reference_comparison.csv floor_over_target"),
]
DD = rows("phaseD_oscillation")
CHECKS.append((4, "0.06", f2(DD["C0_paper"]["amp_ratio"]), "tables/phaseD_oscillation.csv C0_paper amp_ratio"))
CHECKS.append((2, "129.32", f2(Y["Y1_X2_lrsched"]["w_exact"]), "tables/phaseY_oscillation.csv w_exact (analytical)"))
CHECKS.append((2, "3.54", f2(Y["Y1_X2_lrsched"]["decay_exact"]), "tables/phaseY_oscillation.csv decay_exact (analytical)"))
# 611 s (Z4 5k, original run) is quoted from PHASE_Y1_20K_BUDGET_DIAGNOSTIC.md / phaseZ_oscillation.csv
CHECKS.append((7, "611 s", f"{float(Z['Z4_Y1_mb128']['train_seconds']):.0f} s", "tables/phaseZ_oscillation.csv train_seconds"))

fail = 0
for n, shown, recomputed, src in CHECKS:
    ok_val = shown == recomputed
    ok_txt = shown in slide_text(n)
    if not (ok_val and ok_txt):
        fail += 1
    print(f"{'OK  ' if ok_val and ok_txt else 'FAIL'} slide {n}: shown {shown!r:14} source {recomputed!r:14} on-slide={ok_txt}  [{src}]")

# scope / language checks over slides + notes
alltxt = {n: slide_text(n) + " " + slide_text(n, True) for n in range(1, 9)}
for n, t in alltxt.items():
    for bad in ("state of the art", "state-of-the-art", "guarantee", "best PINN", "solves", "outperform"):
        if re.search(r"\b" + re.escape(bad) + r"\b", t, re.I):
            fail += 1; print(f"FAIL slide {n}: forbidden wording {bad!r}")
    for m in re.finditer(r"optimized PINN", t):
        ctx = t[max(0, m.start() - 40):m.start()]
        if not re.search(r"(No validated|not established|NOT established|fully validated)", ctx):
            fail += 1; print(f"FAIL slide {n}: 'optimized PINN' without negation: ...{ctx}")
    if re.search(r"rail", t, re.I) and n != 8:
        fail += 1; print(f"FAIL slide {n}: railway mentioned outside the future-work slide")
    if "Mode-2" in t or "Mode 2" in t:
        for m in re.finditer(r"Mode[- ]2", t):
            ctx = t[max(0, m.start() - 60):m.end() + 40]
            if not re.search(r"(No Mode-2|Mode-2 validation|Mode 2 was not run)", ctx):
                fail += 1; print(f"FAIL slide {n}: Mode-2 mention not marked as absent/future: {ctx}")
s2 = slide_text(2)
if not ("PUBLISHED BASELINE" in s2 and "not a result of this study" in s2):
    fail += 1; print("FAIL slide 2: 4.64e-4 not labelled as the published value")
s8 = slide_text(8)
if "FUTURE WORK (not part of this study)" not in s8 or "future work only" not in s8:
    fail += 1; print("FAIL slide 8: railway not marked as future work")
print("scope/language checks done")
print("ALL CHECKS PASSED" if fail == 0 else f"{fail} FAILURE(S)")
sys.exit(1 if fail else 0)
