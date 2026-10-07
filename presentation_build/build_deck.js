// Final presentation deck, built only from frozen, committed artifacts (no training, no new results).
//   NODE_PATH=<dir with pptxgenjs> node presentation_build/build_deck.js <path to pptx skill apply_theme.js>
// Data:    presentation_tables/deck_data.json (extracted from frozen CSVs / analytical reference)
// Images:  presentation_figures/*.png (crops of frozen figures in results_optimization/figures)
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const OUT = path.join(ROOT, "final_scientific_conference_presentation.pptx");
const FIG = (f) => path.join(ROOT, "presentation_figures", f);
const D = JSON.parse(fs.readFileSync(path.join(ROOT, "presentation_tables", "deck_data.json"), "utf8"));
const { applyTheme } = require(process.argv[2]);

const THEME = {
  name: "Beam Vibration Audit",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B2A35", lt1: "FFFFFF", dk2: "3C5566", lt2: "EEF2F4",
    accent1: "E8743B", accent2: "2A9D8F", accent3: "2F6FBF", accent4: "B5473A",
    accent5: "8A9AA6", accent6: "E9B44C", hlink: "2A9D8F", folHlink: "8A9AA6",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.title = "Reproducibility and Optimization of PINNs for High-Frequency Beam Vibration";
pres.subject = "Controlled reproducibility and optimization investigation (frozen results)";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const W = 13.333;
const MX = 0.6;            // side margin
const CW = W - 2 * MX;     // content width

// ------------------------------------------------------------------ layouts
const FOOT = "Reproducibility study of Söyleyici & Ünver (EAAI 2025) · FE-D-M1, Mode 1 · frozen results only";
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 1.05, w: 9.6, h: 1.9, fontFace: THEME.headFontFace, fontSize: 38, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: MX, y: 3.1, w: 9.6, h: 0.9, fontSize: 18, color: C.background2, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.5, MX, 0.6, MX],
  objects: [
    { placeholder: { options: { name: "kicker", type: "body", x: MX, y: 0.32, w: CW, h: 0.32, fontSize: 12, bold: true, color: C.accent1, charSpacing: 1, margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 0.62, w: CW, h: 0.72, fontFace: THEME.headFontFace, fontSize: 32, bold: true, color: C.text1, valign: "top", align: "left", margin: 0 }, text: "" } },
    { text: { text: FOOT, options: { x: MX, y: 7.05, w: 10.5, h: 0.3, fontSize: 10, color: C.accent5, margin: 0 } } },
  ],
  slideNumber: { x: W - MX - 0.6, y: 7.05, w: 0.6, h: 0.3, fontSize: 10, color: C.accent5, align: "right" },
});
pres.defineSlideMaster({
  title: "CLOSING_DARK",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "kicker", type: "body", x: MX, y: 0.32, w: CW, h: 0.32, fontSize: 12, bold: true, color: C.accent1, charSpacing: 1, margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 0.62, w: CW, h: 0.72, fontFace: THEME.headFontFace, fontSize: 32, bold: true, color: C.background1, valign: "top", align: "left", margin: 0 }, text: "" } },
    { text: { text: FOOT, options: { x: MX, y: 7.05, w: 10.5, h: 0.3, fontSize: 10, color: C.accent5, margin: 0 } } },
  ],
  slideNumber: { x: W - MX - 0.6, y: 7.05, w: 0.6, h: 0.3, fontSize: 10, color: C.accent5, align: "right" },
});

// ------------------------------------------------------------------ helpers
const T = (slide, text, o) => slide.addText(text, { isTextBox: true, margin: 0, fontSize: 14, color: C.text1, valign: "top", ...o });
const card = (slide, x, y, w, h, fill, name, line) =>
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: line ? { color: line, width: 1 } : { type: "none" }, objectName: name });
const badge = (slide, x, y, n, fill, name) => {
  slide.addShape(pres.shapes.OVAL, { x, y, w: 0.42, h: 0.42, fill: { color: fill }, line: { type: "none" }, objectName: name });
  T(slide, String(n), { x, y, w: 0.42, h: 0.42, fontSize: 14, bold: true, color: C.background1, align: "center", valign: "middle", objectName: name + " number" });
};
const sci = (v) => {
  const e = Math.floor(Math.log10(v));
  return `${(v / 10 ** e).toFixed(1)}e${e}`;
};
const AX = { catAxisLabelColor: HEX.dk2, valAxisLabelColor: HEX.dk2, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", catAxisLabelFontSize: 11, valAxisLabelFontSize: 11,
  catAxisTitleColor: HEX.dk2, valAxisTitleColor: HEX.dk2, catAxisTitleFontFace: "+mn-lt", valAxisTitleFontFace: "+mn-lt", catAxisTitleFontSize: 11, valAxisTitleFontSize: 11,
  titleFontFace: "+mn-lt", titleColor: HEX.dk1, titleFontSize: 13, legendFontFace: "+mn-lt", legendFontSize: 11, legendColor: HEX.dk2 };

// ================================================================== slide 1 — title
pres.addSection({ title: "Introduction" });
{
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
  s.addText("Reproducibility and Optimization of Physics-Informed Neural Networks for High-Frequency Beam Vibration", { placeholder: "title" });
  s.addText("Controlled investigation of Fourier features, NTK weighting, hard constraints, optimization, sampling and computational budget", { placeholder: "body" });
  T(s, [
    { text: "[Author name]", options: { bold: true, breakLine: true } },
    { text: "[Institution]", options: { breakLine: true } },
    { text: "October 2026" },
  ], { x: MX, y: 4.25, w: 6, h: 1.0, fontSize: 16, color: C.background1, objectName: "Author block" });
  // signal motif: analytical mid-span reference response (not a training result)
  const u = D.exact.u_mm, t = D.exact.t;
  s.addChart(pres.charts.LINE, [{ name: "exact mid-span displacement", labels: t.map((v) => v.toFixed(3)), values: u }], {
    x: 0.35, y: 5.25, w: W - 0.7, h: 1.55, objectName: "Analytical reference signal",
    chartColors: [HEX.accent1], lineSize: 1.5, lineDataSymbol: "none", showLegend: false,
    catAxisHidden: true, valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" },
    valAxisMinVal: -85, valAxisMaxVal: 85, plotArea: { fill: { color: HEX.dk1 } }, chartArea: { fill: { color: HEX.dk1 } },
  });
  T(s, "Analytical reference: mid-span displacement of the fixed–fixed beam, Mode 1 (20.6 Hz), t = 0–1 s", { x: MX, y: 6.85, w: 9, h: 0.3, fontSize: 10, color: C.accent5, objectName: "Signal caption" });
  // beam schematic (top right): clamped ends + Mode-1 shape
  const bx = 10.35, by = 1.55, bw = 2.35;
  s.addShape(pres.shapes.RECTANGLE, { x: bx - 0.14, y: by - 0.25, w: 0.14, h: 0.75, fill: { color: C.accent5 }, line: { type: "none" }, objectName: "Clamp left" });
  s.addShape(pres.shapes.RECTANGLE, { x: bx + bw, y: by - 0.25, w: 0.14, h: 0.75, fill: { color: C.accent5 }, line: { type: "none" }, objectName: "Clamp right" });
  s.addChart(pres.charts.LINE, [{ name: "Mode-1 shape", labels: D.mode_shape.map((_, i) => String(i)), values: D.mode_shape.map((v) => -v) }], {
    x: bx - 0.05, y: by - 0.1, w: bw + 0.1, h: 1.15, objectName: "Mode-1 shape",
    chartColors: [HEX.lt1], lineSize: 2.5, lineDataSymbol: "none", showLegend: false,
    catAxisHidden: true, valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" },
    valAxisMinVal: -1.1, valAxisMaxVal: 0.1, plotArea: { fill: { color: HEX.dk1 } }, chartArea: { fill: { color: HEX.dk1 } },
  });
  T(s, "Fixed–fixed Euler–Bernoulli beam, first mode", { x: bx - 0.3, y: by + 1.15, w: bw + 0.6, h: 0.3, fontSize: 10, color: C.accent5, align: "center", objectName: "Beam caption" });
  s.addNotes([
    "WHAT THIS SLIDE SHOWS: Title. This talk reports a controlled reproducibility and optimization investigation of a published Fourier-feature / NTK-weighted PINN for damped beam vibration.",
    "WHY IT MATTERS: High-frequency, long-window transient vibration is a known hard case for PINNs; a reproducibility audit tells us what is and is not established.",
    "KEY RESULT: We identified conditioning and optimization bottlenecks and quantified budget effects; a fully validated optimized PINN was NOT established within the tested budget.",
    "The orange trace is the analytical reference solution (Eq. 28, exact root), not a network output. Author and institution fields are placeholders to be filled in.",
  ].join("\n\n"));
}

// ================================================================== slide 2 — problem + published baseline
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  s.addText("PROBLEM AND TARGET PAPER", { placeholder: "kicker" });
  s.addText("A damped, high-frequency beam transient", { placeholder: "title" });
  // left: the physics
  const lx = MX, lw = 6.0, top = 1.65;
  T(s, "The physical problem", { x: lx, y: top, w: lw, h: 0.4, fontSize: 20, bold: true, fontFace: THEME.headFontFace, objectName: "Physics heading" });
  // beam schematic
  const bx = lx + 0.35, by = top + 0.65, bw = 5.0;
  s.addShape(pres.shapes.RECTANGLE, { x: bx - 0.18, y: by - 0.15, w: 0.18, h: 1.05, fill: { color: C.text2 }, line: { type: "none" }, objectName: "Clamp left" });
  s.addShape(pres.shapes.RECTANGLE, { x: bx + bw, y: by - 0.15, w: 0.18, h: 1.05, fill: { color: C.text2 }, line: { type: "none" }, objectName: "Clamp right" });
  s.addChart(pres.charts.LINE, [{ name: "Mode-1 shape", labels: D.mode_shape.map((_, i) => String(i)), values: D.mode_shape.map((v) => -v) }], {
    x: bx - 0.05, y: by - 0.05, w: bw + 0.1, h: 0.95, objectName: "Mode-1 shape",
    chartColors: [HEX.accent1], lineSize: 2.5, lineDataSymbol: "none", showLegend: false,
    catAxisHidden: true, valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" }, valAxisMinVal: -1.1, valAxisMaxVal: 0.1,
  });
  T(s, "L = 2.75 m, both ends clamped (u = u_x = 0); initial shape = Mode 1, A₀ = 0.08 m", { x: lx, y: by + 1.0, w: lw, h: 0.3, fontSize: 12, color: C.text2, objectName: "Beam caption" });
  card(s, lx, top + 2.15, lw, 0.75, C.background2, "PDE card");
  T(s, [
    { text: "c²·u_xxxx + u_tt + γ·u_t = 0", options: { bold: true, fontFace: "Cambria", fontSize: 20, breakLine: true } },
    { text: "Euler–Bernoulli beam with viscous damping (paper Eq. 49): c² = 43.73², γ = 7.08", options: { fontSize: 12, color: C.text2 } },
  ], { x: lx + 0.2, y: top + 2.22, w: lw - 0.4, h: 0.65, objectName: "PDE text" });
  // three stat callouts
  const stats = [["20.6 Hz", "Mode 1, ω_d = 129.32 rad/s"], ["20.6", "cycles in the 1 s window"], ["3.54 s⁻¹", "physical decay rate"]];
  stats.forEach(([v, l], i) => {
    const x = lx + i * 2.05;
    T(s, v, { x, y: top + 3.1, w: 1.95, h: 0.6, fontSize: 28, bold: true, color: C.accent1, fontFace: THEME.headFontFace, objectName: `Stat ${i + 1} value` });
    T(s, l, { x, y: top + 3.7, w: 1.9, h: 0.5, fontSize: 12, color: C.text2, objectName: `Stat ${i + 1} label` });
  });
  // right: published baseline card
  const rx = 7.1, rw = W - MX - rx;
  card(s, rx, top, rw, 5.05, C.background2, "Published baseline card");
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: rx + 0.3, y: top + 0.3, w: 2.45, h: 0.38, rectRadius: 0.06, fill: { color: C.text1 }, line: { type: "none" }, objectName: "Baseline tag" });
  T(s, "PUBLISHED BASELINE", { x: rx + 0.3, y: top + 0.3, w: 2.45, h: 0.38, fontSize: 12, bold: true, color: C.background1, align: "center", valign: "middle", charSpacing: 1, objectName: "Baseline tag text" });
  T(s, "Söyleyici & Ünver, Eng. Appl. Artif. Intell. 141 (2025) 109804", { x: rx + 0.3, y: top + 0.85, w: rw - 0.6, h: 0.3, fontSize: 12, italic: true, color: C.text2, objectName: "Citation" });
  T(s, [
    { text: "6 × 200 tanh network", options: { bullet: true, breakLine: true } },
    { text: "Spatio-temporal multiscale Fourier features (σ_x = 1; σ_t = 10, 1)", options: { bullet: true, breakLine: true } },
    { text: "NTK-trace loss weighting", options: { bullet: true, breakLine: true } },
    { text: "Adam, lr 1e-4, 45 000 epochs", options: { bullet: true } },
  ], { x: rx + 0.3, y: top + 1.3, w: rw - 0.6, h: 1.5, fontSize: 15, paraSpaceAfter: 4, objectName: "Method bullets" });
  T(s, "4.64 × 10⁻⁴", { x: rx + 0.3, y: top + 2.95, w: rw - 0.6, h: 0.75, fontSize: 40, bold: true, color: C.text1, fontFace: THEME.headFontFace, objectName: "Paper L2 value" });
  T(s, "Relative L2 error reported by the paper for this case (FE-D-M1, Mode 1). Published value — not a result of this study.", { x: rx + 0.3, y: top + 3.7, w: rw - 0.6, h: 0.6, fontSize: 12, color: C.text2, objectName: "Paper L2 label" });
  T(s, "Fourier + NTK is the paper's method, not our contribution. Our work audits and stress-tests it.", { x: rx + 0.3, y: top + 4.35, w: rw - 0.6, h: 0.55, fontSize: 13, bold: true, color: C.accent4, objectName: "Attribution note" });
  s.addNotes([
    "WHAT THIS SLIDE SHOWS: The benchmark — a fixed–fixed Euler–Bernoulli beam released from its first mode shape, with viscous damping, over a 1 s window — and the published method we audited.",
    "WHY IT MATTERS: At 20.6 Hz the window contains about 20.6 damped cycles. A PINN must propagate the correct oscillation from the initial condition through the whole window, which is the hard part.",
    "KEY RESULT: The paper reports a relative L2 error of 4.64e-4 with Fourier features + NTK weighting. That number is the paper's, shown here as the published baseline. Fourier + NTK is not our contribution.",
    "Sources: paper Eq. 49 and Table 3; ω_d and decay rate from our analytical reference (src/beampinn/physics/benchmarks.py).",
  ].join("\n\n"));
}

// ================================================================== slide 3 — audit workflow
pres.addSection({ title: "Investigation" });
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Investigation" });
  s.addText("REPRODUCIBILITY AUDIT", { placeholder: "kicker" });
  s.addText("A controlled, one-change-at-a-time workflow", { placeholder: "title" });
  const steps = ["Published method", "Implementation audit", "Paper-faithful baseline", "Diagnostic variants", "Conditioning experiment", "Optimizer experiment", "Sampling / batch experiment", "Compute-budget study"];
  const y0 = 1.75, gap = 0.12, nw = (CW - 7 * gap) / 8;
  steps.forEach((st, i) => {
    const x = MX + i * (nw + gap);
    card(s, x, y0, nw, 1.25, i === 0 ? C.text1 : C.background2, `Step ${i + 1} card`);
    T(s, String(i + 1), { x: x + 0.15, y: y0 + 0.12, w: 0.5, h: 0.4, fontSize: 18, bold: true, color: C.accent1, fontFace: THEME.headFontFace, objectName: `Step ${i + 1} number` });
    T(s, st, { x: x + 0.12, y: y0 + 0.52, w: nw - 0.16, h: 0.65, fontSize: 12, bold: true, color: i === 0 ? C.background1 : C.text1, objectName: `Step ${i + 1} label` });
  });
  T(s, "Each step changed one factor relative to its parent run; every run used one CPU thread, seed 1234, float32 and an identical evaluation grid (201 × 2001).",
    { x: MX, y: y0 + 1.4, w: CW, h: 0.35, fontSize: 12, color: C.text2, objectName: "Workflow caption" });
  T(s, "Verified audit findings", { x: MX, y: 3.75, w: CW, h: 0.4, fontSize: 20, bold: true, fontFace: THEME.headFontFace, objectName: "Findings heading" });
  const F = [
    ["Fourier convention", "Legacy code omitted the 2π factor of paper Eq. 38–39."],
    ["Frequency metric", "Legacy metric used f = mode² and 1 Hz FFT bins; replaced by a damped-cosine fit."],
    ["Sparse evaluation", "An 81 × 81 grid under-resolves 20.6 cycles; replaced by 201 × 2001."],
    ["Reference-code differences", "Public code uses standardised inputs, no 2π, N(0,1) biases and a decaying LR."],
    ["Two references", "Printed root (4.7300) vs exact root: they differ by 4.386e-4 relative L2, 94.5 % of the 4.64e-4 target."],
  ];
  const fw = (CW - 0.3) / 2;
  F.forEach(([h, d], i) => {
    const col = i < 3 ? 0 : 1, row = i < 3 ? i : i - 3;
    const x = MX + col * (fw + 0.3), y = 4.3 + row * 0.82;
    badge(s, x, y + 0.04, i + 1, i === 4 ? C.accent2 : C.text2, `Finding ${i + 1} badge`);
    T(s, [
      { text: h, options: { bold: true, fontSize: 14, breakLine: true } },
      { text: d, options: { fontSize: 12, color: C.text2 } },
    ], { x: x + 0.6, y, w: fw - 0.6, h: 0.75, objectName: `Finding ${i + 1}` });
  });
  s.addNotes([
    "WHAT THIS SLIDE SHOWS: The order of the investigation, from the published method to the compute-budget study, and the five audit findings that had to be fixed before any comparison was meaningful.",
    "WHY IT MATTERS: Without a faithful baseline and a correct metric, a negative (or positive) result cannot be attributed. The dual reference matters most: the printed eigenvalue root alone accounts for 4.386e-4 relative L2, i.e. 94.5% of the published 4.64e-4, so we score against both a paper-faithful and an exact-physics reference.",
    "KEY RESULT: Five verified implementation/evaluation issues (Stage 0 audit and Stage 0.1 corrections). The damped-cosine frequency fit has a 95th-percentile error of about 1.4e-5 at the target error level.",
    "Sources: results_optimization/reports/STAGE0_AUDIT.md, STAGE01_CORRECTIONS.md, CONVENTIONS.md; tables/stage01_reference_comparison.csv, stage01_frequency_extractor_uncertainty.csv.",
  ].join("\n\n"));
}

// ================================================================== slide 4 — first failure (C0)
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Investigation" });
  s.addText("FIRST FAILURE: PAPER-FAITHFUL FOURIER + NTK (C0)", { placeholder: "kicker" });
  s.addText("The faithful baseline settles on a near-static field", { placeholder: "title" });
  const top = 1.6, pw = (CW - 0.4) / 2, ph = 3.55;
  s.addImage({ path: FIG("fig_C0_midspan_static.png"), x: MX, y: top, w: pw, h: pw * 485 / 820, objectName: "C0 mid-span trace", altText: "C0 mid-span displacement vs exact, 0 to 0.15 s: PINN nearly flat at about 70 mm while the exact solution oscillates" });
  T(s, "C0 at 20 000 steps: mid-span displacement, first 0.15 s (blue: exact; dashed: PINN)", { x: MX, y: top + pw * 485 / 820 + 0.05, w: pw, h: 0.3, fontSize: 11, color: C.text2, objectName: "Trace caption" });
  const n = D.ntk;
  s.addChart(pres.charts.LINE, [
    { name: "λ_ux (slope BC)", labels: n.steps.map(String), values: n.lam_ux },
    { name: "λ_ut (velocity IC)", labels: n.steps.map(String), values: n.lam_ut },
    { name: "λ_u (displacement IC/BC)", labels: n.steps.map(String), values: n.lam_u },
  ], {
    x: MX + pw + 0.4, y: top, w: pw, h: ph, objectName: "NTK weights chart", ...AX,
    showTitle: true, title: "NTK loss weights during C0 training (PDE weight λ_f = 1)",
    chartColors: [HEX.accent1, HEX.accent4, HEX.accent5], lineSize: 2, lineDataSymbol: "none",
    valAxisLogScaleBase: 10, valAxisMinVal: 1e10, valAxisMaxVal: 1e16, valAxisLabelFormatCode: "0E+0",
    showValAxisTitle: true, valAxisTitle: "weight (log scale)", showCatAxisTitle: true, catAxisTitle: "optimizer step",
    catAxisLabelFrequency: 5, valGridLine: { color: "E3E8EB", size: 0.75 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "b",
  });
  const sy = 5.55, sw = (CW - 0.6) / 4;
  const S4 = [["3.26", "relative L2 at 20 000 steps"], ["0.06", "amplitude ratio, PINN / exact"], ["~10¹⁴–10¹⁵", "final λ_ut 8.5e14, λ_ux 9.9e14"], ["0 cycles", "static / low-frequency attractor"]];
  S4.forEach(([v, l], i) => {
    const x = MX + i * (sw + 0.2);
    T(s, v, { x, y: sy, w: sw, h: 0.55, fontSize: 28, bold: true, color: i === 2 ? C.accent4 : C.text1, fontFace: THEME.headFontFace, objectName: `Stat ${i + 1} value` });
    T(s, l, { x, y: sy + 0.55, w: sw, h: 0.35, fontSize: 12, color: C.text2, objectName: `Stat ${i + 1} label` });
  });
  T(s, "Observed optimization failure in our implementation. The weight growth coincides with the static solution; it does not prove that NTK weighting causes the collapse.",
    { x: MX, y: 6.55, w: CW, h: 0.35, fontSize: 12, italic: true, color: C.accent4, objectName: "Interpretation caveat" });
  s.addNotes([
    "WHAT THIS SLIDE SHOWS: The paper-faithful Fourier+NTK implementation (run C0, 20 000 steps, 6.4e5 PDE evaluations). Left: the network stays near 70 mm while the exact beam oscillates. Right: the NTK weights for the IC/BC terms rise to 1e14–1e15 relative to the PDE term.",
    "WHY IT MATTERS: The published method did not reproduce the target dynamics in our controlled implementation. Bias/normalisation repairs (D1–D4), Fourier alone, the reference-code convention and an LR schedule on C0 (Y4) also gave static or non-oscillating fields.",
    "KEY RESULT: L2 = 3.26, amplitude ratio 0.06, fitted frequency ≈ 0, PDE residual 21.7. We describe this as an observed optimization failure (a static/low-frequency attractor). The large NTK weights coincide with it; we do not claim they cause it — no fixed-λ ablation was run.",
    "Sources: reports/PHASE_A1_CHECKPOINT.md; tables/phaseA_summary.csv; logs/C0_paper__s1234__c41ccb1cdf/history.csv (λ values); figure crop of figures/phaseA_A1_C0_paper_diagnostics.png.",
  ].join("\n\n"));
}

// ================================================================== slide 5 — conditioning + representation
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Investigation" });
  s.addText("CONDITIONING AND REPRESENTATION", { placeholder: "kicker" });
  s.addText("The bottleneck is optimization, not capacity", { placeholder: "title" });
  const top = 1.6, lw = 5.6;
  T(s, "Hard-constraint ansatz:  u = u₀(x) + g(t)·Φ(x)·A₀·N(x, t)", { x: MX, y: top, w: lw, h: 0.4, fontSize: 15, bold: true, objectName: "Ansatz" });
  const rows = [
    ["g(t) = (t/T)²", "original time factor", "≈ −8 400", "required network output N near t → 0 (range −8 369 to −1.0)", C.accent4],
    ["g(t) = tanh²(ω₁t)", "ω₁ = 129.37 rad/s from PDE coefficient + clamped-beam eigenproblem", "O(1)", "required N in −1.9 to −0.16", C.accent2],
  ];
  rows.forEach(([f, sub, big, cap, col], i) => {
    const y = top + 0.6 + i * 1.75;
    card(s, MX, y, lw, 1.55, C.background2, `Ansatz card ${i + 1}`);
    T(s, [
      { text: f, options: { bold: true, fontFace: "Cambria", fontSize: 18, breakLine: true } },
      { text: sub, options: { fontSize: 12, color: C.text2 } },
    ], { x: MX + 0.2, y: y + 0.15, w: 2.9, h: 1.2, objectName: `Ansatz ${i + 1} text` });
    T(s, big, { x: MX + 3.2, y: y + 0.12, w: 2.25, h: 0.6, fontSize: 28, bold: true, color: col, fontFace: THEME.headFontFace, align: "left", objectName: `Ansatz ${i + 1} value` });
    T(s, cap, { x: MX + 3.2, y: y + 0.72, w: 2.25, h: 0.65, fontSize: 12, color: C.text2, objectName: `Ansatz ${i + 1} caption` });
  });
  T(s, [
    { text: "Both factors enforce the IC and BCs exactly (IC error 4.8e-6, BC error 1.4e-5).", options: { breakLine: true } },
    { text: "PDE-loss gradient norm, early (t < 0.02 s) vs late (t > 0.5 s): 7.2e3 vs 1.5e8 for (t/T)² (E4); 8.4e5 vs 9.8e4 for tanh² (X2)." },
  ], { x: MX, y: top + 4.25, w: lw, h: 1.0, paraSpaceAfter: 6, fontSize: 12, color: C.text2, objectName: "Gradient note" });
  // right: representation check
  const rx = MX + lw + 0.45, rw = W - MX - rx;
  T(s, "Representation check: supervised fit of the exact solution", { x: rx, y: top, w: rw, h: 0.4, fontSize: 15, bold: true, objectName: "Representation heading" });
  const ih = rw * 340 / 1650;
  s.addImage({ path: FIG("fig_X3_supervised_trace.png"), x: rx, y: top + 0.5, w: rw, h: ih, objectName: "X3 supervised trace", altText: "X3 supervised fit vs exact mid-span displacement over the full 1 s window" });
  s.addTable([
    [{ text: "Diagnostic (5 000 steps)", options: { bold: true } }, { text: "Rel. L2", options: { bold: true, align: "right" } }, { text: "ω [rad/s]", options: { bold: true, align: "right" } }],
    ["X3 supervised (paper convention)", { text: "0.463", options: { align: "right" } }, { text: "129.25", options: { align: "right" } }],
    ["X4 supervised (reference-code conv.)", { text: "0.316", options: { align: "right" } }, { text: "129.37", options: { align: "right" } }],
    ["Exact solution", { text: "—", options: { align: "right" } }, { text: "129.32", options: { align: "right" } }],
  ], { x: rx, y: top + 0.65 + ih, w: rw, colW: [rw - 2.4, 1.1, 1.3], fontSize: 13, color: C.text1, fill: { color: C.background1 },
    border: { type: "solid", pt: 0.75, color: C.background2 }, rowH: 0.34, margin: [0.03, 0.08, 0.03, 0.08], objectName: "Supervised results table" });
  card(s, rx, 5.3, rw, 1.55, C.text1, "Conclusion card");
  T(s, "“The network has sufficient representational capacity; the principal difficulty is physics-constrained optimization/conditioning rather than inability to represent the target frequency.”",
    { x: rx + 0.25, y: 5.42, w: rw - 0.5, h: 1.3, fontSize: 15, italic: true, color: C.background1, valign: "middle", objectName: "Conclusion quote" });
  s.addNotes([
    "WHAT THIS SLIDE SHOWS: Left: the hard-constraint ansatz satisfies the initial and boundary conditions exactly (IC error 4.8e-6, BC error 1.4e-5). With g(t) = (t/T)² the network must output about −8 400 near t → 0 to produce the initial velocity behaviour; with g(t) = tanh²(ω₁t) it needs only O(1) values. ω₁ is computed from the PDE coefficient and the clamped-beam eigenproblem — a declared problem-specific prior, not fitted to the solution. Right: the same network architecture, trained on data from the exact solution instead of the PDE, reproduces the 20.6 Hz oscillation.",
    "WHY IT MATTERS: It separates two explanations for the failure: inability to represent the solution vs difficulty of optimizing the physics loss. The supervised fits rule out the first.",
    "KEY RESULT: Supervised X3/X4 recover ω = 129.25 / 129.37 rad/s (exact 129.32). The PDE-trained tanh² run X2 reached L2 0.909 and oscillated, but at 112.2 rad/s and over-damped. Conclusion: the network has sufficient representational capacity; the principal difficulty is physics-constrained optimization/conditioning.",
    "Sources: reports/PHASE_E_SCREEN.md, PHASE_X_DIAGNOSTIC.md; profiles/phaseE_ansatz_conditioning.txt; tables/phaseX_conditioning.csv, phaseX_oscillation.csv; figure crop of figures/phaseX_midspan_traces.png.",
  ].join("\n\n"));
}

// ================================================================== slide 6 — E -> X -> Y -> Z
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Investigation" });
  s.addText("OPTIMIZATION RESULTS: PHASES E → X → Y → Z", { placeholder: "kicker" });
  s.addText("Later collapse, but never the full window", { placeholder: "title" });
  const top = 1.55, tw = 7.6;
  const hdr = (t, a) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text1 }, align: a || "left" } });
  const r = (cells, hl) => cells.map((c, j) => ({ text: c, options: { align: j >= 2 ? "right" : "left", bold: !!hl, fill: { color: hl ? "FCEADF" : HEX.lt1 } } }));
  s.addTable([
    [hdr("Phase · run"), hdr("Single change vs parent"), hdr("Rel. L2", "right"), hdr("Persistence", "right")],
    r(["E · E4", "hard constraints, g = (t/T)²", "2.83", "no oscillation"]),
    r(["X · X2", "g = tanh²(ω₁t), ref.-code convention", "0.909", "ω 112.2, over-damped"]),
    r(["Y · Y1", "+ LR schedule 1e-3·0.9^(k/1000)", "0.774", "1.2 cycles"]),
    r(["Z · Z1", "Y1 + RAD sampling", "0.718", "1.6 cycles"]),
    r(["Z · Z4", "Y1 with mini-batch 128", "0.526", "3.2 cycles"]),
    r(["Z4-20K", "Z4 with 20 000 steps", "0.263", "8.1 cycles"], true),
  ], { x: MX, y: top, w: tw, colW: [1.2, 3.35, 0.95, 2.1], fontSize: 13, color: C.text1, rowH: 0.36,
    border: { type: "solid", pt: 0.75, color: C.background2 }, margin: [0.03, 0.08, 0.03, 0.08], objectName: "Phase progression table" });
  T(s, "5 000 steps unless noted. Persistence: cycles until local amplitude < 50 % of exact (full window 20.6).",
    { x: MX, y: top + 2.6, w: tw, h: 0.3, fontSize: 11, color: C.text2, objectName: "Table caption" });
  // right callouts
  const rx = MX + tw + 0.45, rw = W - MX - rx;
  card(s, rx, top, rw, 2.95, C.background2, "Z4-20K card");
  T(s, "Z4-20K at 20 000 steps", { x: rx + 0.25, y: top + 0.15, w: rw - 0.5, h: 0.35, fontSize: 14, bold: true, objectName: "Card heading" });
  const cs = [["129.2 rad/s", "frequency (exact 129.32)", C.accent2], ["5.40 s⁻¹", "fitted decay (exact 3.54): over-damped", C.accent4], ["8.1 / 20.6", "cycles before collapse", C.accent1]];
  cs.forEach(([v, l, col], i) => {
    const y = top + 0.55 + i * 0.78;
    T(s, v, { x: rx + 0.25, y, w: 1.95, h: 0.5, fontSize: 22, bold: true, color: col, fontFace: THEME.headFontFace, objectName: `Callout ${i + 1} value` });
    T(s, l, { x: rx + 2.2, y: y + 0.05, w: rw - 2.4, h: 0.6, fontSize: 12, color: C.text2, objectName: `Callout ${i + 1} label` });
  });
  // representative trace
  const iy = 4.72, iw = CW, ih = iw * 365 / 1800;
  s.addImage({ path: FIG("fig_Z4_20K_step20k_trace.png"), x: MX, y: iy - 0.05, w: iw * 0.86, h: ih * 0.86, objectName: "Z4-20K trace", altText: "Z4-20K mid-span displacement vs exact at 20 000 steps; collapse marked at 0.393 s" });
  T(s, "Mid-span displacement, exact (blue) vs Z4-20K (purple); dashed line: collapse at 0.393 s. Full-window reproduction not achieved.",
    { x: MX + iw * 0.86 + 0.15, y: iy + 0.3, w: iw * 0.14 - 0.15, h: 1.6, fontSize: 11, color: C.text2, objectName: "Trace caption" });
  s.addNotes([
    "WHAT THIS SLIDE SHOWS: The sequence of single-factor changes after the conditioning fix, with the relative L2 error and the measured persistence (cycles before the local amplitude drops below half the exact amplitude), and one representative trace of the most persistent candidate.",
    "WHY IT MATTERS: Frequency is now correct and the first cycles are reproduced, but the solution decays too fast and then collapses toward a low-amplitude trajectory. That is the remaining failure mode.",
    "KEY RESULT: Y1 L2 0.774 (1.2 cycles) → Z4 0.526 (3.2 cycles) → Z4-20K 0.263 (8.1 cycles of 20.6). Frequency 129.2 rad/s (exact 129.32); decay 5.40 1/s vs 3.54 — still over-damped. RAD (Z1) gave 0.718 / 1.6 cycles and did not resolve the collapse; a second seed (Z2) showed the same failure mode. Z4-20K's L2 of 0.263 is roughly 570× the published 4.64e-4; full-window reproduction was not achieved.",
    "Sources: reports/PHASE_E_SCREEN.md, PHASE_X_DIAGNOSTIC.md, PHASE_Y_OPTIMIZER_DIAGNOSTIC.md, PHASE_Z_DIAGNOSTIC.md, PHASE_Y1_20K_BUDGET_DIAGNOSTIC.md, PHASE_Z4_20K_BUDGET_DIAGNOSTIC.md; tables/phaseY1_20K_checkpoints.csv, phaseZ4_20K_checkpoints.csv; figure crop of figures/phaseZ4_20K_disp_traces.png.",
  ].join("\n\n"));
}

// ================================================================== slide 7 — compute budget
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Investigation" });
  s.addText("COMPUTE BUDGET AND DYNAMIC PERSISTENCE", { placeholder: "kicker" });
  s.addText("More physics computation, longer persistence", { placeholder: "title" });
  const Y = D.collapse.Y1, Z = D.collapse.Z4;
  const xs = [...Y.map((p) => p.pde / 1e6), ...Z.map((p) => p.pde / 1e6)];
  const nul = (k) => Array(k).fill(null);
  const top = 1.55, cw = 7.2, ch = 5.15;
  s.addChart(pres.charts.SCATTER, [
    { name: "x", values: xs },
    { name: "Y1 recipe, mini-batch 32 (5k–20k steps)", values: [...Y.map((p) => p.collapse), ...nul(4)] },
    { name: "Z4 recipe, mini-batch 128 (5k–20k steps)", values: [...nul(4), ...Z.map((p) => p.collapse)] },
  ], {
    x: MX, y: top, w: cw, h: ch, objectName: "Collapse vs PDE evaluations chart", ...AX,
    showTitle: true, title: "Collapse time vs cumulative PDE-residual evaluations (checkpoints every 5 000 steps)",
    chartColors: [HEX.accent2, HEX.accent1], lineSize: 2, lineDataSymbol: "circle", lineDataSymbolSize: 9,
    valAxisMinVal: 0, valAxisMaxVal: 1.0, valAxisMajorUnit: 0.2, valAxisLabelFormatCode: "0.0",
    catAxisMinVal: 0, catAxisMaxVal: 3.0, catAxisMajorUnit: 0.5, catAxisLabelFormatCode: "0.0",
    showValAxisTitle: true, valAxisTitle: "collapse time [s]  (full window = 1.0 s = 20.6 cycles)",
    showCatAxisTitle: true, catAxisTitle: "cumulative PDE evaluations [millions]",
    valGridLine: { color: "E3E8EB", size: 0.75 }, catGridLine: { style: "none" }, showLegend: true, legendPos: "b",
  });
  const rx = MX + cw + 0.45, rw = W - MX - rx;
  s.addTable([
    [{ text: "Checkpoint", options: { bold: true } }, { text: "PDE evals", options: { bold: true, align: "right" } }, { text: "Collapse", options: { bold: true, align: "right" } }, { text: "Cycles", options: { bold: true, align: "right" } }],
    ...[["Y1 5k", Y[0]], ["Y1 20k", Y[3]], ["Z4 5k", Z[0]], ["Z4 20k", Z[3]]].map(([n, p], i) => [
      { text: n, options: { bold: i === 3 } },
      { text: p.pde >= 1e6 ? `${(p.pde / 1e6).toFixed(2)}M` : `${Math.round(p.pde / 1e3)}k`, options: { align: "right", bold: i === 3 } },
      { text: `${p.collapse.toFixed(3)} s`, options: { align: "right", bold: i === 3 } },
      { text: p.cycles.toFixed(1), options: { align: "right", bold: i === 3 } },
    ]),
  ], { x: rx, y: top, w: rw, colW: [1.25, 1.2, 1.15, rw - 3.6], fontSize: 13, color: C.text1, rowH: 0.34,
    border: { type: "solid", pt: 0.75, color: C.background2 }, margin: [0.03, 0.08, 0.03, 0.08], objectName: "Checkpoint table" });
  card(s, rx, top + 1.95, rw, 1.55, C.text1, "Key conclusion card");
  T(s, "Within the tested range, increased physics computation produced progressively longer dynamic persistence and lower error, but did not yield full-window reproduction.",
    { x: rx + 0.2, y: top + 2.05, w: rw - 0.4, h: 1.35, fontSize: 14, bold: true, color: C.background1, valign: "middle", objectName: "Key conclusion" });
  T(s, [
    { text: "Batch size cannot be attributed independently. ", options: { bold: true } },
    { text: "At equal 6.4e5 evaluations, mini-batch 128 (Z4 5k) and 32 (Y1 20k) reach 3.2 vs 3.1 cycles and L2 0.526 vs 0.550. The measured difference is wall-clock (611 s vs 1 078 s); there is no matched small-batch run at 2.56M. The last 6.4e5 block advanced the collapse by +0.047 s vs +0.096 s before; no extrapolation is claimed." },
  ], { x: rx, y: top + 3.65, w: rw, h: 1.55, fontSize: 12, color: C.text2, objectName: "Attribution note" });
  s.addNotes([
    "WHAT THIS SLIDE SHOWS: The collapse time (first time the local amplitude ratio falls below 0.5) at every 5 000-step checkpoint of the two 20 000-step runs, plotted against cumulative PDE-residual evaluations (steps × mini-batch).",
    "WHY IT MATTERS: It tests whether the collapse is a fixed attractor or budget-limited. Within the tested range the collapse front moves monotonically later with physics computation.",
    "KEY RESULT: Y1-20K: 6.4e5 evaluations, 0.151 s, 3.1 cycles, L2 0.550. Z4-20K: 2.56e6 evaluations, 0.393 s, 8.1 cycles, L2 0.263. Within the tested range, increased physics computation produced progressively longer dynamic persistence and lower error, but did not yield full-window reproduction. Batch size cannot be isolated: at equal evaluations the two recipes reach the same state; the larger batch only reached it in 0.57× the wall-clock time on this CPU. The last block's advance was about half of the previous ones, so we do not extrapolate.",
    "Sources: tables/phaseY1_20K_checkpoints.csv, phaseZ4_20K_checkpoints.csv; reports/PHASE_Y1_20K_BUDGET_DIAGNOSTIC.md, PHASE_Z4_20K_BUDGET_DIAGNOSTIC.md (also figures/budget_collapse_vs_pde_evals.png).",
  ].join("\n\n"));
}

// ================================================================== slide 8 — conclusion
pres.addSection({ title: "Conclusion" });
{
  const s = pres.addSlide({ masterName: "CLOSING_DARK", sectionTitle: "Conclusion" });
  s.addText("CONCLUSION", { placeholder: "kicker" });
  s.addText("What is established — and what is not", { placeholder: "title" });
  const top = 1.45, gap = 0.3, cw = (CW - 2 * gap) / 3, ch = 4.1;
  const cols = [
    ["ESTABLISHED", HEX.accent2, [
      "The published Fourier + NTK baseline did not reproduce the target dynamics in our controlled implementation",
      "The original hard-constraint time factor (t/T)² was poorly conditioned",
      "The network can represent the 20.6 Hz solution",
      "LR, batch size and physics-computation budget materially affect optimization",
      "RAD did not resolve the observed failure",
    ]],
    ["NOT ESTABLISHED", HEX.accent1, [
      "No validated optimized PINN",
      "No claim of superiority over the paper",
      "No full-window Mode-1 reproduction",
      "No Mode-2 generalization result",
    ]],
    ["NEXT", HEX.accent6, [
      "Independent confirmation once a genuinely successful Mode-1 candidate is found",
      "Higher-frequency / Mode-2 validation",
      "Extension to railway vibration / SHM physics — future work only",
    ]],
  ];
  cols.forEach(([h, col, items], i) => {
    const x = MX + i * (cw + gap);
    card(s, x, top, cw, ch, "26394A", `${h} card`);
    s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: top + 0.25, w: 0.22, h: 0.22, fill: { color: col }, line: { type: "none" }, objectName: `${h} marker` });
    T(s, h, { x: x + 0.6, y: top + 0.17, w: cw - 0.8, h: 0.4, fontSize: 16, bold: true, color: col, charSpacing: 1, objectName: `${h} heading` });
    T(s, items.map((t, k) => ({ text: t, options: { bullet: i === 2 ? { type: "number" } : true, breakLine: k < items.length - 1 } })),
      { x: x + 0.25, y: top + 0.7, w: cw - 0.45, h: ch - 0.85, fontSize: 14, color: C.background1, paraSpaceAfter: 4, objectName: `${h} items` });
  });
  // future-work flow
  const fy = 5.7;
  T(s, "FUTURE WORK (not part of this study)", { x: MX, y: fy, w: CW, h: 0.3, fontSize: 11, bold: true, color: C.accent6, charSpacing: 1, objectName: "Future work label" });
  const nodes = ["Beam PINN", "Railway structural / component dynamics", "Sensor-informed PINN", "Axle / bearing vibration or AE", "Physics-constrained fault estimation"];
  const ng = 0.35, nw = (CW - 4 * ng) / 5;
  nodes.forEach((n, i) => {
    const x = MX + i * (nw + ng);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: fy + 0.36, w: nw, h: 0.7, rectRadius: 0.06, fill: { color: C.text1 }, line: { color: C.accent6, width: 1, dashType: "dash" }, objectName: `Future node ${i + 1}` });
    T(s, n, { x: x + 0.1, y: fy + 0.36, w: nw - 0.2, h: 0.7, fontSize: 12, color: C.background1, align: "center", valign: "middle", objectName: `Future node ${i + 1} text` });
    if (i < 4) s.addShape(pres.shapes.LINE, { x: x + nw + 0.05, y: fy + 0.71, w: ng - 0.1, h: 0, line: { color: C.accent6, width: 1.25, endArrowType: "triangle" }, objectName: `Future arrow ${i + 1}` });
  });
  s.addNotes([
    "WHAT THIS SLIDE SHOWS: The final scientific position of the study, separated into what is established, what is not, and what comes next.",
    "CENTRAL CONCLUSION: We performed a controlled reproducibility and optimization investigation of a Fourier/NTK PINN for damped beam vibration, identified conditioning and optimization bottlenecks, verified that the network can represent the target dynamics, and quantified how training strategy and physics-computation budget affect late-time dynamic persistence. A fully validated optimized PINN was not established within the tested budget.",
    "WHY IT MATTERS: The negative and diagnostic results are informative: they locate the difficulty in physics-constrained optimization rather than representation, and they show the published result is not reproduced by a faithful implementation at the tested budgets.",
    "KEY RESULT: Best measured candidate (Z4-20K) reaches 8.1 of 20.6 cycles, L2 0.263 — not a validated method. Mode 2 was not run, because no Mode-1 candidate passed the gate. The railway / SHM pathway on the bottom row is future work only; no railway result exists or is claimed.",
  ].join("\n\n"));
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("written", OUT);
})();
