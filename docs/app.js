import { fit, predict, analytic, evaluate, NAMES, POWERS } from "./core.js";
const $ = (s) => document.querySelector(s),
  el = (tag, text, cls) => {
    const n = document.createElement(tag);
    if (text !== undefined) n.textContent = text;
    if (cls) n.className = cls;
    return n;
  };
const model = fit(),
  audit = evaluate(model),
  units = [
    "per +1 spot",
    "per spot²",
    "per +1 vol point",
    "per calendar day",
    "per +1 rate point",
  ];
let applied,
  slice = [];
const format = (v) => (Math.abs(v) < 1e-12 ? "0.000000" : v.toFixed(6)),
  signed = (v) => (v >= 0 ? "+" : "") + format(v);
function draw() {
  if (!applied) return;
  const j = Number($("#greek").value),
    canvas = $("#curve"),
    dpr = Math.min(devicePixelRatio || 1, 2),
    w = canvas.clientWidth,
    h = canvas.clientHeight;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const c = canvas.getContext("2d");
  c.scale(dpr, dpr);
  c.font = "9px monospace";
  const values = slice.flatMap((r) => [r.actual[j], r.predicted[j]]),
    min = Math.min(...values),
    max = Math.max(...values),
    pad = (max - min) * 0.12 || 0.01,
    lo = min - pad,
    hi = max + pad,
    left = 48,
    right = w - 15,
    top = 20,
    bottom = h - 32,
    x = (s) => left + ((s - 85) / 30) * (right - left),
    y = (v) => bottom - ((v - lo) / (hi - lo)) * (bottom - top);
  c.strokeStyle = "#e0e6e1";
  c.fillStyle = "#75848a";
  c.textAlign = "right";
  for (let i = 0; i < 4; i++) {
    const v = lo + ((hi - lo) * i) / 3;
    c.beginPath();
    c.moveTo(left, y(v));
    c.lineTo(right, y(v));
    c.stroke();
    c.fillText(v.toFixed(j === 1 || j === 3 ? 3 : 2), left - 7, y(v) + 3);
  }
  c.textAlign = "center";
  for (const s of [85, 100, 115]) c.fillText(String(s), x(s), h - 14);
  c.textAlign = "right";
  c.fillText("spot S", right, h - 3);
  for (const field of ["actual", "predicted"]) {
    c.strokeStyle = field === "actual" ? "#23333c" : "#a47d3e";
    c.lineWidth = 2;
    c.setLineDash(field === "actual" ? [] : [6, 5]);
    c.beginPath();
    slice.forEach((r, i) =>
      i
        ? c.lineTo(x(r.spot), y(r[field][j]))
        : c.moveTo(x(r.spot), y(r[field][j])),
    );
    c.stroke();
  }
  c.setLineDash([]);
  const selectedX = x(applied.point[0]);
  c.strokeStyle = "#a9b5b6";
  c.lineWidth = 1;
  c.beginPath();
  c.moveTo(selectedX, top);
  c.lineTo(selectedX, bottom);
  c.stroke();
  $("#chart-caption").textContent =
    `${NAMES[j]} / ${units[j]}. Applied ${applied.kind}, T=${applied.point[1]} years, σ=${(100 * applied.point[2]).toFixed(2)}%. Vertical marker: selected S. The 61-point slice includes training or holdout coordinates by coincidence; it is not an independent evaluation.`;
  $("#slice").replaceChildren(
    ...slice.map((r) => {
      const tr = el("tr");
      tr.append(
        el("td", r.spot.toFixed(1)),
        el("td", format(r.actual[j])),
        el("td", format(r.predicted[j])),
        el("td", signed(r.predicted[j] - r.actual[j]), "error"),
      );
      return tr;
    }),
  );
}
function render() {
  const actual = analytic(applied.point, applied.kind),
    approx = predict(model, applied.point, applied.kind);
  $("#price").textContent = actual.price.toFixed(6);
  $("#status").textContent = `${applied.kind.toUpperCase()} / APPLIED POINT`;
  $("#model-meta").textContent =
    `${model.training_rows} train rows / ${model.features} features`;
  $("#greeks").replaceChildren(
    ...NAMES.map((name, j) => {
      const tr = el("tr"),
        title = el("td", name[0].toUpperCase() + name.slice(1));
      title.append(el("small", units[j]));
      tr.append(
        title,
        el("td", format(actual.greeks[j])),
        el("td", format(approx[j])),
        el("td", signed(approx[j] - actual.greeks[j]), "error"),
      );
      return tr;
    }),
  );
  slice = Array.from({ length: 61 }, (_, i) => {
    const spot = 85 + i * 0.5,
      p = [spot, applied.point[1], applied.point[2]];
    return {
      spot,
      actual: analytic(p, applied.kind).greeks,
      predicted: predict(model, p, applied.kind),
    };
  });
  $("#holdout").replaceChildren(
    ...audit.rows
      .filter((r) => r.kind === applied.kind)
      .map((r) => {
        const tr = el("tr");
        tr.append(
          el(
            "td",
            `${r.point[0].toFixed(1)} / ${r.point[1].toFixed(4)} / ${(100 * r.point[2]).toFixed(1)}%`,
          ),
          ...r.errors.map((v) => el("td", signed(v), "error")),
        );
        return tr;
      }),
  );
  draw();
}
function dirty() {
  if (!applied) return;
  $(".desk").classList.add("stale");
  $("#greek").disabled = true;
  $("#notice").textContent =
    "Inputs changed. Previous point is stale; Apply point to update. The audit retains the last applied option type.";
}
function apply() {
  try {
    const values = ["#spot", "#years", "#vol"].map((s) => {
      const raw = $(s).value.trim();
      if (!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(raw))
        throw Error("Enter finite decimal numbers");
      return Number(raw);
    });
    const point = [values[0], values[1], values[2] / 100],
      kind = $("#kind").value;
    analytic(point, kind);
    applied = { point, kind };
    $(".desk").classList.remove("stale");
    $("#greek").disabled = false;
    $("#notice").textContent =
      "Applied. Frozen model unchanged; no clipping or refitting.";
    render();
  } catch (e) {
    $(".desk").classList.add("stale");
    $("#greek").disabled = true;
    $("#notice").textContent =
      `Not applied: ${e.message}. Previous result is stale.`;
  }
}
$("#audit-metrics").replaceChildren(
  ...audit.metrics.map((r) => {
    const tr = el("tr");
    tr.append(
      el("td", `${r.kind} / ${r.greek}`),
      el("td", format(r.mae)),
      el("td", format(r.rmse)),
      el("td", format(r.max_abs)),
      el(
        "td",
        `${r.worst_point[0].toFixed(1)}, ${r.worst_point[1].toFixed(4)}, ${(100 * r.worst_point[2]).toFixed(1)}%`,
      ),
    );
    return tr;
  }),
);
$("#scalers").replaceChildren(
  ...["spot", "years", "volatility"].map((name, j) =>
    el(
      "p",
      `${name}: train mean ${model.means[j].toPrecision(9)}, scale ${model.scales[j].toPrecision(9)} (population SD)`,
    ),
  ),
  el(
    "p",
    `Output means: ${model.target_means.map((v) => v.toPrecision(8)).join(", ")} (call five, then put five)`,
  ),
  el(
    "p",
    `Output scales: ${model.target_scales.map((v) => v.toPrecision(8)).join(", ")} (same order)`,
  ),
);
$("#residual").textContent =
  `Max |(XᵀX + λD)w - Xᵀy| = ${model.normal_residual.toExponential(3)} in standardized target normal equations. This checks the solve, not prediction accuracy.`;
$("#weights").replaceChildren(
  ...POWERS.map((p, i) => {
    const tr = el("tr");
    tr.append(
      el("td", p.join(",")),
      ...model.weights.map((w) => el("td", w[i].toPrecision(9))),
    );
    return tr;
  }),
);
$("#compare").addEventListener("click", apply);
for (const s of ["#spot", "#years", "#vol"])
  $(s).addEventListener("input", dirty);
$("#kind").addEventListener("change", dirty);
$("#greek").addEventListener("change", draw);
new ResizeObserver(draw).observe($("#curve"));
apply();
