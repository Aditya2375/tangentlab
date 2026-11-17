// Original independently implemented polynomial-ridge Greeks surrogate.
export const BOUNDS = [
    [85, 115],
    [0.25, 2],
    [0.15, 0.45],
  ],
  NAMES = ["delta", "gamma", "vega", "theta", "rho"];
export const POWERS = [];
for (let d = 0; d < 5; d++)
  for (let a = 0; a <= d; a++)
    for (let b = 0; b <= d - a; b++) POWERS.push([a, b, d - a - b]);
function check(p) {
  if (
    !Array.isArray(p) ||
    p.length !== 3 ||
    Array.from(p).some(
      (v, j) =>
        typeof v !== "number" ||
        !Number.isFinite(v) ||
        v < BOUNDS[j][0] ||
        v > BOUNDS[j][1],
    )
  )
    throw Error(
      "Need [spot, years, volatility] within [85,115], [0.25,2], [0.15,0.45]",
    );
}
export function cdf(x) {
  let term = x,
    total = x;
  for (let n = 1; n < 100; n++) {
    term *= (-x * x) / (2 * n);
    const addition = term / (2 * n + 1);
    total += addition;
    if (Math.abs(addition) < 1e-17) break;
  }
  return 0.5 + total / Math.sqrt(2 * Math.PI);
}
export function analytic(point, kind = "call") {
  check(point);
  if (!["call", "put"].includes(kind))
    throw Error("Option kind must be call or put");
  const [s, t, v] = point,
    k = 100,
    r = 0.03,
    q = 0.01,
    root = Math.sqrt(t),
    dq = Math.exp(-q * t),
    dr = Math.exp(-r * t),
    d1 = (Math.log(s / k) + (r - q + 0.5 * v * v) * t) / (v * root),
    d2 = d1 - v * root,
    pdf = Math.exp(-0.5 * d1 * d1) / Math.sqrt(2 * Math.PI);
  let price, delta, theta, rho;
  if (kind === "call") {
    price = s * dq * cdf(d1) - k * dr * cdf(d2);
    delta = dq * cdf(d1);
    theta =
      (-s * dq * pdf * v) / (2 * root) -
      r * k * dr * cdf(d2) +
      q * s * dq * cdf(d1);
    rho = k * t * dr * cdf(d2);
  } else {
    price = k * dr * cdf(-d2) - s * dq * cdf(-d1);
    delta = -dq * cdf(-d1);
    theta =
      (-s * dq * pdf * v) / (2 * root) +
      r * k * dr * cdf(-d2) -
      q * s * dq * cdf(-d1);
    rho = -k * t * dr * cdf(-d2);
  }
  return {
    price,
    greeks: [
      delta,
      (dq * pdf) / (s * v * root),
      (s * dq * pdf * root) / 100,
      theta / 365,
      rho / 100,
    ],
  };
}
export function grid(holdout = false) {
  const n = holdout ? 6 : 7,
    out = [];
  for (let a = 0; a < n; a++)
    for (let b = 0; b < n; b++)
      for (let c = 0; c < n; c++)
        out.push(
          [a, b, c].map((i, j) =>
            Math.min(
              BOUNDS[j][1],
              BOUNDS[j][0] +
                ((BOUNDS[j][1] - BOUNDS[j][0]) * (holdout ? i + 0.5 : i)) / 6,
            ),
          ),
        );
  return out;
}
export function basis(p, means, scales) {
  const z = p.map((v, j) => (v - means[j]) / scales[j]);
  return POWERS.map((power) => power.reduce((m, v, j) => m * z[j] ** v, 1));
}
const sum = (a) => a.reduce((a, b) => a + b, 0);
export function solve(matrix, rhs) {
  const n = matrix.length,
    l = Array.from({ length: n }, () => Array(n).fill(0));
  for (let i = 0; i < n; i++)
    for (let j = 0; j <= i; j++) {
      const rem =
        matrix[i][j] - sum(l[i].slice(0, j).map((v, k) => v * l[j][k]));
      if (i === j) {
        if (rem <= 0 || !Number.isFinite(rem))
          throw Error("Non-positive Cholesky pivot");
        l[i][j] = Math.sqrt(rem);
      } else l[i][j] = rem / l[j][j];
    }
  return rhs.map((target) => {
    const y = [],
      x = Array(n).fill(0);
    for (let i = 0; i < n; i++)
      y.push(
        (target[i] - sum(l[i].slice(0, i).map((v, j) => v * y[j]))) / l[i][i],
      );
    for (let i = n - 1; i >= 0; i--)
      x[i] =
        (y[i] - sum(x.slice(i + 1).map((v, j) => l[i + 1 + j][i] * v))) /
        l[i][i];
    return x;
  });
}
export function fit() {
  const points = grid(),
    n = points.length,
    means = [0, 1, 2].map((j) => sum(points.map((p) => p[j])) / n),
    scales = [0, 1, 2].map((j) =>
      Math.sqrt(sum(points.map((p) => (p[j] - means[j]) ** 2)) / n),
    ),
    x = points.map((p) => basis(p, means, scales)),
    targets = points.map((p) => [
      ...analytic(p, "call").greeks,
      ...analytic(p, "put").greeks,
    ]),
    ym = Array.from(
      { length: 10 },
      (_, j) => sum(targets.map((row) => row[j])) / n,
    ),
    ys = ym.map((m, j) =>
      Math.sqrt(sum(targets.map((row) => (row[j] - m) ** 2)) / n),
    ),
    yz = targets.map((row) => row.map((v, j) => (v - ym[j]) / ys[j])),
    size = POWERS.length,
    ridge = 0.01,
    gram = Array.from({ length: size }, (_, i) =>
      Array.from(
        { length: size },
        (_, j) =>
          sum(x.map((row) => row[i] * row[j])) +
          (i === j && i !== 0 ? ridge : 0),
      ),
    ),
    rhs = ym.map((_, j) =>
      Array.from({ length: size }, (_, i) =>
        sum(x.map((row, k) => row[i] * yz[k][j])),
      ),
    ),
    weights = solve(gram, rhs);
  let residual = 0;
  weights.forEach((w, k) =>
    w.forEach(
      (_, i) =>
        (residual = Math.max(
          residual,
          Math.abs(sum(gram[i].map((v, j) => v * w[j])) - rhs[k][i]),
        )),
    ),
  );
  return {
    means,
    scales,
    target_means: ym,
    target_scales: ys,
    weights,
    ridge,
    training_rows: n,
    features: size,
    normal_residual: residual,
  };
}
export function predict(model, p, kind = "call") {
  check(p);
  if (!["call", "put"].includes(kind))
    throw Error("Option kind must be call or put");
  const x = basis(p, model.means, model.scales),
    offset = kind === "call" ? 0 : 5;
  return Array.from({ length: 5 }, (_, k) => {
    const j = k + offset;
    return (
      model.target_means[j] +
      model.target_scales[j] * sum(x.map((v, i) => v * model.weights[j][i]))
    );
  });
}
export function evaluate(model, points = grid(true)) {
  const rows = [];
  for (const p of points)
    for (const kind of ["call", "put"]) {
      const actual = analytic(p, kind).greeks,
        predicted = predict(model, p, kind);
      rows.push({
        point: [...p],
        kind,
        actual,
        predicted,
        errors: actual.map((v, j) => predicted[j] - v),
      });
    }
  const metrics = [];
  for (const kind of ["call", "put"]) {
    const group = rows.filter((r) => r.kind === kind);
    NAMES.forEach((greek, j) => {
      const errors = group.map((r) => r.errors[j]);
      let worst = 0;
      errors.forEach((v, i) => {
        if (Math.abs(v) > Math.abs(errors[worst])) worst = i;
      });
      metrics.push({
        kind,
        greek,
        mae: sum(errors.map(Math.abs)) / errors.length,
        rmse: Math.sqrt(sum(errors.map((e) => e * e)) / errors.length),
        max_abs: Math.abs(errors[worst]),
        worst_point: [...group[worst].point],
      });
    });
  }
  return { points: points.length, rows, metrics };
}
