import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import {
  fit,
  predict,
  analytic,
  evaluate,
  grid,
  POWERS,
  solve,
  cdf,
} from "../docs/core.js";
let checks = 0;
function near(a, b, path = "root") {
  if (typeof a === "number") {
    assert.ok(
      Math.abs(a - b) < 1e-9 * Math.max(1, Math.abs(a), Math.abs(b)),
      `${path}:${a} vs ${b}`,
    );
    checks++;
  } else if (Array.isArray(a)) {
    assert.equal(a.length, b.length, path);
    a.forEach((v, i) => near(v, b[i], path + "." + i));
  } else if (a && typeof a === "object") {
    assert.deepEqual(Object.keys(a).sort(), Object.keys(b).sort());
    for (const k in a) near(a[k], b[k], path + "." + k);
  } else {
    assert.equal(a, b, path);
    checks++;
  }
}
const points = [
    ...grid().filter((_, i) => i % 9 === 0),
    ...grid(true).filter((_, i) => i % 7 === 0),
    [100, 1, 0.3],
  ],
  m = fit(),
  p = spawnSync("python3", ["-m", "tests.parity"], {
    input: JSON.stringify(points),
    encoding: "utf8",
    maxBuffer: 5e6,
  });
assert.equal(p.status, 0, p.stderr);
near(
  {
    model: m,
    points: points.flatMap((point) =>
      ["call", "put"].map((kind) => ({
        point,
        kind,
        analytic: analytic(point, kind),
        predicted: predict(m, point, kind),
      })),
    ),
    holdout: evaluate(m),
  },
  JSON.parse(p.stdout),
);
near(
  solve(
    [
      [4, 1],
      [1, 3],
    ],
    [[6, 7]],
  )[0],
  [1, 2],
);
near(cdf(0), 0.5);
near(cdf(1), 0.8413447460685429);
assert.equal(POWERS.length, 35);
assert.ok(m.normal_residual < 1e-9);
for (const x of [
  null,
  [100, 1],
  [true, 1, 0.3],
  [100, NaN, 0.3],
  [100, 1, 0.45001],
  [84.99, 1, 0.3],
  [100, 0.2499, 0.3],
  ["100", 1, 0.3],
  Array(3),
]) {
  assert.throws(() => analytic(x));
  assert.throws(() => predict(m, x));
}
assert.throws(() => analytic([100, 1, 0.3], "bad"));
assert.throws(() => solve([[-1]], [[1]]));
console.log(
  JSON.stringify({
    scalar_checks: checks,
    native_parity_points: points.length * 2,
    status: "PASS",
  }),
);
