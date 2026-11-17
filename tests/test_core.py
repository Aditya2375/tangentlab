import copy
import math
import random
import subprocess
import sys
import unittest
from tangentlab import BOUNDS, POWERS, analytic, evaluate, fit, grid, predict
from tangentlab.core import basis, cdf, solve


def price(p, kind="call", r=0.03, q=0.01):
    s, t, v = p
    k = 100
    d1 = (math.log(s / k) + (r - q + 0.5 * v * v) * t) / (v * math.sqrt(t))
    d2 = d1 - v * math.sqrt(t)
    n = lambda x: (1 + math.erf(x / math.sqrt(2))) / 2
    return (
        s * math.exp(-q * t) * n(d1) - k * math.exp(-r * t) * n(d2)
        if kind == "call"
        else k * math.exp(-r * t) * n(-d2) - s * math.exp(-q * t) * n(-d1)
    )


class Core(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = fit()

    def test_analytic_fixture(self):
        a = analytic([100, 1, 0.3])
        self.assertAlmostEqual(a["price"], 12.69400452623556, places=11)
        for x, y in zip(
            a["greeks"],
            [
                0.5799374680175788,
                0.012860326945011229,
                0.38580980835033685,
                -0.017989594316446967,
                0.45299742275522314,
            ],
        ):
            self.assertAlmostEqual(x, y, places=12)

    def test_independent_erf_and_finite_differences(self):
        rng = random.Random(981)
        for _ in range(100):
            p = [rng.uniform(lo, hi) for lo, hi in BOUNDS]
            for kind in ["call", "put"]:
                a = analytic(p, kind)
                self.assertAlmostEqual(a["price"], price(p, kind), places=10)
                s, t, v = p
                h = 0.005
                ds = (price([s + h, t, v], kind) - price([s - h, t, v], kind)) / (2 * h)
                gamma = (
                    price([s + h, t, v], kind)
                    - 2 * price(p, kind)
                    + price([s - h, t, v], kind)
                ) / h**2
                hv = 1e-5
                vega = (
                    (price([s, t, v + hv], kind) - price([s, t, v - hv], kind))
                    / (2 * hv)
                    / 100
                )
                ht = 1e-5
                theta = (
                    -(price([s, t + ht, v], kind) - price([s, t - ht, v], kind))
                    / (2 * ht)
                    / 365
                )
                hr = 1e-5
                rho = (
                    (price(p, kind, r=0.03 + hr) - price(p, kind, r=0.03 - hr))
                    / (2 * hr)
                    / 100
                )
                for x, y in zip(a["greeks"], [ds, gamma, vega, theta, rho]):
                    self.assertAlmostEqual(x, y, delta=5e-7)
        for i in range(-300, 301):
            self.assertAlmostEqual(
                cdf(i / 100), 0.5 * (1 + math.erf(i / 100 / math.sqrt(2))), delta=3e-14
            )

    def test_put_call_identities(self):
        for p in grid(True)[::5]:
            c = analytic(p, "call")
            u = analytic(p, "put")
            s, t, v = p
            dq = math.exp(-0.01 * t)
            dr = math.exp(-0.03 * t)
            self.assertAlmostEqual(c["price"] - u["price"], s * dq - 100 * dr)
            self.assertAlmostEqual(c["greeks"][0] - u["greeks"][0], dq)
            self.assertEqual(c["greeks"][1:3], u["greeks"][1:3])
            self.assertAlmostEqual(c["greeks"][4] - u["greeks"][4], t * dr)
            self.assertAlmostEqual(
                (c["greeks"][3] - u["greeks"][3]) * 365, 0.01 * s * dq - 3 * dr
            )

    def test_grid_and_basis(self):
        tr = grid()
        ho = grid(True)
        self.assertEqual(len(tr), 343)
        self.assertEqual(len(ho), 216)
        self.assertFalse(set(map(tuple, tr)) & set(map(tuple, ho)))
        self.assertEqual(len(POWERS), 35)
        self.assertEqual(len(set(map(tuple, POWERS))), 35)
        self.assertTrue(all(sum(p) <= 4 for p in POWERS))
        self.assertEqual(basis([0, 0, 0], [0, 0, 0], [1, 1, 1]), [1] + [0] * 34)
        self.assertTrue(
            all(
                basis([2, 3, 4], [0, 0, 0], [1, 1, 1])[i] == 2**a * 3**b * 4**c
                for i, (a, b, c) in enumerate(POWERS)
            )
        )
        for j in range(3):
            self.assertAlmostEqual(
                sum(
                    (p[j] - self.model["means"][j]) / self.model["scales"][j]
                    for p in tr
                ),
                0,
                delta=1e-10,
            )

    def test_cholesky_independent_known_solutions(self):
        rng = random.Random(19)
        for n in range(1, 13):
            for _ in range(5):
                a = [[rng.uniform(-1, 1) for _ in range(n)] for _ in range(n)]
                matrix = [
                    [
                        sum(a[k][i] * a[k][j] for k in range(n)) + (1 if i == j else 0)
                        for j in range(n)
                    ]
                    for i in range(n)
                ]
                expected = [rng.uniform(-3, 3) for _ in range(n)]
                rhs = [sum(row[j] * expected[j] for j in range(n)) for row in matrix]
                for x, y in zip(solve(matrix, [rhs])[0], expected):
                    self.assertAlmostEqual(x, y, places=11)
        with self.assertRaises(ValueError):
            solve([[-1]], [[1]])

    def test_model_and_holdout(self):
        m = self.model
        self.assertEqual(m["training_rows"], 343)
        self.assertEqual(m["features"], 35)
        self.assertEqual(len(m["weights"]), 10)
        self.assertLess(m["normal_residual"], 1e-9)
        e = evaluate(m)
        self.assertEqual(e["points"], 216)
        self.assertEqual(len(e["rows"]), 432)
        self.assertEqual(len(e["metrics"]), 10)
        for k, r in enumerate(e["metrics"]):
            errors = [
                row["errors"][k % 5] for row in e["rows"] if row["kind"] == r["kind"]
            ]
            self.assertAlmostEqual(r["mae"], sum(map(abs, errors)) / 216)
            self.assertAlmostEqual(r["max_abs"], max(map(abs, errors)))
            self.assertLessEqual(r["mae"], r["rmse"] + 1e-15)
            self.assertLessEqual(r["rmse"], r["max_abs"] + 1e-15)
        self.assertLess(e["metrics"][0]["mae"], 0.01)

    def test_no_mutation_or_test_refit(self):
        old = copy.deepcopy(self.model)
        points = grid(True)[:4]
        snap = copy.deepcopy(points)
        evaluate(self.model, points)
        self.assertEqual(self.model, old)
        self.assertEqual(points, snap)
        self.assertEqual(
            predict(self.model, [100, 1, 0.3]), predict(fit(), [100, 1, 0.3])
        )

    def test_bounds(self):
        for point in [[85, 0.25, 0.15], [115, 2, 0.45]]:
            self.assertTrue(all(math.isfinite(v) for v in predict(self.model, point)))
        for point in [
            None,
            [100, 1],
            [True, 1, 0.3],
            [100, math.nan, 0.3],
            [100, 1, 0.45001],
            [84.99, 1, 0.3],
            [100, 0.2499, 0.3],
            ["100", 1, 0.3],
        ]:
            for f in [analytic, lambda p: predict(self.model, p)]:
                with self.assertRaises(ValueError):
                    f(point)
        with self.assertRaises(ValueError):
            analytic([100, 1, 0.3], "bad")
        with self.assertRaises(ValueError):
            predict(self.model, [100, 1, 0.3], "bad")

    def test_cli(self):
        for args in [
            ["--spot", "nan"],
            ["--years", "0"],
            ["--volatility", "1"],
            ["--kind", "bad"],
        ]:
            p = subprocess.run(
                [sys.executable, "-m", "tangentlab", *args], capture_output=True
            )
            self.assertEqual(p.returncode, 2)
            self.assertEqual(p.stdout, b"")
        p = subprocess.run([sys.executable, "-m", "tangentlab"], capture_output=True)
        self.assertEqual(p.returncode, 0)


if __name__ == "__main__":
    unittest.main()
