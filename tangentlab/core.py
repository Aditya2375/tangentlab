"""Original bounded polynomial ridge surrogate of synthetic Black-Scholes Greeks."""

import math

BOUNDS = [[85.0, 115.0], [0.25, 2.0], [0.15, 0.45]]
NAMES = ["delta", "gamma", "vega", "theta", "rho"]
# Explicit total-degree order, constant first; no learned feature selection.
POWERS = [
    [a, b, d - a - b] for d in range(5) for a in range(d + 1) for b in range(d - a + 1)
]


def point_check(point):
    if (
        type(point) is not list
        or len(point) != 3
        or any(
            type(v) not in (int, float) or not math.isfinite(v) or not lo <= v <= hi
            for v, (lo, hi) in zip(point, BOUNDS)
        )
    ):
        raise ValueError(
            "Need [spot, years, volatility] within [85,115], [0.25,2], [0.15,0.45]"
        )


def cdf(x):
    # Symmetric normal CDF via a convergent erf power series on this bounded domain.
    # Integrate exp(-u^2/2): Phi(x)=1/2+phi(0)*sum (-1)^n*x^(2n+1)/(2^n*n!*(2n+1)).
    term = x
    total = x
    for n in range(1, 100):
        term *= -x * x / (2 * n)
        addition = term / (2 * n + 1)
        total += addition
        if abs(addition) < 1e-17:
            break
    return 0.5 + total / math.sqrt(2 * math.pi)


def analytic(point, kind="call"):
    point_check(point)
    if kind not in ("call", "put"):
        raise ValueError("Option kind must be call or put")
    s, t, v = point
    k = 100.0
    r = 0.03
    q = 0.01
    root = math.sqrt(t)
    dq = math.exp(-q * t)
    dr = math.exp(-r * t)
    d1 = (math.log(s / k) + (r - q + 0.5 * v * v) * t) / (v * root)
    d2 = d1 - v * root
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
    if kind == "call":
        price = s * dq * cdf(d1) - k * dr * cdf(d2)
        delta = dq * cdf(d1)
        theta = (
            -s * dq * pdf * v / (2 * root) - r * k * dr * cdf(d2) + q * s * dq * cdf(d1)
        )
        rho = k * t * dr * cdf(d2)
    else:
        price = k * dr * cdf(-d2) - s * dq * cdf(-d1)
        delta = -dq * cdf(-d1)
        theta = (
            -s * dq * pdf * v / (2 * root)
            + r * k * dr * cdf(-d2)
            - q * s * dq * cdf(-d1)
        )
        rho = -k * t * dr * cdf(-d2)
    return {
        "price": price,
        "greeks": [
            delta,
            dq * pdf / (s * v * root),
            s * dq * pdf * root / 100,
            theta / 365,
            rho / 100,
        ],
    }


def grid(holdout=False):
    count = 6 if holdout else 7
    return [
        [
            min(hi, lo + (hi - lo) * (i + 0.5 if holdout else i) / 6)
            for i, (lo, hi) in zip(indices, BOUNDS)
        ]
        for indices in (
            (a, b, c) for a in range(count) for b in range(count) for c in range(count)
        )
    ]


def basis(point, means, scales):
    z = [(v - m) / s for v, m, s in zip(point, means, scales)]
    return [math.prod(z[j] ** power[j] for j in range(3)) for power in POWERS]


def solve(matrix, rhs):
    """Own Cholesky factorization/triangular solve. Internal SPD bounded matrices."""
    n = len(matrix)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            rem = matrix[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if rem <= 0 or not math.isfinite(rem):
                    raise ValueError("Non-positive Cholesky pivot")
                lower[i][j] = math.sqrt(rem)
            else:
                lower[i][j] = rem / lower[j][j]
    outputs = []
    for target in rhs:
        y = []
        for i in range(n):
            y.append(
                (target[i] - sum(lower[i][j] * y[j] for j in range(i))) / lower[i][i]
            )
        x = [0.0] * n
        for i in reversed(range(n)):
            x[i] = (y[i] - sum(lower[j][i] * x[j] for j in range(i + 1, n))) / lower[i][
                i
            ]
        outputs.append(x)
    return outputs


def fit():
    points = grid()
    n = len(points)
    means = [sum(p[j] for p in points) / n for j in range(3)]
    scales = [
        math.sqrt(sum((p[j] - means[j]) ** 2 for p in points) / n) for j in range(3)
    ]
    x = [basis(p, means, scales) for p in points]
    targets = [
        analytic(p, "call")["greeks"] + analytic(p, "put")["greeks"] for p in points
    ]
    ym = [sum(row[j] for row in targets) / n for j in range(10)]
    ys = [
        math.sqrt(sum((row[j] - ym[j]) ** 2 for row in targets) / n) for j in range(10)
    ]
    yz = [[(row[j] - ym[j]) / ys[j] for j in range(10)] for row in targets]
    size = len(POWERS)
    ridge = 0.01
    gram = [
        [
            sum(row[i] * row[j] for row in x) + (ridge if i == j and i != 0 else 0)
            for j in range(size)
        ]
        for i in range(size)
    ]
    rhs = [
        [sum(row[i] * y[j] for row, y in zip(x, yz)) for i in range(size)]
        for j in range(10)
    ]
    weights = solve(gram, rhs)
    residual = max(
        abs(sum(gram[i][j] * w[j] for j in range(size)) - b[i])
        for w, b in zip(weights, rhs)
        for i in range(size)
    )
    return {
        "means": means,
        "scales": scales,
        "target_means": ym,
        "target_scales": ys,
        "weights": weights,
        "ridge": ridge,
        "training_rows": n,
        "features": size,
        "normal_residual": residual,
    }


def predict(model, point, kind="call"):
    point_check(point)
    if kind not in ("call", "put"):
        raise ValueError("Option kind must be call or put")
    x = basis(point, model["means"], model["scales"])
    offset = 0 if kind == "call" else 5
    return [
        model["target_means"][j]
        + model["target_scales"][j] * sum(a * b for a, b in zip(x, model["weights"][j]))
        for j in range(offset, offset + 5)
    ]


def evaluate(model, points=None):
    points = grid(True) if points is None else points
    # Trusted bounded internal evaluation set; public point validation still applies.
    rows = []
    for p in points:
        for kind in ("call", "put"):
            actual = analytic(p, kind)["greeks"]
            predicted = predict(model, p, kind)
            rows.append(
                {
                    "point": p[:],
                    "kind": kind,
                    "actual": actual,
                    "predicted": predicted,
                    "errors": [b - a for a, b in zip(actual, predicted)],
                }
            )
    metrics = []
    for kind in ("call", "put"):
        group = [r for r in rows if r["kind"] == kind]
        for j, name in enumerate(NAMES):
            errors = [r["errors"][j] for r in group]
            worst = max(range(len(group)), key=lambda i: abs(errors[i]))
            metrics.append(
                {
                    "kind": kind,
                    "greek": name,
                    "mae": sum(abs(e) for e in errors) / len(errors),
                    "rmse": math.sqrt(sum(e * e for e in errors) / len(errors)),
                    "max_abs": abs(errors[worst]),
                    "worst_point": group[worst]["point"][:],
                }
            )
    return {"points": len(points), "rows": rows, "metrics": metrics}
