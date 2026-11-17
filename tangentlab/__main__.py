import argparse
import json
from . import analytic, evaluate, fit, predict

p = argparse.ArgumentParser(
    description="Synthetic Greeks surrogate, not financial advice"
)
p.add_argument("--spot", type=float, default=100.0)
p.add_argument("--years", type=float, default=1.0)
p.add_argument("--volatility", type=float, default=0.3)
p.add_argument("--kind", choices=["call", "put"], default="call")
a = p.parse_args()
try:
    point = [a.spot, a.years, a.volatility]
    reference = analytic(point, a.kind)
    model = fit()
    print(
        json.dumps(
            {
                "point": point,
                "kind": a.kind,
                "reference": reference,
                "surrogate": predict(model, point, a.kind),
                "holdout_metrics": evaluate(model)["metrics"],
            },
            allow_nan=False,
        )
    )
except ValueError as e:
    p.error(str(e))
