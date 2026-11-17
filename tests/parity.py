import json
import sys
from tangentlab import fit, predict, analytic, evaluate

m = fit()
points = json.load(sys.stdin)
print(
    json.dumps(
        {
            "model": m,
            "points": [
                {
                    "point": p,
                    "kind": k,
                    "analytic": analytic(p, k),
                    "predicted": predict(m, p, k),
                }
                for p in points
                for k in ["call", "put"]
            ],
            "holdout": evaluate(m),
        },
        allow_nan=False,
    )
)
