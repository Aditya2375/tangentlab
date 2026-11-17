# Tangentlab

An original trained polynomial-ridge surrogate for European option Greeks, compared with analytic Black-Scholes-Merton on a separate synthetic off-grid holdout. Independent Python and JavaScript cores implement the formula, train-only scalers, polynomial features, normal equations, Cholesky solve and frozen prediction. A responsive static lab shows five Greeks for a selected point, reference-versus-surrogate curves, every plotted value, the complete holdout error ledger and all 350 learned coefficients.

**This predicts formula outputs, not the market.** No observed stock or option prices, market data, implied-volatility fitting, broker keys, trades, portfolio, money, investment advice, PnL, speedup or empirical accuracy claim. Training and reference labels come from the same idealized formula. An accurate surrogate can still inherit every financial-model error of that teacher. This is ordinary supervised regression, not a neural network, LLM, calibrated risk model or reproduction of a cited paper.

## Run

Python 3.10+ standard library only:

```sh
python3 -m tangentlab
python3 -m tangentlab --spot 97.5 --years 0.6875 --volatility 0.225 --kind put
```

The CLI prints JSON: applied point, analytic price and Greeks, learned Greeks and off-grid holdout metrics. Invalid arguments exit with status 2. It reads no files, writes no files, runs no shell commands and makes no network requests. No key, model download or paid service.

```python
from tangentlab import analytic, fit, predict

model = fit()
point = [100, 1, 0.3]
reference = analytic(point, "call")
surrogate = predict(model, point, "call")
assert len(surrogate) == 5
assert abs(reference["price"] - 12.69400452623556) < 1e-10
```

Serve the independent static browser:

```sh
python3 -m http.server 8000 --directory docs --bind 127.0.0.1
```

Open `http://127.0.0.1:8000`. No build is needed. It fits at page load, not on a remote Python server. Inputs mark the last point stale and disable chart selection until **Apply point** succeeds. Invalid input retains the visibly stale last result. The global error audit is for the same fixed model; its option-type row ledger retains the last applied option type. Changing the plotted Greek is a display choice, not a refit. Every chart point is available in the 61-row slice table. Wide tables scroll locally rather than hiding content or overflowing the page.

## Exact financial contract

All examples use abstract currency units, one option unit, no lot/contract multiplier, and fixed constants:

| Input                                           | Value or supported range  |
| ----------------------------------------------- | ------------------------- |
| Strike K                                        | 100                       |
| Continuously compounded annual risk-free rate r | 0.03                      |
| Continuous annual dividend yield q              | 0.01                      |
| Spot S                                          | 85 to 115 inclusive       |
| Remaining time T                                | 0.25 to 2 years inclusive |
| Annual volatility σ                             | 0.15 to 0.45 inclusive    |
| Exercise                                        | European call or put      |

The browser takes volatility in percent, so 30 means σ = 0.30. The CLI/library take decimal volatility, so 0.30 means 30%. There is no expiry-date, trading-calendar or timezone calculation. T is already a year fraction. No support for zero volatility, expiry, early exercise, discrete dividends, changing strike/rates/yield or out-of-box extrapolation. Bounds fail explicitly; inputs and outputs are never silently clipped. Grid generation only caps a rounded upper-endpoint coordinate at its declared upper bound.

With Φ the standard normal CDF and φ its density:

```text
d1 = (log(S/K) + (r-q+σ²/2)*T) / (σ*sqrt(T))
d2 = d1 - σ*sqrt(T)
Call price = S*exp(-qT)*Φ(d1) - K*exp(-rT)*Φ(d2)
Put price  = K*exp(-rT)*Φ(-d2) - S*exp(-qT)*Φ(-d1)
```

| Output | Definition and display unit                                                    |
| ------ | ------------------------------------------------------------------------------ |
| Delta  | ∂V/∂S, option-value change per +1 spot unit                                    |
| Gamma  | ∂²V/∂S², change in delta per +1 spot unit                                      |
| Vega   | (∂V/∂σ)/100, option-value change per +1 percentage point of volatility         |
| Theta  | -(∂V/∂T)/365, calendar-time decay per day, holding spot/volatility/rates fixed |
| Rho    | (∂V/∂r)/100, option-value change per +1 percentage point of annual rate        |

These sensitivities are local derivatives, not exact finite-shock PnL. Theta uses 365 calendar days, not 252 trading days, and its sign is calendar-time passage, not remaining-maturity derivative. Rho is a partial derivative at fixed q and volatility, not a scenario recalibration. The analytic price is shown as reference only; **no price model is trained**. The order of Greek arrays is delta, gamma, vega, theta, rho.

The CDF uses an original recurrence for the integrated normal-density power series, equivalent to the erf series. It stops when an added term is below 1e-17 or after 99 updates after the initial term. The supported financial domain keeps the required normal arguments within [-3, 3]; the internal series helper is not a general tail-CDF service. Native tests compare 601 points in this interval against the standard library's independently implemented erf at an absolute tolerance of 3e-14. The implementation is finite floating arithmetic, not symbolic exactness or guaranteed correctly rounded output.

## What actually gets learned

1. Training coordinates form a Cartesian grid with seven equally spaced values on each of the three input axes, including endpoints. There are 7³ = 343 coordinates. For each, compute all five analytic Greeks for calls and puts, yielding ten supervised targets. No observed data, randomized seed, hidden weights or memorized lookup interpolation.
2. Fit each input's training mean and population standard deviation, using denominator 343. Standardize inputs with those fixed scalers. Fit each of the ten targets' training mean and population SD the same way. All scales are strictly positive on the fixed training grid, so no variance fallback is needed or implied. Neither holdout nor selected points change these statistics.
3. Expand standardized inputs into all monomials of total degree at most four. There are 35 features, including the constant. The public `POWERS` order is deterministic: total degree d increases from 0 to 4, spot exponent a from 0 to d, time exponent b from 0 to d-a, volatility exponent d-a-b. Feature = standardized spot^a × standardized time^b × standardized volatility^(d-a-b). Powers are not raw-unit financial coefficients.
4. For each standardized target, minimize `sum((Xw-y)^2) + 0.01*sum(w[j]^2 for j>0)`. The loss is a **sum**, not a mean. The constant coefficient is unpenalized. Penalty and degree are fixed teaching choices; there is no hyperparameter search, optimality claim or holdout tuning. Ten target regressions share the same design matrix and factorization but have separate fitted coefficients, including separate call/put gamma and vega regressions.
5. Form `A = XᵀX + 0.01*D`, where D is diagonal with first entry 0 and all others 1, and `b = Xᵀy`. Solve `Aw=b` using an original lower-triangular Cholesky factorization followed by forward/back substitution. This supported matrix is positive definite. A nonpositive or nonfinite pivot fails rather than inventing a result. No explicit inverse, optimizer loop, external linear-algebra runtime or external model package.
6. Freeze scalers and weights. For a new bounded point, expand its standardized features, multiply by the learned coefficients and unscale each target. Inference does not call the analytic formula. The UI independently calls that formula for comparison. Fitting and evaluation do call the formula to create labels. No hidden correction forces surrogate values to equal the teacher or satisfy financial bounds.

The model has 35 coefficients for each of ten outputs, totaling 350. It predicts Greeks **directly**, not derivatives of a learned price. It need not satisfy put-call derivative identities, no-arbitrage bounds, integrability or cross-Greek consistency. Polynomial artifacts can appear at boundaries. The identical analytic call/put gamma and vega targets may produce equal learned outputs here; this is not a general structural guarantee across arbitrary financial models.

The displayed normal-equation residual is `max(abs(Aw-b))` over all ten standardized-target solves. It checks numerical solution of those equations, not generalization, market fit, certainty or economic validity. Normal equations can square a design matrix's condition number; this bounded degree, scaling and fixed positive regularization are part of the supported setup, not a recommendation for arbitrary ill-conditioned regression.

## The off-grid audit

Six midpoints between adjacent training values on each axis form 6³ = 216 holdout coordinates. They do not overlap any training coordinate. Both option types are evaluated at each point, so the full result contains 432 rows and ten per-output error summaries. This is deterministic **interior interpolation testing**, not an independent market sample, chronological prediction, broad extrapolation test or proof of uniform error over the domain. No labels from the holdout enter scalers or coefficients. Boundary/corner performance is not certified by midpoint results.

Signed error is surrogate minus analytic reference in each Greek's display units. MAE is the mean absolute error, RMSE is the square root of mean squared error, and maximum absolute error is the largest observed holdout error for that output. The worst coordinate is the first in deterministic row order on an exact tie. Errors across unlike units are not combined. Relative error is deliberately omitted because a Greek can be near zero. No percentage-accuracy or confidence number is inferred from these values. The complete UI error ledger follows the last applied call/put selection; all ten summary metrics remain visible.

The plotted slice uses S = 85, 85.5, ..., 115 at the applied T, volatility and option type. Its 61 values are for inspection, not an additional independent holdout. Some coordinates can coincide with training or midpoint coordinates. Its vertical marker is selected spot. Chart axes use the displayed Greek's native units. The conceptual hero curves are clearly labeled illustration, not model output. No measured runtime advantage over the already-cheap analytic formula is claimed.

## Library and security boundaries

- `fit()` has no parameters and always fits the fixed synthetic grid above. It returns detached ordinary arrays/dicts/objects containing scalers, weights, dimensions, penalty and solve residual. It is not an arbitrary dataset loader or user-controlled model trainer.
- `analytic(point, kind="call")` and `predict(model, point, kind="call")` validate exact Python lists / ordinary JS arrays of three finite numeric coordinates in the domain above. Numeric strings, booleans, sparse arrays, NaN, Infinity, missing or extra coordinates and unsupported option kinds fail. Python numeric values are built-in int/float, not arbitrary numeric subclasses. Inputs are not mutated.
- `predict` expects an unmodified model from `fit()`. No serialized-model loading, hostile getters/proxies, concurrent mutation, subclasses or process-isolation guarantee. Returned output belongs to the caller; mutating a model changes later predictions and is outside supported use.
- `evaluate(model, points=None)` defaults to the fixed midpoint grid. Its optional points argument is a trusted nonempty bounded list of valid coordinates for engineering tests, not an untrusted evaluation API. An empty evaluation is unsupported. Internal/exported `basis`, `solve`, `cdf`, constants and powers support implementation/testing only; they do not validate arbitrary matrices, extreme tail arguments or mutated globals. Treat every exported constant as read-only.
- Fitting is bounded by 343 rows, 35 features and ten targets. Holdout evaluation is bounded by 216 coordinates in normal product use; the UI slice has 61 points. No recursion, external callbacks, user code, shell, file input, model download or continuous background loop. Host memory/runtime failure is not transactionally isolated.
- The browser uses native numeric/select/button controls, canvas plus full tabular chart alternatives, and explicit local scrolling for wide tables. No upload, storage, cookies, telemetry or runtime network. Static hosting may log ordinary asset requests. Reload/close normally discards state; session restore and secure erasure are outside the app's control. Source links navigate away.

## Verify

```sh
make test
npm ci --ignore-scripts
npx playwright install chromium
npm test
npm run test:browser
```

Native tests include a worked analytic fixture, 100 seeded independent erf-price/finite-difference checks for both option types and all five Greek units/signs, 601 CDF values, put-call price/Greek identities, exact grid separation, all polynomial powers, train-scaler means, 60 positive-definite linear systems with independently chosen solutions, learned dimensions/solve residual, independently recomputed holdout metrics, model/input immutability, bounds and invalid CLI. Finite differences use separate standard-library erf pricing and do not call this CDF or analytic implementation for expected values.

The independent JavaScript suite compares the full trained model, 142 option-type/point predictions, all 432 holdout rows and ten metric summaries with native Python, plus analytic linear-system/CDF and invalid-input fixtures. Numerical parity supplements independent oracles; it can otherwise preserve shared errors. Comparisons use finite tolerances rather than claiming bit-exact floating behavior across runtimes.

Real Chromium tests at 1440/1024/768/390/320 px verify actual model-derived reference/predicted values, price, holdout metrics, full row counts, actual painted chart pixels, all 61 slice values present, 350 real coefficients, all five Greek chart choices, call/put, boundary parameters, decimal volatility units, dirty and invalid input, table containment, no page overflow/errors/failed or third-party requests/storage/cookies. No other-browser or assistive-technology certification. Small finite-difference tolerances and finite sample tests are not proofs of uniform approximation accuracy, financial correctness or absence of bugs.

CI tests Python 3.10, 3.12 and 3.13, independent browser math, real Chromium and static/security scans. Actions are commit-pinned, permissions read-only, the Gitleaks binary checksum-verified and dev-only Playwright locked. The runtime has no third-party packages. Source contains no private design/security/process evidence or personal data.

## Background

Concept references, not copied implementation, weights, data or inherited performance claims:

- [QuantPie Black-Scholes summary](https://www.quantpie.co.uk/bsm_formula/bs_summary.php): call/put prices and Greek formulas, with foreign/domestic rates interpreted as dividend yield/discount rate.
- [GreekGeek methodology](https://www.greekgeek.online/methodology): displayed Greek units, calendar-day theta and percentage-point vega/rho. No live feeds or implied-volatility algorithm from that site are used.
- [Scikit-learn linear models](https://scikit-learn.org/stable/modules/linear_model.html): ridge regression's penalized squared-error objective.
- [Scikit-learn polynomial features](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html): total-degree feature expansion as a model class, not an implementation dependency.
- [NumPy Cholesky reference](https://numpy.org/doc/stable/reference/generated/numpy.linalg.cholesky.html): positive-definite factorization and triangular solution steps. The product uses its own standard-library implementation, not NumPy.
- [NIST DLMF series](https://dlmf.nist.gov/7.6): erf power-series foundation for the bounded normal-CDF recurrence.
- [Approximating Option Greeks using ANNs (2024)](https://www.mdpi.com/1911-8074/17/4/140): a different synthetic-training/Greeks-approximation research direction. Tangentlab is a small polynomial baseline, not that network, experiment or reported market validation.

MIT. Original code and design by Aditya Kulkarni.
