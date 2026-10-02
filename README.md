# Volatility Surface Fitting

An end-to-end implied-volatility surface builder in Python for equity index and ETF options (Rutgers MQF Numerical Methods, solo project, reference date 21 March 2025). It runs on SPX (spot 5667.56, European) and SPY (spot 563.98, American) option chains at three tenors (about 1, 3 and 9 months), calls and puts, with roughly 40 to 80 quotes per name.

## Files
- `vol_fitting.py`: the full pipeline (about 670 lines, type hints and docstrings).
- `iv_surface_*.png`, `vol_surface_*.png`: implied-vol scatter plots and fitted 3D surfaces for SPX and SPY, European and American, including an SPX European versus American comparison.

## Numerical methods
- **Rate curve:** linear interpolation in `r*t` (log discount factor) space with flat extrapolation.
- **Dividends:** continuous yield for SPX; discrete cash dividends for SPY through the escrowed-spot model (spot less the PV of dividends with ex-date before expiry).
- **Black-Scholes pricer** with carry `r - q`, and **implied vol by Brent's method** (`scipy.optimize.brentq`) bracketed on [1e-7, 10], returning NaN safely when there is no sign change.
- **Cox-Ross-Rubinstein binomial tree** for American options: 200 steps, vectorized backward induction, early-exercise check at every node, risk-neutral probability clipped to [0, 1]. American implied vol is obtained by Brent inversion of the tree.
- **Repo / borrow cost bootstrapping**, tenor by tenor:
  - European: put-call parity at the strike nearest the forward, `F = (C - P) e^{rt} + K`, then `q_total = r - ln(F/S)/t` and `q_repo = q_total - q_div`.
  - American: root-solve `q_repo` so the tree-implied call vol equals the tree-implied put vol at the ATM-forward strike, with adaptive bracket widening.
- **Smile fit:** per tenor, OLS `IV(x) = a x^2 + b x + c` with `x = ln(K/F)` through `np.linalg.lstsq`, using only out-of-the-money quotes (calls above the forward, puts below) to avoid the unreliable in-the-money wing.
- **Surface:** the `(a, b, c)` coefficients are interpolated across tenor onto a fine log-moneyness by tenor mesh and drawn with `plot_surface`.

## Outputs
Four tables (repo rate and implied forward per tenor, IV per strike and type, `(a, b, c)` per tenor, fitted versus market IV) and the 3D surface plots.

## Limitations and next steps
Option chains are hardcoded rather than loaded from a snapshot. The quadratic smile is simple: add SVI or SABR for comparison, and add static no-arbitrage checks (butterfly convexity in strike, calendar monotonicity in total variance) to turn this into a validation-grade deliverable.

## Run
`pip install numpy scipy matplotlib` then `python vol_fitting.py`.
