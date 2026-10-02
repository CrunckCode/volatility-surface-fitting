"""
Name:    Deepak Chaudhary
Project: Volatility Surface Fitting
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Tuple, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import brentq
from scipy.stats import norm

warnings.filterwarnings("ignore")

# 0. MARKET DATA
SPOT_SPX = 5667.56
SPOT_SPY = 563.98

YIELD_CURVE: Dict[float, float] = {
    0.083: 0.0436,
    0.25:  0.0433,
    0.50:  0.0426,
    1.00:  0.0404,
    2.00:  0.0394,
}

SPX_DIV_YIELD = 0.0134

SPY_DIV_FORECAST: Dict[float, float] = {
    0.25: 1.90,
    0.50: 2.10,
    0.75: 1.90,
    1.00: 1.92,
}

SPX_OPTIONS: List[Tuple[float, float, bool, float]] = [
    (0.083,5000,True,689.85),(0.083,5100,True,592.5),(0.083,5200,True,496.65),
    (0.083,5300,True,403.4),(0.083,5400,True,314.3),(0.083,5500,True,231.6),
    (0.083,5600,True,157.45),(0.083,5650,True,124.65),(0.083,5700,True,95.25),
    (0.083,5800,True,48.2),(0.083,5900,True,18.95),(0.083,6000,True,5.55),
    (0.083,6100,True,1.43),(0.083,6200,True,0.5),
    (0.083,5000,False,6.7),(0.083,5100,False,9),(0.083,5200,False,12.85),
    (0.083,5300,False,19.2),(0.083,5400,False,29.9),(0.083,5500,False,46.8),
    (0.083,5600,False,72.45),(0.083,5650,False,89.45),(0.083,5700,False,109.8),
    (0.083,5800,False,162.3),(0.083,5900,False,232.6),(0.083,6000,False,318.95),
    (0.083,6100,False,414.45),(0.083,6200,False,513.15),
    (0.25,5000,True,749.05),(0.25,5100,True,658.95),(0.25,5200,True,571.35),
    (0.25,5300,True,486.85),(0.25,5400,True,406.05),(0.25,5500,True,330.05),
    (0.25,5600,True,258.5),(0.25,5650,True,225.45),(0.25,5700,True,194.05),
    (0.25,5800,True,137.45),(0.25,5900,True,90.8),(0.25,6000,True,55.15),
    (0.25,6100,True,30.65),(0.25,6200,True,15.85),
    (0.25,5000,False,37.65),(0.25,5100,False,46.4),(0.25,5200,False,57.75),
    (0.25,5300,False,72.05),(0.25,5400,False,90.1),(0.25,5500,False,112.5),
    (0.25,5600,False,140.25),(0.25,5650,False,156.55),(0.25,5700,False,174.65),
    (0.25,5800,False,217.3),(0.25,5900,False,269.15),(0.25,6000,False,332.65),
    (0.25,6100,False,407.15),(0.25,6200,False,491.25),
    (0.75,4500,True,1331.55),(0.75,4700,True,1154.75),(0.75,4900,True,983.85),
    (0.75,5100,True,820.25),(0.75,5300,True,665.35),(0.75,5500,True,521.75),
    (0.75,5700,True,389.25),(0.75,5900,True,271.7),(0.75,6100,True,173.05),
    (0.75,6300,True,98),(0.75,6500,True,49.55),(0.75,6700,True,23.3),
    (0.75,6900,True,10.85),(0.75,7100,True,5.2),
    (0.75,4500,False,61.35),(0.75,4700,False,78.35),(0.75,4900,False,101.05),
    (0.75,5100,False,131.15),(0.75,5300,False,170),(0.75,5500,False,219.05),
    (0.75,5700,False,280.45),(0.75,5900,False,356.75),(0.75,6100,False,451.55),
    (0.75,6300,False,570.05),(0.75,6500,False,715.25),(0.75,6700,False,883.05),
    (0.75,6900,False,1064.35),(0.75,7100,False,1252.35),
]

SPY_OPTIONS: List[Tuple[float, float, bool, float]] = [
    (0.083,500,True,67.02),(0.083,510,True,57.32),(0.083,520,True,47.78),
    (0.083,530,True,38.53),(0.083,540,True,29.77),(0.083,550,True,21.68),
    (0.083,560,True,14.5),(0.083,565,True,11.34),(0.083,570,True,8.54),
    (0.083,580,True,4.18),(0.083,590,True,1.57),(0.083,600,True,0.45),
    (0.083,610,True,0.12),(0.083,620,True,0.05),
    (0.083,500,False,0.74),(0.083,510,False,1.01),(0.083,520,False,1.43),
    (0.083,530,False,2.16),(0.083,540,False,3.38),(0.083,550,False,5.29),
    (0.083,560,False,8.15),(0.083,565,False,10.04),(0.083,570,False,12.33),
    (0.083,580,False,18.26),(0.083,590,False,26.04),(0.083,600,False,35.62),
    (0.083,610,False,45.56),(0.083,620,False,55.56),
    (0.25,500,True,74.11),(0.25,510,True,65.1),(0.25,520,True,56.33),
    (0.25,530,True,47.88),(0.25,540,True,39.8),(0.25,550,True,32.03),
    (0.25,560,True,24.97),(0.25,565,True,21.69),(0.25,570,True,18.59),
    (0.25,580,True,13.04),(0.25,590,True,8.5),(0.25,600,True,5.09),
    (0.25,610,True,2.79),(0.25,620,True,1.44),
    (0.25,500,False,3.98),(0.25,510,False,4.92),(0.25,520,False,6.12),
    (0.25,530,False,7.65),(0.25,540,False,9.57),(0.25,550,False,11.97),
    (0.25,560,False,14.97),(0.25,565,False,16.73),(0.25,570,False,18.71),
    (0.25,580,False,23.55),(0.25,590,False,29.49),(0.25,600,False,36.89),
    (0.25,610,False,45.57),(0.25,620,False,56.35),
    (0.75,450,True,131.94),(0.75,470,True,114.28),(0.75,490,True,97.21),
    (0.75,510,True,80.9),(0.75,530,True,65.44),(0.75,550,True,51.03),
    (0.75,570,True,37.85),(0.75,590,True,26.33),(0.75,610,True,16.55),
    (0.75,630,True,9.27),(0.75,650,True,4.64),(0.75,670,True,2.18),
    (0.75,690,True,1.02),(0.75,710,True,0.51),
    (0.75,450,False,6.34),(0.75,470,False,8.12),(0.75,490,False,10.51),
    (0.75,510,False,13.71),(0.75,530,False,17.86),(0.75,550,False,23.2),
    (0.75,570,False,30),(0.75,590,False,38.45),(0.75,610,False,49.85),
    (0.75,630,False,65.41),(0.75,650,False,85.54),(0.75,670,False,105.54),
    (0.75,690,False,125.54),(0.75,710,False,145.54),
]



# 1. CURVES (linear interpolation on r*t, the log discount factor)
def interp_rate(t: float, rate_curve: Dict[float, float] = YIELD_CURVE) -> float:
    """Linear interpolation in r*t (log discount factor) with flat extrapolation."""
    if not rate_curve:
        return 0.0
    tenors = sorted(rate_curve.keys())
    rates  = [rate_curve[k] for k in tenors]
    if t <= tenors[0]:
        return rates[0]
    if t >= tenors[-1]:
        return rates[-1]
    for i in range(len(tenors) - 1):
        if tenors[i] <= t <= tenors[i + 1]:
            t1, t2 = tenors[i], tenors[i + 1]
            r1, r2 = rates[i], rates[i + 1]
            rt = r1 * t1 + (r2 * t2 - r1 * t1) * (t - t1) / (t2 - t1)
            return rt / t
    return rates[-1]

def interp_repo(t: float, repo_curve: Dict[float, float]) -> float:
    """Linear interpolation in q with flat extrapolation."""
    if not repo_curve:
        return 0.0
    tenors = sorted(repo_curve.keys())
    rates  = [repo_curve[k] for k in tenors]
    if t <= tenors[0]:
        return rates[0]
    if t >= tenors[-1]:
        return rates[-1]
    return float(np.interp(t, tenors, rates))

# 2. DIVIDEND HANDLING (escrowed-spot model for discrete cash dividends)
def pv_dividends(t: float, div_forecast: Dict[float, float]) -> float:
    """PV at t=0 of all discrete cash dividends with ex-date <= t."""
    return sum(
        D * np.exp(-interp_rate(td) * td)
        for td, D in div_forecast.items()
        if td <= t
    )


def adjusted_spot(spot: float, t: float, div_forecast: Dict[float, float]) -> float:
    return spot - pv_dividends(t, div_forecast)

# 3. BLACK-SCHOLES (European)
def bs_price(S: float, K: float, t: float, r: float, q: float,
             sigma: float, is_call: bool) -> float:
    if t <= 1e-10 or sigma <= 1e-10:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    sqrt_t = np.sqrt(t)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    if is_call:
        return S * np.exp(-q * t) * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
    return K * np.exp(-r * t) * norm.cdf(-d2) - S * np.exp(-q * t) * norm.cdf(-d1)


def bs_implied_vol(price: float, S: float, K: float, t: float,
                   r: float, q: float, is_call: bool, tol: float = 1e-9) -> float:
    """Brent inversion of Black-Scholes. Returns NaN if no bracket."""
    f_lo = bs_price(S, K, t, r, q, 1e-7, is_call) - price
    f_hi = bs_price(S, K, t, r, q, 10.0,  is_call) - price
    if f_lo * f_hi > 0:
        return np.nan
    try:
        return brentq(lambda s: bs_price(S, K, t, r, q, s, is_call) - price,
                      1e-7, 10.0, xtol=tol)
    except Exception:
        return np.nan

# 4. CRR BINOMIAL TREE (American)
def crr_american_price(S: float, K: float, t: float, r: float, q: float,
                       sigma: float, is_call: bool, N: int = 200) -> float:
    """
    Cox-Ross-Rubinstein binomial tree for American options.
    For SPY with discrete dividends: pass S = S_adj = S - PV(divs <= t),
    q = q_repo (escrowed-spot model).
    """
    if t <= 1e-10 or sigma <= 1e-10:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)

    dt   = t / N
    u    = np.exp(sigma * np.sqrt(dt))
    d    = 1.0 / u
    disc = np.exp(-r * dt)
    p    = (np.exp((r - q) * dt) - d) / (u - d)
    p    = float(np.clip(p, 0.0, 1.0))   # asymptotic at sigma -> 0
    q_p  = 1.0 - p

    j  = np.arange(N + 1)
    ST = S * (u ** (N - 2 * j))
    V  = np.maximum(ST - K, 0.0) if is_call else np.maximum(K - ST, 0.0)

    for i in range(N - 1, -1, -1):
        V = disc * (p * V[: i + 1] + q_p * V[1 : i + 2])
        idx       = np.arange(i + 1)
        S_node    = S * (u ** (i - 2 * idx))
        intrinsic = (S_node - K) if is_call else (K - S_node)
        V = np.maximum(V, intrinsic)

    return float(V[0])


def american_implied_vol(price: float, S: float, K: float, t: float,
                         r: float, q: float, is_call: bool,
                         N: int = 200, tol: float = 1e-7) -> float:
    """Bracketed Brent inversion of the CRR tree."""
    def obj(sigma: float) -> float:
        return crr_american_price(S, K, t, r, q, sigma, is_call, N) - price

    try:
        lo, hi = 1e-4, 5.0
        if obj(lo) * obj(hi) > 0:
            return np.nan
        return brentq(obj, lo, hi, xtol=tol)
    except Exception:
        return np.nan

# 5. REPO FITTING — EUROPEAN (Put-Call Parity)
def fit_repo_european_single(
    tenor: float,
    options: Sequence[Tuple[float, float, bool, float]],
    spot: float,
    rate_curve: Dict[float, float],
    repo_curve: Dict[float, float],
    div_yield: float = 0.0,
) -> Tuple[float, float, float]:
    """
    Fit q at one tenor using PCP at the strike closest to the estimated forward.
    Returns (repo_rate, implied_forward, strike_used).
    """
    r = interp_rate(tenor, rate_curve)

    # Estimated forward F_e using already-fitted repo (only used to pick K_j)
    q_repo_prev  = interp_repo(tenor, repo_curve) if repo_curve else 0.0
    q_total_prev = div_yield + q_repo_prev
    Fe = spot * np.exp((r - q_total_prev) * tenor)

    calls = {k: p for (t, k, ic, p) in options if abs(t - tenor) < 1e-9 and ic}
    puts  = {k: p for (t, k, ic, p) in options if abs(t - tenor) < 1e-9 and not ic}
    common = sorted(set(calls) & set(puts))
    if not common:
        raise ValueError(f"No common call/put strikes at tenor {tenor:.3f}")

    Kj   = min(common, key=lambda k: abs(k - Fe))
    C, P = calls[Kj], puts[Kj]

    # PCP:  C - P = exp(-r t) * (F - K)  =>  F = (C - P) * exp(r t) + K
    F_implied = (C - P) * np.exp(r * tenor) + Kj

    if F_implied <= 0 or spot <= 0:
        q_repo = 0.0
    else:
        q_total = r - np.log(F_implied / spot) / tenor
        q_repo  = q_total - div_yield

    return q_repo, F_implied, Kj

# 6. REPO FITTING — AMERICAN (root-solve q so call_IV = put_IV)
def fit_repo_american_single(
    tenor: float,
    options: Sequence[Tuple[float, float, bool, float]],
    spot: float,
    rate_curve: Dict[float, float],
    repo_curve: Dict[float, float],
    div_forecast: Dict[float, float],
    N: int = 100,
) -> Tuple[float, float, float]:
    """
    For American options, solve q_repo such that CRR-implied call vol equals
    CRR-implied put vol at the strike closest to F_e.
    """
    r     = interp_rate(tenor, rate_curve)
    S_adj = adjusted_spot(spot, tenor, div_forecast)

    q_repo_prev = interp_repo(tenor, repo_curve) if repo_curve else 0.0
    Fe = S_adj * np.exp((r - q_repo_prev) * tenor)

    calls = {k: p for (t, k, ic, p) in options if abs(t - tenor) < 1e-9 and ic}
    puts  = {k: p for (t, k, ic, p) in options if abs(t - tenor) < 1e-9 and not ic}
    common = sorted(set(calls) & set(puts))
    if not common:
        raise ValueError(f"No common strikes at tenor {tenor:.3f}")

    Kj           = min(common, key=lambda k: abs(k - Fe))
    C_mkt, P_mkt = calls[Kj], puts[Kj]

    def iv_diff(q_repo: float) -> float:
        call_iv = american_implied_vol(C_mkt, S_adj, Kj, tenor, r, q_repo,
                                       is_call=True,  N=N)
        put_iv  = american_implied_vol(P_mkt, S_adj, Kj, tenor, r, q_repo,
                                       is_call=False, N=N)
        if np.isnan(call_iv) or np.isnan(put_iv):
            return np.nan
        return call_iv - put_iv

    # Try a tight bracket first, widen on sign mismatch
    for lo, hi in [(-0.10, 0.15), (-0.20, 0.25), (-0.40, 0.40)]:
        f_lo, f_hi = iv_diff(lo), iv_diff(hi)
        if np.isnan(f_lo) or np.isnan(f_hi):
            continue
        if f_lo * f_hi <= 0:
            try:
                q_solved  = brentq(iv_diff, lo, hi, xtol=1e-6)
                F_implied = S_adj * np.exp((r - q_solved) * tenor)
                return q_solved, F_implied, Kj
            except Exception:
                continue

    raise RuntimeError(f"American repo solve failed at tenor {tenor:.3f}")

# 7. REPO BOOTSTRAPPING (full curve, tenor by tenor)
def bootstrap_repo_curve(
    options: Sequence[Tuple[float, float, bool, float]],
    spot: float,
    rate_curve: Dict[float, float],
    div_param: Union[float, Dict[float, float]],
    is_american: bool = False,
    label: str = "",
) -> Tuple[Dict[float, float], Dict[float, float]]:
    """Bootstrap repo curve in chronological order: t1, t2, ..., tn."""
    tenors     = sorted(set(t for (t, k, ic, p) in options))
    repo_curve: Dict[float, float] = {}
    fwd_curve : Dict[float, float] = {}

    method = "American (CRR root-solve)" if is_american else "European (PCP)"
    print(f"\n{'=' * 70}")
    print(f"  OUTPUT 1 — REPO RATE AND IMPLIED FORWARD [{label}]   ({method})")
    print(f"{'=' * 70}")
    print(f"  {'Tenor':>8}  {'Repo Rate':>12}  {'Impl. Fwd':>12}  {'Strike Used':>12}")
    print(f"  {'-' * 8}  {'-' * 12}  {'-' * 12}  {'-' * 12}")

    for tenor in tenors:
        if is_american:
            assert isinstance(div_param, dict)
            q, F, Kj = fit_repo_american_single(
                tenor, options, spot, rate_curve, repo_curve,
                div_forecast=div_param,
            )
        else:
            div_yield = div_param if isinstance(div_param, float) else 0.0
            q, F, Kj = fit_repo_european_single(
                tenor, options, spot, rate_curve, repo_curve, div_yield
            )
        repo_curve[tenor] = q
        fwd_curve[tenor]  = F
        print(f"  {tenor:>8.3f}  {q:>12.6f}  {F:>12.4f}  {Kj:>12.1f}")

    return repo_curve, fwd_curve

# 8. IMPLIED VOLATILITY SURFACE (per-quote IV)
def compute_iv_surface_european(
    options: Sequence[Tuple[float, float, bool, float]],
    spot: float,
    rate_curve: Dict[float, float],
    repo_curve: Dict[float, float],
    div_yield: float = 0.0,
) -> List[Tuple[float, float, bool, float]]:
    """Black-Scholes IV for every quote (continuous-yield model)."""
    iv_surface: List[Tuple[float, float, bool, float]] = []
    for (tenor, strike, is_call, price) in options:
        r       = interp_rate(tenor, rate_curve)
        q_repo  = interp_repo(tenor, repo_curve)
        q_total = div_yield + q_repo
        iv = bs_implied_vol(price, spot, strike, tenor, r, q_total, is_call)
        iv_surface.append((tenor, strike, is_call, iv))
    return iv_surface


def compute_iv_surface_american(
    options: Sequence[Tuple[float, float, bool, float]],
    spot: float,
    rate_curve: Dict[float, float],
    repo_curve: Dict[float, float],
    div_forecast: Dict[float, float],
    N: int = 200,
) -> List[Tuple[float, float, bool, float]]:
    """CRR-tree IV for every quote (escrowed-spot model). NaN on failure."""
    iv_surface: List[Tuple[float, float, bool, float]] = []
    for (tenor, strike, is_call, price) in options:
        r      = interp_rate(tenor, rate_curve)
        q_repo = interp_repo(tenor, repo_curve)
        S_adj  = adjusted_spot(spot, tenor, div_forecast)
        iv = american_implied_vol(price, S_adj, strike, tenor, r, q_repo,
                                  is_call, N=N)
        iv_surface.append((tenor, strike, is_call, iv))
    return iv_surface

# 9. QUADRATIC VOL SURFACE FIT (OLS, OTM only, K > F for calls, K < F for puts)
def fit_vol_surface(
    iv_surface: Sequence[Tuple[float, float, bool, float]],
    fwd_curve: Dict[float, float],
) -> Dict[float, Tuple[float, float, float]]:
    """
    For each tenor fit IV(x) = a*x^2 + b*x + c by OLS, x = log(K/F).
    OTM only: call when K > F, put when K < F (per PDF).
    NaN IVs are skipped.
    """
    tenors = sorted(set(t for (t, k, ic, iv) in iv_surface))
    poly_coeffs: Dict[float, Tuple[float, float, float]] = {}

    for tenor in tenors:
        F = fwd_curve.get(tenor)
        if F is None:
            continue

        xs: List[float] = []
        ivs: List[float] = []
        for (t, k, ic, iv) in iv_surface:
            if abs(t - tenor) > 1e-9 or np.isnan(iv):
                continue
            include = (k > F) if ic else (k < F)
            if include:
                xs.append(np.log(k / F))
                ivs.append(iv)

        if len(xs) < 3:
            continue

        x = np.asarray(xs)
        y = np.asarray(ivs)
        A = np.column_stack([x * x, x, np.ones_like(x)])
        coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
        a, b, c = coeffs
        poly_coeffs[tenor] = (float(a), float(b), float(c))

    return poly_coeffs


# 10. DISPLAY (the four required outputs)
def display_iv_table(
    iv_surface: Sequence[Tuple[float, float, bool, float]],
    fwd_curve: Dict[float, float],
    label: str,
) -> None:
    """Output 2: IV per (tenor, strike, type)."""
    tenors = sorted(set(t for (t, k, ic, iv) in iv_surface))
    print(f"\n{'=' * 70}")
    print(f"  OUTPUT 2 — IMPLIED VOLATILITY [{label}]")
    print(f"{'=' * 70}")
    for tenor in tenors:
        F = fwd_curve.get(tenor)
        Fs = f"{F:.4f}" if F else "N/A"
        print(f"\n  Tenor = {tenor:.3f}  |  Implied Forward = {Fs}")
        print(f"  {'Strike':>8}  {'Type':>6}  {'IV':>10}")
        print(f"  {'-' * 8}  {'-' * 6}  {'-' * 10}")
        rows = sorted(
            [(k, ic, iv) for (t, k, ic, iv) in iv_surface if abs(t - tenor) < 1e-9],
            key=lambda r: (r[0], not r[1]),
        )
        for k, ic, iv in rows:
            typ   = "Call" if ic else "Put"
            iv_str = f"{iv * 100:>9.4f}%" if not np.isnan(iv) else f"{'N/A':>10}"
            print(f"  {k:>8.1f}  {typ:>6}  {iv_str}")


def display_vol_surface(
    poly_coeffs: Dict[float, Tuple[float, float, float]],
    label: str,
) -> None:
    """Output 3: (a, b, c) per tenor."""
    print(f"\n{'=' * 70}")
    print(f"  OUTPUT 3 — VOL SURFACE COEFFICIENTS [{label}]")
    print(f"  IV(x) = a*x^2 + b*x + c   where x = log(K/F)")
    print(f"{'=' * 70}")
    print(f"  {'Tenor':>8}  {'a':>15}  {'b':>15}  {'c':>15}")
    print(f"  {'-' * 8}  {'-' * 15}  {'-' * 15}  {'-' * 15}")
    for tenor, (a, b, c) in sorted(poly_coeffs.items()):
        print(f"  {tenor:>8.3f}  {a:>15.6f}  {b:>15.6f}  {c:>15.6f}")


def display_fitted_iv_at_strikes(
    iv_surface: Sequence[Tuple[float, float, bool, float]],
    poly_coeffs: Dict[float, Tuple[float, float, float]],
    fwd_curve: Dict[float, float],
    label: str,
) -> None:
    """Output 4: f(x) = a*x^2 + b*x + c at every actual strike, vs market IV."""
    tenors = sorted(set(t for (t, k, ic, iv) in iv_surface))
    print(f"\n{'=' * 70}")
    print(f"  OUTPUT 4 - FITTED IV AT EACH STRIKE [{label}]")
    print(f"  Using f(x) = a*x^2 + b*x + c,  x = log(K/F)")
    print(f"{'=' * 70}")

    for tenor in tenors:
        if tenor not in poly_coeffs or tenor not in fwd_curve:
            continue
        a, b, c = poly_coeffs[tenor]
        F       = fwd_curve[tenor]

        print(f"\n  Tenor = {tenor:.3f}  |  F = {F:.4f}")
        print(f"  a = {a:.6f}   b = {b:.6f}   c = {c:.6f}")
        print(f"  {'Strike':>8}  {'Type':>6}  {'x=log(K/F)':>12}  "
              f"{'Mkt IV':>10}  {'Fitted IV':>10}")
        print(f"  {'-' * 8}  {'-' * 6}  {'-' * 12}  "
              f"{'-' * 10}  {'-' * 10}")

        rows = sorted(
            [(k, ic, iv) for (t, k, ic, iv) in iv_surface if abs(t - tenor) < 1e-9],
            key=lambda r: (r[0], not r[1]),
        )
        for k, ic, iv in rows:
            x         = np.log(k / F)
            fitted_iv = a * x * x + b * x + c
            typ       = "Call" if ic else "Put"
            mkt_str   = f"{iv * 100:>9.4f}%" if not np.isnan(iv) else f"{'N/A':>10}"
            print(f"  {k:>8.1f}  {typ:>6}  {x:>12.6f}  "
                  f"{mkt_str}  {fitted_iv * 100:>9.4f}%")

# 10b. VOL SURFACE 3D PLOTS
def make_grid(fwd_curve, poly_coeffs, n_x=60, n_t=60):
    """Build a (X, T, IV) mesh by linearly interpolating (a, b, c) across tenors."""
    tenors = sorted(poly_coeffs.keys())
    if len(tenors) < 2:
        return None
    t_fine = np.linspace(tenors[0], tenors[-1], n_t)
    xs_all = []
    for tenor in tenors:
        F = fwd_curve[tenor]
        xs_all += [np.log(K / F) for K in np.linspace(0.6 * F, 1.4 * F, 50)]
    x_fine = np.linspace(min(xs_all), max(xs_all), n_x)
    X, T   = np.meshgrid(x_fine, t_fine)
    IV     = np.full_like(X, np.nan)
    for i, t in enumerate(t_fine):
        if t <= tenors[0]:
            a, b, c = poly_coeffs[tenors[0]]
        elif t >= tenors[-1]:
            a, b, c = poly_coeffs[tenors[-1]]
        else:
            for j in range(len(tenors) - 1):
                if tenors[j] <= t <= tenors[j + 1]:
                    w = (t - tenors[j]) / (tenors[j + 1] - tenors[j])
                    a0, b0, c0 = poly_coeffs[tenors[j]]
                    a1, b1, c1 = poly_coeffs[tenors[j + 1]]
                    a = (1 - w) * a0 + w * a1
                    b = (1 - w) * b0 + w * b1
                    c = (1 - w) * c0 + w * c1
                    break
        IV[i, :] = a * x_fine ** 2 + b * x_fine + c
    return X, T, IV


def plot_vol_surface(fwd_curve, poly_coeffs, label, save_path=None):
    """Render a single 3D vol surface for the given (label) market."""
    grid = make_grid(fwd_curve, poly_coeffs)
    if grid is None:
        print(f"  [Plot] Not enough tenors to plot {label} surface.")
        return
    X, T, IV = grid

    fig = plt.figure(figsize=(9, 6))
    fig.patch.set_facecolor("#0f0f1a")
    ax  = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, T, IV * 100, cmap=cm.plasma,
                           edgecolor="none", alpha=0.92,
                           rstride=1, cstride=1)
    ax.set_facecolor("#0f0f1a")
    ax.set_xlabel("log(K/F)",    color="white", labelpad=8)
    ax.set_ylabel("Tenor (yrs)", color="white", labelpad=8)
    ax.set_zlabel("IV (%)",      color="white", labelpad=8)
    ax.set_title(f"Implied Volatility Surface - {label}",
                 color="white", fontsize=13, pad=12)
    ax.tick_params(colors="white")
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#333")
    ax.grid(True, color="#333333", linewidth=0.4)
    cb = fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1)
    cb.set_label("IV (%)", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    return fig

# 11. PIPELINES
def run_european_pipeline(
    options: Sequence[Tuple[float, float, bool, float]],
    spot: float,
    rate_curve: Dict[float, float],
    div_yield: float,
    label: str,
) -> None:
    print(f"\n{'#' * 70}")
    print(f"  PIPELINE: {label}  (European, continuous dividend yield)")
    print(f"  Spot = {spot:.4f}    Dividend Yield = {div_yield * 100:.4f}%")
    print(f"{'#' * 70}")

    repo_curve, fwd_curve = bootstrap_repo_curve(
        options, spot, rate_curve,
        div_param=div_yield, is_american=False, label=label,
    )
    iv_surface  = compute_iv_surface_european(
        options, spot, rate_curve, repo_curve, div_yield=div_yield,
    )
    display_iv_table(iv_surface, fwd_curve, label)
    poly_coeffs = fit_vol_surface(iv_surface, fwd_curve)
    display_vol_surface(poly_coeffs, label)
    display_fitted_iv_at_strikes(iv_surface, poly_coeffs, fwd_curve, label)
    plot_vol_surface(fwd_curve, poly_coeffs, label,
                     save_path=f"vol_surface_{label}.png")


def run_american_pipeline(
    options: Sequence[Tuple[float, float, bool, float]],
    spot: float,
    rate_curve: Dict[float, float],
    div_forecast: Dict[float, float],
    label: str,
    tree_steps_iv: int = 200,
) -> None:
    print(f"\n{'#' * 70}")
    print(f"  PIPELINE: {label}  (American, discrete cash dividends)")
    print(f"  Spot = {spot:.4f}")
    print(f"  Dividend Forecast (ex-date -> amount):")
    for td, D in sorted(div_forecast.items()):
        pv = D * np.exp(-interp_rate(td) * td)
        print(f"    t = {td:.2f}    D = {D:.2f}    PV(D) = {pv:.4f}")
    print(f"{'#' * 70}")

    repo_curve, fwd_curve = bootstrap_repo_curve(
        options, spot, rate_curve,
        div_param=div_forecast, is_american=True, label=label,
    )
    iv_surface  = compute_iv_surface_american(
        options, spot, rate_curve, repo_curve,
        div_forecast=div_forecast, N=tree_steps_iv,
    )
    display_iv_table(iv_surface, fwd_curve, label)
    poly_coeffs = fit_vol_surface(iv_surface, fwd_curve)
    display_vol_surface(poly_coeffs, label)
    display_fitted_iv_at_strikes(iv_surface, poly_coeffs, fwd_curve, label)
    plot_vol_surface(fwd_curve, poly_coeffs, label,
                     save_path=f"vol_surface_{label}.png")

# 12. ENTRY POINT
def main() -> None:
    print("=" * 70)
    print("  VOLATILITY SURFACE FITTING")
    print("  Reference Date: March 21, 2025")
    print("  Name: Deepak Chaudhary")
    print("=" * 70)

    run_european_pipeline(
        options    = SPX_OPTIONS,
        spot       = SPOT_SPX,
        rate_curve = YIELD_CURVE,
        div_yield  = SPX_DIV_YIELD,
        label      = "SPX (European)",
    )

    run_american_pipeline(
        options=SPY_OPTIONS,
        spot=SPOT_SPY,
        rate_curve=YIELD_CURVE,
        div_forecast=SPY_DIV_FORECAST,
        label="SPY (American)",
        tree_steps_iv=200,
    )

    plt.show()


if __name__ == "__main__":
    main()