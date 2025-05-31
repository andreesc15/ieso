#!/usr/bin/env python3
# Fixed Indonesian ESO Monte Carlo Model
# Based on "Pricing Employee Stock Options with an Asian Style Using a Modified Binomial Method" by Chendra et. al. 2022
# Written by Andree Sulistio Chandra (20824003) and Aya Sofia (20124015)
# ---------------------------------------------------------------------
from __future__ import annotations
import numpy as np
from scipy.stats import norm
from typing      import List, Tuple, Optional
from datetime import datetime
from numba import njit, prange

# ──────────────────────────────────────────────────────────────────────
# 0. CONSTANTS
# ──────────────────────────────────────────────────────────────────────
DAYS_PER_MONTH  = 25
MONTHS_PER_YEAR = 12
DAYS_PER_YEAR   = DAYS_PER_MONTH * MONTHS_PER_YEAR          # = 300

# ──────────────────────────────────────────────────────────────────────
# 1. JIT-compiled single-path simulator
# ──────────────────────────────────────────────────────────────────────
@njit(fastmath=True, nogil=True)
def _single_path_jit(z: np.ndarray,
                     S0: float, r: float, q: float, sigma: float,
                     lam: float, M: float, dt: float, quit_p: float,
                     final_day: int,
                     window_start: np.ndarray,   # shape (n_win,)
                     window_end:   np.ndarray,   # shape (n_win,)
                     window_map:   np.ndarray,   # shape (final_day+1,)
                     Tv_days: int) -> float:
    """
    Pure-Numba replication of original _single_path logic.
    All inputs are plain NumPy/primitive objects so the function can
    compile in nopython mode.
    """
    n_win = window_start.size
    S     = np.empty(final_day + 1, dtype=np.float64)
    S[0]  = S0
    mu    = (r - q - 0.5 * sigma * sigma) * dt
    vol   = sigma * np.sqrt(dt)

    # 1. Generate GBM path ------------------------------------------------
    for k in range(1, final_day + 1):
        S[k] = S[k - 1] * np.exp(mu + vol * z[k - 1])

    # 2. Compute strike for every execution window ------------------------
    strikes = np.empty(n_win, dtype=np.float64)
    for i in range(n_win):
        avg_start = window_start[i] - DAYS_PER_MONTH  # inclusive (1-based days)
        avg_end   = window_start[i] - 1               # inclusive
        # Slice indices are already correct for S (S[k] = price at end of day k)
        strikes[i] = 0.9 * S[avg_start:avg_end + 1].mean()

    # 3. Walk through each calendar day -----------------------------------
    vested = False
    for day in range(1, final_day + 1):
        time_yr = day * dt
        df      = np.exp(-r * time_yr)
        current_S = S[day]

        # --- vesting day --------------------------------------------------
        if (not vested) and (day >= Tv_days):
            vested = True

        # --- quit event ---------------------------------------------------
        if np.random.rand() < quit_p:
            # before vesting  → lapse
            if not vested:
                return 0.0

            win_idx = window_map[day]
            if win_idx == -1:        # quit outside any window  → lapse
                return 0.0

            # quit inside a window
            K_quit = strikes[win_idx]
            return df * max(current_S - K_quit, 0.0)

        # --- voluntary early exercise ------------------------------------
        if vested:
            win_idx = window_map[day]
            if win_idx != -1:        # only possible inside a window
                K_vol = strikes[win_idx]
                if current_S >= M * K_vol:
                    return df * (current_S - K_vol)

    # 4. Reached final day (expiry) ---------------------------------------
    K_expiry = strikes[-1]
    df_exp   = np.exp(-r * final_day * dt)
    return df_exp * max(S[final_day] - K_expiry, 0.0)

# ──────────────────────────────────────────────────────────────────────
# 2.  Core model class
# ──────────────────────────────────────────────────────────────────────
class IESOMonteCarlo:
    """
    Numba-accelerated Indonesian ESO Monte-Carlo engine.
    """
    # ----- constructor ---------------------------------------------------
    def __init__(
        self,
        S0: float              = 5_000.0,   # initial stock price       
        r : float              = 0.05,      # risk-free interest rate   (per year, continouus)
        q : float              = 0.025,     # dividend yield            (per year, continuous)
        sigma: float           = 0.30,      # volatility                (per year, continous)
        lam:  float            = 0.06,      # probability of exits λ    (0..1 assumes Poisson process --> time-to-exit assumed exponential)
        M: float               = 1.0,       # trigger S ≥ M·K for voluntary exercise
        maturity_years: float  = 5.0,       # option life               (calendar year)
        vesting_years:  float  = 1.0,       # vesting period            (calendar year)
        seed: Optional[int]    = None,      # for experiments purpose, can set seed to fixed integer such that experiments are replicable
        debug: bool            = False,     # if True, will print process throughout
    ):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.lam = lam
        self.M = M
        self.q = q
        self.T = maturity_years
        self.Tv = vesting_years
        self.debug        = debug

        # time grid
        self.n_days   = int(round(self.T * DAYS_PER_YEAR))
        self.dt       = 1.0 / DAYS_PER_YEAR
        self.quit_p   = 1.0 - np.exp(-self.lam * self.dt)
        self.Tv_days = int(round(self.Tv * DAYS_PER_YEAR))

        # execution windows
        self.windows  = self._build_windows()
        if not self.windows:
            raise ValueError("No execution windows could be built.")
        self.final_day = self.windows[-1][-1]

        # convert window info into NumPy forms friendly to Numba
        starts = [w[0] for w in self.windows]
        ends   = [w[-1] for w in self.windows]
        self.window_start = np.asarray(starts, dtype=np.int64)
        self.window_end   = np.asarray(ends,   dtype=np.int64)

        # map each calendar day → window index (-1 if none)
        self.window_map = np.full(self.final_day + 1, -1, dtype=np.int64)
        for idx, w in enumerate(self.windows):
            self.window_map[w[0]:w[-1] + 1] = idx

        

        if seed is not None:
            np.random.seed(seed)
        if self.debug:
            self._print_calendar()

    # ----- public pricing routine ---------------------------------------
    def price(
        self,
        n_paths:   int   = 250_000,         # Number of Brownian path simulated
        conf:      float = 0.95,            # Confidence Interval
        antithetic: bool = True,            # Whether to use Antithetic Sample (if generate Z~Normal, then the next simulation generates -Z)
        verbose:    bool = True,            # verbose terminal output for debugging
    ) -> Tuple[float, Tuple[float, float]]:
        """
        This is the primary function to call to get option price given parameters set in Constructor IESOMonteCarlo(). 
        Monte-Carlo price with identical semantics to the reference code.
        Heavy inner loops are executed by Numba.
        """
        if antithetic and n_paths % 2:
            n_paths += 1  # make even

        payoffs = np.empty(n_paths, dtype=np.float64)
        half    = n_paths // 2 if antithetic else n_paths

        for i in range(half):
            z = np.random.normal(size=self.final_day)  # fresh Gaussian draws

            p  = _single_path_jit(
                    z, self.S0, self.r, self.q, self.sigma,
                    self.lam, self.M, self.dt, self.quit_p,
                    self.final_day,
                    self.window_start, self.window_end, self.window_map,
                    self.Tv_days)

            if antithetic:
                p2 = _single_path_jit(
                        -z, self.S0, self.r, self.q, self.sigma,
                        self.lam, self.M, self.dt, self.quit_p,
                        self.final_day,
                        self.window_start, self.window_end, self.window_map,
                        self.Tv_days)
                payoffs[2*i], payoffs[2*i+1] = p, p2
            else:
                payoffs[i] = p

            if verbose and (i + 1) % 256 == 0:
                done = 2*(i+1) if antithetic else i+1
                pct  = 100 * done / n_paths
                print(f"\r{done:,}/{n_paths:,} paths  [{pct:5.1f}%]", end='', flush=True)

        if verbose:
            print()

        mean = payoffs.mean()
        se   = payoffs.std(ddof=1) / np.sqrt(n_paths)
        z_   = norm.ppf(0.5 + conf/2)

        return round(mean, 4), (round(mean - z_*se, 4), round(mean + z_*se, 4))

    # ----- helpers (unchanged logic) ------------------------------------
    def _build_windows(self):
        """Build execution windows according to Indonesian ESO rules"""
        windows = []
        
        # Vesting ends at day index (converted to 0-indexed)
        vest_end_day = int(round(self.Tv * DAYS_PER_YEAR))
        
        # Total days in option life
        total_days = int(round(self.T * DAYS_PER_YEAR))
        
        # Indonesian ESO: execution periods twice per year, 6 months apart
        # Starting from month 13 (after 1-year vesting), then every 6 months
        
        months_after_vesting = []
        current_month = 13  # First execution in month 13 (1-indexed)
        
        while current_month <= self.T * 12:  # Convert years to months
            months_after_vesting.append(current_month)
            current_month += 6  # Every 6 months
        
        for month in months_after_vesting:
            # Convert month to day index (0-indexed)
            # Month 13 = days 300-324 (0-indexed: 300-324)
            start_day = (month - 1) * 25  # 0-indexed
            end_day = start_day + 24      # 25-day window
            
            # Ensure we don't go beyond option maturity
            if start_day < total_days:
                end_day = min(end_day, total_days - 1)
                if start_day <= end_day:
                    windows.append(np.arange(start_day, end_day + 1))
        
        return windows

    def _print_calendar(self):
        print("=== CALENDAR ===")
        print(f"Nominal option life:    {self.n_days} days")
        print(f"Vesting period:         {self.Tv_days} days")
        for i, w in enumerate(self.windows, 1):
            print(f"Window {i:2}: Day {w[0]:4} – {w[-1]:4}")
        print(f"Simulation ends on day: {self.final_day}")
        print(f"Daily quit probability: {self.quit_p:.6f}\n")

# ──────────────────────────────────────────────────────────────────────
# 3. Quick numerical smoke test
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running IESO-1 (13 mo maturity)…")
    eso1 = IESOMonteCarlo(maturity_years=13/12, vesting_years=1.0,
                          S0=5000, r=0.05, q=0.025, sigma=0.30,
                          lam=0.06, M=1.0, seed=7, debug=True)
    p1, ci1 = eso1.price(n_paths=250_000, verbose=True)
    print(f"\nIESO-1  Price: {p1}   95% CI: {ci1[0]}, {ci1[1]}")
    print("-"*50)
    print(eso1._build_windows())

    print("Running IESO-2 (19 mo maturity)…")
    eso2 = IESOMonteCarlo(maturity_years=19/12, vesting_years=1.0,
                          S0=5000, r=0.05, q=0.025, sigma=0.30,
                          lam=0.06, M=1.0, seed=7, debug=False)
    p2, ci2 = eso2.price(n_paths=250_000, verbose=True)
    print(f"IESO-2  Price: {p2}   95% CI: {ci2}")
    print("-"*50)
    print(eso2._build_windows())

    print("Running IESO-8 (5 yr maturity)…")
    eso8 = IESOMonteCarlo(maturity_years=5.0, vesting_years=1.0,
                          S0=5000, r=0.05, q=0.025, sigma=0.30,
                          lam=0.06, M=1.0, seed=7, debug=False)
    p8, ci8 = eso8.price(n_paths=250_000, verbose=True)
    print(f"IESO-8  Price: {p8}   95% CI: {ci8}")
    print("-"*50)
    print(eso8._build_windows())
