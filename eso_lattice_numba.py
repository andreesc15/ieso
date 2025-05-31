#!/usr/bin/env python3
# Fixed Indonesian ESO Lattice Model
# Based on "Pricing Employee Stock Options with an Asian Style Using a Modified Binomial Method" by Chendra et. al. 2022
# Written by Andree Sulistio Chandra (20824003) and Aya Sofia (20124015)
# ----------------------------------------------------------------------
import numpy as np
from math import exp, sqrt
from numba import njit, prange
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────
# 0. CONSTANTS
# ──────────────────────────────────────────────────────────────────────

STRIKE_WINDOW = 25      # 25-day averaging window
DAYS_PER_YEAR = 300   # 300 trading days per year (25 days/month × 12 months)

# ──────────────────────────────────────────────────────────────────────
# 1. JIT-compiled single-path simulator
# ──────────────────────────────────────────────────────────────────────

@njit(parallel=True, fastmath=True)
def sweep_level(i: int,
                u: float, d: float, p_rn: float,
                r: float, dt: float, prob_survive: float,
                M_mult: float,
                win_today: bool, vested: bool,
                strike_multiplier: float,  # 0.9 for Indonesian ESO
                preN, avgN, valN,  # Next level data
                preH, avgH, valH): # Current level data (output)
    
    disc = exp(-r * dt)
    prob_leave = 1.0 - prob_survive
    
    for j in prange(i + 1):
        loH, hiH = preH[j], preH[j + 1]
        loU, hiU = preN[j + 1], preN[j + 2] 
        loD, hiD = preN[j], preN[j + 1]
        
        Au, Vu = avgN[loU:hiU], valN[loU:hiU]
        Ad, Vd = avgN[loD:hiD], valN[loD:hiD]
        
        s_ij_div_s0 = u**j * d**(i - j)
        
        for k_idx in range(loH, hiH):
            avg_div_s0 = avgH[k_idx]
            strike_scaled = strike_multiplier * avg_div_s0
            
            # Find continuation values
            ku = min(np.searchsorted(Au, avg_div_s0), Au.size - 1)
            kd = min(np.searchsorted(Ad, avg_div_s0), Ad.size - 1)
            expected_future_val = disc * (p_rn * Vu[ku] + (1.0 - p_rn) * Vd[kd])
            
            payoff_now = s_ij_div_s0 - strike_scaled
            
            if not vested:
                # Before vesting: only continuation value with survival probability
                valH[k_idx] = prob_survive * expected_future_val
            else:
                # After vesting
                if win_today:
                    # In execution window
                    if s_ij_div_s0 >= M_mult * strike_scaled:
                        # Early exercise due to barrier
                        valH[k_idx] = payoff_now
                    else:
                        # No early exercise, but account for potential departure
                        exercise_if_leave = prob_leave * max(payoff_now, 0.0)
                        continue_if_survive = prob_survive * expected_future_val
                        valH[k_idx] = exercise_if_leave + continue_if_survive
                else:
                    # Outside execution window, only continuation with survival
                    valH[k_idx] = prob_survive * expected_future_val

# ──────────────────────────────────────────────────────────────────────
# 2.  Core model class
# ──────────────────────────────────────────────────────────────────────

class IESOLattice:
    """
    Numba-accelerated Indonesian ESO Lattice engine.
    """
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
        debug: bool            = False,     # if True, will print process throughout
        print_freq=50                       # steps count to be printed for debugging purpose
    ):
        
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.lam = lam
        self.M = M
        self.T = maturity_years
        self.Tv = vesting_years
        
        # Time step
        self.dt = 1.0 / DAYS_PER_YEAR
        
        # CRR parameters
        self.u = exp(sigma * sqrt(self.dt))
        self.d = 1.0 / self.u
        
        exp_r_dt = exp((r-q) * self.dt)
        self.p = (exp_r_dt - self.d) / (self.u - self.d)
        
        if not (0 <= self.p <= 1):
            print(f"Warning: Risk-neutral probability p={self.p} is outside [0,1]")
        
        # Employee survival probability
        self.survive = exp(-lam * self.dt)
        
        # Build execution windows
        self.windows = self._build_execution_windows()
        self.N = int(round(maturity_years * DAYS_PER_YEAR)) - 1  # Last day index (0-indexed)
        
        self.print_freq = max(print_freq, 1)
        
        if debug:
            self._print_debug_info()
    # ----- helper functions, to be declared before price() -------------------------------
    def _build_execution_windows(self):
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
    
    def _is_execution_day(self, day_idx):
        """Check if given day is in any execution window"""
        for window in self.windows:
            if day_idx in window:
                return True
        return False
    # ----- public pricing routine ---------------------------------------  
    def price(self):
        """
        This is the primary function to call to get option price given parameters set in Constructor IESOLattice() which follows Modified Binomial Method.
        Heavy inner loops are executed by Numba.
        """
        if self.N < 0:
            return 0.0
        
        # Initialize terminal condition
        avgN, valN, preN = self._compute_level(self.N, terminal=True)
        
        # Backward induction
        for i in range(self.N - 1, -1, -1):
            if (self.N - i) % self.print_freq == 0:
                pct = 100 * (self.N - i) / self.N if self.N > 0 else 100.0
                print(f"Processing day {i:4d} [{pct:5.1f}%]", flush=True)
            
            # Check if current day is in execution window
            win_today = self._is_execution_day(i)
            
            # Check if vested
            vest_end_day = int(round(self.Tv * DAYS_PER_YEAR))
            vested = i >= vest_end_day
            
            # Compute current level
            avgH, valH, preH = self._compute_level(i, terminal=False)
            
            # Sweep current level
            sweep_level(i, self.u, self.d, self.p,
                       self.r, self.dt, self.survive, self.M,
                       win_today, vested, 0.9,  # 0.9 multiplier for Indonesian ESO
                       preN, avgN, valN,
                       preH, avgH, valH)
            
            # Update for next iteration
            avgN, valN, preN = avgH, valH, preH
        
        return round(valN[0] * self.S0, 4)
    
    def _compute_level(self, i, terminal=False):
        """Compute representative averages for level i"""
        # Number of nodes at level i
        num_nodes = i + 1
        
        # Compute sizes for each node (number of representative averages)
        sizes = np.zeros(num_nodes, dtype=np.int64)
        
        for j in range(num_nodes):
            if i < STRIKE_WINDOW - 1:
                # Not enough history for full window
                sizes[j] = 1
            else:
                # Use paper's methodology for representative averages
                # Approximate number of distinct averages at node (i,j)
                max_ups_in_window = min(j, STRIKE_WINDOW)
                max_downs_in_window = min(i - j, STRIKE_WINDOW)
                # This gives rough estimate of distinct paths through window
                sizes[j] = max(1, max_ups_in_window * max_downs_in_window // 4 + 1)
        
        # Build prefix sum array
        prefix = np.zeros(num_nodes + 1, dtype=np.int64)
        np.cumsum(sizes, out=prefix[1:])
        
        total_size = prefix[-1]
        avg_array = np.zeros(total_size)
        val_array = np.zeros(total_size)
        
        # Fill arrays for each node
        for j in range(num_nodes):
            lo, hi = prefix[j], prefix[j + 1]
            
            if i < STRIKE_WINDOW - 1:
                # Current stock price as average
                s_ij_div_s0 = self.u**j * self.d**(i - j)
                avg_array[lo] = s_ij_div_s0
            else:
                # Compute min/max representative averages
                amin, amax = self._compute_avg_bounds(i, j)
                
                if sizes[j] == 1:
                    avg_array[lo] = (amin + amax) / 2.0
                else:
                    avg_array[lo:hi] = np.linspace(amin, amax, sizes[j])
                    # since k = 0,...,j(1-j) then it's an evenly spaced array. Easier to use linspace.
            
            if terminal:
                # Terminal payoff: max(S - 0.9*A, 0)
                s_N_div_s0 = self.u**j * self.d**(i - j)
                strike_scaled = 0.9 * avg_array[lo:hi]
                val_array[lo:hi] = np.maximum(s_N_div_s0 - strike_scaled, 0.0)
        
        return avg_array, val_array, prefix
    
    def _compute_avg_bounds(self, i, j):
        """Compute min/max representative averages using paper's method"""
        # For node (i,j), find paths that give min/max average over last STRIKE_WINDOW steps
        min_ups = max(0, j - (i - STRIKE_WINDOW + 1))  # Can't have more ups than in total path to (i,j)
        max_ups = min(j, STRIKE_WINDOW)  # Can't exceed total ups to reach (i,j) or window size
        
        # For minimum average: minimize ups in window (more downs = lower prices)
        ups_for_min = min_ups
        downs_for_min = STRIKE_WINDOW - ups_for_min
        
        # For maximum average: maximize ups in window  
        ups_for_max = max_ups
        downs_for_max = STRIKE_WINDOW - ups_for_max
        
        # Calculate minimum average
        if ups_for_min == 0:
            sum_min = STRIKE_WINDOW * self.d * (1 - self.d**STRIKE_WINDOW) / (1 - self.d) if abs(self.d - 1) > 1e-12 else STRIKE_WINDOW
        else:
            # Sum of geometric series for downs, then ups
            if downs_for_min > 0:
                sum_downs = self.d * (1 - self.d**downs_for_min) / (1 - self.d) if abs(self.d - 1) > 1e-12 else downs_for_min
            else:
                sum_downs = 0
            
            price_after_downs = self.d**downs_for_min
            if ups_for_min > 0:
                sum_ups = price_after_downs * self.u * (self.u**ups_for_min - 1) / (self.u - 1) if abs(self.u - 1) > 1e-12 else price_after_downs * ups_for_min
            else:
                sum_ups = 0
            
            sum_min = sum_downs + sum_ups
        
        amin = sum_min / STRIKE_WINDOW
        
        # Calculate maximum average  
        if ups_for_max == 0:
            sum_max = STRIKE_WINDOW * self.d * (1 - self.d**STRIKE_WINDOW) / (1 - self.d) if abs(self.d - 1) > 1e-12 else STRIKE_WINDOW
        else:
            # Sum ups first, then downs
            if ups_for_max > 0:
                sum_ups = self.u * (self.u**ups_for_max - 1) / (self.u - 1) if abs(self.u - 1) > 1e-12 else ups_for_max
            else:
                sum_ups = 0
            
            price_after_ups = self.u**ups_for_max  
            if downs_for_max > 0:
                sum_downs = price_after_ups * self.d * (1 - self.d**downs_for_max) / (1 - self.d) if abs(self.d - 1) > 1e-12 else price_after_ups * downs_for_max
            else:
                sum_downs = 0
            
            sum_max = sum_ups + sum_downs
        
        amax = sum_max / STRIKE_WINDOW
        
        # Ensure proper ordering
        if amin > amax:
            amin, amax = amax, amin
        
        return amin, amax
    
    def _print_debug_info(self):
        """Print debug information about the ESO structure"""
        print("=== Indonesian ESO Debug Info ===")
        print(f"Maturity: {self.T} years ({int(self.T * DAYS_PER_YEAR)} days)")
        print(f"Vesting: {self.Tv} years ({int(self.Tv * DAYS_PER_YEAR)} days)")
        print(f"Total lattice steps: {self.N + 1}")
        print(f"u={self.u:.6f}, d={self.d:.6f}, p={self.p:.6f}")
        print(f"Survival prob per day: {self.survive:.6f}")
        
        print("\n=== Execution Windows (0-indexed days) ===")
        for i, window in enumerate(self.windows):
            print(f"Window {i+1}: days {window[0]} to {window[-1]} ({len(window)} days)")


# ──────────────────────────────────────────────────────────────────────
# 3. Quick numerical smoke test
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running IESO-1 (13 mo maturity)…")
    eso1 = IESOLattice(maturity_years=13/12, vesting_years=1.0,
                          S0=5000, r=0.05, q=0.025, sigma=0.30,
                          lam=0.06, M=1.0, debug=True)
    p1 = eso1.price()
    print(f"\nIESO-1  Price: {p1}")
    print("-"*50)

    print("Running IESO-2 (19 mo maturity)…")
    eso2 = IESOLattice(maturity_years=19/12, vesting_years=1.0,
                          S0=5000, r=0.05, q=0.0025, sigma=0.30,
                          lam=0.06, M=1.0, debug=False)
    p2 = eso2.price()
    print(f"IESO-2  Price: {p2}")
    print("-"*50)

    print("Running IESO-8 (5 yr maturity)…")
    eso8 = IESOLattice(maturity_years=5.0, vesting_years=1.0,
                          S0=5000, r=0.05, q=0.0025, sigma=0.30,
                          lam=0.06, M=1.0, debug=False)
    p8 = eso8.price()
    print(f"IESO-8  Price: {p8}")
    print("-"*50)
