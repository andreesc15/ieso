#!/usr/bin/env python3
# ieso_mc.py  –  Reference Monte-Carlo engine for Indonesian ESOs
# (Chendra et al., Int. J. Appl. Math. 35-2 2022 233-247)
# ---------------------------------------------------------------------
from __future__ import annotations
import numpy as np
from scipy.stats import norm
from typing      import List, Dict, Tuple, Optional

# ---------------------------------------------------------------------
#  0.  CONSTANTS that are not parameter
# ---------------------------------------------------------------------
DAYS_PER_MONTH  = 25
MONTHS_PER_YEAR = 12
DAYS_PER_YEAR   = DAYS_PER_MONTH * MONTHS_PER_YEAR    # = 300

# ---------------------------------------------------------------------
#  1.  Core model class
# ---------------------------------------------------------------------
class IESOMonteCarlo:
    """
    Sets Monte Carlo compute engine as a class which contains all necessary function.
    Parameters are set via constructor. Default value of this constructor matches paper, but can be passed through parameter passing when constructing this class.
    """
    # ----- constructor -------------------------------------------------
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
        # --- set self parameter from constructor ---
        self.S0, self.r, self.q, self.sigma = S0, r, q, sigma
        self.lam, self.M  = lam, M
        self.T,  self.Tv  = maturity_years, vesting_years
        self.debug        = debug

        # --- set calculated parameters from constructor ---
        self.n_days = int(round(self.T * DAYS_PER_YEAR))      # Total calendar days = T (year) * days per year (300)
        self.dt     = 1.0 / DAYS_PER_YEAR                     # Time step in days 
        self.quit_p = 1.0 - np.exp(-self.lam * self.dt)       # the probability of said person quitting at time t follows exponential distribution since employee quitting is Poisson process.

        # ––– generate execution window on calendar days & final day of ESOP simulation. Wrapped with error handling to reject model should there be no valid exercise window based on maturity time.
        self.windows = self._build_windows()
        if not self.windows: # Ensure windows were actually created
             raise ValueError("No execution windows could be built with the given maturity and vesting periods.")
        self.final_day = self.windows[-1][-1]                 # ESOP simulation ends HERE

        if seed is not None:
            np.random.seed(seed)

        if self.debug:
            self._print_calendar()

    # =================================================================
    # 2. Monte-Carlo driver
    # =================================================================
    def price(    
        self,
        n_paths:   int   = 250_000,         # Number of Brownian path simulated
        conf:      float = 0.95,            # Confidence Interval
        antithetic: bool = True,            # Whether to use Antithetic Sample (if generate Z~Normal, then the next simulation generates -Z)
        verbose:    bool = True,            # verbose terminal output for debugging
    ) -> Tuple[float, Tuple[float, float]]:
        """
        This is the primary function to call to get option price given parameters set in Constructor IESOMonteCarlo(). 
        """
        actual_n_paths = n_paths
        if antithetic and n_paths % 2 != 0: # Ensure n_paths is even for antithetic
            actual_n_paths = n_paths + 1
            if verbose:
                print(f"Adjusting n_paths to {actual_n_paths} for antithetic sampling.")

        half = actual_n_paths // 2 if antithetic else actual_n_paths
        pay  = np.empty(actual_n_paths)

        for i in range(half):
            # Generate random numbers for the entire effective life of the option
            z = np.random.normal(size=self.final_day) 
            
            # Pass self.debug combined with i==0 to trace only the first path if debug is on
            trace_this_path = self.debug and (i == 0)

            if antithetic:
                pay[2*i  ] = self._single_path(z, trace_this_path) 
                pay[2*i+1] = self._single_path(-z, False) # Don't trace the antithetic path by default
            else:
                pay[i]     = self._single_path(z, trace_this_path)

            if verbose and (i+1) % 256 == 0:
                done = 2*(i+1) if antithetic else i+1
                pct  = 100*done/actual_n_paths
                print(f"\r{done:,}/{actual_n_paths:,} paths  [{pct:5.1f}%]", end='', flush=True)

        if verbose:  print() # Newline after progress indicator

        m  = pay.mean()
        se = pay.std(ddof=1)/np.sqrt(actual_n_paths)
        z_ = norm.ppf(0.5 + conf/2)
        return round(m,4), (round(m-z_*se,4), round(m+z_*se,4))

    # =================================================================
    # 3. One-path simulator
    # =================================================================
    def _single_path(self, z:np.ndarray, trace:bool) -> float:
        """Simulate ONE GBM + behaviour path."""
        # S is stock price array, S[0] is S0, S[k] is price at end of day k
        # Array size needs to accommodate indices up to self.final_day
        S = np.empty(self.final_day + 1) 
        S[0] = self.S0
        mu   = (self.r - self.q - 0.5*self.sigma**2) * self.dt
        vol  = self.sigma * np.sqrt(self.dt)
        
        # Generate stock prices up to self.final_day
        # z is 0-indexed, z[k-1] drives price for S[k] when k starts from 1
        for day_k_idx in range(1, self.final_day + 1): # S_indices from 1 to self.final_day
            S[day_k_idx] = S[day_k_idx-1] * np.exp(mu + vol * z[day_k_idx-1]) # z_indices from 0 to self.final_day-1

        strikes = self._calc_strikes(S, trace) # Path-dependent strikes
        vested  = False

        # Loop through each day of the option's effective life
        for day_num in range(1, self.final_day + 1): # day_num is 1-based (Day 1, Day 2, ..., self.final_day)
            current_S = S[day_num] # Stock price at the end of day_num
            time_in_years = day_num * self.dt
            df = np.exp(-self.r * time_in_years) # Discount factor for this day

            # --- Vesting ---
            if not vested and time_in_years >= self.Tv:
                vested = True
                if trace: 
                    print(f"[vest @ day {day_num}]")

            # --- Quit Event ---
            if np.random.rand() < self.quit_p:
                if not vested: # Quit before vesting
                    if trace: print(f"[quit day {day_num} PRE-VESTING – lapse]")
                    return 0.0
                else: # Quit after vesting
                    K_for_quit = self._strike_today(strikes, day_num) 
                    
                    if K_for_quit is not None: # Quit day is IN an execution window
                        if current_S > K_for_quit: # Exercise if S > K for quitters
                            payout = current_S - K_for_quit
                            if trace: print(f"[quit+exercise day {day_num} (in window)] S={current_S:.2f} K={K_for_quit:.2f} Payout={payout:.2f} PV={df*payout:.2f}")
                            return df * payout
                        else: # Vested, quit in window, but not ITM -> lapse
                            if trace: print(f"[quit day {day_num} (in window, not ITM: S={current_S:.2f} <= K={K_for_quit:.2f}) – lapse]")
                            return 0.0
                    else: # Vested, but quit day is OUTSIDE any execution window -> option lapses
                        if trace: print(f"[quit day {day_num} (OUTSIDE window) – lapse]")
                        return 0.0
            
            # --- Voluntary Early Exercise (if not quit today) ---
            if vested: # Must be vested to consider voluntary exercise
                if self._in_window(day_num): # Check if current day_num is within any execution window
                    # Only if in a window, get the window index and proceed
                    window_idx = self._which_window(day_num) 
                    K_voluntary = strikes[window_idx]
                    
                    # Voluntary exercise condition S >= M*K
                    if current_S >= self.M * K_voluntary: 
                        payout = current_S - K_voluntary
                        if trace:
                            print(f"[early exercise day {day_num}] win={window_idx+1} S={current_S:.2f} M*K={self.M*K_voluntary:.2f} K={K_voluntary:.2f} Payout={payout:.2f} PV={df*payout:.2f}")
                        return df * payout
        
        # --- Reached Final Day (self.final_day) without prior exercise/quit ---
        K_at_expiry = strikes[len(self.windows) - 1] 
        S_at_expiry = S[self.final_day]
        
        payoff_at_expiry = max(S_at_expiry - K_at_expiry, 0.0)
        df_expiry = np.exp(-self.r * self.final_day * self.dt)

        if trace:
            print(f"[expiry day {self.final_day}] S={S_at_expiry:.2f} K={K_at_expiry:.2f} Payout={payoff_at_expiry:.2f} PV={df_expiry*payoff_at_expiry:.2f}")
        
        return df_expiry * payoff_at_expiry

    # =================================================================
    # 4. Helpers
    # =================================================================
    def _build_windows(self) -> List[np.ndarray]:
        wins = []
        start_day_of_vested_period = int(round(self.Tv * DAYS_PER_YEAR)) + 1 
        
        current_window_start_day = start_day_of_vested_period
        while True:
            window_end_day = current_window_start_day + DAYS_PER_MONTH - 1
            if window_end_day > self.n_days: # self.n_days is based on overall maturity_years
                break 
            wins.append(np.arange(current_window_start_day, current_window_start_day + DAYS_PER_MONTH))
            current_window_start_day += 6 * DAYS_PER_MONTH 
            
        return wins

    def _calc_strikes(self, S:np.ndarray, trace:bool) -> Dict[int,float]:
        K_dict = {} 
        for i, w_days in enumerate(self.windows):
            first_day_of_window = w_days[0]
            avg_period_start_idx = first_day_of_window - DAYS_PER_MONTH
            avg_period_end_idx   = first_day_of_window - 1 

            if avg_period_start_idx < 0:
                 # This case should be rare if vesting period is reasonably long (e.g. >= 1 month)
                 # and implies an issue with parameter setup or window logic for very short vestings.
                raise ValueError(f"Strike averaging period (days {avg_period_start_idx}-{avg_period_end_idx}) for window {i+1} (starts day {first_day_of_window}) is invalid. Vesting period might be too short.")

            # Ensure indices are valid for S array (S is 0-indexed for S0, so S[k] is price at end of day k)
            # avg_period_start_idx and avg_period_end_idx are 1-based day numbers for S array access
            avg_price = S[avg_period_start_idx : avg_period_end_idx + 1].mean() # Slice S using day numbers as indices
            K_dict[i] = 0.9 * avg_price
            if trace:
                print(f"[strike win {i+1}] starts day {first_day_of_window}. Averaging S[{avg_period_start_idx}]..S[{avg_period_end_idx}]. Avg_S={avg_price:.2f}. K={K_dict[i]:.2f}")
        return K_dict

    def _in_window(self, day:int) -> bool:
        return any(day in w for w in self.windows)

    def _which_window(self, day:int) -> int:
        for i,w in enumerate(self.windows):
            if day in w: return i
        raise ValueError(f"Day {day} is not in any execution window, but _which_window was called.") # Should be guarded

    def _strike_today(self, K_dict:Dict[int,float], day:int) -> float|None:
        if not self._in_window(day): # Guard added for safety, though _which_window usually called after _in_window
            return None
        window_idx = self._which_window(day)
        return K_dict[window_idx]

    def _print_calendar(self):
        print("=== CALENDAR ===")
        print(f"Nominal option life (maturity_years * DAYS_PER_YEAR): {self.n_days} days")
        print(f"Vesting period: {self.Tv*DAYS_PER_YEAR:.0f} days (ends after day {int(round(self.Tv*DAYS_PER_YEAR))})")
        if not self.windows:
            print("No execution windows defined.")
            return
        for i,w in enumerate(self.windows,1):
            print(f"Window {i:>2}: Day {w[0]:>4} – {w[-1]:>4} (Strike K{i} applies)")
        if self.windows: # Check if windows list is not empty before accessing self.final_day
             print(f"Effective option simulation ENDS on day {self.final_day} (last day of Window {len(self.windows)})")
        print(f"Daily quit probability: {self.quit_p:.6f}")
        print()

# ---------------------------------------------------------------------
# 5.  Quick numerical smoke test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("Running IESO-1 (13 months total maturity, 1 year vesting)...")
    eso1 = IESOMonteCarlo(maturity_years=13/12, vesting_years=1.0, S0=5000, r=0.05, q=0.025, sigma=0.30, lam=0.06, M=1.0, seed=7, debug=True)
    # Using fewer paths for the debug run to make it faster and easier to see trace output
    p1, ci1 = eso1.price(n_paths=10, verbose=True) 
    print(f"\nIESO-1  Price: {p1}  95% CI: {ci1}")
    print("-" * 50)

    print("\nRunning IESO-2 (19 months total maturity, 1 year vesting)...")
    eso2 = IESOMonteCarlo(maturity_years=19/12, vesting_years=1.0, S0=5000, r=0.05, q=0.025, sigma=0.30, lam=0.06, M=1.0, seed=7, debug=False)
    p2, ci2 = eso2.price(n_paths=100, verbose=True)
    print(f"IESO-2  Price: {p2}  95% CI: {ci2}")
    print("-" * 50)

    print("\nRunning IESO-8 (5 years total maturity, 1 year vesting)...")
    eso8 = IESOMonteCarlo(maturity_years=5.0, vesting_years=1.0, S0=5000, r=0.05, q=0.025, sigma=0.30, lam=0.06, M=1.0, seed=7, debug=False)
    p8, ci8 = eso8.price(n_paths=250, verbose=True)
    print(f"IESO-8  Price: {p8}  95% CI: {ci8}")
    print("-" * 50)