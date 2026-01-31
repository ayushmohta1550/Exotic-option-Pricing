import numpy as np
import pandas as pd
import io
from scipy.stats import norm
from scipy.optimize import brentq

# --- Calibration Data as a String ---
calibration_text = """CalibIdx,Stock,Type,Strike,Maturity,Price
                        1,DTC,Call,50,1y,52.44
                        2,DTC,Call,50,2y,54.77
                        3,DTC,Call,50,5y,61.23
                        4,DTC,Call,75,1y,28.97
                        5,DTC,Call,75,2y,33.04
                        6,DTC,Call,75,5y,43.47
                        7,DTC,Call,100,1y,10.45
                        8,DTC,Call,100,2y,16.13
                        9,DTC,Call,100,5y,29.14
                        10,DTC,Call,125,1y,2.32
                        11,DTC,Call,125,2y,6.54
                        12,DTC,Call,125,5y,18.82
                        13,DTC,Call,150,1y,0.36
                        14,DTC,Call,150,2y,2.34
                        15,DTC,Call,150,5y,11.89
                        16,DFC,Call,50,1y,52.45
                        17,DFC,Call,50,2y,54.9
                        18,DFC,Call,50,5y,61.87
                        19,DFC,Call,75,1y,29.11
                        20,DFC,Call,75,2y,33.34
                        21,DFC,Call,75,5y,43.99
                        22,DFC,Call,100,1y,10.45
                        23,DFC,Call,100,2y,16.13
                        24,DFC,Call,100,5y,29.14
                        25,DFC,Call,125,1y,2.8
                        26,DFC,Call,125,2y,7.39
                        27,DFC,Call,125,5y,20.15
                        28,DFC,Call,150,1y,1.26
                        29,DFC,Call,150,2y,4.94
                        30,DFC,Call,150,5y,17.46
                        31,DEC,Call,50,1y,52.44
                        32,DEC,Call,50,2y,54.8
                        33,DEC,Call,50,5y,61.42
                        34,DEC,Call,75,1y,29.08
                        35,DEC,Call,75,2y,33.28
                        36,DEC,Call,75,5y,43.88
                        37,DEC,Call,100,1y,10.45
                        38,DEC,Call,100,2y,16.13
                        39,DEC,Call,100,5y,29.14
                        40,DEC,Call,125,1y,1.96
                        41,DEC,Call,125,2y,5.87
                        42,DEC,Call,125,5y,17.74
                        43,DEC,Call,150,1y,0.16
                        44,DEC,Call,150,2y,1.49
                        45,DEC,Call,150,5y,9.7
                        """

# --- Load Calibration Data ---
calibration_df = pd.read_csv(io.StringIO(calibration_text.strip()))

# --- Load Basket Option Data ---
basket_options_text = '''Id,Asset,KnockOut,Maturity,Strike,Type
                            1,Basket,150,2y,50,Call
                            2,Basket,175,2y,50,Call
                            3,Basket,200,2y,50,Call
                            4,Basket,150,5y,50,Call
                            5,Basket,175,5y,50,Call
                            6,Basket,200,5y,50,Call
                            7,Basket,150,2y,100,Call
                            8,Basket,175,2y,100,Call
                            9,Basket,200,2y,100,Call
                            10,Basket,150,5y,100,Call
                            11,Basket,175,5y,100,Call
                            12,Basket,200,5y,100,Call
                            13,Basket,150,2y,125,Call
                            14,Basket,175,2y,125,Call
                            15,Basket,200,2y,125,Call
                            16,Basket,150,5y,125,Call
                            17,Basket,175,5y,125,Call
                            18,Basket,200,5y,125,Call
                            19,Basket,150,2y,75,Put
                            20,Basket,175,2y,75,Put
                            21,Basket,200,2y,75,Put
                            22,Basket,150,5y,75,Put
                            23,Basket,175,5y,75,Put
                            24,Basket,200,5y,75,Put
                            25,Basket,150,2y,100,Put
                            26,Basket,175,2y,100,Put
                            27,Basket,200,2y,100,Put
                            28,Basket,150,5y,100,Put
                            29,Basket,175,5y,100,Put
                            30,Basket,200,5y,100,Put
                            31,Basket,150,2y,125,Put
                            32,Basket,175,2y,125,Put
                            33,Basket,200,2y,125,Put
                            34,Basket,150,5y,125,Put
                            35,Basket,175,5y,125,Put
                            36,Basket,200,5y,125,Put
                            '''
basket_df = pd.read_csv(io.StringIO(basket_options_text))


# --- Define Market Settings ---
initial_spot = 100.0
risk_free_rate = 0.05

# --- Define Asset Correlations and Compute Cholesky Factor ---
correlation_matrix = np.array([
    [1.0, 0.75, 0.5],
    [0.75, 1.0, 0.25],
    [0.5, 0.25, 1.0]
])
cholesky_factor = np.linalg.cholesky(correlation_matrix)

def calc_implied_vol(S, K, T, r, market_price):
    def diff_func(vol):
        if vol < 1e-8:
            price = max(S - K * np.exp(-r * T), 0)
        else:
            d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
            d2 = d1 - vol * np.sqrt(T)
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return price - market_price

    try:
        return brentq(diff_func, 1e-6, 5)
    except Exception:
        return 0.2  # fallback


# --- Build Volatility Surface ---
strike_levels = [50, 75, 100, 125, 150]
maturity_levels = [1, 2, 5]
asset_list = ['DTC', 'DFC', 'DEC']

# 3D grid: Asset index, Maturity index, Strike index
vol_surface = np.zeros((len(asset_list), len(maturity_levels), len(strike_levels)))

for asset_idx, asset in enumerate(asset_list):
    for m_idx, year in enumerate(maturity_levels):
        for s_idx, strike in enumerate(strike_levels):
            calib_row = calibration_df[
                (calibration_df.Stock == asset) &
                (calibration_df.Strike == strike) &
                (calibration_df.Maturity == f"{year}y")
            ]
            if not calib_row.empty:
                market_val = calib_row.iloc[0].Price
                vol = calc_implied_vol(initial_spot, strike, year, risk_free_rate, market_val)
                vol_surface[asset_idx, m_idx, s_idx] = vol
            else:
                vol_surface[asset_idx, m_idx, s_idx] = 0.2

strike_array = np.array(strike_levels)
maturity_array = np.array(maturity_levels)

# --- Price Basket Option with Knockout Feature ---
def compute_basket_option_price(option_row, num_paths=5000, num_steps=150):
    # Parse option parameters
    T_years = float(str(option_row['Maturity']).replace('y', ''))
    option_strike = float(option_row['Strike'])
    knockout_barrier = float(option_row['KnockOut'])
    opt_type = option_row['Type'].lower()
    delta_t = T_years / num_steps

    # Initialize simulation arrays
    asset_prices = np.full((num_paths, 3), initial_spot)
    active_paths = np.ones(num_paths, dtype=bool)

    # Generate random increments for each time step
    rand_normals = np.random.normal(size=(num_steps, num_paths, 3))
    # Create correlated increments using the Cholesky factor
    correlated_increments = np.matmul(rand_normals, cholesky_factor.T) * np.sqrt(delta_t)

    # Time stepping for simulation
    for step in range(num_steps):
        remaining_time = T_years - step * delta_t
        # Find closest maturity index in our volatility grid for remaining time
        maturity_idx = np.abs(maturity_array - remaining_time).argmin()

        for asset in range(3):
            current_prices = asset_prices[active_paths, asset]
            if current_prices.size == 0:
                continue
            # Match strikes via nearest neighbor search for volatility lookup
            nearest_strike_idx = np.abs(strike_array - current_prices[:, None]).argmin(axis=1)
            current_vols = vol_surface[asset, maturity_idx, nearest_strike_idx]
            # Update prices using an exponential Euler step
            asset_prices[active_paths, asset] = current_prices * np.exp(
                (risk_free_rate - 0.5 * current_vols**2) * delta_t +
                current_vols * correlated_increments[step, active_paths, asset]
            )

        # Monitor basket average price for knockout condition
        current_basket = asset_prices[active_paths].mean(axis=1)
        indices = np.where(active_paths)[0]
        active_paths[indices[current_basket >= knockout_barrier]] = False
        if not active_paths.any():
            break

    # Terminal basket value is the average of asset prices per path
    final_basket = asset_prices.mean(axis=1)
    if opt_type == 'call':
        option_payoff = np.maximum(final_basket - option_strike, 0)
    else:
        option_payoff = np.maximum(option_strike - final_basket, 0)
    # Nullify payoffs for knocked out paths
    option_payoff[~active_paths] = 0

    # Discount and average the payoff
    option_value = np.exp(-risk_free_rate * T_years) * np.mean(option_payoff)
    return max(0, round(option_value, 2))


# --- Price Options in Batch ---
def run_batch_pricing(num_sim_paths=5000):
    output = []
    for _, opt_row in basket_df.iterrows():
        val = compute_basket_option_price(opt_row, num_paths=num_sim_paths)
        output.append((opt_row['Id'], val))
    return output

# --- Display Results ---
pricing_results = run_batch_pricing(num_sim_paths=5000)
print("Id,Price")
for opt_id, opt_price in pricing_results:
    print(f"{opt_id},{opt_price}")
