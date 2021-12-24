import numpy as np
import pandas as pd
import swifter

"""1.	Simple risk parity: (i) calculate N-day rolling window standard deviation Omega of daily returns of individual 
assets (you can set N to 252 for the time being). (ii) calculate normalized asset return. Ret_t*Min(Maxlev, 
Voltarget/Omega_t-1). Let's set voltarget to 30% and make sure omega is annualized. Maxlev is specified in the 
asset_spec file too. (iii) Calculate simple average of Ret_t across all assets as the portfolio return. 2. 3.	Asset 
class risk parity: the first two steps are the same as 1. (iii) calculate simple average of Ret_t across all assets in 
the same asset class to form 4 asset class portfolios. (iv) calculate the average return across the four asset class 
portfolios. 4. 5.	Momentum risk parity: the first two steps are the same as 1. (iii) calculate N-day rolling window 
cumulative return Mom_t-1. (iv) calculate simple average of those assets with Mom_t-1>0. 4.	Momentum asset class risk 
parity: Combine 2 and 3. 

"""

N = 252
VOL_TARGET = 0.3


def adjusted_volume_weighted(matrix: pd.DataFrame,
                             start_date: str = '2013-1-1',
                             end_date: str = '2025-1-1',
                             lag: int = 1) -> pd.DataFrame:
    ret_df = matrix.xs("%change", level='Features')

    mask = (ret_df.index > start_date) & (ret_df.index < end_date)

    ret_df = ret_df[mask]

    rolling_std_df = ret_df.shift(lag).swifter.rolling(N, min_periods=2).apply(np.nanstd) * np.sqrt(
        N)  # shift 1 to avoid look ahead bias

    rolling_lev = VOL_TARGET / rolling_std_df

    volume_df = matrix.xs("Vol", level='Features').shift(1)[mask]

    adjusted_volume = volume_df.divide(rolling_lev, axis=0)

    scaling_factor_df = (1 / adjusted_volume.sum(axis=1)).replace([np.inf, -np.inf], np.nan)

    weights = volume_df.multiply(scaling_factor_df, axis=0)

    return weights


if __name__ == "__main__":
    data_dir = './data/consolidated_table_wo_features'
    xarray = pd.read_pickle(data_dir)
    print(adjusted_volume_weighted(xarray))
