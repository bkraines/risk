import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from collections import OrderedDict
from scipy.optimize import minimize

# Define the full list of tickers
#tickers = ['SPY', 'IEF', 'QQQ', 'IWM', 'EEM']
tickers = [ 'MWTIX', 'SPY', 'IWM', 'MDY', 'RSP', 'QQQ', 'XLK', 'XLI', 'XLF', 'XLC', 'XLE', 'XLY', 'XLB', 'XLV', 'XLU', 'XLP', 'VNQ', 'AIQ', 'ICLN', 'PFF', 'FEZ', 'EEM', 'FXI', 'ASHR',  'LQD', 'HYG', 'LQDH', 'HYGH', 'AGG',  'SHY', 'IEI', 'IEF', 'TLT', 'TIP', 'VTIP', 'AGNC', 'VMBS', 'CMBS', 'EMB', 'EMHY', 'GLD', 'SLV', 'USO', 'DBC', 'UUP', 'FXE', 'FXY' ]
x_tickers=[ 'AGG', 'IEF', 'VMBS', 'IEI', 'LQD', 'TLT', 'TIP', 'SHY', 'EMB']

#************ assert x_tickers is a subset of ticker
# Define the start and end dates
end_date = datetime.today()
start_date = end_date - timedelta(days=30 * 365)

# Download the adjusted closing prices for all tickers
# price_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
price_data = yf.download(tickers, start=start_date, end=end_date)['Close'] # API Changed

# Calculate daily returns - *************************************   fix to account for diffrent bus days carry forward
returns = price_data.pct_change().dropna()

# Generate month-end rebalancing dates using 'ME'
# rebalancing_dates = returns.resample('ME').last().index
rebalancing_dates = returns.resample('M').last().index # ME is deprecated

# Initialize a DataFrame to store portfolio weights in long format
portfolio_weights_long = pd.DataFrame(columns=["portfolio_name", "date", "ticker", "weight"])

# Set half-life for exponential weighting (6 months ~ 126 trading days)
halflife = 126
min_periods = 60  # Minimum number of data points required

# Set the lag parameter (number of business days)
lag = 1  # Adjust as needed call it   ******** execution delay


def construct_fixed_weight_portfolio(returns_up_to_date, available_assets, **kwargs):
    """
    Returns fixed weights for the date that is closest but before the rebalancing date.

    Parameters:
    - returns_up_to_date: DataFrame of historical returns up to the current date.
    - available_assets: List of tickers with sufficient data.
    - kwargs: Additional arguments (e.g., fixed_weights_data).

    Returns:
    - weights_fixed: Series with portfolio weights.
    """
    # Get the rebalancing date from kwargs
    rebalancing_date = kwargs.get('rebalancing_date')
    if rebalancing_date is None:
        raise ValueError("Rebalancing date must be provided in kwargs.")

    # Remove timezone information from rebalancing_date
    rebalancing_date = rebalancing_date.replace(tzinfo=None)

    # Get the fixed weights data
    fixed_weights_df = kwargs.get('fixed_weights_data')
    if fixed_weights_df is None:
        raise ValueError("Fixed weights data must be provided in kwargs.")

    # Find the date in fixed_weights_df that is closest but before the rebalancing date
    available_dates = fixed_weights_df.index[fixed_weights_df.index <= rebalancing_date]
    if available_dates.empty:
        # No weights available before the rebalancing date
        return pd.Series(0.0, index=available_assets)
    else:
        closest_date = available_dates.max()
        weights_row = fixed_weights_df.loc[closest_date]

        # Extract weights for available assets
        weights_fixed = weights_row[available_assets].dropna()

        # Normalize weights to sum to 1 (optional, if weights are not guaranteed to sum to 1)
        total_weight = weights_fixed.sum()
        if total_weight != 0:
            weights_fixed = weights_fixed / total_weight
        else:
            # If total weight is zero, return zero weights
            weights_fixed = pd.Series(0.0, index=available_assets)

        return weights_fixed

def construct_equal_weight_portfolio(returns_up_to_date, available_assets, **kwargs):
    """
    Constructs equal weight portfolio weights over available_assets.
    """
    num_assets = len(available_assets)
    weights_equal = pd.Series(dtype=float)
    if num_assets > 0:
        weights_equal = pd.Series(1 / num_assets, index=available_assets)
    return weights_equal

def construct_inverse_volatility_portfolio(returns_up_to_date, available_assets, **kwargs):
    """
    Constructs inverse volatility portfolio weights over available_assets.
    """
    if len(available_assets) == 0:
        return pd.Series(dtype=float)
    # Compute exponentially weighted standard deviation
    ewm_std = returns_up_to_date[available_assets].ewm(halflife=halflife, min_periods=min_periods).std().iloc[-1]
    # Drop NaNs to avoid division errors
    ewm_std = ewm_std.dropna()
    if ewm_std.empty:
        return pd.Series(dtype=float)
    inv_vol = 1 / ewm_std
    weights_invvol = inv_vol / inv_vol.sum()
    return weights_invvol

def construct_tracking_portfolio(returns_up_to_date, available_assets, **kwargs):
    """
    Constructs a portfolio by minimizing the exponentially weighted standard deviation of
    the difference between the target portfolio returns and the weighted returns of assets,
    with weights summing to one.
    """
    # Ensure the target portfolio returns are provided
    target_portfolio_name = kwargs.get('target_portfolio_name')
    if target_portfolio_name is None:
        raise ValueError("Target portfolio name must be provided in kwargs.")

    if len(available_assets) == 0:
        return pd.Series(dtype=float)

    # Ensure initial_weights are provided
    initial_weights = kwargs.get('initial_weights')
    if initial_weights is None or len(initial_weights) != len(available_assets):
        initial_weights = np.array([1 / len(available_assets)] * len(available_assets))
        initial_weights = np.array([1] + [0] * (len(available_assets) - 1))

    # Get target portfolio returns
    returns_y = returns_up_to_date[target_portfolio_name].dropna()
    # Get asset returns
    returns_x = returns_up_to_date[available_assets]

    # Align the returns
    combined_returns = returns_x.join(returns_y, how='inner')
    returns_x = combined_returns[available_assets]
    returns_y = combined_returns[target_portfolio_name]

    if returns_x.empty or returns_y.empty:
        # Not enough data to compute weights
        return pd.Series(dtype=float)

    # Objective function: Exponentially weighted variance of residuals
    def objective(w):
        w = np.array(w)
        residuals = returns_y - returns_x.dot(w)
        ewm_var = residuals.ewm(halflife=halflife, min_periods=min_periods).var().iloc[-1]
        return ewm_var

    # Constraints: Weights must be non-negative and sum to 1
    bounds = [(0, None) for _ in available_assets]
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

    # Run the optimizer
    result = minimize(
        objective,
        x0=initial_weights,
        bounds=bounds,
        constraints=constraints,
        method='SLSQP',
        options={'ftol': 1e-9, 'disp': False}
    )

    if not result.success:
        # Optimization failed; return equal weights as fallback
        weights = pd.Series(1 / len(available_assets), index=available_assets)
    else:
        weights = pd.Series(result.x, index=available_assets)

    return weights


def construct_tracking_with_penalty(returns_up_to_date, available_assets, **kwargs):
    """
    Constructs a tracking portfolio without the sum(w) = 1 constraint.
    Includes a penalty term in the objective function for deviation of sum(w) from 1.
    """
    from scipy.optimize import minimize

    # Ensure the target portfolio returns are provided
    target_portfolio_name = kwargs.get('target_portfolio_name')
    if target_portfolio_name is None:
        raise ValueError("Target portfolio name must be provided in kwargs.")

    if len(available_assets) == 0:
        return pd.Series(dtype=float)

    # Ensure initial_weights are provided
    initial_weights = kwargs.get('initial_weights')
    if initial_weights is None or len(initial_weights) != len(available_assets):
        initial_weights = np.array([1 / len(available_assets)] * len(available_assets))
    initial_weights = np.array([1] + [0] * (len(available_assets) - 1))

    # Get penalty parameter
    penalty_weight = kwargs.get('penalty_weight', 1e3)  # Default penalty_weight if not provided

    # Get target portfolio returns
    returns_y = returns_up_to_date[target_portfolio_name].dropna()
    # Get asset returns
    returns_x = returns_up_to_date[available_assets]

    # Align the returns
    combined_returns = returns_x.join(returns_y, how='inner')
    returns_x = combined_returns[available_assets]
    returns_y = combined_returns[target_portfolio_name]

    if returns_x.empty or returns_y.empty:
        # Not enough data to compute weights
        return pd.Series(dtype=float)

    # Objective function: Exponentially weighted variance of residuals with penalty
    def objective(w):
        w = np.array(w)
        residuals = returns_y - returns_x.dot(w)
        ewm_var = residuals.ewm(halflife=halflife, min_periods=min_periods).var().iloc[-1]
        penalty = penalty_weight * (np.sum(w) - 1) ** 2
        return ewm_var + penalty

    # Constraints: Weights must be non-negative
    bounds = [(0, None) for _ in available_assets]

    # Run the optimizer
    result = minimize(
        objective,
        x0=initial_weights,
        bounds=bounds,
        method='SLSQP',
        options={'ftol': 1e-9, 'disp': False}
    )

    if not result.success:
        # Optimization failed; return equal weights as fallback
        weights = pd.Series(1 / len(available_assets), index=available_assets)
    else:
        weights = pd.Series(result.x, index=available_assets)

    return weights



# Portfolio Construction




# Map function names to actual functions
portfolio_functions = {
    "EW": construct_equal_weight_portfolio,
    "VW": construct_inverse_volatility_portfolio,
    "TRACK": construct_tracking_portfolio,
    "TRACK_PENALTY": construct_tracking_with_penalty,
    "FIXED": construct_fixed_weight_portfolio,
}

# Prepare the fixed weights data
fixed_weights_data = pd.DataFrame({
    'date': pd.to_datetime(['2015-01-31', '2024-10-31', '2024-06-30']),
    'MWTIX': [1, 0.5, 0.75],
    'SHY': [0, 0.5, 0.1],
    # Add other tickers as needed
})

# Set the date as index for easier lookup
fixed_weights_data.set_index('date', inplace=True)



# Define the portfolios using an OrderedDict
portfolios = OrderedDict({
    "TCW": {
        "function_to_call": "FIXED",
        "ticker_subset": ['MWTIX',  'SHY'],
        "other_options": {
            "fixed_weights_data": fixed_weights_data,
        },
    },
    # Include other portfolios as before
    "Vol_wtd_ptfl": {
        "function_to_call": "VW",
        "ticker_subset": x_tickers,
    },
    "EW_ptfl": {
        "function_to_call": "EW",
        "ticker_subset": x_tickers,
    },
    "Tracking_ptfl": {
        "function_to_call": "TRACK",
        "ticker_subset": x_tickers,
        "other_options": {
            "target_portfolio_name": 'MWTIX',
            # Initial weights will be set dynamically
        },
    },

    "Tracking_ptfl_penalty": {  # New tracking portfolio with penalty
        "function_to_call": "TRACK_PENALTY",
        "ticker_subset": x_tickers,
        "other_options": {
            "target_portfolio_name": 'MWTIX',
            "penalty_weight": 1e0,  # Set penalty parameter
            # Initial weights will be set dynamically
        },
    },
})

# Initialize a dictionary to store portfolio returns
portfolio_returns_dict = {}

# Store previous weights for tracking portfolio
previous_weights_tracking = None

# Loop over each portfolio
for portfolio_name, portfolio_info in portfolios.items():
    function_name = portfolio_info["function_to_call"]
    subset_tickers = portfolio_info["ticker_subset"]
    other_options = portfolio_info.get("other_options", {}).copy()
    portfolio_function = portfolio_functions[function_name]

    # Initialize a DataFrame to store weights for this portfolio
    portfolio_weights = pd.DataFrame(index=rebalancing_dates, columns=subset_tickers, dtype=float)

    # Loop over each rebalancing date to compute weights
    for date in rebalancing_dates:
        # Returns up to the current date
        returns_up_to_date = returns.loc[:date]

        # Identify available assets for the portfolio
        available_assets = []
        for ticker in subset_tickers:
            asset_returns = returns_up_to_date[ticker].dropna()
            if len(asset_returns) >= min_periods:
                available_assets.append(ticker)

        if len(available_assets) == 0:
            # Set weights to zero for all subset_tickers
            weights = pd.Series(0.0, index=subset_tickers)
            # Store weights
            portfolio_weights.loc[date] = weights
            # For tracking portfolio, reset previous_weights_tracking
            if function_name == "TRACK":
                previous_weights_tracking = None
            continue

        # For the tracking portfolio, set initial weights
        if function_name == "TRACK":
            if previous_weights_tracking is not None and len(previous_weights_tracking) == len(available_assets):
                other_options['initial_weights'] = previous_weights_tracking
                other_options['initial_weights'] = np.array([1] + [0] * (len(available_assets) - 1))  # override to delete

            else:
                other_options['initial_weights'] = np.array([1] + [0] * (len(available_assets) - 1))

        # For the fixed weight portfolio, provide rebalancing date in kwargs
        if function_name == "FIXED":
            other_options['rebalancing_date'] = date

        # Construct the portfolio weights
        try:
            weights = portfolio_function(
                returns_up_to_date, available_assets, **other_options
            )
        except Exception as e:
            print(f"Error constructing weights for {portfolio_name} on {date}: {e}")
            weights = pd.Series(dtype=float)

        # Reindex weights to subset_tickers, setting weights to zero for unavailable assets
        weights = weights.reindex(subset_tickers, fill_value=0.0)

        # Store weights
        portfolio_weights.loc[date] = weights

        # Store previous weights for tracking portfolio
        if function_name == "TRACK":
            previous_weights_tracking = weights.loc[available_assets].values

        # Store weights in long format
        weights_long = pd.DataFrame({
            "portfolio_name": portfolio_name,
            "date": date,
            "ticker": weights.index,
            "weight": weights.values
        })

        # Remove existing entries for the current date and portfolio_name
        if not portfolio_weights_long.empty:
            portfolio_weights_long = portfolio_weights_long[~(
                (portfolio_weights_long['portfolio_name'] == portfolio_name) & (portfolio_weights_long['date'] == date)
            )]

        # Proceed to concatenate if weights_long is not empty and has valid data
        if not weights_long.empty and weights_long.notna().any().any():
            portfolio_weights_long = pd.concat([portfolio_weights_long, weights_long], ignore_index=True)

    # Set the lag parameter (number of business days)
    #lag = 1  # Adjust as needed

    # Expand weights to daily frequency by forward-filling
    daily_weights = portfolio_weights.reindex(returns.index).ffill().infer_objects(copy=False)

    # Shift the daily weights by the lag to apply them starting from the desired date
    daily_weights = daily_weights.shift(lag)

    # Fill any missing weights with zeros (before weights start to be applied)
    daily_weights = daily_weights.fillna(0.0)

    # Ensure that weights are aligned with the returns index
    daily_weights = daily_weights.loc[returns.index]

    # For the 'TRACK' portfolio, ensure the target portfolio returns are available
    if function_name == "TRACK":
        # Ensure the target portfolio returns are available
        target_portfolio_name = other_options.get('target_portfolio_name')
        if target_portfolio_name not in returns.columns:
            raise ValueError(f"Target portfolio '{target_portfolio_name}' returns are not available.")

    # Calculate portfolio returns
    portfolio_returns = daily_weights.multiply(returns[subset_tickers], axis=1).sum(axis=1)

    # Name the portfolio returns
    portfolio_returns.name = portfolio_name

    # If portfolio column already exists in returns, remove it
    returns = returns.drop(columns=[portfolio_name], errors='ignore')

    # Append the portfolio returns to the returns DataFrame
    returns = returns.join(portfolio_returns)

    # Store the portfolio returns in the dictionary
    portfolio_returns_dict[portfolio_name] = portfolio_returns


# Update the list of tickers with portfolio names
all_tickers_with_portfolios = tickers.copy()
for portfolio_name in portfolios.keys():
    if portfolio_name in all_tickers_with_portfolios:
        all_tickers_with_portfolios.remove(portfolio_name)
    all_tickers_with_portfolios.append(portfolio_name)

# Compute cumulative returns for all assets and portfolios
cumulative_returns = (1 + returns).cumprod()

# Define function to compute summary statistics
def compute_summary_stats(series):
    """
    Computes summary statistics for a given time series of cumulative returns.
    """
    total_period_in_years = (series.index[-1] - series.index[0]).days / 365.25
    cumulative_return = series.iloc[-1] - 1  # cumulative_return over the period
    annualized_return = series.iloc[-1] ** (1 / total_period_in_years) - 1
    # For annualized volatility, use daily returns
    daily_returns = series.pct_change().dropna()
    annualized_volatility = daily_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility
    return pd.Series({
        'Cumulative Return': cumulative_return,
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio
    })

# Initialize summary stats DataFrame
summary_stats_df = pd.DataFrame()

# Compute summary statistics for portfolios
for portfolio_name in portfolios.keys():
    portfolio_cum_returns = cumulative_returns[portfolio_name]
    stats = compute_summary_stats(portfolio_cum_returns)
    summary_stats_df[portfolio_name] = stats

# Transpose for better readability
summary_stats_df = summary_stats_df.T

# Display summary statistics
print("Summary Statistics for Portfolios:")
print(summary_stats_df)
print()

# Plot cumulative returns for portfolios using Plotly
fig = go.Figure()

for portfolio_name in portfolios.keys():
    fig.add_trace(go.Scatter(
        x=cumulative_returns.index,
        y=cumulative_returns[portfolio_name],
        mode='lines',
        name=portfolio_name
    ))

fig.update_layout(
    title='Cumulative Returns of Portfolios',
    xaxis_title='Date',
    yaxis_title='Cumulative Return',
    legend_title='Portfolio',
    template='plotly_white'
)

fig.show()
