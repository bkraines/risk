import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from functools import reduce

# Configuration
CONFIG = {
    'tickers': ['XLE', 'XLF', 'XLU', 'XLI', 'XLK', 'XLV', 'XLY', 'XLP', 'XLB'],
    'start_date': '2004-01-01',
    'end_date': pd.Timestamp.today().strftime('%Y-%m-%d'),
    'halflife': 126,
    'min_periods': 126,
    'shock_quantile': 0.99,
    'num_shock_days_table': 3,
    'num_shock_days_chart': 3,
    'trailing_periods': {'week': 5, 'month': 21}
}

# Basic utility functions
def is_valid_covariance(cov_matrix: np.ndarray, max_condition: float = 1e12) -> bool:
    """Check if covariance matrix is valid for inversion."""
    return (
        not np.isnan(cov_matrix).any() and
        not np.isinf(cov_matrix).any() and
        np.all(np.diag(cov_matrix) > 1e-12) and
        np.linalg.cond(cov_matrix) <= max_condition
    )

# Data acquisition and preprocessing
def download_data(tickers: list, start: str, end: str) -> pd.DataFrame:
    """Download and clean price data."""
    print(f"Downloading data for {len(tickers)} tickers from {start} to {end}")
    
    raw_data = yf.download(tickers, start=start, end=end, progress=False)
    
    if raw_data.empty:
        raise ValueError("No data downloaded")
    
    # Extract close prices
    if isinstance(raw_data.columns, pd.MultiIndex):
        data = raw_data.xs('Close', level=0, axis=1)
    elif len(tickers) == 1:
        data = raw_data[['Close']].rename(columns={'Close': tickers[0]})
    else:
        data = raw_data[tickers]
    
    # Clean and validate
    data = data.dropna()
    if len(data) < CONFIG['min_periods'] + 5:
        raise ValueError(f"Insufficient data: {len(data)} rows")
    
    print(f"Data shape: {data.shape}, range: {data.index.min():%Y-%m-%d} to {data.index.max():%Y-%m-%d}")
    return data

def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate log returns."""
    returns = np.log(prices / prices.shift(1)).dropna()
    print(f"Returns shape: {returns.shape}")
    return returns

def calculate_ewma_covariance(returns: pd.DataFrame, halflife: int, min_periods: int) -> pd.DataFrame:
    """Calculate EWMA covariance matrix series."""
    return returns.ewm(halflife=halflife, min_periods=min_periods).cov()

# Core risk metric calculations
def calculate_metrics_for_date(
    returns_t: np.ndarray, 
    cov_matrix: np.ndarray, 
    weights: np.ndarray
) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate all metrics for a single date."""
    W_diag = np.diag(weights)
    cov_inv = np.linalg.inv(cov_matrix)
    
    # Mahalanobis distance
    mahal_dist = (returns_t.T @ W_diag @ cov_inv @ W_diag @ returns_t).item()
    
    # Volatility shock (using diagonal covariance)
    vol_inv = np.diag(1.0 / np.diag(cov_matrix))
    vol_shock = (returns_t.T @ W_diag @ vol_inv @ W_diag @ returns_t).item()
    
    # Correlation shock (residual)
    corr_shock = mahal_dist - vol_shock
    # corr_shock = mahal_dist / vol_shock
    
    # Contributions
    A_matrix = W_diag @ cov_inv @ W_diag
    v_vector = A_matrix @ returns_t
    contrib_mahal = returns_t.flatten() * v_vector.flatten()
    
    std_devs = np.sqrt(np.diag(cov_matrix))
    contrib_vol = (weights**2) * ((returns_t.flatten() / std_devs)**2)
    contrib_corr = contrib_mahal - contrib_vol
    
    return mahal_dist, vol_shock, corr_shock, contrib_mahal, contrib_vol, contrib_corr

def compute_risk_metrics(returns: pd.DataFrame, cov_series: pd.DataFrame) -> dict[str, pd.Series]:
    """Compute all risk metrics efficiently."""
    tickers = returns.columns.tolist()
    n_assets = len(tickers)
    weights = np.full(n_assets, 1/n_assets)
    
    # Initialize result containers
    results = {
        'mahal_dist': pd.Series(index=returns.index, dtype=float),
        'vol_shock': pd.Series(index=returns.index, dtype=float),
        'corr_shock': pd.Series(index=returns.index, dtype=float),
        'contrib_mahal': pd.DataFrame(index=returns.index, columns=tickers, dtype=float),
        'contrib_vol': pd.DataFrame(index=returns.index, columns=tickers, dtype=float),
        'contrib_corr': pd.DataFrame(index=returns.index, columns=tickers, dtype=float)
    }
    
    valid_cov_dates = cov_series.index.get_level_values(0).unique()
    processed = 0
    
    for i in range(1, len(returns)):
        t_date = returns.index[i]
        t_minus_1 = returns.index[i-1]
        
        if t_minus_1 not in valid_cov_dates:
            continue
            
        try:
            cov_matrix = cov_series.loc[t_minus_1].reindex(
                index=tickers, columns=tickers
            ).values
            
            if not is_valid_covariance(cov_matrix):
                continue
                
            returns_t = returns.loc[t_date].values.reshape(-1, 1)
            
            metrics = calculate_metrics_for_date(returns_t, cov_matrix, weights)
            mahal_dist, vol_shock, corr_shock, contrib_m, contrib_v, contrib_c = metrics
            
            results['mahal_dist'].loc[t_date] = mahal_dist
            results['vol_shock'].loc[t_date] = vol_shock
            results['corr_shock'].loc[t_date] = corr_shock
            results['contrib_mahal'].loc[t_date] = contrib_m
            results['contrib_vol'].loc[t_date] = contrib_v
            results['contrib_corr'].loc[t_date] = contrib_c
            
            processed += 1
            
        except Exception:
            continue
    
    print(f"Successfully processed {processed} days")
    return {k: v.dropna() if isinstance(v, pd.Series) else v.dropna(how='all') 
            for k, v in results.items()}

# Business logic and analysis
def identify_shock_days(mahal_series: pd.Series, quantile: float) -> pd.Series:
    """Identify shock days above threshold."""
    threshold = mahal_series.quantile(quantile)
    shock_days = mahal_series[mahal_series >= threshold].sort_index(ascending=False)
    print(f"Found {len(shock_days)} shock days (threshold: {threshold:.4f})")
    return shock_days

def create_attribution_table(
    shock_date: pd.Timestamp,
    contrib_data: dict[str, pd.DataFrame],
    tickers: list,
    periods: dict[str, int]
) -> pd.DataFrame:
    """Create attribution table for a shock day."""
    def get_period_contrib(df: pd.DataFrame, date: pd.Timestamp, days: int) -> np.ndarray:
        if date not in df.index:
            return np.full(len(tickers), np.nan)
        
        if days == 1:
            return df.loc[date].values
        
        try:
            idx = df.index.get_loc(date)
            start_idx = max(0, idx - days + 1)
            slice_data = df.iloc[start_idx:idx+1]
            return slice_data.abs().mean().values
        except:
            return np.full(len(tickers), np.nan)
    
    # Calculate contributions for different periods
    period_data = {}
    for period_name, days in [('Last Day', 1)] + [(f'Avg {k.title()}', v) for k, v in periods.items()]:
        for metric in ['mahal', 'vol', 'corr']:
            contrib_key = f'contrib_{metric}'
            if contrib_key in contrib_data:
                period_data[f'{metric}_{period_name}'] = get_period_contrib(
                    contrib_data[contrib_key], shock_date, days
                )
    
    # Create DataFrame
    table_data = []
    for i, ticker in enumerate(tickers):
        row = {'Asset': ticker}
        for col_name, values in period_data.items():
            row[col_name] = values[i] if not np.isnan(values[i]) else 'N/A'
        table_data.append(row)
    
    df = pd.DataFrame(table_data).set_index('Asset')
    
    # Sort by absolute contribution
    if 'mahal_Last Day' in df.columns:
        sort_col = pd.to_numeric(df['mahal_Last Day'], errors='coerce')
        df = df.reindex(sort_col.abs().sort_values(ascending=False).index)
    
    return df

# Visualization and reporting
def create_time_series_plot(metrics: dict[str, pd.Series], shock_quantile: float) -> go.Figure:
    """Create interactive time series plot."""
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=['Mahalanobis Distance', 'Correlation Shock', 'Volatility Shock']
    )
    
    # Mahalanobis distance
    mahal = metrics['mahal_dist']
    threshold = mahal.quantile(shock_quantile)
    
    fig.add_trace(go.Scatter(
        x=mahal.index, y=mahal, mode='lines', 
        name='Mahalanobis Distance', line=dict(color='blue')
    ), row=1, col=1)
    
    fig.add_hline(
        y=threshold, line_dash="dash", line_color="red",
        annotation_text=f"{shock_quantile*100:.0f}th percentile ({threshold:.2f})",
        row=1, col=1
    )
    
    # Correlation shock
    corr_shock = metrics['corr_shock']
    fig.add_trace(go.Scatter(
        x=corr_shock.index, y=corr_shock, mode='lines',
        name='Correlation Shock', line=dict(color='purple')
    ), row=2, col=1)
    
    # Volatility shock
    vol_shock = metrics['vol_shock']
    fig.add_trace(go.Scatter(
        x=vol_shock.index, y=vol_shock, mode='lines',
        name='Volatility Shock', line=dict(color='green')
    ), row=3, col=1)
    
    fig.update_layout(
        height=900, 
        title=f'Portfolio Risk Metrics (EWMA Half-life: {CONFIG["halflife"]} days)',
        showlegend=False
    )
    
    return fig

def create_contribution_bar_chart(
    contrib_data: pd.DataFrame,
    shock_days: pd.Series,
    num_shock_days: int
) -> go.Figure:
    """Create grouped bar chart for contributions."""
    chart_data = {}
    
    # Latest observation
    if not contrib_data.empty:
        latest_date = contrib_data.index[-1]
        chart_data[f"Latest ({latest_date:%Y-%m-%d})"] = contrib_data.loc[latest_date]
    
    # Recent shock days
    for i, (shock_date, _) in enumerate(shock_days.head(num_shock_days).items()):
        if shock_date in contrib_data.index:
            chart_data[f"Shock ({shock_date:%Y-%m-%d})"] = contrib_data.loc[shock_date]
    
    if not chart_data:
        return go.Figure()
    
    # Create bar chart
    fig = go.Figure()
    
    for label, values in chart_data.items():
        fig.add_trace(go.Bar(name=label, x=values.index, y=values.values))
    
    fig.update_layout(
        barmode='group',
        title='Asset Contributions to Mahalanobis Distance',
        xaxis_title='Asset',
        yaxis_title='Contribution',
        legend_title='Date'
    )
    
    return fig

# Main orchestration
def main():
    """Main analysis pipeline."""
    try:
        # Data acquisition and processing
        prices = download_data(CONFIG['tickers'], CONFIG['start_date'], CONFIG['end_date'])
        returns = calculate_returns(prices)
        cov_series = calculate_ewma_covariance(returns, CONFIG['halflife'], CONFIG['min_periods'])
        
        # Risk metrics calculation
        print("\nCalculating risk metrics...")
        metrics = compute_risk_metrics(returns, cov_series)
        
        # Time series visualization
        print("\nCreating time series plot...")
        ts_fig = create_time_series_plot(metrics, CONFIG['shock_quantile'])
        ts_fig.show()
        
        # Shock day analysis
        print("\nAnalyzing shock days...")
        shock_days = identify_shock_days(metrics['mahal_dist'], CONFIG['shock_quantile'])
        
        if not shock_days.empty:
            # Attribution tables
            for i, (shock_date, mahal_val) in enumerate(shock_days.head(CONFIG['num_shock_days_table']).items()):
                print(f"\nShock Day {i+1}: {shock_date:%Y-%m-%d} (Mahalanobis: {mahal_val:.4f})")
                
                table = create_attribution_table(
                    shock_date, metrics, CONFIG['tickers'], CONFIG['trailing_periods']
                )
                print(table.to_string(float_format="%.4f"))
            
            # Contribution bar chart
            print("\nCreating contribution bar chart...")
            bar_fig = create_contribution_bar_chart(
                metrics['contrib_mahal'], shock_days, CONFIG['num_shock_days_chart']
            )
            bar_fig.show()
        
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        raise

if __name__ == "__main__":
    main()