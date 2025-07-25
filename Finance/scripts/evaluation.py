
import numpy as np

def annualized_return(returns, freq=252):
    return np.mean(returns) * freq

def annualized_volatility(returns, freq=252):
    return np.std(returns) * np.sqrt(freq)

def sharpe_ratio(returns, risk_free=0.01, freq=252):
    excess = returns - (risk_free / freq)
    return annualized_return(excess, freq) / annualized_volatility(returns, freq)

def max_drawdown(cumulative_returns):
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()
