import matplotlib.pyplot as plt

plt.plot(cumulative_returns_lstm, label='LSTM Portfolio')
plt.plot(cumulative_returns_trans, label='Transformer Portfolio')
plt.plot(cumulative_returns_eq, label='Equal Weight')
plt.legend()
plt.title('Portfolio Performance Comparison')
