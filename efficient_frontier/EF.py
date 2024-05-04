import numpy as np
import pandas as pd

def restrict(portfolio, db):
    portfolio_returns_data = read_portfolio_data(db, portfolio)
    cov_matrix = calculate_covariance_matrix(portfolio_returns_data)
    returns = calculate_expected_returns(portfolio_returns_data)
    cols = cov_matrix.shape[1]
    low_bound = np.ceil(np.min(returns) * 100) / 100 + 0.01
    up_bound = np.floor(np.max(returns) * 100) / 100
    n = min(int((up_bound - low_bound) / 0.01) + 1, 30)

    om = np.zeros((n, 1))
    target_returns = np.linspace(low_bound, up_bound, n)

    A = np.vstack([np.ones(cols), returns])
    matBig = np.block([[cov_matrix, A.T], [A, np.zeros((2, 2))]])

    for k, target_return in enumerate(target_returns):
        b = np.array([1.0, target_return])
        B_ext = np.vstack([np.zeros((cols, 1)), b.reshape(-1, 1)])

        # Solve the linear system
        x = np.linalg.lstsq(matBig, B_ext, rcond=None)[0][:cols]

        # Calculate the portfolio volatility
        vol = np.sqrt(x.T @ cov_matrix @ x)
        om[k, 0] = vol.iloc[0,0]

    return target_returns, om.flatten()


def calculate_expected_returns(assets_data):
    returns = assets_data.pct_change().dropna()
    return returns.mean() * 252

def calculate_covariance_matrix(assets_data):
    returns = assets_data.pct_change().dropna()
    return returns.cov() * 252

def read_portfolio_data(conn, symbols):
    # Initialize an empty DataFrame to store price data
    portfolio_data = pd.DataFrame()

    for symbol in symbols:
        query = f"SELECT date, adj_close FROM Stock_{symbol}_Data ORDER BY date"
        df = pd.read_sql_query(query, conn)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        
        # Rename the 'adj_close' column to the stock symbol
        df.rename(columns={"adj_close": symbol}, inplace=True)

        # Merge the stock data with the portfolio data
        if portfolio_data.empty:
            portfolio_data = df
        else:
            portfolio_data = portfolio_data.merge(df, left_index=True, right_index=True, how="outer")

    return portfolio_data