import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Streamlit Page Config ---
st.set_page_config(page_title="Markowitz Efficient Frontier", layout="wide")

st.title("📊 Interactive Markowitz Efficient Frontier")
st.markdown("This app calculates the efficient frontier based on independent scenario analysis for each stock.")

# --- Sidebar Inputs ---
st.sidebar.header("Global Settings")
num_stocks = st.sidebar.number_input("Number of stocks", min_value=1, max_value=10, value=2)
rf = st.sidebar.number_input("Risk-Free (Rf) Rate %", value=3.0) / 100

stock_tickers = []
for n in range(num_stocks):
    ticker = st.sidebar.text_input(f"Ticker for Stock {n+1}", value=f"Stock{n+1}").upper().strip()
    stock_tickers.append(ticker)

# --- Individual Stock Scenarios ---
st.header("📈 Stock Scenario Inputs")
mean_stock_returns = {}
stock_standard_deviations = {}
cases = ['Case I', 'Case II', 'Case III', 'Case IV', 'Case V']

# Create columns for stock inputs to save space
cols = st.columns(num_stocks)

for i, ticker in enumerate(stock_tickers):
    with cols[i]:
        st.subheader(f"Inputs for {ticker}")
        probs = []
        rets = []
        for c in cases:
            # Unique keys are required for streamlit widgets in loops
            col_a, col_b = st.columns(2)
            with col_a:
                p = st.number_input(f"Prob % ({c})", key=f"p_{ticker}_{c}", value=20.0) / 100
            with col_b:
                r = st.number_input(f"Ret % ({c})", key=f"r_{ticker}_{c}", value=5.0)
            probs.append(p)
            rets.append(r)

        # Check if probabilities sum to 100%
        if not math.isclose(sum(probs), 1.0, rel_tol=1e-5):
            st.warning(f"Probabilities for {ticker} sum to {sum(probs)*100:.1f}%!")

        # Calculate Mean for this stock
        m = sum(p * r for p, r in zip(probs, rets))
        mean_stock_returns[ticker] = m

        # Calculate Variance/Std Dev for this stock
        v = sum(p * (r - m)**2 for p, r in zip(probs, rets))
        stock_standard_deviations[ticker] = math.sqrt(v)

# --- Portfolio Simulation ---
if st.button("🚀 Generate Efficient Frontier"):
    
    # Covariance Matrix (Assuming Independence)
    cov_matrix = pd.DataFrame(0.0, index=stock_tickers, columns=stock_tickers)
    for ticker in stock_tickers:
        cov_matrix.loc[ticker, ticker] = stock_standard_deviations[ticker]**2

    # Simulation setup
    num_portfolios = 10000
    all_weights = np.zeros((num_portfolios, num_stocks))
    exp_returns_array = np.array([mean_stock_returns[t] for t in stock_tickers])
    cov_matrix_array = cov_matrix.values

    for i in range(num_portfolios):
        weights = np.random.random(num_stocks)
        weights /= np.sum(weights)
        all_weights[i,:] = weights

    # Risk and Return Calculations
    port_returns = np.dot(all_weights, exp_returns_array)
    # Using einsum for efficient matrix multiplication over 10k rows
    port_volatility = np.sqrt(np.einsum('ij,jk,ik->i', all_weights, cov_matrix_array, all_weights))

    # Create DataFrame
    df = pd.DataFrame(all_weights, columns=[f"{t} weight" for t in stock_tickers])
    df['Portfolio_Return'] = port_returns
    df['Portfolio_Risk'] = port_volatility
    df['Sharpe_Ratio'] = (df['Portfolio_Return'] - (rf * 100)) / df['Portfolio_Risk']

    # Efficient Frontier Line
    min_vol_idx = df['Portfolio_Risk'].idxmin()
    min_vol_return = df.loc[min_vol_idx, 'Portfolio_Return']
    upper_half = df[df['Portfolio_Return'] >= min_vol_return].copy()

    num_bins = 50
    upper_half['Risk_Bin'] = pd.cut(upper_half['Portfolio_Risk'], bins=num_bins)
    efficient_frontier = upper_half.groupby('Risk_Bin', observed=True).apply(lambda x: x.loc[x['Portfolio_Return'].idxmax()])
    efficient_frontier = efficient_frontier.sort_values('Portfolio_Risk')

    # --- Plotting ---
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Portfolio_Risk'], y=df['Portfolio_Return'],
        mode='markers',
        marker=dict(color=df['Sharpe_Ratio'], colorscale='Viridis', showscale=True, size=5, colorbar=dict(title="Sharpe")),
        name='Simulated Portfolios',
        text=[", ".join([f"{t}: {row[f'{t} weight']:.2%}" for t in stock_tickers]) for _, row in df.iterrows()],
        hoverinfo='text+x+y'
    ))

    fig.add_trace(go.Scatter(
        x=efficient_frontier['Portfolio_Risk'], y=efficient_frontier['Portfolio_Return'],
        mode='lines', line=dict(color='red', width=3),
        name='Efficient Frontier'
    ))

    max_sharpe_idx = df['Sharpe_Ratio'].idxmax()
    fig.add_trace(go.Scatter(
        x=[df.loc[max_sharpe_idx, 'Portfolio_Risk']],
        y=[df.loc[max_sharpe_idx, 'Portfolio_Return']],
        mode='markers', marker=dict(color='orange', size=15, symbol='star'),
        name='Max Sharpe Ratio'
    ))

    fig.update_layout(
        title='Markowitz Efficient Frontier',
        xaxis_title='Expected Risk (Std Dev)',
        yaxis_title='Expected Return (%)',
        template='plotly_white',
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    # Display Summary Table
    st.subheader("Summary Statistics")
    summary_df = pd.DataFrame({
        "Mean Return": mean_stock_returns,
        "Std Deviation": stock_standard_deviations
    }).T
    st.table(summary_df)
