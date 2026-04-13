import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Streamlit Page Config ---
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

st.title("📊 Markowitz Portfolio Optimization Suite")
st.markdown("Define your stock scenarios in the sidebar and use the tools below to find exact portfolio weights.")

# --- Sidebar Inputs ---
st.sidebar.header("1. Global Settings")
num_stocks = st.sidebar.number_input("Number of stocks", min_value=2, max_value=10, value=2)
rf = st.sidebar.number_input("Risk-Free (Rf) Rate %", value=3.0)

stock_tickers = []
for n in range(num_stocks):
    ticker = st.sidebar.text_input(f"Ticker for Stock {n+1}", value=f"Stock {n+1}").upper().strip()
    stock_tickers.append(ticker)

# --- Individual Stock Scenarios ---
st.sidebar.header("2. Stock Scenario Inputs")
mean_stock_returns = {}
stock_standard_deviations = {}
cases = ['Case I', 'Case II', 'Case III', 'Case IV', 'Case V']

for ticker in stock_tickers:
    with st.sidebar.expander(f"Scenarios for {ticker}"):
        probs = []
        rets = []
        for c in cases:
            p = st.number_input(f"Prob % ({c})", key=f"p_{ticker}_{c}", value=20.0) / 100
            r = st.number_input(f"Ret % ({c})", key=f"r_{ticker}_{c}", value=5.0 if c == 'Case III' else 10.0)
            probs.append(p)
            rets.append(r)
        
        # Calculations
        m = sum(p * r for p, r in zip(probs, rets))
        v = sum(p * (r - m)**2 for p, r in zip(probs, rets))
        mean_stock_returns[ticker] = m
        stock_standard_deviations[ticker] = math.sqrt(v)

# --- Optimization Tools ---
st.header("🛠️ Portfolio Calculators")
tool_choice = st.radio(
    "Choose a Calculator:",
    ["Simulated Cloud Only", "Target Return", "Target Risk (Min Vol)", "Minimum Variance Portfolio", "Tangency Portfolio (Max Sharpe)"],
    horizontal=True
)

# Convert to arrays for math
mu = np.array([mean_stock_returns[t] for t in stock_tickers])
# Diagonal Covariance Matrix (Independence assumption)
S = np.diag([stock_standard_deviations[t]**2 for t in stock_tickers])
S_inv = np.linalg.inv(S)
ones = np.ones(len(mu))

# Logic for Exact Calculations
exact_w = None
calc_label = ""

if tool_choice == "Target Return":
    target_r = st.number_input("Enter Target Return (%)", value=float(np.mean(mu)))
    # Solving for Min Var given Target Return
    A = ones @ S_inv @ ones
    B = ones @ S_inv @ mu
    C = mu @ S_inv @ mu
    det = A*C - B**2
    exact_w = ((C - target_r*B)* (S_inv @ ones) + (target_r*A - B)* (S_inv @ mu)) / det
    calc_label = f"Target Return ({target_r}%)"

elif tool_choice == "Minimum Variance Portfolio":
    exact_w = (S_inv @ ones) / (ones @ S_inv @ ones)
    calc_label = "Min Variance Portfolio"

elif tool_choice == "Tangency Portfolio (Max Sharpe)":
    excess_mu = mu - rf
    exact_w = (S_inv @ excess_mu) / (ones @ S_inv @ excess_mu)
    calc_label = "Tangency Portfolio"

elif tool_choice == "Target Risk (Min Vol)":
    target_v = st.number_input("Enter Target Std Dev (%)", value=float(np.mean(list(stock_standard_deviations.values()))))
    st.info("Note: This finds the portfolio with the highest return for this specific risk level.")
    # For target risk, we find the point on the efficient frontier (upper half)
    # This is a simplification using the simulation data for easier UX
    calc_label = f"Target Risk ({target_v}%)"

# --- Simulation for Plotting ---
num_portfolios = 5000
all_weights = np.random.random((num_portfolios, num_stocks))
all_weights /= all_weights.sum(axis=1)[:, np.newaxis]

port_returns = all_weights @ mu
port_volatility = np.sqrt(np.einsum('ij,jk,ik->i', all_weights, S, all_weights))
df = pd.DataFrame(all_weights, columns=[f"{t} weight" for t in stock_tickers])
df['Return'] = port_returns
df['Risk'] = port_volatility
df['Sharpe'] = (df['Return'] - rf) / df['Risk']

# --- Plotting ---
fig = go.Figure()

# Cloud
fig.add_trace(go.Scatter(
    x=df['Risk'], y=df['Return'], mode='markers',
    marker=dict(color=df['Sharpe'], colorscale='Viridis', size=4, opacity=0.5),
    name='Simulated Portfolios',
    text=[", ".join([f"{t}: {row[i]:.1%}" for i, t in enumerate(stock_tickers)]) for row in all_weights],
    hoverinfo='text+x+y'
))

# Exact Point (The Calculators)
if exact_w is not None:
    e_ret = exact_w @ mu
    e_risk = np.sqrt(exact_w.T @ S @ exact_w)
    
    fig.add_trace(go.Scatter(
        x=[e_risk], y=[e_ret], mode='markers+text',
        marker=dict(color='red', size=15, symbol='diamond', line=dict(width=2, color='white')),
        text=[f"  {calc_label}"], textposition="top right",
        name='Exact Calculation'
    ))
    
    # Show Weights for the calculated portfolio
    st.subheader(f"Optimal Weights for {calc_label}")
    weight_display = pd.DataFrame([exact_w], columns=stock_tickers, index=["Weights"])
    st.table(weight_display.applymap(lambda x: f"{x:.2%}"))

fig.update_layout(template='plotly_white', height=600, xaxis_title="Risk", yaxis_title="Return")
st.plotly_chart(fig, use_container_width=True)
