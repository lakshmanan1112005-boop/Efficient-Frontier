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

st.title("📊 Markowitz Portfolio Calculators")
st.markdown("Enter stock scenarios below, then use the sidebar to find specific optimal portfolios.")

# --- Sidebar Configuration ---
st.sidebar.header("1. Global Settings")
num_stocks = st.sidebar.number_input("Number of stocks", min_value=2, max_value=10, value=2)
rf_input = st.sidebar.number_input("Risk-Free (Rf) Rate %", value=3.0)
rf = rf_input  # Keep in same scale as returns (e.g., 3.0 for 3%)

stock_tickers = []
for n in range(num_stocks):
    ticker = st.sidebar.text_input(f"Ticker for Stock {n+1}", value=f"Stock {n+1}").upper().strip()
    stock_tickers.append(ticker)

st.sidebar.header("2. Portfolio Calculators")
calc_option = st.sidebar.selectbox(
    "Choose a Calculator",
    ["None", "Target Return", "Target Risk (Std Dev)", "Minimum Variance Portfolio", "Tangency Portfolio"]
)

target_val = 0.0
if calc_option == "Target Return":
    target_val = st.sidebar.number_input("Enter Target Return (%)", value=10.0)
elif calc_option == "Target Risk (Std Dev)":
    target_val = st.sidebar.number_input("Enter Target Risk (%)", value=15.0)

# --- Individual Stock Scenarios ---
st.header("📈 Stock Scenario Inputs")
mean_stock_returns = {}
stock_standard_deviations = {}
cases = ['Case I', 'Case II', 'Case III', 'Case IV', 'Case V']

cols = st.columns(num_stocks)
for i, ticker in enumerate(stock_tickers):
    with cols[i]:
        st.subheader(f"{ticker} Inputs")
        probs, rets = [], []
        for c in cases:
            col_a, col_b = st.columns(2)
            with col_a:
                p = st.number_input(f"Prob % ({c})", key=f"p_{ticker}_{c}", value=20.0) / 100
            with col_b:
                r = st.number_input(f"Ret % ({c})", key=f"r_{ticker}_{c}", value=5.0 + (i*5))
            probs.append(p)
            rets.append(r)
        
        m = sum(p * r for p, r in zip(probs, rets))
        mean_stock_returns[ticker] = m
        v = sum(p * (r - m)**2 for p, r in zip(probs, rets))
        stock_standard_deviations[ticker] = math.sqrt(v)

# --- Core Matrix Setup ---
exp_rets = np.array([mean_stock_returns[t] for t in stock_tickers])
# Variance-Covariance Matrix (Assuming Independence as requested)
cov_mat = np.diag([stock_standard_deviations[t]**2 for t in stock_tickers])
inv_cov = np.linalg.inv(cov_mat)
ones = np.ones(len(exp_rets))

# --- Mathematical Functions for Exact Weights ---
def get_min_var_weights():
    w = np.dot(inv_cov, ones) / np.dot(ones.T, np.dot(inv_cov, ones))
    return w

def get_tangency_weights():
    excess_rets = exp_rets - rf
    w = np.dot(inv_cov, excess_rets) / np.dot(ones.T, np.dot(inv_cov, excess_rets))
    return w

def get_target_ret_weights(target_ret):
    # Solver for weights on the efficient frontier for a specific return
    A = np.dot(ones.T, np.dot(inv_cov, ones))
    B = np.dot(ones.T, np.dot(inv_cov, exp_rets))
    C = np.dot(exp_rets.T, np.dot(inv_cov, exp_rets))
    D = A*C - B**2
    g = (inv_cov @ (C*ones - B*exp_rets)) / D
    h = (inv_cov @ (A*exp_rets - B*ones)) / D
    return g + h * target_ret

# --- Execution & Plotting ---
if st.button("🚀 Calculate & Plot"):
    # 1. Random Simulation for the Cloud
    num_portfolios = 5000
    sim_w = np.random.random((num_portfolios, num_stocks))
    sim_w /= sim_w.sum(axis=1)[:, None]
    
    sim_rets = np.dot(sim_w, exp_rets)
    sim_vol = np.sqrt(np.einsum('ij,jk,ik->i', sim_w, cov_mat, sim_w))
    
    df = pd.DataFrame(sim_w, columns=[f"{t} weight" for t in stock_tickers])
    df['Return'] = sim_rets
    df['Risk'] = sim_vol
    df['Sharpe'] = (df['Return'] - rf) / df['Risk']

    # 2. Handle Calculator Logic
    calc_results = None
    if calc_option != "None":
        if calc_option == "Minimum Variance Portfolio":
            w_calc = get_min_var_weights()
        elif calc_option == "Tangency Portfolio":
            w_calc = get_tangency_weights()
        elif calc_option == "Target Return":
            w_calc = get_target_ret_weights(target_val)
        elif calc_option == "Target Risk (Std Dev)":
            # Finding exact weights for a target risk requires searching the frontier
            # Simplified: we find the target return that matches this risk on the frontier
            A = np.dot(ones.T, np.dot(inv_cov, ones))
            B = np.dot(ones.T, np.dot(inv_cov, exp_rets))
            C = np.dot(exp_rets.T, np.dot(inv_cov, exp_rets))
            D = A*C - B**2
            # Frontier Variance formula: Var = (A*mu^2 - 2B*mu + C) / D
            # We solve the quadratic for mu (Return)
            v_target = target_val**2
            # Quadratic: A*mu^2 - 2B*mu + (C - D*v_target) = 0
            roots = np.roots([A, -2*B, (C - D*v_target)])
            w_calc = get_target_ret_weights(max(roots)) # Pick the efficient (higher) root

        ret_calc = np.dot(w_calc, exp_rets)
        risk_calc = np.sqrt(np.dot(w_calc.T, np.dot(cov_mat, w_calc)))
        calc_results = {"weights": w_calc, "ret": ret_calc, "risk": risk_calc}

    # 3. Create Plotly Graph
    fig = go.Figure()

    # Cloud
    fig.add_trace(go.Scatter(
        x=df['Risk'], y=df['Return'], mode='markers',
        marker=dict(color=df['Sharpe'], colorscale='Viridis', size=4, opacity=0.5),
        name='Simulated Portfolios', hoverinfo='skip'
    ))

    # Highlight Calculated Portfolio
    if calc_results:
        weight_str = "<br>".join([f"{t}: {w:.2%}" for t, w in zip(stock_tickers, calc_results['weights'])])
        fig.add_trace(go.Scatter(
            x=[calc_results['risk']], y=[calc_results['ret']],
            mode='markers+text',
            marker=dict(color='red', size=15, symbol='x', line=dict(width=2, color='white')),
            name=f"Result: {calc_option}",
            text=[f"<b>{calc_option}</b><br>{weight_str}"],
            textposition="top center"
        ))
        
        # Display weights in a table
        st.subheader(f"Results for {calc_option}")
        res_df = pd.DataFrame([f"{w:.4%}" for w in calc_results['weights']], index=stock_tickers, columns=["Exact Weight"])
        st.table(res_df.T)

    fig.update_layout(
        title="Efficient Frontier & Selected Calculator Result",
        xaxis_title="Risk (Standard Deviation %)", yaxis_title="Expected Return (%)",
        template="plotly_white", height=700
    )
    st.plotly_chart(fig, use_container_width=True)
