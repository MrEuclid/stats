import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 1. Page Configuration
st.set_page_config(page_title="Dental Stats Dashboard", layout="wide")

# 2. Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a Simulation:", ["Central Limit Theorem", "Confidence Intervals"])
st.sidebar.markdown("---")

# ==========================================
# APP 1: CENTRAL LIMIT THEOREM
# ==========================================
if app_mode == "Central Limit Theorem":
    st.title("🦷 The Central Limit Theorem")
    st.markdown("Observe how the **Sampling Distribution** becomes normal as $n$ increases.")

    # Controls
    st.sidebar.header("CLT Settings")
    dist_type = st.sidebar.selectbox(
        "Population Shape:",
        ("Exponential (Skewed)", "Uniform (Flat)", "Bimodal (Two Peaks)")
    )
    n = st.sidebar.slider("Sample Size (n)", 1, 100, 5)
    num_simulations = st.sidebar.slider("Number of Simulations", 100, 5000, 1000)

    # Logic
    np.random.seed(42)
    if "Exponential" in dist_type:
        population = np.random.exponential(scale=1.0, size=10000)
        sample_means = [np.mean(np.random.exponential(scale=1.0, size=n)) for _ in range(num_simulations)]
    elif "Uniform" in dist_type:
        population = np.random.uniform(0, 10, 10000)
        sample_means = [np.mean(np.random.uniform(0, 10, size=n)) for _ in range(num_simulations)]
    else:
        pop1 = np.random.normal(10, 2, 5000)
        pop2 = np.random.normal(30, 3, 5000)
        population = np.concatenate([pop1, pop2])
        sample_means = [np.mean(np.random.choice(population, n)) for _ in range(num_simulations)]

    # Plotting
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Population")
        fig1, ax1 = plt.subplots()
        ax1.hist(population, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        st.pyplot(fig1)
    with col2:
        st.subheader(f"Sampling Distribution (n={n})")
        fig2, ax2 = plt.subplots()
        ax2.hist(sample_means, bins=50, color='orange', edgecolor='black', alpha=0.7, density=True)
        # Normal fit
        if np.std(sample_means) > 0:
            mu, std = stats.norm.fit(sample_means)
            x = np.linspace(min(sample_means), max(sample_means), 100)
            ax2.plot(x, stats.norm.pdf(x, mu, std), 'k', lw=2)
        st.pyplot(fig2)

# ==========================================
# APP 2: CONFIDENCE INTERVALS
# ==========================================
elif app_mode == "Confidence Intervals":
    st.title("🦷 Confidence Interval Explorer")
    st.markdown("Simulating **20 studies**. Green = Captured True Mean. Red = Missed.")

    # Controls
    st.sidebar.header("CI Settings")
    true_mean = st.sidebar.number_input("True Mean (μ)", value=50.0)
    true_sd = st.sidebar.number_input("True SD (σ)", value=10.0)
    n = st.sidebar.slider("Sample Size (n)", 3, 200, 10)
    conf_level = st.sidebar.selectbox("Confidence Level", [0.90, 0.95, 0.99], index=1)

    # Logic
    num_studies = 20
    np.random.seed(None) # Randomize every time
    
    means = []
    intervals = []
    captured = []
    
    for _ in range(num_studies):
        sample = np.random.normal(true_mean, true_sd, n)
        mu_hat = np.mean(sample)
        se = stats.sem(sample)
        ci = stats.t.interval(conf_level, df=n-1, loc=mu_hat, scale=se)
        
        means.append(mu_hat)
        intervals.append(ci)
        captured.append(ci[0] <= true_mean <= ci[1])

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axvline(true_mean, color='black', linestyle='--', label='True Mean')
    for i, (m, ci, cap) in enumerate(zip(means, intervals, captured)):
        color = 'green' if cap else 'red'
        ax.errorbar(m, i, xerr=[[m - ci[0]], [ci[1] - m]], fmt='o', color=color, capsize=4)
    
    ax.set_yticks(range(num_studies))
    ax.set_yticklabels([f"Study {i+1}" for i in range(num_studies)])
    st.pyplot(fig)
    
    st.metric("Success Rate", f"{sum(captured)/num_studies*100}%")
