import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Student Study Performance Estimator",
    page_icon="🎓",
    layout="wide"
)

# --- CUSTOM CSS FOR HERO BANNER & CARDS ---
st.markdown("""
    <style>
    .hero-banner {
        background-color: #0E1A40;
        padding: 28px 32px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
    }
    .hero-title {
        font-size: 32px;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        font-size: 15px;
        color: #C0C8E0;
        margin-bottom: 0px;
    }
    .meta-bar {
        font-size: 15px;
        font-weight: 600;
        color: #1E293B;
        margin-bottom: 18px;
    }
    .summary-card {
        background-color: #F0F4F9;
        border-left: 5px solid #1E3A8A;
        padding: 16px 20px;
        border-radius: 8px;
        margin-top: 15px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR: CONTROLS & INPUTS ---
st.sidebar.title("📚 Study Details")
st.sidebar.caption("Enter study parameters to generate an estimate.")

# Preset / Scenario Selector
scenario = st.sidebar.selectbox(
    "Select Scenario",
    [
        "Study App Effectiveness (1-Sample)",
        "Study Method Comparison (2-Sample)",
        "Exam Target Score (1-Sample)",
        "Custom Input"
    ]
)

# Map scenario choices to parameter defaults
if scenario == "Study App Effectiveness (1-Sample)":
    test_type = "One-Sample t-Test"
    tail_type = "Right-tailed (>)"
    alpha = 0.05
    mu_0 = 70.0
    x_bar1, s1, n1 = 74.2, 8.0, 25
    x_bar2, s2, n2 = 70.0, 7.5, 30
    scenario_desc = "Testing if an interactive study app significantly increases student exam scores above the 70-point baseline."

elif scenario == "Study Method Comparison (2-Sample)":
    test_type = "Two-Sample Independent t-Test"
    tail_type = "Two-tailed (≠)"
    alpha = 0.05
    mu_0 = 0.0
    x_bar1, s1, n1 = 71.0, 7.5, 30
    x_bar2, s2, n2 = 76.5, 8.1, 30
    scenario_desc = "Comparing performance between traditional self-study and app-assisted study groups."

elif scenario == "Exam Target Score (1-Sample)":
    test_type = "One-Sample t-Test"
    tail_type = "Two-tailed (≠)"
    alpha = 0.01
    mu_0 = 75.0
    x_bar1, s1, n1 = 78.4, 6.2, 40
    x_bar2, s2, n2 = 0.0, 1.0, 2
    scenario_desc = "Evaluating if average student performance significantly deviates from the target 75-point benchmark."

else:  # Custom Input
    test_type = st.sidebar.selectbox("Select Test Type", ["One-Sample t-Test", "Two-Sample Independent t-Test"])
    tail_type = st.sidebar.selectbox("Select Tail Type", ["Two-tailed (≠)", "Right-tailed (>)", "Left-tailed (<)"])
    alpha = st.sidebar.select_slider("Significance Level (α)", options=[0.01, 0.05, 0.10], value=0.05)
    scenario_desc = "Custom student performance hypothesis testing evaluation."

    st.sidebar.subheader("Sample 1 Parameters")
    x_bar1 = st.sidebar.number_input("Sample Mean (x̄₁)", value=74.2)
    s1 = st.sidebar.number_input("Sample Std Dev (s₁)", value=8.0, min_value=0.1)
    n1 = st.sidebar.slider("Sample Size (n₁)", min_value=5, max_value=200, value=25)

    if test_type == "One-Sample t-Test":
        mu_0 = st.sidebar.number_input("Target Baseline (μ₀)", value=70.0)
        x_bar2, s2, n2 = 0.0, 1.0, 2
    else:
        mu_0 = 0.0
        st.sidebar.subheader("Sample 2 Parameters")
        x_bar2 = st.sidebar.number_input("Sample Mean (x̄₂)", value=71.0)
        s2 = st.sidebar.number_input("Sample Std Dev (s₂)", value=7.5, min_value=0.1)
        n2 = st.sidebar.slider("Sample Size (n₂)", min_value=5, max_value=200, value=30)


# --- STATISTICAL CALCULATIONS ---
if test_type == "One-Sample t-Test":
    df = n1 - 1
    se = s1 / np.sqrt(n1)
    t_stat = (x_bar1 - mu_0) / se
    
    if tail_type == "Right-tailed (>)":
        p_val = 1 - stats.t.cdf(t_stat, df)
        t_crit_upper = stats.t.ppf(1 - alpha, df)
        t_crit_lower = None
        reject = t_stat > t_crit_upper
    elif tail_type == "Left-tailed (<)":
        p_val = stats.t.cdf(t_stat, df)
        t_crit_lower = stats.t.ppf(alpha, df)
        t_crit_upper = None
        reject = t_stat < t_crit_lower
    else:  # Two-tailed
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        t_crit_upper = stats.t.ppf(1 - alpha / 2, df)
        t_crit_lower = -t_crit_upper
        reject = abs(t_stat) > t_crit_upper

else:  # Two-Sample t-Test
    df = n1 + n2 - 2
    sp2 = ((n1 - 1) * (s1**2) + (n2 - 1) * (s2**2)) / df
    se = np.sqrt(sp2 * (1/n1 + 1/n2))
    t_stat = (x_bar1 - x_bar2) / se
    
    if tail_type == "Right-tailed (>)":
        p_val = 1 - stats.t.cdf(t_stat, df)
        t_crit_upper = stats.t.ppf(1 - alpha, df)
        t_crit_lower = None
        reject = t_stat > t_crit_upper
    elif tail_type == "Left-tailed (<)":
        p_val = stats.t.cdf(t_stat, df)
        t_crit_lower = stats.t.ppf(alpha, df)
        t_crit_upper = None
        reject = t_stat < t_crit_lower
    else:  # Two-tailed
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        t_crit_upper = stats.t.ppf(1 - alpha / 2, df)
        t_crit_lower = -t_crit_upper
        reject = abs(t_stat) > t_crit_upper


# --- HERO BANNER ---
st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🎓 Student Study Performance Estimator</div>
        <div class="hero-subtitle">Estimate student performance impact using sampling, probability distributions, and statistical estimation.</div>
    </div>
""", unsafe_allow_html=True)

# --- META SUMMARY BAR ---
st.markdown(f"""
    <div class="meta-bar">
        📍 <b>Selected Scenario:</b> {scenario} &nbsp;|&nbsp; <b>Test:</b> {test_type} &nbsp;|&nbsp; <b>Significance (α):</b> {alpha}
    </div>
""", unsafe_allow_html=True)

# --- METRICS ROW ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Calculated t-Stat", f"{t_stat:.2f}")

with col2:
    st.metric("p-Value", f"{p_val:.4f}")

with col3:
    st.metric("Sample Std Dev", f"{s1:.2f}")

with col4:
    st.metric("Sample Size", f"{n1}")

# --- SUMMARY CARD ---
if reject:
    verdict_text = f"At α = {alpha}, the t-statistic ({t_stat:.2f}) falls in the rejection region (p-value = {p_val:.4f}). We <b>reject the null hypothesis (H₀)</b>."
else:
    verdict_text = f"At α = {alpha}, the t-statistic ({t_stat:.2f}) does not fall in the rejection region (p-value = {p_val:.4f}). We <b>fail to reject the null hypothesis (H₀)</b>."

st.markdown(f"""
    <div class="summary-card">
        <h4 style="margin-top:0; color:#1E3A8A; font-size:16px;">Prediction Summary</h4>
        <p style="margin-bottom:0; color:#334155; font-size:14px;">
            {scenario_desc}<br><br>
            <b>Result:</b> {verdict_text}
        </p>
    </div>
""", unsafe_allow_html=True)

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📈 Distribution Plot", "📝 Mathematical Proof", "⚖️ Decision Errors"])

with tab1:
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.linspace(-4, 4, 1000)
    y = stats.t.pdf(x, df)
    
    ax.plot(x, y, label=f"t-distribution (df={df})", color="#1E3A8A", lw=2)

    if tail_type == "Right-tailed (>)":
        ax.fill_between(x, 0, y, where=(x >= t_crit_upper), color="#DC2626", alpha=0.4, label="Rejection Region")
        ax.axvline(t_crit_upper, color="#DC2626", linestyle="--", lw=1.5, label=f"Critical Value ({t_crit_upper:.2f})")
    elif tail_type == "Left-tailed (<)":
        ax.fill_between(x, 0, y, where=(x <= t_crit_lower), color="#DC2626", alpha=0.4, label="Rejection Region")
        ax.axvline(t_crit_lower, color="#DC2626", linestyle="--", lw=1.5, label=f"Critical Value ({t_crit_lower:.2f})")
    else:
        ax.fill_between(x, 0, y, where=(x >= t_crit_upper) | (x <= t_crit_lower), color="#DC2626", alpha=0.4, label="Rejection Region")
        ax.axvline(t_crit_upper, color="#DC2626", linestyle="--", lw=1.5, label=f"Critical Upper ({t_crit_upper:.2f})")
        ax.axvline(t_crit_lower, color="#DC2626", linestyle="--", lw=1.5, label=f"Critical Lower ({t_crit_lower:.2f})")

    ax.axvline(t_stat, color="#16A34A", linestyle="-", lw=2.5, label=f"Calculated t ({t_stat:.2f})")
    ax.set_xlabel("t-Score")
    ax.set_ylabel("Probability Density")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    
    st.pyplot(fig)

with tab2:
    if test_type == "One-Sample t-Test":
        st.markdown(f"""
        **1. Formulate Hypotheses:**
        * Null Hypothesis ($H_0$): $\mu \le {mu_0}$
        * Alternative Hypothesis ($H_1$): $\mu > {mu_0}$

        **2. Standard Error ($SE$):**
        $$SE = \\frac{{s}}{{\\sqrt{{n}}}} = \\frac{{{s1:.2f}}}{{\\sqrt{{{n1}}}}} = {se:.4f}$$

        **3. Calculate t-Statistic:**
        $$t = \\frac{{\\bar{{x}} - \\mu_0}}{{SE}} = \\frac{{{x_bar1:.2f} - {mu_0}}}{{{se:.4f}}} = {t_stat:.4f}$$
        """)
    else:
        st.markdown(f"""
        **1. Formulate Hypotheses:**
        * Null Hypothesis ($H_0$): $\mu_1 = \mu_2$
        * Alternative Hypothesis ($H_1$): $\mu_1 \\neq \mu_2$

        **2. Pooled Variance ($s_p^2$):**
        $$s_p^2 = \\frac{{(n_1-1)s_1^2 + (n_2-1)s_2^2}}{{n_1 + n_2 - 2}} = {sp2:.4f}$$

        **3. Standard Error ($SE$):**
        $$SE = \\sqrt{{s_p^2 \\left(\\frac{{1}}{{n_1}} + \\frac{{1}}{{n_2}}\\right)}} = {se:.4f}$$

        **4. Calculate t-Statistic:**
        $$t = \\frac{{\\bar{{x}}_1 - \\bar{{x}}_2}}{{SE}} = \\frac{{{x_bar1:.2f} - {x_bar2:.2f}}}{{{se:.4f}}} = {t_stat:.4f}$$
        """)

with tab3:
    st.table({
        "Reality / Decision": ["H₀ is actually TRUE", "H₀ is actually FALSE"],
        "Fail to Reject H₀": ["Correct Decision (1 - α)", "Type II Error (β) - False Negative"],
        "Reject H₀": [f"Type I Error (α) = {alpha} - False Positive", "Correct Decision / Power (1 - β)"]
    })