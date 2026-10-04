import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Fundamentals of Hypothesis Testing",
    page_icon="📊",
    layout="wide"
)

# --- TITLE & DESCRIPTION ---
st.title("📊 Fundamentals of Hypothesis Testing")
st.markdown("""
An interactive statistical calculator and visualizer for **one-sample** and **two-sample** hypothesis testing.
Choose manual inputs/presets or upload your own raw dataset CSV to calculate test statistics, critical values, $p$-values, and decision rules.
""")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Data Input Method")
input_mode = st.sidebar.radio("Select Input Source:", ["Manual Input / Presets", "Upload CSV File"])

if input_mode == "Manual Input / Presets":
    preset = st.sidebar.selectbox(
        "Choose a Sample Scenario:",
        ["Custom Input", "Study App Scores (1-Sample)", "Study Method Comparison (2-Sample)"]
    )

    if preset == "Study App Scores (1-Sample)":
        test_type = "One-Sample t-Test"
        tail_type = "Right-tailed (>)"
        alpha = 0.05
        mu_0 = 70.0
        x_bar1, s1, n1 = 74.2, 8.0, 25
        x_bar2, s2, n2 = 70.0, 7.5, 30
    elif preset == "Study Method Comparison (2-Sample)":
        test_type = "Two-Sample Independent t-Test"
        tail_type = "Two-tailed (≠)"
        alpha = 0.05
        mu_0 = 0.0
        x_bar1, s1, n1 = 71.0, 7.5, 30
        x_bar2, s2, n2 = 76.5, 8.1, 30
    else:
        test_type = st.sidebar.selectbox("Select Test Type:", ["One-Sample t-Test", "Two-Sample Independent t-Test"])
        tail_type = st.sidebar.selectbox("Select Tail Type:", ["Two-tailed (≠)", "Right-tailed (>)", "Left-tailed (<)"])
        alpha = st.sidebar.slider("Significance Level (α):", 0.01, 0.10, 0.05, step=0.01)
        
        st.sidebar.subheader("Sample 1 Data")
        x_bar1 = st.sidebar.number_input("Sample Mean (x̄₁):", value=74.2)
        s1 = st.sidebar.number_input("Sample Std Dev (s₁):", value=8.0, min_value=0.1)
        n1 = st.sidebar.number_input("Sample Size (n₁):", value=25, min_value=2, step=1)
        
        if test_type == "One-Sample t-Test":
            mu_0 = st.sidebar.number_input("Null Target (μ₀):", value=70.0)
            x_bar2, s2, n2 = 0.0, 1.0, 2
        else:
            mu_0 = 0.0
            st.sidebar.subheader("Sample 2 Data")
            x_bar2 = st.sidebar.number_input("Sample Mean (x̄₂):", value=71.0)
            s2 = st.sidebar.number_input("Sample Std Dev (s₂):", value=7.5, min_value=0.1)
            n2 = st.sidebar.number_input("Sample Size (n₂):", value=30, min_value=2, step=1)

else:  # Upload CSV File Mode
    st.sidebar.subheader("📁 File Upload")
    
    # Sample CSV Generator & Download Button
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    sample_df = pd.DataFrame({
        "Control_Group_Scores": [68, 72, 65, 74, 70, 71, 69, 73, 67, 75, 70, 68, 72, 71, 69],
        "App_Group_Scores":     [75, 80, 78, 82, 76, 79, 77, 81, 74, 83, 78, 76, 80, 79, 77]
    })
    
    st.sidebar.download_button(
        label="📥 Download Sample CSV",
        data=convert_df_to_csv(sample_df),
        file_name="hypothesis_sample_data.csv",
        mime="text/csv",
        help="Download sample data to try out the file uploader."
    )
    
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
    
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
        st.sidebar.success("File uploaded successfully!")
        
        test_type = st.sidebar.selectbox("Select Test Type:", ["One-Sample t-Test", "Two-Sample Independent t-Test"])
        tail_type = st.sidebar.selectbox("Select Tail Type:", ["Two-tailed (≠)", "Right-tailed (>)", "Left-tailed (<)"])
        alpha = st.sidebar.slider("Significance Level (α):", 0.01, 0.10, 0.05, step=0.01)
        
        numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            st.error("No numeric columns found in the uploaded CSV file.")
            st.stop()
            
        if test_type == "One-Sample t-Test":
            col1 = st.sidebar.selectbox("Select Column for Sample 1:", numeric_cols)
            mu_0 = st.sidebar.number_input("Null Target Mean (μ₀):", value=70.0)
            
            data1 = df_raw[col1].dropna()
            x_bar1, s1, n1 = data1.mean(), data1.std(ddof=1), len(data1)
            x_bar2, s2, n2 = 0.0, 1.0, 2
            
        else:  # Two-Sample t-Test
            col1 = st.sidebar.selectbox("Select Column for Sample 1:", numeric_cols, index=0)
            col2_default = 1 if len(numeric_cols) > 1 else 0
            col2 = st.sidebar.selectbox("Select Column for Sample 2:", numeric_cols, index=col2_default)
            mu_0 = 0.0
            
            data1 = df_raw[col1].dropna()
            data2 = df_raw[col2].dropna()
            
            x_bar1, s1, n1 = data1.mean(), data1.std(ddof=1), len(data1)
            x_bar2, s2, n2 = data2.mean(), data2.std(ddof=1), len(data2)

        with st.expander("👀 View Uploaded Dataset Preview"):
            st.dataframe(df_raw.head())

    else:
        st.info("👈 Upload a CSV file or download the sample dataset above to test.")
        st.stop()

# --- CALCULATIONS ENGINE ---
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

# --- MAIN DASHBOARD TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Results & Visualization", "📝 Step-by-Step Proof", "⚖️ Decision Error Matrix"])

with tab1:
    if reject:
        st.error(f"🔴 **VERDICT: REJECT NULL HYPOTHESIS ($H_0$)** at α = {alpha}")
    else:
        st.success(f"🟢 **VERDICT: FAIL TO REJECT NULL HYPOTHESIS ($H_0$)** at α = {alpha}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Calculated t-Statistic", f"{t_stat:.3f}")
    col2.metric("p-Value", f"{p_val:.4f}")
    col3.metric("Significance Level (α)", f"{alpha}")
    col4.metric("Degrees of Freedom (df)", f"{df}")

    st.subheader("Distribution Curve & Rejection Region")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.linspace(-4, 4, 1000)
    y = stats.t.pdf(x, df)
    ax.plot(x, y, label=f"t-distribution (df={df})", color="navy", lw=2)

    if tail_type == "Right-tailed (>)":
        ax.fill_between(x, 0, y, where=(x >= t_crit_upper), color="red", alpha=0.4, label="Rejection Region")
        ax.axvline(t_crit_upper, color="red", linestyle="--", lw=1.5, label=f"Critical Value ({t_crit_upper:.3f})")
    elif tail_type == "Left-tailed (<)":
        ax.fill_between(x, 0, y, where=(x <= t_crit_lower), color="red", alpha=0.4, label="Rejection Region")
        ax.axvline(t_crit_lower, color="red", linestyle="--", lw=1.5, label=f"Critical Value ({t_crit_lower:.3f})")
    else:
        ax.fill_between(x, 0, y, where=(x >= t_crit_upper) | (x <= t_crit_lower), color="red", alpha=0.4, label="Rejection Region")
        ax.axvline(t_crit_upper, color="red", linestyle="--", lw=1.5, label=f"Critical Upper ({t_crit_upper:.3f})")
        ax.axvline(t_crit_lower, color="red", linestyle="--", lw=1.5, label=f"Critical Lower ({t_crit_lower:.3f})")

    ax.axvline(t_stat, color="green", linestyle="-", lw=2.5, label=f"Calculated t ({t_stat:.3f})")
    ax.set_title("Hypothesis Test Visualized", fontsize=12)
    ax.set_xlabel("t-Score")
    ax.set_ylabel("Probability Density")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    
    st.pyplot(fig)

with tab2:
    st.subheader("Mathematical Derivation")
    
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
    st.subheader("Decision Errors Matrix")
    st.table({
        "Reality / Decision": ["H₀ is actually TRUE", "H₀ is actually FALSE"],
        "Fail to Reject H₀": ["Correct Decision (1 - α)", "Type II Error (β) - False Negative"],
        "Reject H₀": [f"Type I Error (α) = {alpha} - False Positive", "Correct Decision / Power (1 - β)"]
    })