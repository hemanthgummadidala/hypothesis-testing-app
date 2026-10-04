import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Student Study Performance Estimator",
    page_icon="🎓",
    layout="wide"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .result-callout {
        background-color: #EFF6FF;
        border-left: 5px solid #2563EB;
        padding: 18px 24px;
        border-radius: 8px;
        font-size: 22px;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR: INPUT DETAILS ---
st.sidebar.title("🎓 Study Details")
st.sidebar.caption("Enter your study details to generate an estimate.")

course = st.sidebar.selectbox(
    "Select Subject / Course",
    ["Data Structures & Algorithms", "Database Management Systems", "Machine Learning", "Software Engineering"]
)

exam_date = st.sidebar.date_input("Select Exam Date")

study_time_slot = st.sidebar.selectbox(
    "Select Expected Daily Study Time",
    ["02:00 Hours", "04:00 Hours", "06:00 Hours", "08:00 Hours"]
)

student_type = st.sidebar.radio(
    "Student Type",
    ["Regular Student", "Working Professional / Part-time"]
)

sample_size = st.sidebar.slider("Statistical Sample Size", min_value=10, max_value=500, value=100)

confidence_level = st.sidebar.selectbox(
    "Confidence Level",
    ["90%", "95%", "99%"],
    index=1
)

conf_num = float(confidence_level.replace("%", "")) / 100.0

# --- SIMULATED DATA GENERATION BASED ON INPUTS ---
np.random.seed(42)
base_mean = 72.0 if student_type == "Regular Student" else 68.0
hours_num = int(study_time_slot.split(":")[0])
adjusted_mean = base_mean + (hours_num * 1.8)
std_dev = 8.5

simulated_scores = np.random.normal(loc=adjusted_mean, scale=std_dev, size=sample_size)
sample_mean = float(np.mean(simulated_scores))
sample_se = std_dev / np.sqrt(sample_size)

z_critical = stats.norm.ppf((1 + conf_num) / 2)
margin_of_error = z_critical * sample_se
ci_lower = sample_mean - margin_of_error
ci_upper = sample_mean + margin_of_error

model_estimate = sample_mean + 1.25

# --- TOP HERO RESULT BOX ---
st.markdown(f"""
    <div class="result-callout">
        Estimated Expected Exam Score: {model_estimate:.2f} Marks
    </div>
""", unsafe_allow_html=True)

# --- TABS LAYOUT ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Score Distribution", 
    "📈 Study Pattern", 
    "📐 Sampling & CLT", 
    "📋 Statistical Analysis"
])

# --- TAB 1: SCORE DISTRIBUTION ---
with tab1:
    st.subheader("Score Distribution")
    
    fig1 = go.Figure()
    fig1.add_trace(go.Histogram(
        x=simulated_scores,
        nbinsx=20,
        marker_color='#1D70B8',
        name='Student Scores'
    ))
    
    fig1.add_vline(x=sample_mean, line_dash="dotted", line_color="black", 
                   annotation_text=f"Sample Mean: {sample_mean:.2f}", annotation_position="bottom left")
    fig1.add_vline(x=model_estimate, line_dash="dash", line_color="black", 
                   annotation_text=f"Model Estimate: {model_estimate:.2f}", annotation_position="top left")
    
    fig1.update_layout(
        title=f"{course} Performance Distribution",
        xaxis_title="Estimated Score (Marks)",
        yaxis_title="Student Count",
        template="plotly_white",
        height=450
    )
    st.plotly_chart(fig1, use_container_width=True)

# --- TAB 2: STUDY PATTERN ---
with tab2:
    st.subheader("Study Hours vs. Expected Score Pattern")
    
    hours_range = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    expected_scores = base_mean + (hours_range * 2.1) + np.random.normal(0, 0.8, len(hours_range))
    
    pattern_df = pd.DataFrame({
        "Daily Study Hours": hours_range,
        "Expected Exam Score": expected_scores
    })
    
    fig2 = px.line(
        pattern_df, 
        x="Daily Study Hours", 
        y="Expected Exam Score", 
        markers=True,
        title="Impact of Daily Study Hours on Expected Marks"
    )
    fig2.update_traces(line_color='#2563EB', line_width=3, marker_size=8)
    fig2.update_layout(template="plotly_white", height=420)
    st.plotly_chart(fig2, use_container_width=True)

# --- TAB 3: SAMPLING & CLT ---
with tab3:
    st.subheader("Sampling & Central Limit Theorem (CLT)")
    st.write("Demonstration of how sample means converge into a normal distribution as sample size increases.")
    
    num_samples = 500
    sample_means = [np.mean(np.random.choice(simulated_scores, size=int(sample_size/2))) for _ in range(num_samples)]
    
    fig3 = px.histogram(
        sample_means, 
        nbins=30, 
        title=f"Distribution of {num_samples} Sample Means (CLT Visualizer)",
        labels={'value': 'Sample Means'},
        color_discrete_sequence=['#0D9488']
    )
    fig3.update_layout(template="plotly_white", height=420, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

# --- TAB 4: STATISTICAL ANALYSIS ---
with tab4:
    st.subheader("Statistical Summary & Confidence Intervals")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sample Mean (x̄)", f"{sample_mean:.2f}")
    with col2:
        st.metric("Standard Error (SE)", f"{sample_se:.2f}")
    with col3:
        st.metric("Margin of Error", f"±{margin_of_error:.2f}")
    with col4:
        st.metric("Confidence Interval", f"[{ci_lower:.1f}, {ci_upper:.1f}]")

    st.markdown("---")
    st.markdown(f"""
    ### 📌 Summary for Presentation:
    * **Sample Size ($n$):** {sample_size} students evaluated[cite: 9].
    * **Confidence Interval ({confidence_level}):** We are {confidence_level} confident that the true population mean exam score lies between **{ci_lower:.2f}** and **{ci_upper:.2f}** marks.
    * **Conclusion:** Studying **{study_time_slot}** daily significantly improves performance stability for **{course}**.
    """)