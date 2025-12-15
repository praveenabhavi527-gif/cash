import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Salary Predictor AI",
    page_icon="💰",
    layout="wide"
)

# Custom CSS for a "very design" look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4B4B4B;
        text-align: center;
        font-weight: 700;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 40px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Data Loading Logic
# -----------------------------------------------------------------------------
@st.cache_data
def generate_mock_data():
    """Generates synthetic data if no file is uploaded."""
    np.random.seed(42)
    experience = np.random.uniform(0, 25, 100)
    # y = mx + c + noise (approx 10k per year + 30k base)
    salary = 10000 * experience + 30000 + np.random.normal(0, 5000, 100) 
    return pd.DataFrame({
        'Years of Experience': experience, 
        'Salary': salary
    })

def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    else:
        return generate_mock_data()

# -----------------------------------------------------------------------------
# 3. Sidebar (Controls)
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Configuration")
st.sidebar.write("Upload your dataset or use the built-in demo data.")

uploaded_file = st.sidebar.file_uploader("Upload Salary Data (CSV)", type=['csv'])
use_mock = False

if uploaded_file is None:
    st.sidebar.info("ℹ️ Using Demo Data (Linear Pattern)")
    df = generate_mock_data()
else:
    df = load_data(uploaded_file)

# -----------------------------------------------------------------------------
# 4. Main Application Logic
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">💰 Salary Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Linear Regression Model based on Years of Experience</div>', unsafe_allow_html=True)

if df is not None:
    # --- Data Preprocessing (Cleaning based on PDF logic) ---
    # Dropping missing values
    df_clean = df.dropna()
    
    # Ensure necessary columns exist
    required_cols = ['Years of Experience', 'Salary']
    if not all(col in df_clean.columns for col in required_cols):
        st.error(f"Dataset must contain columns: {required_cols}")
        st.stop()

    # Define Features (X) and Target (y)
    X = df_clean[['Years of Experience']]
    y = df_clean['Salary']

    # --- Model Training ---
    lr = LinearRegression()
    lr.fit(X, y)

    # --- Layout: Split into Prediction and Visualization ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 🔮 Predict Salary")
        
        # User Input
        min_exp = float(df_clean['Years of Experience'].min())
        max_exp = float(df_clean['Years of Experience'].max())
        
        user_exp = st.slider(
            "Select Years of Experience:",
            min_value=0.0,
            max_value=max_exp + 5.0, # Allow predicting slightly beyond data
            value=5.0,
            step=0.5
        )

        # Make Prediction
        prediction = lr.predict([[user_exp]])[0]

        # Display Result nicely
        st.markdown(f"""
        <div class="metric-card">
            <h3>Estimated Salary</h3>
            <h1 style="color: #2e7bcf;">${prediction:,.2f}</h1>
            <p>For {user_exp} years of experience</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📊 Dataset Stats")
        st.write(df_clean.describe().round(2))

    with col2:
        st.markdown("### 📈 Regression Analysis")
        
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter plot of actual data
        ax.scatter(X, y, color='#2e7bcf', alpha=0.6, label='Actual Data')
        
        # Regression Line
        # Create a range for the line
        x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_pred_line = lr.predict(x_range)
        ax.plot(x_range, y_pred_line, color='#ff4b4b', linewidth=3, label='Regression Line')
        
        # Plot the user's specific prediction point
        ax.scatter([user_exp], [prediction], color='black', s=150, zorder=5, label='Your Prediction')
        
        ax.set_xlabel("Years of Experience")
        ax.set_ylabel("Salary")
        ax.set_title("Experience vs. Salary")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        
        st.pyplot(fig)

    # --- Show Raw Data Expander ---
    with st.expander("🔎 View Raw Data"):
        st.dataframe(df_clean)

else:
    st.warning("Please upload a valid CSV file.")
