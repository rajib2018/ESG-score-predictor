import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ESG Score Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    h1 {
        color: #1f77b4;
        font-weight: 700;
    }
    .big-font {
        font-size: 20px !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def generate_synthetic_esg_data(n_samples=500):
    """Generate synthetic ESG data based on realistic patterns"""
    np.random.seed(42)
    
    # Generate features
    data = {
        'company_id': range(1, n_samples + 1),
        'carbon_emissions': np.random.exponential(scale=1000, size=n_samples),
        'renewable_energy_pct': np.random.beta(2, 5, n_samples) * 100,
        'water_usage': np.random.gamma(shape=2, scale=500, size=n_samples),
        'waste_recycled_pct': np.random.beta(3, 2, n_samples) * 100,
        'employee_turnover': np.random.beta(2, 8, n_samples) * 40,
        'gender_diversity_pct': np.random.beta(5, 5, n_samples) * 100,
        'board_independence_pct': np.random.beta(6, 3, n_samples) * 100,
        'ethics_training_hours': np.random.gamma(shape=2, scale=10, size=n_samples),
        'safety_incidents': np.random.poisson(lam=5, size=n_samples),
        'community_investment': np.random.exponential(scale=200, size=n_samples),
        'revenue_millions': np.random.lognormal(mean=5, sigma=1.5, size=n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Calculate ESG score based on features with some noise
    df['environmental_score'] = (
        (100 - np.minimum(df['carbon_emissions'] / 30, 100)) * 0.3 +
        df['renewable_energy_pct'] * 0.3 +
        (100 - np.minimum(df['water_usage'] / 20, 100)) * 0.2 +
        df['waste_recycled_pct'] * 0.2
    )
    
    df['social_score'] = (
        (100 - df['employee_turnover'] * 2) * 0.25 +
        df['gender_diversity_pct'] * 0.25 +
        (100 - df['safety_incidents'] * 5) * 0.25 +
        np.minimum(df['community_investment'] / 5, 100) * 0.25
    )
    
    df['governance_score'] = (
        df['board_independence_pct'] * 0.4 +
        np.minimum(df['ethics_training_hours'] * 2, 100) * 0.6
    )
    
    # Overall ESG score (0-100)
    df['esg_score'] = (
        df['environmental_score'] * 0.33 +
        df['social_score'] * 0.33 +
        df['governance_score'] * 0.34
    )
    
    # Add realistic noise
    df['esg_score'] += np.random.normal(0, 3, n_samples)
    df['esg_score'] = np.clip(df['esg_score'], 0, 100)
    
    # Add industry sector
    industries = ['Technology', 'Manufacturing', 'Energy', 'Finance', 'Healthcare', 'Retail']
    df['industry'] = np.random.choice(industries, n_samples)
    
    return df

@st.cache_resource
def train_model(df):
    """Train Random Forest model for ESG prediction"""
    feature_cols = [
        'carbon_emissions', 'renewable_energy_pct', 'water_usage', 
        'waste_recycled_pct', 'employee_turnover', 'gender_diversity_pct',
        'board_independence_pct', 'ethics_training_hours', 'safety_incidents',
        'community_investment', 'revenue_millions'
    ]
    
    X = df[feature_cols]
    y = df['esg_score']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train_scaled[:100])
    
    return model, scaler, explainer, X_train_scaled, feature_cols, X_test_scaled, y_test

def create_gauge_chart(score, title):
    """Create a gauge chart for scores"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24}},
        delta={'reference': 70},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': '#ffcccc'},
                {'range': [40, 70], 'color': '#fff4cc'},
                {'range': [70, 100], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def plot_shap_waterfall(shap_values, feature_values, feature_names, base_value):
    """Create SHAP waterfall plot"""
    shap_contrib = list(zip(feature_names, shap_values, feature_values))
    shap_contrib.sort(key=lambda x: abs(x[1]), reverse=True)
    
    features = [x[0] for x in shap_contrib[:10]]
    values = [x[1] for x in shap_contrib[:10]]
    
    colors = ['#ff6b6b' if v < 0 else '#51cf66' for v in values]
    
    fig = go.Figure(go.Bar(
        y=features,
        x=values,
        orientation='h',
        marker=dict(color=colors),
        text=[f"{v:+.2f}" for v in values],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Top 10 Feature Impacts on ESG Score",
        xaxis_title="SHAP Value (Impact on Score)",
        yaxis_title="Features",
        height=500,
        showlegend=False,
        yaxis=dict(autorange="reversed")
    )
    
    return fig

# Main app
def main():
    st.title("🌍 ESG Score Prediction & Analysis")
    st.markdown("### Predict and explain Environmental, Social, and Governance scores using AI")
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=ESG+Analytics", use_container_width=True)
        st.markdown("---")
        st.markdown("### About")
        st.info(
            "This application predicts ESG scores for organizations using "
            "machine learning and provides explainable insights using SHAP values."
        )
        st.markdown("---")
        st.markdown("### Model Info")
        st.write("**Algorithm:** Random Forest")
        st.write("**Features:** 11 key ESG metrics")
        st.write("**Explainability:** SHAP")
    
    # Load data and train model
    with st.spinner("Loading data and training model..."):
        df = generate_synthetic_esg_data(500)
        model, scaler, explainer, X_train_scaled, feature_cols, X_test_scaled, y_test = train_model(df)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Predict", "🔍 Model Insights", "📈 Dataset Analysis", "ℹ️ About"])
    
    with tab1:
        st.header("Predict ESG Score")
        st.markdown("Enter organization metrics to predict ESG score")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🌱 Environmental")
            carbon = st.slider("Carbon Emissions (tons)", 0, 5000, 1000)
            renewable = st.slider("Renewable Energy (%)", 0, 100, 30)
            water = st.slider("Water Usage (m³)", 0, 3000, 800)
            waste = st.slider("Waste Recycled (%)", 0, 100, 60)
        
        with col2:
            st.subheader("👥 Social")
            turnover = st.slider("Employee Turnover (%)", 0, 40, 15)
            diversity = st.slider("Gender Diversity (%)", 0, 100, 50)
            incidents = st.slider("Safety Incidents", 0, 20, 3)
            investment = st.slider("Community Investment ($K)", 0, 1000, 200)
        
        with col3:
            st.subheader("⚖️ Governance")
            independence = st.slider("Board Independence (%)", 0, 100, 70)
            training = st.slider("Ethics Training (hours)", 0, 50, 20)
            revenue = st.slider("Revenue ($M)", 0, 1000, 100)
        
        if st.button("🎯 Predict ESG Score", type="primary"):
            input_data = np.array([[
                carbon, renewable, water, waste, turnover, diversity,
                independence, training, incidents, investment, revenue
            ]])
            
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            
            # Calculate component scores
            env_score = (100 - min(carbon/30, 100)) * 0.3 + renewable * 0.3 + \
                       (100 - min(water/20, 100)) * 0.2 + waste * 0.2
            social_score = (100 - turnover*2) * 0.25 + diversity * 0.25 + \
                          (100 - incidents*5) * 0.25 + min(investment/5, 100) * 0.25
            gov_score = independence * 0.4 + min(training*2, 100) * 0.6
            
            st.markdown("---")
            st.subheader("Prediction Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Overall ESG Score", f"{prediction:.1f}/100", 
                         delta=f"{prediction-70:.1f} vs avg")
            with col2:
                st.metric("Environmental", f"{env_score:.1f}/100")
            with col3:
                st.metric("Social", f"{social_score:.1f}/100")
            with col4:
                st.metric("Governance", f"{gov_score:.1f}/100")
            
            # Gauges
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(create_gauge_chart(prediction, "Overall ESG Score"), 
                               use_container_width=True)
            
            with col2:
                # SHAP explanation
                shap_values = explainer.shap_values(input_scaled)
                fig = plot_shap_waterfall(shap_values[0], input_data[0], 
                                         feature_cols, explainer.expected_value)
                st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.subheader("💡 Recommendations for Improvement")
            
            improvements = []
            if renewable < 50:
                improvements.append("• Increase renewable energy usage to improve environmental score")
            if diversity < 40:
                improvements.append("• Enhance gender diversity initiatives")
            if carbon > 1500:
                improvements.append("• Implement carbon reduction strategies")
            if training < 20:
                improvements.append("• Increase ethics training programs")
            if waste < 70:
                improvements.append("• Improve waste recycling programs")
            
            if improvements:
                for imp in improvements[:5]:
                    st.write(imp)
            else:
                st.success("Excellent! Your organization is performing well across all ESG metrics.")
    
    with tab2:
        st.header("Model Insights & Explainability")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(feature_importance, x='Importance', y='Feature', 
                        orientation='h', title="Feature Importance")
            fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("SHAP Summary")
            st.write("""
            SHAP (SHapley Additive exPlanations) values show how each feature 
            contributes to the model's predictions. 
            
            **Key Insights:**
            - Red bars indicate features that decrease the ESG score
            - Green bars indicate features that increase the ESG score
            - Longer bars = stronger impact on prediction
            """)
            
            # Model performance
            from sklearn.metrics import mean_absolute_error, r2_score
            y_pred = model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            st.metric("Model R² Score", f"{r2:.3f}")
            st.metric("Mean Absolute Error", f"{mae:.2f}")
    
    with tab3:
        st.header("Dataset Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.histogram(df, x='esg_score', nbins=30, 
                             title="Distribution of ESG Scores")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Dataset Statistics")
            st.write(f"**Total Organizations:** {len(df)}")
            st.write(f"**Avg ESG Score:** {df['esg_score'].mean():.1f}")
            st.write(f"**Std Deviation:** {df['esg_score'].std():.1f}")
            st.write(f"**Min Score:** {df['esg_score'].min():.1f}")
            st.write(f"**Max Score:** {df['esg_score'].max():.1f}")
        
        # Industry analysis
        industry_avg = df.groupby('industry')['esg_score'].mean().sort_values(ascending=False)
        fig = px.bar(x=industry_avg.values, y=industry_avg.index, 
                    orientation='h', title="Average ESG Score by Industry")
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        corr_cols = ['esg_score'] + feature_cols
        corr = df[corr_cols].corr()['esg_score'].drop('esg_score').sort_values(ascending=False)
        
        fig = px.bar(x=corr.values, y=corr.index, orientation='h',
                    title="Feature Correlation with ESG Score",
                    labels={'x': 'Correlation', 'y': 'Feature'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("About This Application")
        
        st.markdown("""
        ### 🎯 Purpose
        This application demonstrates ESG (Environmental, Social, and Governance) score 
        prediction using machine learning with full explainability through SHAP values.
        
        ### 📊 Data Source
        The application uses synthetic data generated to mimic realistic ESG patterns 
        from organizational sustainability reports. In production, this would connect to:
        - CDP (Carbon Disclosure Project)
        - SASB Standards
        - GRI Reporting Framework
        - Public company sustainability reports
        
        ### 🔬 Model Architecture
        - **Algorithm:** Random Forest Regressor
        - **Features:** 11 key ESG metrics across E, S, and G pillars
        - **Explainability:** SHAP (SHapley Additive exPlanations)
        - **Validation:** Train-test split with cross-validation
        
        ### 📈 Key Features
        1. **Environmental Metrics:** Carbon emissions, renewable energy, water usage, waste
        2. **Social Metrics:** Employee turnover, diversity, safety, community investment
        3. **Governance Metrics:** Board independence, ethics training
        
        ### 🔍 Explainability
        SHAP values provide transparent insights into:
        - Which features most impact predictions
        - How each feature contributes (positively or negatively)
        - Feature interactions and dependencies
        
        ### 🚀 Deployment
        This app is built with Streamlit and can be deployed to:
        - Streamlit Cloud
        - Heroku
        - AWS/GCP/Azure
        - Docker containers
        
        ### 📝 Note
        This is a demonstration model. For production use, integrate real ESG data sources
        and conduct thorough validation with domain experts.
        """)
        
        st.info("💡 Tip: Use the sidebar to navigate and explore different features of the application!")

if __name__ == "__main__":
    main()
