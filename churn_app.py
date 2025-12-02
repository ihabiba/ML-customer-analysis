"""
Customer Churn Prediction App
Multi-page Streamlit application with navigation.
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Churn Prediction",
    page_icon="◐",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .main .block-container {
        padding: 1rem 2rem 2rem 2rem;
        max-width: 1400px;
    }
    
    /* Navigation tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8fafc;
        padding: 0.5rem;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1.5rem;
        border-radius: 8px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
    }
    
    /* Result cards */
    .result-card {
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 1rem 0;
    }
    .result-churn {
        background: linear-gradient(135deg, #fecaca 0%, #fca5a5 100%);
        border: 2px solid #ef4444;
    }
    .result-churn h2 { color: #b91c1c; margin: 0; }
    .result-no-churn {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border: 2px solid #10b981;
    }
    .result-no-churn h2 { color: #047857; margin: 0; }
    
    /* Action box */
    .action-box {
        background: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .action-box-warning {
        border-left-color: #f59e0b;
        background: #fffbeb;
    }
    
    /* Section styling */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #374151;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

# Load models and data with error handling
@st.cache_resource
def load_models():
    try:
        models = {
            'Logistic Regression': joblib.load('models/churn_logistic_regression.pkl'),
            'K-Nearest Neighbors': joblib.load('models/churn_knn.pkl'),
            'Decision Tree': joblib.load('models/churn_decision_tree_tuned.pkl'),
            'SVM': joblib.load('models/churn_svm.pkl')
        }
        scaler = joblib.load('models/churn_scaler.pkl')
        features = joblib.load('models/churn_features.pkl')
        metrics = joblib.load('models/churn_metrics.pkl')
        metrics_fixed = {
            'Logistic Regression': metrics['Logistic Regression'],
            'K-Nearest Neighbors': metrics['K-Nearest Neighbors'],
            'Decision Tree': metrics['Decision Tree (Tuned)'],
            'SVM': metrics['Support Vector Machine']
        }
        return models, scaler, features, metrics_fixed
    except FileNotFoundError as e:
        st.error(f"Model files not found. Please ensure the 'models/' folder exists with trained models. Error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

@st.cache_data
def load_dataset():
    try:
        return pd.read_csv('Datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'Datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv' exists.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

models, scaler, features, metrics = load_models()
dataset = load_dataset()

# ============ HEADER ============
st.title("Customer Churn Prediction")

# ============ NAVIGATION TABS ============
tab1, tab2, tab3 = st.tabs(["Predict", "Model Comparison", "Dataset Explorer"])

# ============ TAB 1: PREDICT ============
with tab1:
    st.markdown("Enter customer information to predict churn risk")
    
    # Sidebar for model selection
    with st.sidebar:
        st.markdown("### Settings")
        selected_model_name = st.selectbox("Select Model", list(models.keys()))
        
        st.markdown("---")
        st.markdown("### Selected Model Metrics")
        m = metrics[selected_model_name]
        st.metric("Accuracy", f"{m['accuracy']:.1%}")
        st.metric("F1 Score", f"{m['f1_score']:.1%}")
        if m['auc']:
            st.metric("AUC-ROC", f"{m['auc']:.1%}")
        
        st.markdown("---")
        st.markdown("### Dataset Info")
        st.caption(f"Samples: {len(dataset):,}")
        churn_rate = (dataset['Churn'] == 'Yes').mean()
        st.caption(f"Churn rate: {churn_rate:.1%}")
    
    # Customer input form wrapped in st.form to prevent reruns
    with st.form("churn_prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Account Details**")
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 50.0, step=5.0)
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 600.0, step=50.0)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            paperless_billing = st.toggle("Paperless Billing", value=True)

        with col2:
            st.markdown("**Services**")
            phone_service = st.toggle("Phone Service", value=True)
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])

        with col3:
            st.markdown("**Profile & More Services**")
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
            senior_citizen = st.toggle("Senior Citizen")
            partner = st.toggle("Has Partner")
            dependents = st.toggle("Has Dependents")

        st.markdown("---")
        submitted = st.form_submit_button("Predict Churn Risk", type="primary", use_container_width=True)
    
    if submitted:
        # Initialize all features to 0 first for safety, then update
        row = {col: 0 for col in features}
        
        # Update with user inputs
        row.update({
            'gender': 1 if gender == "Male" else 0,
            'SeniorCitizen': 1 if senior_citizen else 0,
            'Partner': 1 if partner else 0,
            'Dependents': 1 if dependents else 0,
            'tenure': tenure,
            'PhoneService': 1 if phone_service else 0,
            'MultipleLines': 1 if multiple_lines == "Yes" else 0,
            'OnlineSecurity': 1 if online_security == "Yes" else 0,
            'OnlineBackup': 1 if online_backup == "Yes" else 0,
            'DeviceProtection': 1 if device_protection == "Yes" else 0,
            'TechSupport': 1 if tech_support == "Yes" else 0,
            'StreamingTV': 1 if streaming_tv == "Yes" else 0,
            'StreamingMovies': 1 if streaming_movies == "Yes" else 0,
            'PaperlessBilling': 1 if paperless_billing else 0,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'InternetService_DSL': 1 if internet_service == "DSL" else 0,
            'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
            'InternetService_No': 1 if internet_service == "No" else 0,
            'Contract_Month-to-month': 1 if contract == "Month-to-month" else 0,
            'Contract_One year': 1 if contract == "One year" else 0,
            'Contract_Two year': 1 if contract == "Two year" else 0,
            'PaymentMethod_Bank transfer (automatic)': 1 if payment_method == "Bank transfer (automatic)" else 0,
            'PaymentMethod_Credit card (automatic)': 1 if payment_method == "Credit card (automatic)" else 0,
            'PaymentMethod_Electronic check': 1 if payment_method == "Electronic check" else 0,
            'PaymentMethod_Mailed check': 1 if payment_method == "Mailed check" else 0,
        })
        
        input_df = pd.DataFrame([row])
        input_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
            input_df[['tenure', 'MonthlyCharges', 'TotalCharges']]
        )
        
        X_input = input_df[features]
        
        model = models[selected_model_name]
        prediction = model.predict(X_input)[0]
        
        if hasattr(model, 'predict_proba'):
            churn_prob = model.predict_proba(X_input)[0][1]
        elif hasattr(model, 'decision_function'):
            churn_prob = 1 / (1 + np.exp(-model.decision_function(X_input)[0]))
        else:
            churn_prob = None
        
        res_col1, res_col2 = st.columns([2, 1])
        
        with res_col1:
            if prediction == 1:
                st.markdown("""
                <div class="result-card result-churn">
                    <h2>High Churn Risk</h2>
                    <p>This customer is likely to leave</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Suggested action for high churn
                st.markdown("""
                <div class="action-box action-box-warning">
                    <strong>Suggested Actions:</strong>
                    <ul style="margin: 0.5rem 0 0 0; padding-left: 1.5rem;">
                        <li>Offer retention incentives (discounts, loyalty rewards)</li>
                        <li>Consider plan downgrade options to reduce monthly costs</li>
                        <li>Schedule personalized outreach from customer success team</li>
                        <li>Review service issues and address any complaints</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-card result-no-churn">
                    <h2>Low Churn Risk</h2>
                    <p>This customer is likely to stay</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Suggested action for low churn
                st.markdown("""
                <div class="action-box">
                    <strong>Suggested Actions:</strong>
                    <ul style="margin: 0.5rem 0 0 0; padding-left: 1.5rem;">
                        <li>Maintain current service quality and plan</li>
                        <li>Monitor engagement metrics periodically</li>
                        <li>Consider upselling premium features when appropriate</li>
                        <li>Include in referral program outreach</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        with res_col2:
            if churn_prob is not None:
                st.metric("Churn Probability", f"{churn_prob:.1%}")
                st.progress(churn_prob)
                if churn_prob >= 0.7:
                    st.error("Critical Risk")
                elif churn_prob >= 0.4:
                    st.warning("Moderate Risk")
                else:
                    st.success("Low Risk")

# ============ TAB 2: MODEL COMPARISON ============
with tab2:
    st.markdown("Compare performance metrics across all trained models")
    
    metrics_df = pd.DataFrame(metrics).T.reset_index()
    metrics_df.columns = ['Model', 'Accuracy', 'F1 Score', 'AUC-ROC']
    
    st.dataframe(
        metrics_df.style.format({'Accuracy': '{:.2%}', 'F1 Score': '{:.2%}', 'AUC-ROC': '{:.2%}'}),
        use_container_width=True,
        hide_index=True
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Fixed: Use discrete colors - all bars visible now
        fig_acc = px.bar(
            metrics_df, x='Model', y='Accuracy',
            title='Model Accuracy Comparison',
            color='Model',
            color_discrete_sequence=['#1e3a5f', '#2563eb', '#3b82f6', '#0ea5e9'],
            text=metrics_df['Accuracy'].apply(lambda x: f'{x:.1%}')
        )
        fig_acc.update_traces(textposition='outside')
        fig_acc.update_layout(showlegend=False, yaxis_tickformat='.0%', yaxis_range=[0, 1])
        st.plotly_chart(fig_acc, use_container_width=True)
    
    with col2:
        # Fixed: Use discrete colors - all bars visible now
        fig_f1 = px.bar(
            metrics_df, x='Model', y='F1 Score',
            title='Model F1 Score Comparison',
            color='Model',
            color_discrete_sequence=['#064e3b', '#047857', '#10b981', '#14b8a6'],
            text=metrics_df['F1 Score'].apply(lambda x: f'{x:.1%}')
        )
        fig_f1.update_traces(textposition='outside')
        fig_f1.update_layout(showlegend=False, yaxis_tickformat='.0%', yaxis_range=[0, 1])
        st.plotly_chart(fig_f1, use_container_width=True)
    
    # Best Model Summary Card
    st.markdown("### Best Model Recommendation")
    best_model = max(metrics.items(), key=lambda x: x[1]['accuracy'])
    best_name = best_model[0]
    best_metrics = best_model[1]
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); 
                border: 2px solid #22c55e; border-radius: 16px; padding: 1.5rem; margin: 1rem 0;">
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
            <span style="font-size: 2rem;">🏆</span>
            <div>
                <h3 style="margin: 0; color: #166534;">{best_name}</h3>
                <p style="margin: 0; color: #4ade80; font-size: 0.9rem;">Recommended for Production</p>
            </div>
        </div>
        <div style="display: flex; gap: 2rem; flex-wrap: wrap;">
            <div style="text-align: center;">
                <p style="margin: 0; font-size: 1.5rem; font-weight: bold; color: #166534;">{best_metrics['accuracy']:.1%}</p>
                <p style="margin: 0; color: #6b7280; font-size: 0.85rem;">Accuracy</p>
            </div>
            <div style="text-align: center;">
                <p style="margin: 0; font-size: 1.5rem; font-weight: bold; color: #166534;">{best_metrics['f1_score']:.1%}</p>
                <p style="margin: 0; color: #6b7280; font-size: 0.85rem;">F1 Score</p>
            </div>
            <div style="text-align: center;">
                <p style="margin: 0; font-size: 1.5rem; font-weight: bold; color: #166534;">{best_metrics['auc']:.1%}</p>
                <p style="margin: 0; color: #6b7280; font-size: 0.85rem;">AUC-ROC</p>
            </div>
        </div>
        <p style="margin: 1rem 0 0 0; color: #374151; font-size: 0.9rem;">
            <strong>Why this model?</strong> Highest accuracy with balanced precision-recall trade-off, 
            making it reliable for identifying at-risk customers without excessive false alarms.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ROC Curves from training
    st.markdown("### ROC Curves")
    try:
        st.image("assets/combined_roc_curves.png", caption="ROC Curves - All Models", use_container_width=True)
    except:
        st.info("ROC curves image not found. Run the notebook to generate 'assets/combined_roc_curves.png'.")

# ============ TAB 3: DATASET EXPLORER ============
with tab3:
    st.markdown("Explore the training dataset used to build the models")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(dataset):,}")
    col2.metric("Features", len(dataset.columns))
    col3.metric("Churned", f"{(dataset['Churn'] == 'Yes').sum():,}")
    col4.metric("Retained", f"{(dataset['Churn'] == 'No').sum():,}")
    
    st.markdown("---")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        churn_counts = dataset['Churn'].value_counts()
        fig_churn = px.pie(
            values=churn_counts.values, names=churn_counts.index,
            title='Churn Distribution', color_discrete_sequence=['#10b981', '#ef4444'], hole=0.4
        )
        fig_churn.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_churn, use_container_width=True)
    
    with chart_col2:
        contract_churn = dataset.groupby(['Contract', 'Churn']).size().reset_index(name='Count')
        fig_contract = px.bar(
            contract_churn, x='Contract', y='Count', color='Churn',
            title='Churn by Contract Type', barmode='group',
            color_discrete_map={'Yes': '#ef4444', 'No': '#10b981'}
        )
        st.plotly_chart(fig_contract, use_container_width=True)
    
    chart_col3, chart_col4 = st.columns(2)
    
    with chart_col3:
        fig_charges = px.histogram(
            dataset, x='MonthlyCharges', color='Churn',
            title='Monthly Charges Distribution by Churn', nbins=30,
            color_discrete_map={'Yes': '#ef4444', 'No': '#10b981'}, opacity=0.7
        )
        fig_charges.update_layout(barmode='overlay')
        st.plotly_chart(fig_charges, use_container_width=True)
    
    with chart_col4:
        fig_tenure = px.histogram(
            dataset, x='tenure', color='Churn',
            title='Tenure Distribution by Churn', nbins=20,
            color_discrete_map={'Yes': '#ef4444', 'No': '#10b981'}, opacity=0.7
        )
        fig_tenure.update_layout(barmode='overlay')
        st.plotly_chart(fig_tenure, use_container_width=True)
    
    st.markdown("### Raw Data")
    st.dataframe(dataset, use_container_width=True, height=400)

st.markdown("---")
st.caption("ML Customer Analysis Project")
