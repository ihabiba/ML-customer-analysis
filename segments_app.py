"""
Customer Segmentation Dashboard
Streamlit app for visualizing and exploring customer segments.
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="◎",
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
        background-color: #8b5cf6 !important;
        color: white !important;
    }
    
    /* Segment cards */
    .segment-card {
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .segment-0 {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 2px solid #f59e0b;
    }
    .segment-0 h3 { color: #b45309; margin: 0; }
    .segment-1 {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border: 2px solid #3b82f6;
    }
    .segment-1 h3 { color: #1d4ed8; margin: 0; }
    .segment-2 {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border: 2px solid #10b981;
    }
    .segment-2 h3 { color: #047857; margin: 0; }
</style>
""", unsafe_allow_html=True)

# Load models and data with error handling
@st.cache_resource
def load_models():
    try:
        kmeans = joblib.load('models/segments_kmeans.pkl')
        scaler = joblib.load('models/segments_scaler.pkl')
        pca = joblib.load('models/segments_pca.pkl')
        features = joblib.load('models/segments_features.pkl')
        cluster_map = joblib.load('models/segments_cluster_map.pkl')
        cluster_stats = joblib.load('models/segments_cluster_stats.pkl')
        return kmeans, scaler, pca, features, cluster_map, cluster_stats
    except FileNotFoundError as e:
        st.error(f"Model files not found. Please run the Unsupervised notebook first. Error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

@st.cache_data
def load_dataset():
    try:
        return pd.read_csv('Datasets/data.csv', encoding='ISO-8859-1')
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'Datasets/data.csv' exists.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

kmeans, scaler, pca, features, cluster_map, cluster_stats = load_models()
raw_dataset = load_dataset()

# ============ HEADER ============
st.title("Customer Segmentation Dashboard")

# ============ NAVIGATION TABS ============
tab1, tab2, tab3 = st.tabs(["Segments Overview", "Cluster Analysis", "Dataset Explorer"])

# ============ TAB 1: SEGMENTS OVERVIEW ============
with tab1:
    st.markdown("Explore customer segments identified through K-Means clustering")
    
    # Segment cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        stats = cluster_stats[0]
        st.markdown(f"""
        <div class="segment-card segment-0">
            <h3>{stats['name']}</h3>
            <p style="color: #78716c; margin: 0.5rem 0;">Cluster 0</p>
            <div style="display: flex; justify-content: space-around; margin-top: 1rem;">
                <div>
                    <p style="font-size: 1.3rem; font-weight: bold; margin: 0; color: #b45309;">{stats['avg_quantity']:.1f}</p>
                    <p style="font-size: 0.8rem; color: #78716c; margin: 0;">Avg Quantity</p>
                </div>
                <div>
                    <p style="font-size: 1.3rem; font-weight: bold; margin: 0; color: #b45309;">${stats['avg_spend']:.2f}</p>
                    <p style="font-size: 0.8rem; color: #78716c; margin: 0;">Avg Spend</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        stats = cluster_stats[1]
        st.markdown(f"""
        <div class="segment-card segment-1">
            <h3>{stats['name']}</h3>
            <p style="color: #78716c; margin: 0.5rem 0;">Cluster 1</p>
            <div style="display: flex; justify-content: space-around; margin-top: 1rem;">
                <div>
                    <p style="font-size: 1.3rem; font-weight: bold; margin: 0; color: #1d4ed8;">{stats['avg_quantity']:.1f}</p>
                    <p style="font-size: 0.8rem; color: #78716c; margin: 0;">Avg Quantity</p>
                </div>
                <div>
                    <p style="font-size: 1.3rem; font-weight: bold; margin: 0; color: #1d4ed8;">${stats['avg_spend']:.2f}</p>
                    <p style="font-size: 0.8rem; color: #78716c; margin: 0;">Avg Spend</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        stats = cluster_stats[2]
        st.markdown(f"""
        <div class="segment-card segment-2">
            <h3>{stats['name']}</h3>
            <p style="color: #78716c; margin: 0.5rem 0;">Cluster 2</p>
            <div style="display: flex; justify-content: space-around; margin-top: 1rem;">
                <div>
                    <p style="font-size: 1.3rem; font-weight: bold; margin: 0; color: #047857;">{stats['avg_quantity']:.1f}</p>
                    <p style="font-size: 0.8rem; color: #78716c; margin: 0;">Avg Quantity</p>
                </div>
                <div>
                    <p style="font-size: 1.3rem; font-weight: bold; margin: 0; color: #047857;">${stats['avg_spend']:.2f}</p>
                    <p style="font-size: 0.8rem; color: #78716c; margin: 0;">Avg Spend</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Comparison chart - built dynamically from cluster_stats
    st.markdown("### Segment Comparison")
    
    # Build comparison_df from cluster_stats (not hardcoded)
    comparison_rows = []
    for cid in sorted(cluster_stats.keys()):
        stats = cluster_stats[cid]
        comparison_rows.append({'Segment': stats['name'], 'Metric': 'Avg Quantity', 'Value': stats['avg_quantity']})
        comparison_rows.append({'Segment': stats['name'], 'Metric': 'Avg Spend ($)', 'Value': stats['avg_spend']})
    comparison_df = pd.DataFrame(comparison_rows)
    
    fig_compare = px.bar(
        comparison_df, x='Segment', y='Value', color='Metric',
        barmode='group', title='Average Quantity and Spend by Segment',
        color_discrete_sequence=['#8b5cf6', '#10b981']
    )
    fig_compare.update_layout(xaxis_title='', yaxis_title='Value')
    st.plotly_chart(fig_compare, use_container_width=True)
    
    # Business recommendations
    st.markdown("### Business Recommendations")
    
    rec_col1, rec_col2, rec_col3 = st.columns(3)
    
    with rec_col1:
        st.markdown("""
        **🟡 Low Spenders**
        - Send promotional offers
        - Introduce loyalty programs
        - Target with entry-level products
        - Email campaigns for engagement
        """)
    
    with rec_col2:
        st.markdown("""
        **🔵 Regular Customers**
        - Upsell premium products
        - Offer bundle deals
        - Personalized recommendations
        - Early access to sales
        """)
    
    with rec_col3:
        st.markdown("""
        **🟢 High-Value Customers**
        - VIP treatment & exclusive offers
        - Priority customer support
        - Personal account manager
        - Invite to exclusive events
        """)
    
    st.markdown("---")
    st.info("""
    **Note:** This segmentation model was trained on transaction-level data using 40+ engineered features 
      (e.g., Country, InvoiceHour, StockCode, UnitPrice, Quantity, etc.). Such high-dimensional patterns 
      cannot be meaningfully entered manually. For real-time customer-level predictions, a dedicated 
      RFM-based segmentation model should be built instead.
    """)

# ============ TAB 2: CLUSTER ANALYSIS ============
with tab2:
    st.markdown("Visualize customer clusters in reduced dimensions")
    
    # Load PCA scatter plots from assets
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 2D PCA Visualization")
        try:
            st.image("assets/pca_2d_scatter.png", use_container_width=True)
        except:
            st.info("2D scatter plot not found. Run the notebook to generate 'assets/pca_2d_scatter.png'.")
    
    with col2:
        st.markdown("### 3D PCA Visualization")
        try:
            st.image("assets/pca_3d_scatter.png", use_container_width=True)
        except:
            st.info("3D scatter plot not found. Run the notebook to generate 'assets/pca_3d_scatter.png'.")
    
    st.markdown("---")
    
    # Additional analysis plots
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("### Elbow Method (Optimal K)")
        try:
            st.image("assets/elbow_method.png", use_container_width=True)
        except:
            st.info("Elbow plot not found.")
    
    with col4:
        st.markdown("### Silhouette Scores")
        try:
            st.image("assets/silhouette_scores_bar.png", use_container_width=True)
        except:
            st.info("Silhouette scores plot not found.")
    
    st.markdown("---")
    
    # Cluster distribution
    st.markdown("### Customers per Segment")
    try:
        st.image("assets/customers_per_cluster_count.png", use_container_width=True)
    except:
        st.info("Customer count plot not found.")

# ============ TAB 3: DATASET EXPLORER ============
with tab3:
    st.markdown("Explore the raw transaction dataset")
    
    # Clean dataset for display
    df_display = raw_dataset.dropna().head(1000)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{len(raw_dataset):,}")
    col2.metric("Unique Customers", f"{raw_dataset['CustomerID'].nunique():,}")
    col3.metric("Unique Products", f"{raw_dataset['StockCode'].nunique():,}")
    col4.metric("Countries", f"{raw_dataset['Country'].nunique():,}")
    
    st.markdown("---")
    
    # Charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Top countries
        country_counts = raw_dataset['Country'].value_counts().head(10)
        fig_country = px.bar(
            x=country_counts.values, y=country_counts.index,
            orientation='h', title='Top 10 Countries by Transactions',
            color=country_counts.values, color_continuous_scale='Purples'
        )
        fig_country.update_layout(showlegend=False, coloraxis_showscale=False,
                                  xaxis_title='Transactions', yaxis_title='')
        st.plotly_chart(fig_country, use_container_width=True)
    
    with chart_col2:
        # Quantity distribution
        df_clean = raw_dataset[(raw_dataset['Quantity'] > 0) & (raw_dataset['Quantity'] < 100)]
        fig_qty = px.histogram(
            df_clean, x='Quantity', nbins=50,
            title='Quantity Distribution (1-100)',
            color_discrete_sequence=['#8b5cf6']
        )
        fig_qty.update_layout(xaxis_title='Quantity', yaxis_title='Frequency')
        st.plotly_chart(fig_qty, use_container_width=True)
    
    st.markdown("### Sample Data")
    st.dataframe(df_display, use_container_width=True, height=400)

st.markdown("---")
st.caption("ML Customer Analysis Project")
