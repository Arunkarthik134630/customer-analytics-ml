# ============================================================
# app.py — Customer Analytics Streamlit Dashboard
# ============================================================
# 4 Pages:
#   1. Home — Project overview and key metrics
#   2. Segmentation — Customer segments and personas
#   3. Churn Prediction — Predict individual customer churn
#   4. Business Insights — EDA charts and model performance
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG — Must be first Streamlit command
# ============================================================
st.set_page_config(
    page_title="Customer Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — Makes the app look professional
# ============================================================
st.markdown("""
<style>
    /* ── Global font & background ── */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
    }

    /* ── Main content area ── */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #0f1117;
        border-right: 1px solid #2d2d2d;
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #c0c0c0 !important;
        font-size: 0.9rem;
    }
    [data-testid="stSidebar"] hr {
        border-color: #2d2d2d;
    }

    /* ── Page headers ── */
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
        border-left: 5px solid #0066cc;
        padding-left: 0.8rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
        padding-left: 1rem;
        letter-spacing: 0.2px;
    }

    /* ── KPI metric cards ── */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e8ecf0;
        border-top: 3px solid #0066cc;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        color: #6c757d !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #1a1a2e !important;
    }

    /* ── Section divider ── */
    hr {
        border: none;
        border-top: 1px solid #e8ecf0;
        margin: 1.5rem 0;
    }

    /* ── Insight cards ── */
    .card {
        background: #ffffff;
        border: 1px solid #e8ecf0;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .card-title {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 0.4rem;
    }
    .card-body {
        font-size: 0.92rem;
        color: #444;
        line-height: 1.5;
    }

    /* ── Status cards ── */
    .success-box {
        background: #f0faf4;
        border: 1px solid #b7dfc8;
        border-left: 4px solid #1a7a4a;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .insight-box {
        background: #f0f6ff;
        border: 1px solid #b3d1f7;
        border-left: 4px solid #0066cc;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .warning-box {
        background: #fffbf0;
        border: 1px solid #f5dfa0;
        border-left: 4px solid #c97a00;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .danger-box {
        background: #fff5f5;
        border: 1px solid #f5b8b8;
        border-left: 4px solid #c0392b;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }

    /* ── Subheaders ── */
    h2, h3 {
        color: #1a1a2e !important;
        font-weight: 600 !important;
    }

    /* ── Plotly chart container ── */
    .stPlotlyChart {
        border: 1px solid #e8ecf0;
        border-radius: 8px;
        padding: 0.5rem;
        background: #ffffff;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        border: 1px solid #e8ecf0 !important;
        border-radius: 8px !important;
    }

    /* ── Primary button ── */
    .stButton > button[kind="primary"] {
        background: #0066cc;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0.6rem 1.5rem;
        letter-spacing: 0.3px;
        transition: background 0.2s;
    }
    .stButton > button[kind="primary"]:hover {
        background: #0052a3;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 2px solid #e8ecf0;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.88rem;
        color: #6c757d;
        padding: 0.5rem 1rem;
        border-radius: 6px 6px 0 0;
    }
    .stTabs [aria-selected="true"] {
        color: #0066cc !important;
        border-bottom: 2px solid #0066cc !important;
    }

    /* ── Selectbox / Slider labels ── */
    .stSelectbox label, .stSlider label, .stNumberInput label {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: #444 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)
# ============================================================
# DATA AND MODEL LOADING
# ============================================================
# We use @st.cache_data so data loads once and is reused
# Without caching, data reloads on every user interaction
# — making the app very slow

@st.cache_data
def load_data():
    customers = pd.read_csv(
        'data/processed/customer_features.csv',
        dtype={'Customer ID': str}
    )
    transactions = pd.read_csv(
        'data/processed/clean_transactions.csv',
        parse_dates=['InvoiceDate'],
        dtype={'Customer ID': str}
    )
    return customers, transactions

@st.cache_resource
def load_model():
    model          = joblib.load('models/saved/churn_model_catboost.pkl')
    scaler         = joblib.load('models/saved/scaler.pkl')
    features       = joblib.load('models/saved/feature_cols.pkl')
    le_country     = joblib.load('models/saved/le_country.pkl')
    clv_model      = joblib.load('models/saved/clv_model.pkl')
    scaler_clv     = joblib.load('models/saved/scaler_clv.pkl')
    le_clv         = joblib.load('models/saved/le_clv.pkl')
    clv_features   = joblib.load('models/saved/clv_feature_cols.pkl')
    return (model, scaler, features, le_country,
            clv_model, scaler_clv, le_clv, clv_features)

# Load everything
customers, transactions = load_data()
(model, scaler, feature_cols, le_country,
 clv_model, scaler_clv, le_clv, clv_features) = load_model()

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.markdown(
    "<h2 style='color:#ffffff; font-size:1.1rem; "
    "font-weight:700; letter-spacing:0.5px;'>"
    "📊 CUSTOMER ANALYTICS</h2>",
    unsafe_allow_html=True
)
st.sidebar.markdown(
    "<p style='color:#888; font-size:0.75rem; "
    "text-transform:uppercase; letter-spacing:1px;'>"
    "Portfolio Project</p>",
    unsafe_allow_html=True
)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to:",
    ["🏠 Home",
     "👥 Customer Segments",
     "🎯 Churn Prediction",
     "📈 Business Insights",
     "🔄 Retention Strategy"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-size:0.78rem; color:#aaa; line-height:1.8;'>
<b style='color:#ccc;'>DATASET</b><br>
Online Retail II<br>
Dec 2009 – Dec 2011<br>
5,878 Customers<br>
779,425 Transactions
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-size:0.78rem; color:#aaa; line-height:1.8;'>
<b style='color:#ccc;'>BEST MODEL</b><br>
CatBoost Classifier<br>
ROC-AUC: 0.90<br>
F1 Score: 0.80<br>
Precision: 0.82
</div>
""", unsafe_allow_html=True)
# ============================================================
# PAGE 1 — HOME
# ============================================================
if page == "🏠 Home":

    st.markdown('<p class="main-header">Customer Analytics Dashboard</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Customer Segmentation & Churn Prediction — Online Retail II Dataset</p>',
                unsafe_allow_html=True)

    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)

    total_customers  = len(customers)
    total_revenue    = customers['Monetary'].sum()
    churn_rate       = customers['Churned'].mean() * 100
    avg_order_value  = customers['AvgOrderValue'].median()
    total_orders     = customers['Frequency'].sum()

    col1.metric("Total Customers",   f"{total_customers:,}")
    col2.metric("Total Revenue",     f"£{total_revenue:,.0f}")
    col3.metric("Churn Rate",        f"{churn_rate:.1f}%")
    col4.metric("Median Order Value",f"£{avg_order_value:,.0f}")
    col5.metric("Total Orders",      f"{total_orders:,}")

    st.markdown("---")

    # Two column layout
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Customer Segments")
        segment_data = customers['Persona'].value_counts().reset_index()
        segment_data.columns = ['Persona', 'Count']

        colors = {
            'Champions':        '#2ecc71',
            'Loyal Developing': '#3498db',
            'At Risk':          '#e67e22',
            'One-Time Lost':    '#e74c3c'
        }

        fig = px.pie(
            segment_data,
            values='Count',
            names='Persona',
            color='Persona',
            color_discrete_map=colors,
            hole=0.4
        )
        fig.update_layout(
            height=350,
            margin=dict(t=20, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom",
                       y=-0.2, xanchor="center", x=0.5)
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Churn Rate by Segment")
        churn_by_segment = customers.groupby('Persona')['Churned'].mean().reset_index()
        churn_by_segment.columns = ['Persona', 'ChurnRate']
        churn_by_segment['ChurnRate'] = (churn_by_segment['ChurnRate'] * 100).round(1)
        churn_by_segment = churn_by_segment.sort_values('ChurnRate', ascending=True)

        fig2 = px.bar(
            churn_by_segment,
            x='ChurnRate',
            y='Persona',
            orientation='h',
            color='ChurnRate',
            color_continuous_scale='RdYlGn_r',
            text='ChurnRate'
        )
        fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig2.update_layout(
            height=350,
            margin=dict(t=20, b=20, l=20, r=20),
            xaxis_title="Churn Rate (%)",
            yaxis_title="",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Key Business Insights")

    ins1, ins2, ins3, ins4 = st.columns(4)

    with ins1:
        st.markdown("""
        <div class="success-box">
        <b>🏆 Champions</b><br>
        1,520 customers generate <b>78.4%</b> of revenue
        with only 14% churn risk
        </div>""", unsafe_allow_html=True)

    with ins2:
        st.markdown("""
        <div class="insight-box">
        <b>💛 Loyal Developing</b><br>
        2,031 customers at <b>43% churn risk</b>
        — biggest growth opportunity
        </div>""", unsafe_allow_html=True)

    with ins3:
        st.markdown("""
        <div class="warning-box">
        <b>⚠️ At Risk</b><br>
        700 customers at <b>73% churn risk</b>
        — urgent win-back campaigns needed
        </div>""", unsafe_allow_html=True)

    with ins4:
        st.markdown("""
        <div class="danger-box">
        <b>💤 One-Time Lost</b><br>
        1,627 customers at <b>85% churn risk</b>
        — tried once, never returned
        </div>""", unsafe_allow_html=True)

# ============================================================
# PAGE 2 — CUSTOMER SEGMENTS
# ============================================================
elif page == "👥 Customer Segments":

    st.markdown('<p class="main-header">👥 Customer Segmentation</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">K-Means Clustering on RFM + Behavioral Features (K=4)</p>',
                unsafe_allow_html=True)

    # Segment selector
    selected_segment = st.selectbox(
        "Select a segment to explore:",
        ["All Segments", "Champions", "Loyal Developing",
         "At Risk", "One-Time Lost"]
    )

    if selected_segment == "All Segments":
        df_view = customers.copy()
    else:
        df_view = customers[customers['Persona'] == selected_segment].copy()

    st.markdown(f"**Showing:** {len(df_view):,} customers")
    st.markdown("---")

    # Metrics for selected segment
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers",       f"{len(df_view):,}")
    col2.metric("Avg Monetary",    f"£{df_view['Monetary'].median():,.0f}")
    col3.metric("Avg Frequency",   f"{df_view['Frequency'].median():.0f} orders")
    col4.metric("Churn Rate",      f"{df_view['Churned'].mean()*100:.1f}%")

    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Revenue Distribution by Segment")
        rev_data = customers.groupby('Persona')['Monetary'].sum().reset_index()
        rev_data.columns = ['Persona', 'Revenue']
        rev_data['Revenue_pct'] = (
            rev_data['Revenue'] / rev_data['Revenue'].sum() * 100
        ).round(1)

        fig = px.bar(
            rev_data.sort_values('Revenue', ascending=True),
            x='Revenue', y='Persona',
            orientation='h',
            color='Persona',
            color_discrete_map={
                'Champions':        '#2ecc71',
                'Loyal Developing': '#3498db',
                'At Risk':          '#e67e22',
                'One-Time Lost':    '#e74c3c'
            },
            text='Revenue_pct'
        )
        fig.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside'
        )
        fig.update_layout(
            height=300, showlegend=False,
            xaxis_title="Total Revenue (£)",
            yaxis_title="",
            margin=dict(t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Frequency vs Monetary by Segment")
        sample = customers.sample(
            min(1000, len(customers)), random_state=42
        )
        fig2 = px.scatter(
            sample,
            x='Frequency',
            y='Monetary',
            color='Persona',
            color_discrete_map={
                'Champions':        '#2ecc71',
                'Loyal Developing': '#3498db',
                'At Risk':          '#e67e22',
                'One-Time Lost':    '#e74c3c'
            },
            hover_data=['Tenure', 'ReturnRate'],
            opacity=0.6
        )
        fig2.update_layout(
            height=300,
            margin=dict(t=10, b=10),
            xaxis_title="Purchase Frequency",
            yaxis_title="Total Monetary Value (£)"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Segment profile table
    st.markdown("---")
    st.subheader("Segment Profile Summary")

    profile = customers.groupby('Persona').agg(
        Customers=('Customer ID', 'count'),
        Median_Recency=('Recency', 'median'),
        Median_Frequency=('Frequency', 'median'),
        Median_Monetary=('Monetary', 'median'),
        Median_Tenure=('Tenure', 'median'),
        Churn_Rate=('Churned', 'mean'),
        Revenue_Share=('Monetary', 'sum')
    ).reset_index()

    profile['Churn_Rate'] = (profile['Churn_Rate'] * 100).round(1)
    total_rev = profile['Revenue_Share'].sum()
    profile['Revenue_Share'] = (
        profile['Revenue_Share'] / total_rev * 100
    ).round(1)
    profile['Median_Monetary'] = profile['Median_Monetary'].round(0)

    st.dataframe(
        profile.style.background_gradient(
            subset=['Churn_Rate'], cmap='RdYlGn_r'
        ).background_gradient(
            subset=['Revenue_Share'], cmap='Greens'
        ).format({
            'Churn_Rate': '{:.1f}%',
            'Revenue_Share': '{:.1f}%',
            'Median_Monetary': '£{:.0f}'
        }),
        use_container_width=True,
        height=200
    )

    # Business recommendations
    st.markdown("---")
    st.subheader("Business Recommendations")

    recs = {
        'Champions': (
            '✅', 'success-box',
            'Reward with VIP loyalty program and early product access. '
            'Low churn risk — focus on upselling and referral programs.'
        ),
        'Loyal Developing': (
            '💡', 'insight-box',
            'Nurture with personalized email campaigns and cross-sell '
            'recommendations. Highest potential for conversion to Champions.'
        ),
        'At Risk': (
            '⚠️', 'warning-box',
            'Launch immediate win-back campaign with targeted discounts. '
            'High return rate signals dissatisfaction — prioritize service recovery.'
        ),
        'One-Time Lost': (
            '🚨', 'danger-box',
            'Re-engagement campaign with strong incentive. '
            'Filter by Monetary value — focus budget on higher-value lost customers.'
        )
    }

    col1, col2 = st.columns(2)
    for i, (segment, (icon, style, text)) in enumerate(recs.items()):
        col = col1 if i % 2 == 0 else col2
        with col:
            st.markdown(
                f'<div class="{style}"><b>{icon} {segment}</b><br>{text}</div>',
                unsafe_allow_html=True
            )
            st.markdown("")

# ============================================================
# PAGE 3 — CHURN PREDICTION
# ============================================================
elif page == "🎯 Churn Prediction":

    st.markdown('<p class="main-header">🎯 Churn Prediction</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict churn probability for a customer using CatBoost model (ROC-AUC: 0.90)</p>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    <b>How to use:</b> Enter a customer's behavioral metrics below.
    The model will predict their churn probability and explain
    which factors are driving the prediction.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Input form
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Purchase Behavior")
        frequency = st.number_input(
            "Purchase Frequency", min_value=1,
            max_value=200, value=5, step=1,
            key="rs_freq"
        )
        monetary = st.number_input(
            "Total Spend (£)",
            min_value=10.0, max_value=100000.0,
            value=500.0, step=50.0,
            help="Total revenue generated by customer"
        )
        avg_order_value = st.number_input(
            "Avg Order Value (£)",
            min_value=10.0, max_value=50000.0,
            value=200.0, step=10.0,
            help="Average spend per order"
        )

    with col2:
        st.subheader("Shopping Profile")
        avg_basket = st.number_input(
            "Avg Basket Size", min_value=1,
            max_value=500, value=20, step=1,
            key="rs_bas"
        )
        unique_products = st.number_input(
            "Unique Products", min_value=1,
            max_value=3000, value=50, step=1,
            key="rs_uni"
        )
        tenure = st.number_input(
            "Tenure (days)", min_value=0,
            max_value=738, value=180, step=1,
            key="rs_ten"
        )

    with col3:
        st.subheader("Return & Timing")
        return_rate = st.number_input(
            "Return Rate", min_value=0.0,
            max_value=1.0, value=0.05, step=0.01,
            key="rs_ret"
        )
        return_count = st.number_input(
            "Return Count", min_value=0,
            max_value=50, value=0, step=1,
            key="rs_rc"
        )
        preferred_day = st.selectbox(
            "Preferred Shopping Day",
            options=[0,1,2,3,4,5,6],
            format_func=lambda x: ['Monday','Tuesday','Wednesday',
                                    'Thursday','Friday','Saturday',
                                    'Sunday'][x],
            index=3,
            help="Day of week with most purchases"
        )
        preferred_month = st.selectbox(
            "Preferred Shopping Month",
            options=list(range(1,13)),
            format_func=lambda x: ['Jan','Feb','Mar','Apr','May',
                                    'Jun','Jul','Aug','Sep','Oct',
                                    'Nov','Dec'][x-1],
            index=10,
            help="Month with most purchases"
        )

    # Country selector
    st.markdown("---")
    available_countries = sorted(le_country.classes_.tolist())
    country = st.selectbox(
        "Country",
        options=available_countries,
        index=available_countries.index('United Kingdom')
        if 'United Kingdom' in available_countries else 0
    )
    country_encoded = le_country.transform([country])[0]

    # Predict button
    st.markdown("---")
    predict_btn = st.button(
        "🔍 Predict Churn Probability",
        type="primary",
        use_container_width=True
    )

    if predict_btn:
        # Build input array
        input_data = pd.DataFrame([[
            frequency, monetary, avg_order_value,
            avg_basket, unique_products, tenure,
            return_rate, return_count,
            preferred_day, preferred_month,
            country_encoded
        ]], columns=feature_cols)

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        churn_prob  = model.predict_proba(input_scaled)[0][1]
        churn_label = model.predict(input_scaled)[0]

        st.markdown("---")
        st.subheader("Prediction Result")

        # Result display
        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            st.metric(
                "Churn Probability",
                f"{churn_prob:.1%}",
                delta=f"{'High Risk' if churn_prob > 0.6 else 'Low Risk'}"
            )

        with res_col2:
            if churn_prob >= 0.7:
                risk_level = "🔴 HIGH RISK"
                box_class  = "danger-box"
            elif churn_prob >= 0.4:
                risk_level = "🟡 MEDIUM RISK"
                box_class  = "warning-box"
            else:
                risk_level = "🟢 LOW RISK"
                box_class  = "success-box"

            st.markdown(
                f'<div class="{box_class}" style="text-align:center;">'
                f'<h3>{risk_level}</h3>'
                f'<p>Churn probability: <b>{churn_prob:.1%}</b></p>'
                f'</div>',
                unsafe_allow_html=True
            )

        with res_col3:
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_prob * 100,
                domain={'x': [0,1], 'y': [0,1]},
                title={'text': "Churn Risk %"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar':  {'color': "darkred"},
                    'steps': [
                        {'range': [0,  40], 'color': '#d4edda'},
                        {'range': [40, 70], 'color': '#fff3cd'},
                        {'range': [70,100], 'color': '#f8d7da'}
                    ],
                    'threshold': {
                        'line':  {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': churn_prob * 100
                    }
                }
            ))
            fig.update_layout(height=200, margin=dict(t=30,b=0,l=20,r=20))
            st.plotly_chart(fig, use_container_width=True)

        # Feature contribution table
        st.markdown("---")
        st.subheader("Key Factors Driving This Prediction")

        # Simple feature impact display without SHAP
        # (SHAP in Streamlit requires extra setup)
        feature_values = dict(zip(feature_cols, [
            frequency, monetary, avg_order_value,
            avg_basket, unique_products, tenure,
            return_rate, return_count,
            preferred_day, preferred_month,
            country_encoded
        ]))

        # Compare to median values
        medians = customers[feature_cols[:-1]].median()

        impact_data = []
        for feat in feature_cols[:-1]:
            val    = feature_values[feat]
            median = medians[feat]
            if median > 0:
                diff_pct = ((val - median) / median * 100)
            else:
                diff_pct = 0
            impact_data.append({
                'Feature':         feat,
                'Your Value':      round(val, 2),
                'Median Customer': round(median, 2),
                'vs Median':       f"{'↑' if diff_pct > 0 else '↓'} {abs(diff_pct):.0f}%"
            })

        impact_df = pd.DataFrame(impact_data)
        st.dataframe(impact_df, use_container_width=True, height=350)

        # Recommendation
        st.markdown("---")
        st.subheader("Recommended Action")

        if churn_prob >= 0.7:
            st.markdown("""
            <div class="danger-box">
            <b>🚨 Immediate Action Required</b><br>
            This customer is at high churn risk. Recommended actions:<br>
            • Send win-back email with personalized discount (15-20% off)<br>
            • Assign to customer success team for outreach<br>
            • Offer loyalty points or free shipping incentive<br>
            • Flag for monthly monitoring
            </div>""", unsafe_allow_html=True)
        elif churn_prob >= 0.4:
            st.markdown("""
            <div class="warning-box">
            <b>⚠️ Proactive Engagement Needed</b><br>
            This customer shows moderate churn signals. Recommended actions:<br>
            • Include in next email marketing campaign<br>
            • Send product recommendations based on purchase history<br>
            • Offer early access to new products<br>
            • Monitor for next 30 days
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="success-box">
            <b>✅ Customer Appears Healthy</b><br>
            Low churn risk. Recommended actions:<br>
            • Include in loyalty rewards program<br>
            • Cross-sell complementary products<br>
            • Request product review or referral<br>
            • Continue standard engagement
            </div>""", unsafe_allow_html=True)

# ============================================================
# PAGE 4 — BUSINESS INSIGHTS
# ============================================================
elif page == "📈 Business Insights":

    st.markdown('<p class="main-header">📈 Business Insights</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Revenue trends, customer behavior, and model performance</p>',
                unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "💰 Revenue Analysis",
        "🌍 Geographic Analysis",
        "🤖 Model Performance",
        "💎 CLV Analysis"
    ])

    # --- TAB 1: Revenue Analysis ---
    with tab1:
        st.subheader("Monthly Revenue Trend")

        transactions['YearMonth'] = transactions['InvoiceDate'].dt.to_period('M')
        monthly = transactions.groupby('YearMonth')['TotalPrice'].sum().reset_index()
        monthly['YearMonth_dt'] = monthly['YearMonth'].dt.to_timestamp()

        fig = px.area(
            monthly,
            x='YearMonth_dt',
            y='TotalPrice',
            title='Monthly Revenue (2009–2011)',
            labels={'TotalPrice': 'Revenue (£)', 'YearMonth_dt': 'Month'}
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Orders by Day of Week")
            transactions['DayOfWeek'] = transactions['InvoiceDate'].dt.day_name()
            day_order = ['Monday','Tuesday','Wednesday',
                        'Thursday','Friday','Saturday','Sunday']
            day_data = transactions.groupby('DayOfWeek')['Invoice'].nunique()
            day_data = day_data.reindex(day_order).reset_index()
            day_data.columns = ['Day', 'Orders']

            fig2 = px.bar(
                day_data, x='Day', y='Orders',
                color='Orders',
                color_continuous_scale='Blues'
            )
            fig2.update_layout(
                height=300, coloraxis_showscale=False,
                margin=dict(t=10)
            )
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            st.subheader("Top 10 Products by Revenue")
            top_products = (
                transactions.groupby('Description')['TotalPrice']
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )
            top_products.columns = ['Product', 'Revenue']

            fig3 = px.bar(
                top_products.sort_values('Revenue'),
                x='Revenue', y='Product',
                orientation='h',
                color='Revenue',
                color_continuous_scale='Greens'
            )
            fig3.update_layout(
                height=300, coloraxis_showscale=False,
                margin=dict(t=10), yaxis_title=""
            )
            st.plotly_chart(fig3, use_container_width=True)

    # --- TAB 2: Geographic Analysis ---
    with tab2:
        st.subheader("Revenue by Country (Top 10)")

        country_data = (
            transactions.groupby('Country')
            .agg(
                Revenue=('TotalPrice', 'sum'),
                Customers=('Customer ID', 'nunique'),
                Orders=('Invoice', 'nunique')
            )
            .sort_values('Revenue', ascending=False)
            .head(10)
            .reset_index()
        )
        country_data['AvgOrderValue'] = (
            country_data['Revenue'] / country_data['Orders']
        ).round(0)

        fig = px.bar(
            country_data.sort_values('Revenue'),
            x='Revenue', y='Country',
            orientation='h',
            color='Revenue',
            color_continuous_scale='Blues',
            text='Revenue'
        )
        fig.update_traces(
            texttemplate='£%{text:,.0f}',
            textposition='outside'
        )
        fig.update_layout(
            height=400, coloraxis_showscale=False,
            margin=dict(t=10, r=120)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Country Summary Table")
        country_data['Revenue'] = country_data['Revenue'].apply(
            lambda x: f"£{x:,.0f}"
        )
        country_data['AvgOrderValue'] = country_data['AvgOrderValue'].apply(
            lambda x: f"£{x:,.0f}"
        )
        st.dataframe(country_data, use_container_width=True)

    # --- TAB 3: Model Performance ---
    with tab3:
        st.subheader("Model Comparison — All 5 Models")

        model_results = pd.DataFrame({
            'Model': ['CatBoost','LightGBM','XGBoost',
                      'Random Forest','Logistic Regression'],
            'ROC-AUC':   [0.9004, 0.8958, 0.8895, 0.8850, 0.8401],
            'F1':        [0.8000, 0.7898, 0.7790, 0.7833, 0.7639],
            'Accuracy':  [0.8010, 0.7891, 0.7781, 0.7840, 0.7466],
            'Precision': [0.8182, 0.8007, 0.7890, 0.7997, 0.7259],
            'Recall':    [0.7826, 0.7793, 0.7692, 0.7676, 0.8060]
        })

        fig = px.bar(
            model_results.sort_values('ROC-AUC'),
            x='ROC-AUC',
            y='Model',
            orientation='h',
            color='ROC-AUC',
            color_continuous_scale='RdYlGn',
            text='ROC-AUC',
            title='ROC-AUC Score by Model'
        )
        fig.update_traces(
            texttemplate='%{text:.4f}',
            textposition='outside'
        )
        fig.update_layout(
            height=350,
            coloraxis_showscale=False,
            xaxis_range=[0.8, 0.92],
            margin=dict(t=40, r=80)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Detailed Metrics Table")
        st.dataframe(
            model_results.style.background_gradient(
                subset=['ROC-AUC','F1','Accuracy'],
                cmap='RdYlGn'
            ).format({
                'ROC-AUC':   '{:.4f}',
                'F1':        '{:.4f}',
                'Accuracy':  '{:.4f}',
                'Precision': '{:.4f}',
                'Recall':    '{:.4f}'
            }),
            use_container_width=True
        )

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="success-box">
            <b>✅ Best Model: CatBoost</b><br>
            • ROC-AUC: 0.9004<br>
            • F1 Score: 0.8000<br>
            • Precision: 0.8182<br>
            • Recall: 0.7826<br>
            • PR-AUC: 0.9090
            </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="insight-box">
            <b>💡 Why CatBoost Won</b><br>
            • Handles categorical features natively<br>
            • Built-in regularization reduces overfitting<br>
            • Best precision — minimizes false alarms<br>
            • Consistent across all cross-validation folds
            </div>""", unsafe_allow_html=True)
    # --- TAB 4: CLV Analysis ---
    with tab4:
        st.subheader("Customer Lifetime Value Analysis")

        # Key CLV metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Median CLV",        f"£{customers['CLV'].median():,.0f}")
        col2.metric("Mean CLV",          f"£{customers['CLV'].mean():,.0f}")
        col3.metric("High-CLV Churners", "109 customers")
        col4.metric("CLV at Risk",       "£484,992")

        st.markdown("---")
        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("Median CLV by Segment")
            clv_persona = customers.groupby('Persona')['CLV'].median(
            ).sort_values().reset_index()
            clv_persona.columns = ['Persona', 'Median_CLV']

            fig = px.bar(
                clv_persona,
                x='Median_CLV', y='Persona',
                orientation='h',
                color='Median_CLV',
                color_continuous_scale='Blues',
                text='Median_CLV'
            )
            fig.update_traces(
                texttemplate='£%{text:,.0f}',
                textposition='outside'
            )
            fig.update_layout(
                height=320,
                coloraxis_showscale=False,
                xaxis_title="Median CLV (£)",
                yaxis_title="",
                margin=dict(t=10, r=100)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.subheader("CLV Tier Distribution by Segment")
            clv_tier_data = customers.groupby(
                ['Persona', 'CLV_Tier']
            ).size().reset_index(name='Count')

            fig2 = px.bar(
                clv_tier_data,
                x='Persona', y='Count',
                color='CLV_Tier',
                color_discrete_map={
                    'High':   '#2ecc71',
                    'Medium': '#f39c12',
                    'Low':    '#e74c3c'
                },
                barmode='stack'
            )
            fig2.update_layout(
                height=320,
                margin=dict(t=10),
                xaxis_title="",
                yaxis_title="Number of Customers",
                legend_title="CLV Tier"
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.subheader("Revenue at Risk — High CLV Churners")

        st.markdown("""
        <div class="danger-box">
        <b>🚨 £484,992 in customer lifetime value is at risk</b><br>
        109 high-CLV customers are predicted to churn.
        Average CLV per customer: £4,449.
        Immediate win-back campaigns targeting these customers
        have the highest ROI of any retention initiative.
        </div>
        """, unsafe_allow_html=True)

        # CLV tier breakdown table
        clv_summary = customers.groupby('Persona').agg(
            Customers=('Customer ID', 'count'),
            Median_CLV=('CLV', 'median'),
            Total_CLV=('CLV', 'sum'),
            High_CLV_Pct=('CLV_Tier',
                          lambda x: f"{(x=='High').mean()*100:.1f}%"),
            Churn_Rate=('Churned',
                        lambda x: f"{x.mean()*100:.1f}%")
        ).reset_index()

        Median_CLV_fmt = Median_CLV_fmt = customers.groupby(
            'Persona'
        )['CLV'].median().apply(lambda x: f'£{x:,.0f}')

        clv_summary['Median_CLV'] = clv_summary['Persona'].map(
            Median_CLV_fmt
        )
        clv_summary['Total_CLV'] = customers.groupby(
            'Persona'
        )['CLV'].sum().apply(lambda x: f'£{x:,.0f}').values

        st.dataframe(clv_summary, use_container_width=True)

# ============================================================
# PAGE 5 — RETENTION STRATEGY GENERATOR
# ============================================================
elif page == "🔄 Retention Strategy":

    st.markdown(
        '<p class="main-header">🔄 Retention Strategy Generator</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">'
        'AI-powered retention recommendations based on '
        'churn risk, CLV tier, and customer segment'
        '</p>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="insight-box">
    <b>How this works:</b> Enter a customer's profile below.
    The system combines their churn probability, predicted
    CLV tier, and segment membership to generate a
    personalised retention strategy with specific actions,
    budget guidance, and expected ROI.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Input section
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Customer Profile")
        rs_frequency = st.number_input(
            "Purchase Frequency", min_value=1,
            max_value=200, value=5, step=1,
            key="rs_freq"
        )
        
        rs_monetary = st.number_input(
            "Total Spend (£)", 10.0, 100000.0,
            500.0, 50.0, key="rs_mon"
        )
        rs_tenure = st.number_input(
            "Tenure (days)", min_value=0,
            max_value=738, value=180, step=1,
            key="rs_ten"
        )
        rs_recency = st.number_input(
            "Days Since Last Purchase", min_value=0,
            max_value=738, value=60, step=1,
            key="rs_rec"
        )

    with col2:
        st.subheader("Behavioral Signals")
        rs_aov = st.number_input(
            "Avg Order Value (£)", 10.0, 50000.0,
            200.0, 10.0, key="rs_aov"
        )
        rs_basket = st.number_input(
            "Avg Basket Size", min_value=1,
            max_value=500, value=20, step=1,
            key="rs_bas"
        )
        rs_unique = st.number_input(
            "Unique Products", min_value=1,
            max_value=3000, value=50, step=1,
            key="rs_uni"
        )
        rs_return_rate = st.number_input(
            "Return Rate", min_value=0.0,
            max_value=1.0, value=0.05, step=0.01,
            key="rs_ret"
        )

    with col3:
        st.subheader("Timing")
        rs_return_count = st.number_input(
            "Return Count", min_value=0,
            max_value=50, value=0, step=1,
            key="rs_rc"
        )
        rs_day = st.selectbox(
            "Preferred Day", [0,1,2,3,4,5,6],
            format_func=lambda x: ['Mon','Tue','Wed',
                                    'Thu','Fri','Sat',
                                    'Sun'][x],
            index=3, key="rs_day"
        )
        rs_month = st.selectbox(
            "Preferred Month", list(range(1,13)),
            format_func=lambda x: ['Jan','Feb','Mar',
                                    'Apr','May','Jun',
                                    'Jul','Aug','Sep',
                                    'Oct','Nov','Dec'
                                    ][x-1],
            index=10, key="rs_mon2"
        )
        rs_country = st.selectbox(
            "Country",
            sorted(le_country.classes_.tolist()),
            index=sorted(
                le_country.classes_.tolist()
            ).index('United Kingdom'),
            key="rs_country"
        )

    st.markdown("---")
    generate_btn = st.button(
        "⚡ Generate Retention Strategy",
        type="primary",
        use_container_width=True
    )

    if generate_btn:

        # --- Churn prediction ---
        rs_country_enc = le_country.transform([rs_country])[0]
        churn_input = pd.DataFrame([[
            rs_frequency, rs_monetary, rs_aov,
            rs_basket, rs_unique, rs_tenure,
            rs_return_rate, rs_return_count,
            rs_day, rs_month, rs_country_enc
        ]], columns=feature_cols)
        churn_scaled = scaler.transform(churn_input)
        churn_prob   = model.predict_proba(churn_scaled)[0][1]

        # --- CLV prediction ---
        clv_input = pd.DataFrame([[
            rs_frequency, rs_basket, rs_unique,
            rs_tenure, rs_return_rate, rs_return_count,
            rs_day, rs_month, rs_recency,
            rs_country_enc
        ]], columns=clv_features)
        clv_scaled    = scaler_clv.transform(clv_input)
        clv_tier_enc  = clv_model.predict(clv_scaled)[0]
        clv_tier      = le_clv.inverse_transform([clv_tier_enc])[0]

        # --- Segment assignment ---
        if rs_recency <= 30 and rs_frequency >= 8:
            segment = 'Champions'
        elif rs_recency <= 90 and rs_frequency >= 3:
            segment = 'Loyal Developing'
        elif rs_return_rate >= 0.5:
            segment = 'At Risk'
        elif rs_frequency == 1:
            segment = 'One-Time Lost'
        else:
            segment = 'Loyal Developing'

        # --- Display results ---
        st.markdown("---")
        st.subheader("Customer Assessment")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Churn Probability", f"{churn_prob:.1%}")
        r2.metric("CLV Tier",          clv_tier)
        r3.metric("Segment",           segment)

        if churn_prob >= 0.7:
            risk = "🔴 HIGH"
        elif churn_prob >= 0.4:
            risk = "🟡 MEDIUM"
        else:
            risk = "🟢 LOW"
        r4.metric("Risk Level", risk)

        st.markdown("---")
        st.subheader("Recommended Retention Strategy")

        # Strategy matrix
        # Logic: CLV Tier × Churn Risk → specific strategy
        strategies = {
            ('High',   'high'):   {
                'priority': '🚨 CRITICAL — Act within 24 hours',
                'style':    'danger-box',
                'budget':   '£50–£100 discount or gift voucher',
                'actions':  [
                    'Personal phone call from account manager',
                    'Exclusive VIP discount (20-25% off next order)',
                    'Free express shipping for 3 months',
                    'Early access to new product launches',
                    'Dedicated customer success support'
                ],
                'message':  'High-value customer at critical churn risk. '
                            'Every day of delay increases permanent loss '
                            'probability. ROI on retention spend is very high.',
                'expected_roi': '8–12x return on retention spend'
            },
            ('High',   'medium'): {
                'priority': '⚠️ HIGH PRIORITY — Act within 1 week',
                'style':    'warning-box',
                'budget':   '£20–£50 incentive',
                'actions':  [
                    'Personalised email with product recommendations',
                    'Loyalty points double-up promotion',
                    'Exclusive member discount (10-15% off)',
                    'Invite to VIP customer feedback session',
                    'Free shipping on next 2 orders'
                ],
                'message':  'High-value customer showing early churn signals. '
                            'Proactive engagement now prevents escalation '
                            'to critical risk.',
                'expected_roi': '5–8x return on retention spend'
            },
            ('High',   'low'):    {
                'priority': '✅ MAINTAIN — Standard VIP treatment',
                'style':    'success-box',
                'budget':   '£10–£20 loyalty reward',
                'actions':  [
                    'Include in VIP loyalty programme',
                    'Send personalised thank you + product rec',
                    'Offer referral bonus',
                    'Early sale access',
                    'Birthday or anniversary reward'
                ],
                'message':  'High-value, low-risk customer. '
                            'Focus on deepening relationship '
                            'and maximising upsell opportunity.',
                'expected_roi': '3–5x return on retention spend'
            },
            ('Medium', 'high'):   {
                'priority': '⚠️ URGENT — Act within 3 days',
                'style':    'warning-box',
                'budget':   '£15–£30 incentive',
                'actions':  [
                    'Win-back email with 15% discount code',
                    'Showcase new arrivals relevant to past purchases',
                    'Free shipping incentive',
                    'Customer satisfaction survey + follow-up',
                    'Re-engagement email sequence (3 emails)'
                ],
                'message':  'Mid-value customer at high churn risk. '
                            'Has growth potential — worth targeted '
                            'retention investment.',
                'expected_roi': '3–5x return on retention spend'
            },
            ('Medium', 'medium'): {
                'priority': '💡 NURTURE — Act within 2 weeks',
                'style':    'insight-box',
                'budget':   '£5–£15 incentive',
                'actions':  [
                    'Include in regular email marketing campaign',
                    'Cross-sell complementary products',
                    'Offer bundle discount',
                    'Newsletter with product education content',
                    'Monitor for 30 days'
                ],
                'message':  'Mid-value customer with moderate churn risk. '
                            'Standard nurture programme is sufficient. '
                            'Monitor for escalation.',
                'expected_roi': '2–3x return on retention spend'
            },
            ('Medium', 'low'):    {
                'priority': '✅ GROW — Standard engagement',
                'style':    'success-box',
                'budget':   '£5–£10 reward',
                'actions':  [
                    'Cross-sell and upsell campaigns',
                    'Product recommendation emails',
                    'Seasonal promotions',
                    'Loyalty points accumulation',
                    'Standard email cadence'
                ],
                'message':  'Mid-value, low-risk customer. '
                            'Focus on growing their order value '
                            'and purchase frequency.',
                'expected_roi': '2–4x return on retention spend'
            },
            ('Low',    'high'):   {
                'priority': '📊 LOW ROI — Minimal intervention',
                'style':    'warning-box',
                'budget':   '£2–£5 (email only)',
                'actions':  [
                    'Single automated re-engagement email',
                    'Low-cost incentive (free shipping only)',
                    'No personal outreach — not cost effective',
                    'Remove from high-touch campaign lists',
                    'Monitor only'
                ],
                'message':  'Low-value customer at high churn risk. '
                            'Retention spend ROI is low. '
                            'Minimal automated intervention only.',
                'expected_roi': '0.5–1x return on retention spend'
            },
            ('Low',    'medium'): {
                'priority': '📊 MINIMAL — Automated only',
                'style':    'insight-box',
                'budget':   'Email only (no discount)',
                'actions':  [
                    'Include in standard email newsletter',
                    'Seasonal promotion emails',
                    'No personalised outreach',
                    'Standard automated journey only'
                ],
                'message':  'Low-value customer with moderate risk. '
                            'Automated touchpoints only. '
                            'Do not allocate manual retention resources.',
                'expected_roi': '1–2x return on retention spend'
            },
            ('Low',    'low'):    {
                'priority': '✅ MAINTAIN — No action needed',
                'style':    'success-box',
                'budget':   'Standard email cadence only',
                'actions':  [
                    'Standard newsletter inclusion',
                    'Seasonal campaigns only',
                    'No specific retention action needed'
                ],
                'message':  'Low-value, low-risk customer. '
                            'No specific retention action needed. '
                            'Standard communication cadence.',
                'expected_roi': 'N/A — maintenance mode'
            }
        }

        # Map churn probability to risk level key
        if churn_prob >= 0.7:
            risk_key = 'high'
        elif churn_prob >= 0.4:
            risk_key = 'medium'
        else:
            risk_key = 'low'

        strategy = strategies.get(
            (clv_tier, risk_key),
            strategies[('Medium', 'medium')]
        )

        # Display strategy
        st.markdown(
            f'<div class="{strategy["style"]}">'
            f'<b>{strategy["priority"]}</b><br>'
            f'{strategy["message"]}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown("")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Recommended Actions:**")
            for i, action in enumerate(strategy['actions'], 1):
                st.markdown(f"{i}. {action}")

        with col_b:
            st.markdown("**Investment Guidance:**")
            st.markdown(
                f"- **Budget per customer:** "
                f"{strategy['budget']}"
            )
            st.markdown(
                f"- **Expected ROI:** "
                f"{strategy['expected_roi']}"
            )
            st.markdown(
                f"- **CLV Tier:** {clv_tier}"
            )
            st.markdown(
                f"- **Churn Probability:** {churn_prob:.1%}"
            )
            st.markdown(
                f"- **Segment:** {segment}"
            )