"""
Real Estate Investment Advisor — Streamlit App
Features:
  ✅ Property input form
  ✅ Classification: Is this a Good Investment?
  ✅ Regression:    Estimated Price After 5 Years
  ✅ Visual insights: heatmaps, charts, feature importance
  ✅ Model confidence scores
  ✅ Filter & explore dataset
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import warnings
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# THEME / CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
.stApp { background-color: #0f1117; }

/* Sidebar */
[data-testid="stSidebar"] { background-color: #1a1d27; }

/* Cards */
.metric-card {
    background: linear-gradient(135deg, #1e2235, #252a3d);
    border: 1px solid #3a3f5c;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin-bottom: 12px;
}
.metric-card h2 { color: #6c63ff; font-size: 2rem; margin: 0; }
.metric-card p  { color: #a0a8c0; margin: 4px 0 0; font-size: 0.9rem; }

/* Good / Bad badges */
.badge-good { background:#00c896; color:#000; border-radius:8px; padding:6px 18px; font-weight:700; }
.badge-bad  { background:#ff6b6b; color:#fff; border-radius:8px; padding:6px 18px; font-weight:700; }

/* Section headers */
.section-title {
    font-size: 1.3rem; font-weight: 700; color: #6c63ff;
    border-left: 4px solid #6c63ff; padding-left: 10px;
    margin: 20px 0 10px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    clf_data = reg_data = None
    try:
        with open("models/classifier.pkl", "rb") as f:
            clf_data = pickle.load(f)
    except FileNotFoundError:
        pass
    try:
        with open("models/regressor.pkl", "rb") as f:
            reg_data = pickle.load(f)
    except FileNotFoundError:
        pass
    return clf_data, reg_data


@st.cache_data
def load_dataset():
    paths = ["data/processed_housing.csv", "data/india_housing_prices.csv"]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None


clf_bundle, reg_bundle = load_models()
df_raw = load_dataset()


# ─────────────────────────────────────────────
# SIDEBAR — NAVIGATION
# ─────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/real-estate.png", width=60)
st.sidebar.title("🏠 RE Advisor")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🔮 Predict", "📊 EDA Dashboard", "🔍 Explore Data", "ℹ️ About"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small style='color:#5a5f7a'>Real Estate Investment Advisor<br>"
    "Powered by ML + Streamlit</small>",
    unsafe_allow_html=True
)


# ═══════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT
# ═══════════════════════════════════════════════════════════════
if page == "🔮 Predict":
    st.title("🏠 Real Estate Investment Advisor")
    st.markdown(
        "Enter property details below to get an **investment recommendation** "
        "and an **estimated price forecast for the next 5 years**."
    )

    if clf_bundle is None or reg_bundle is None:
        st.warning(
            "⚠️  Model files not found. Please run `python scripts/03_model_training.py` first.\n\n"
            "Showing demo prediction mode."
        )

    # ── INPUT FORM ────────────────────────────────────
    with st.form("prediction_form"):
        st.markdown('<div class="section-title">📋 Property Details</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            state        = st.selectbox("State", ["Maharashtra", "Delhi", "Karnataka", "Tamil Nadu",
                                                   "Gujarat", "Rajasthan", "Uttar Pradesh", "Other"])
            city         = st.selectbox("City", ["Mumbai", "Delhi", "Bengaluru", "Chennai",
                                                  "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Other"])
            locality     = st.text_input("Locality", "Andheri East")
            property_type = st.selectbox("Property Type", ["Apartment", "Villa", "House",
                                                            "Studio", "Penthouse", "Row House"])

        with col2:
            bhk          = st.selectbox("BHK", [1, 2, 3, 4, 5])
            size_sqft    = st.number_input("Size (Sq Ft)", min_value=200, max_value=10000,
                                           value=1200, step=50)
            price_lakhs  = st.number_input("Current Price (₹ Lakhs)", min_value=5.0,
                                            max_value=5000.0, value=85.0, step=1.0)
            year_built   = st.number_input("Year Built", min_value=1950, max_value=2024, value=2015)
            floor_no     = st.number_input("Floor Number", min_value=0, max_value=100, value=5)
            total_floors = st.number_input("Total Floors", min_value=1, max_value=100, value=15)

        with col3:
            furnished    = st.selectbox("Furnished Status", ["Unfurnished", "Semi-Furnished", "Fully Furnished"])
            transport    = st.selectbox("Transport Accessibility", ["Low", "Medium", "High"])
            security     = st.selectbox("Security", ["None", "CCTV", "Gated", "Guard", "Full"])
            nearby_schools   = st.slider("Nearby Schools", 0, 20, 5)
            nearby_hospitals = st.slider("Nearby Hospitals", 0, 15, 3)
            parking      = st.selectbox("Parking Spaces", [0, 1, 2, 3])
            owner_type   = st.selectbox("Owner Type", ["Individual", "Builder", "Agent"])

        st.markdown("---")
        amenities = st.multiselect(
            "Amenities Available",
            ["Gym", "Swimming Pool", "Clubhouse", "Garden", "Lift", "Power Backup",
             "24×7 Water", "Children Play Area", "Intercom", "Jogging Track"],
            default=["Gym", "Lift", "Power Backup"]
        )
        facing = st.selectbox("Property Facing", ["North", "South", "East", "West",
                                                    "North-East", "North-West", "South-East", "South-West"])
        availability = st.selectbox("Availability Status", ["Available", "Under Construction", "Sold"])

        submitted = st.form_submit_button("🔮 Get Investment Prediction", use_container_width=True)

    # ── PREDICTION LOGIC ──────────────────────────────
    if submitted:
        age = 2024 - int(year_built)
        price_per_sqft = price_lakhs * 1e5 / size_sqft
        transport_map = {"Low": 1, "Medium": 2, "High": 3}
        transport_score = transport_map[transport]
        appreciation_rate = transport_score / max(age, 1)
        amenity_count = len(amenities)
        school_density = nearby_schools / 20.0
        hospital_prox  = nearby_hospitals / 15.0

        label_maps = {
            "State": {"Maharashtra":0,"Delhi":1,"Karnataka":2,"Tamil Nadu":3,
                      "Gujarat":4,"Rajasthan":5,"Uttar Pradesh":6,"Other":7},
            "City":  {"Mumbai":0,"Delhi":1,"Bengaluru":2,"Chennai":3,
                      "Hyderabad":4,"Pune":5,"Ahmedabad":6,"Jaipur":7,"Other":8},
            "Property_Type": {"Apartment":0,"Villa":1,"House":2,"Studio":3,"Penthouse":4,"Row House":5},
            "Furnished_Status": {"Unfurnished":0,"Semi-Furnished":1,"Fully Furnished":2},
            "Owner_Type": {"Individual":0,"Builder":1,"Agent":2},
            "Availability_Status": {"Available":0,"Under Construction":1,"Sold":2},
            "Security": {"None":0,"CCTV":1,"Gated":2,"Guard":3,"Full":4},
        }

        input_dict = {
            "BHK": bhk,
            "Size_in_SqFt": size_sqft,
            "Price_in_Lakhs": price_lakhs,
            "Price_per_SqFt": price_per_sqft,
            "Age_of_Property": age,
            "Nearby_Schools": nearby_schools,
            "Nearby_Hospitals": nearby_hospitals,
            "Parking_Space": parking,
            "Floor_No": floor_no,
            "Total_Floors": total_floors,
            "School_Density_Score": school_density,
            "Hospital_Proximity_Score": hospital_prox,
            "Appreciation_Rate": appreciation_rate,
            "Amenity_Count": amenity_count,
            "Transport_Score": transport_score,
            "Property_Type_enc": label_maps["Property_Type"].get(property_type, 0),
            "City_enc": label_maps["City"].get(city, 0),
            "State_enc": label_maps["State"].get(state, 0),
            "Furnished_Status_enc": label_maps["Furnished_Status"].get(furnished, 0),
            "Owner_Type_enc": label_maps["Owner_Type"].get(owner_type, 0),
            "Availability_Status_enc": label_maps["Availability_Status"].get(availability, 0),
            "Security_enc": label_maps["Security"].get(security, 0),
        }

        st.markdown("---")
        st.markdown("## 🎯 Prediction Results")

        col_r1, col_r2 = st.columns(2)

        # ── CLASSIFICATION ────────────────────────────
        with col_r1:
            st.markdown('<div class="section-title">📌 Investment Classification</div>',
                        unsafe_allow_html=True)
            if clf_bundle:
                feats = clf_bundle["features"]
                X_in = pd.DataFrame([input_dict])[
                    [f for f in feats if f in input_dict]
                ].reindex(columns=feats, fill_value=0)
                X_sc = clf_bundle["scaler"].transform(X_in)
                pred = clf_bundle["model"].predict(X_sc)[0]
                prob = clf_bundle["model"].predict_proba(X_sc)[0]
                confidence = max(prob) * 100
            else:
                # Demo mode
                pred = 1 if appreciation_rate > 0.05 else 0
                confidence = 72.0 if pred else 63.0

            if pred == 1:
                st.markdown(
                    '<div class="metric-card">'
                    '<h2>✅ GOOD INVESTMENT</h2>'
                    f'<p>Confidence: {confidence:.1f}%</p>'
                    '</div>',
                    unsafe_allow_html=True
                )
                st.success("This property shows strong investment potential based on appreciation rate, connectivity, and pricing metrics.")
            else:
                st.markdown(
                    '<div class="metric-card" style="border-color:#ff6b6b">'
                    '<h2 style="color:#ff6b6b">❌ NOT RECOMMENDED</h2>'
                    f'<p>Confidence: {confidence:.1f}%</p>'
                    '</div>',
                    unsafe_allow_html=True
                )
                st.warning("This property may not yield optimal returns. Consider location, age, or pricing alternatives.")

            # Confidence gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                number={"suffix": "%", "font": {"color": "#6c63ff"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#a0a8c0"},
                    "bar": {"color": "#6c63ff"},
                    "bgcolor": "#1a1d27",
                    "steps": [
                        {"range": [0, 40], "color": "#2a1f1f"},
                        {"range": [40, 70], "color": "#1f2a25"},
                        {"range": [70, 100], "color": "#1a2535"},
                    ],
                    "threshold": {"line": {"color": "#00c896", "width": 4},
                                  "thickness": 0.75, "value": 70},
                },
                title={"text": "Model Confidence", "font": {"color": "#a0a8c0"}},
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#0f1117", font_color="#a0a8c0", height=250, margin=dict(t=40, b=0)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        # ── REGRESSION ────────────────────────────────
        with col_r2:
            st.markdown('<div class="section-title">📈 5-Year Price Forecast</div>',
                        unsafe_allow_html=True)
            if reg_bundle:
                feats_r = reg_bundle["features"]
                X_in_r = pd.DataFrame([input_dict])[
                    [f for f in feats_r if f in input_dict]
                ].reindex(columns=feats_r, fill_value=0)
                X_sc_r = reg_bundle["scaler"].transform(X_in_r)
                predicted_price = reg_bundle["model"].predict(X_sc_r)[0]
            else:
                # Demo mode
                annual_rate = min(0.08, max(0.03, appreciation_rate * 0.05 + 0.03))
                predicted_price = price_lakhs * ((1 + annual_rate) ** 5)

            gain = predicted_price - price_lakhs
            gain_pct = (gain / price_lakhs) * 100
            annual_return = ((predicted_price / price_lakhs) ** (1 / 5) - 1) * 100

            st.markdown(
                f'<div class="metric-card">'
                f'<h2>₹{predicted_price:.1f}L</h2>'
                f'<p>Estimated Price in 2029</p>'
                f'</div>',
                unsafe_allow_html=True
            )

            mc1, mc2 = st.columns(2)
            mc1.metric("Expected Gain", f"₹{gain:.1f}L", f"+{gain_pct:.1f}%")
            mc2.metric("Annual Return", f"{annual_return:.1f}%", "CAGR")

            # Year-by-year forecast chart
            years = list(range(2024, 2030))
            annual_r = annual_return / 100
            forecast = [price_lakhs * ((1 + annual_r) ** i) for i in range(6)]
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=years, y=forecast,
                mode="lines+markers",
                line=dict(color="#6c63ff", width=3),
                marker=dict(size=8, color="#00c896"),
                fill="tozeroy",
                fillcolor="rgba(108,99,255,0.1)",
                name="Forecast"
            ))
            fig_line.add_hline(y=price_lakhs, line_dash="dash",
                               line_color="#ff6b6b", annotation_text="Current Price")
            fig_line.update_layout(
                paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                font_color="#a0a8c0",
                xaxis=dict(gridcolor="#2a2f45", title="Year"),
                yaxis=dict(gridcolor="#2a2f45", title="Price (₹ Lakhs)"),
                title=dict(text="Price Forecast 2024–2029", font=dict(color="white")),
                margin=dict(t=40, b=20),
                height=300,
            )
            st.plotly_chart(fig_line, use_container_width=True)

        # ── KEY METRICS SUMMARY ───────────────────────
        st.markdown("---")
        st.markdown("### 📊 Property Metrics Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price per Sq Ft", f"₹{price_per_sqft:,.0f}")
        c2.metric("Property Age",    f"{age} years")
        c3.metric("Appreciation Score", f"{appreciation_rate:.3f}")
        c4.metric("Amenity Count", str(amenity_count))


# ═══════════════════════════════════════════════════════════════
# PAGE 2 — EDA DASHBOARD
# ═══════════════════════════════════════════════════════════════
elif page == "📊 EDA Dashboard":
    st.title("📊 Exploratory Data Analysis Dashboard")

    if df_raw is None:
        st.error("Dataset not found. Please add `data/india_housing_prices.csv`.")
        st.stop()

    df = df_raw.copy()

    # Compute derived columns if missing
    if "Age_of_Property" not in df.columns and "Year_Built" in df.columns:
        df["Age_of_Property"] = 2024 - df["Year_Built"]
    if "Price_per_SqFt" not in df.columns:
        df["Price_per_SqFt"] = df["Price_in_Lakhs"] * 1e5 / df["Size_in_SqFt"]

    tab1, tab2, tab3, tab4 = st.tabs(
        ["💰 Price & Size", "🗺️ Location", "🔗 Relationships", "🏆 Investment"]
    )

    # ── TAB 1: PRICE & SIZE ───────────────────────────
    with tab1:
        st.subheader("Price Distribution")
        fig = px.histogram(df, x="Price_in_Lakhs", nbins=60,
                           title="Property Price Distribution",
                           color_discrete_sequence=["#6c63ff"])
        fig.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                          font_color="#a0a8c0")
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Size Distribution")
            fig2 = px.histogram(df, x="Size_in_SqFt", nbins=50,
                                color_discrete_sequence=["#00c896"])
            fig2.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                               font_color="#a0a8c0")
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            if "Property_Type" in df.columns:
                st.subheader("Price per SqFt by Property Type")
                fig3 = px.box(df, x="Property_Type", y="Price_per_SqFt",
                              color="Property_Type",
                              color_discrete_sequence=px.colors.qualitative.Set2)
                fig3.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                                   font_color="#a0a8c0", showlegend=False)
                st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Size vs Price (Scatter)")
        sample = df.sample(min(3000, len(df)), random_state=42)
        fig4 = px.scatter(sample, x="Size_in_SqFt", y="Price_in_Lakhs",
                          color="BHK" if "BHK" in df.columns else None,
                          opacity=0.5, trendline="ols",
                          color_continuous_scale="viridis")
        fig4.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                           font_color="#a0a8c0")
        st.plotly_chart(fig4, use_container_width=True)

    # ── TAB 2: LOCATION ───────────────────────────────
    with tab2:
        if "State" in df.columns:
            st.subheader("Average Price per SqFt by State")
            data = df.groupby("State")["Price_per_SqFt"].mean().sort_values(ascending=False).head(15).reset_index()
            fig5 = px.bar(data, x="State", y="Price_per_SqFt",
                          color="Price_per_SqFt", color_continuous_scale="Purples")
            fig5.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                               font_color="#a0a8c0")
            st.plotly_chart(fig5, use_container_width=True)

        if "City" in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Average Price by City (Top 15)")
                data2 = df.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=False).head(15).reset_index()
                fig6 = px.bar(data2, x="Price_in_Lakhs", y="City", orientation="h",
                              color="Price_in_Lakhs", color_continuous_scale="Blues")
                fig6.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                                   font_color="#a0a8c0", height=400)
                st.plotly_chart(fig6, use_container_width=True)

            with col2:
                if "BHK" in df.columns:
                    st.subheader("BHK Distribution by City (Top 8)")
                    top_cities = df["City"].value_counts().head(8).index
                    df_city = df[df["City"].isin(top_cities)]
                    fig7 = px.histogram(df_city, x="City", color="BHK",
                                        barmode="group",
                                        color_discrete_sequence=px.colors.qualitative.Bold)
                    fig7.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                                       font_color="#a0a8c0", height=400)
                    st.plotly_chart(fig7, use_container_width=True)

    # ── TAB 3: RELATIONSHIPS ──────────────────────────
    with tab3:
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if not c.endswith("_enc")][:14]
        corr = df[num_cols].corr()
        st.subheader("Correlation Heatmap")
        fig8 = px.imshow(corr, text_auto=".2f", aspect="auto",
                         color_continuous_scale="RdBu_r")
        fig8.update_layout(paper_bgcolor="#0f1117", font_color="#a0a8c0", height=550)
        st.plotly_chart(fig8, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if "Furnished_Status" in df.columns:
                st.subheader("Price by Furnished Status")
                fig9 = px.violin(df, x="Furnished_Status", y="Price_in_Lakhs",
                                 color="Furnished_Status",
                                 color_discrete_sequence=px.colors.qualitative.Pastel)
                fig9.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                                   font_color="#a0a8c0", showlegend=False)
                st.plotly_chart(fig9, use_container_width=True)

        with col2:
            if "Facing" in df.columns:
                st.subheader("Price per SqFt by Facing Direction")
                data3 = df.groupby("Facing")["Price_per_SqFt"].mean().reset_index()
                fig10 = px.bar_polar(data3, r="Price_per_SqFt", theta="Facing",
                                     color="Price_per_SqFt",
                                     color_continuous_scale="Viridis")
                fig10.update_layout(paper_bgcolor="#0f1117", font_color="#a0a8c0")
                st.plotly_chart(fig10, use_container_width=True)

    # ── TAB 4: INVESTMENT ─────────────────────────────
    with tab4:
        if "Good_Investment" not in df.columns:
            st.info("Run preprocessing to generate `Good_Investment` labels.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Good Investment Distribution")
                data4 = df["Good_Investment"].map({1: "Good", 0: "Not Good"}).value_counts().reset_index()
                data4.columns = ["Category", "Count"]
                fig11 = px.pie(data4, values="Count", names="Category",
                               color_discrete_sequence=["#00c896", "#ff6b6b"])
                fig11.update_layout(paper_bgcolor="#0f1117", font_color="#a0a8c0")
                st.plotly_chart(fig11, use_container_width=True)

            with col2:
                if "City" in df.columns:
                    st.subheader("Investment Rate by City (%)")
                    top8 = df["City"].value_counts().head(8).index
                    inv = (df[df["City"].isin(top8)]
                           .groupby("City")["Good_Investment"]
                           .mean()
                           .sort_values(ascending=False)
                           .reset_index())
                    inv["Rate (%)"] = inv["Good_Investment"] * 100
                    fig12 = px.bar(inv, x="City", y="Rate (%)",
                                   color="Rate (%)", color_continuous_scale="Greens")
                    fig12.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                                        font_color="#a0a8c0")
                    st.plotly_chart(fig12, use_container_width=True)

            if "Public_Transport_Accessibility" in df.columns:
                st.subheader("Investment Rate by Transport Accessibility")
                inv2 = (df.groupby("Public_Transport_Accessibility")["Good_Investment"]
                        .mean()
                        .reset_index())
                inv2["Rate (%)"] = inv2["Good_Investment"] * 100
                fig13 = px.bar(inv2, x="Public_Transport_Accessibility", y="Rate (%)",
                               color="Public_Transport_Accessibility",
                               color_discrete_sequence=["#ff6b6b", "#ffd166", "#00c896"])
                fig13.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                                    font_color="#a0a8c0", showlegend=False)
                st.plotly_chart(fig13, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 3 — EXPLORE DATA
# ═══════════════════════════════════════════════════════════════
elif page == "🔍 Explore Data":
    st.title("🔍 Explore Properties")

    if df_raw is None:
        st.error("Dataset not found. Please add `data/india_housing_prices.csv`.")
        st.stop()

    df = df_raw.copy()

    st.sidebar.markdown("### Filter Properties")

    # Filters
    if "City" in df.columns:
        cities = ["All"] + sorted(df["City"].unique().tolist())
        city_f = st.sidebar.selectbox("City", cities)
        if city_f != "All":
            df = df[df["City"] == city_f]

    if "Property_Type" in df.columns:
        ptypes = ["All"] + sorted(df["Property_Type"].unique().tolist())
        ptype_f = st.sidebar.selectbox("Property Type", ptypes)
        if ptype_f != "All":
            df = df[df["Property_Type"] == ptype_f]

    if "BHK" in df.columns:
        bhk_vals = ["All"] + sorted(df["BHK"].unique().tolist())
        bhk_f = st.sidebar.selectbox("BHK", bhk_vals)
        if bhk_f != "All":
            df = df[df["BHK"] == bhk_f]

    price_min = float(df_raw["Price_in_Lakhs"].min())
    price_max = float(df_raw["Price_in_Lakhs"].max())
    price_range = st.sidebar.slider("Price Range (₹ Lakhs)",
                                    min_value=price_min, max_value=price_max,
                                    value=(price_min, min(price_max, 500.0)))
    df = df[(df["Price_in_Lakhs"] >= price_range[0]) & (df["Price_in_Lakhs"] <= price_range[1])]

    size_min = float(df_raw["Size_in_SqFt"].min())
    size_max = float(df_raw["Size_in_SqFt"].max())
    size_range = st.sidebar.slider("Size Range (SqFt)",
                                   min_value=size_min, max_value=size_max,
                                   value=(size_min, min(size_max, 3000.0)))
    df = df[(df["Size_in_SqFt"] >= size_range[0]) & (df["Size_in_SqFt"] <= size_range[1])]

    st.markdown(f"**{len(df):,} properties match your filters**")

    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Price",  f"₹{df['Price_in_Lakhs'].mean():.1f}L")
    col2.metric("Avg Size",   f"{df['Size_in_SqFt'].mean():.0f} sqft")
    col3.metric("Median Price", f"₹{df['Price_in_Lakhs'].median():.1f}L")
    if "Price_per_SqFt" in df.columns or "Size_in_SqFt" in df.columns:
        ppsqft = (df["Price_in_Lakhs"] * 1e5 / df["Size_in_SqFt"]).mean()
        col4.metric("Avg ₹/SqFt", f"₹{ppsqft:,.0f}")

    st.markdown("---")
    show_cols = [c for c in ["City", "Locality", "Property_Type", "BHK",
                              "Size_in_SqFt", "Price_in_Lakhs", "Furnished_Status",
                              "Age_of_Property", "Good_Investment"] if c in df.columns]
    st.dataframe(df[show_cols].head(200).reset_index(drop=True), use_container_width=True)

    st.download_button(
        "⬇️ Download Filtered Data (CSV)",
        df.to_csv(index=False),
        file_name="filtered_properties.csv",
        mime="text/csv"
    )


# ═══════════════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ═══════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("""
## 🏠 Real Estate Investment Advisor

**Domain:** Real Estate / Investment / Financial Analytics

### Problem Statement
A machine learning application to help investors make data-driven real estate decisions:

| Task | Type | Algorithm |
|------|------|-----------|
| Is this a Good Investment? | Classification | XGBoost / Random Forest |
| Estimated Price After 5 Years | Regression | XGBoost / Random Forest |

### Dataset Features
- **Location:** State, City, Locality
- **Property:** Type, BHK, Size, Floor, Age
- **Pricing:** Price in Lakhs, Price per SqFt
- **Amenities:** Schools, Hospitals, Transport, Parking, Security
- **Status:** Furnished, Owner Type, Availability

### Tech Stack
`Python` · `Pandas` · `Scikit-learn` · `XGBoost` · `Streamlit` · `MLflow` · `Plotly`

### Project Steps
1. **Preprocessing** — Imputation, encoding, outlier handling, feature engineering
2. **EDA** — 20 analytical questions answered with visualizations
3. **Model Training** — Classification + Regression with MLflow tracking
4. **Deployment** — This Streamlit app

### Target Variables
- `Good_Investment` — Binary (1 = Good, 0 = Not Good)
- `Price_After_5_Years` — Continuous (₹ Lakhs)

---
*Built as a Capstone Project | Timeline: 10 Days*
    """)
