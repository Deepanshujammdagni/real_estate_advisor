"""
Real Estate Investment Advisor
Step 2: Exploratory Data Analysis (EDA)
Covers all 20 analysis questions from the project brief.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
import warnings

warnings.filterwarnings("ignore")

# ── Global style ────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#3a3f5c",
    "axes.labelcolor":  "#e0e4f0",
    "axes.titlecolor":  "#ffffff",
    "xtick.color":      "#a0a8c0",
    "ytick.color":      "#a0a8c0",
    "text.color":       "#e0e4f0",
    "grid.color":       "#2a2f45",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
})
PALETTE   = ["#6c63ff", "#00c896", "#ff6b6b", "#ffd166", "#06d6a0", "#118ab2"]
OUT_DIR   = "eda_outputs"
os.makedirs(OUT_DIR, exist_ok=True)


def _save(fig: plt.Figure, name: str):
    path = os.path.join(OUT_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved → {path}")


def _fig(title: str, figsize=(10, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold", color="white", y=1.02)
    return fig, ax


# ═══════════════════════════════════════════════════════════════
# SECTION A — PRICE & SIZE ANALYSIS  (Q1–Q5)
# ═══════════════════════════════════════════════════════════════

def q1_price_distribution(df):
    """Q1: Distribution of property prices"""
    fig, ax = _fig("Q1 · Distribution of Property Prices (₹ Lakhs)", (10, 5))
    col = "Price_in_Lakhs"
    ax.hist(df[col].dropna(), bins=60, color=PALETTE[0], edgecolor="none", alpha=0.85)
    ax.axvline(df[col].median(), color=PALETTE[1], lw=2, ls="--", label=f"Median ₹{df[col].median():.0f}L")
    ax.axvline(df[col].mean(),   color=PALETTE[2], lw=2, ls="--", label=f"Mean ₹{df[col].mean():.0f}L")
    ax.set_xlabel("Price (₹ Lakhs)")
    ax.set_ylabel("Count")
    ax.legend()
    _save(fig, "q01_price_distribution")


def q2_size_distribution(df):
    """Q2: Distribution of property sizes"""
    fig, ax = _fig("Q2 · Distribution of Property Sizes (Sq Ft)", (10, 5))
    col = "Size_in_SqFt"
    ax.hist(df[col].dropna(), bins=60, color=PALETTE[1], edgecolor="none", alpha=0.85)
    ax.axvline(df[col].median(), color=PALETTE[0], lw=2, ls="--", label=f"Median {df[col].median():.0f} sqft")
    ax.set_xlabel("Size (Sq Ft)")
    ax.set_ylabel("Count")
    ax.legend()
    _save(fig, "q02_size_distribution")


def q3_price_per_sqft_by_type(df):
    """Q3: Price per sq ft by property type"""
    if "Property_Type" not in df.columns:
        return
    fig, ax = _fig("Q3 · Price per Sq Ft by Property Type", (10, 5))
    order = df.groupby("Property_Type")["Price_per_SqFt"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="Property_Type", y="Price_per_SqFt", order=order,
                palette=PALETTE, ax=ax, flierprops=dict(marker=".", color="gray", alpha=0.3))
    ax.set_xlabel("Property Type")
    ax.set_ylabel("Price per Sq Ft (₹)")
    ax.tick_params(axis="x", rotation=30)
    _save(fig, "q03_price_per_sqft_by_type")


def q4_size_vs_price(df):
    """Q4: Relationship between size and price"""
    fig, ax = _fig("Q4 · Property Size vs Price", (9, 5))
    sample = df.sample(min(3000, len(df)), random_state=42)
    sc = ax.scatter(sample["Size_in_SqFt"], sample["Price_in_Lakhs"],
                    alpha=0.4, s=10, c=PALETTE[0])
    # Trend line
    z = np.polyfit(sample["Size_in_SqFt"].dropna(), sample["Price_in_Lakhs"].dropna(), 1)
    p = np.poly1d(z)
    xs = np.linspace(df["Size_in_SqFt"].min(), df["Size_in_SqFt"].max(), 200)
    ax.plot(xs, p(xs), color=PALETTE[2], lw=2, label="Trend")
    ax.set_xlabel("Size (Sq Ft)")
    ax.set_ylabel("Price (₹ Lakhs)")
    ax.legend()
    _save(fig, "q04_size_vs_price")


def q5_outliers(df):
    """Q5: Outliers in price per sq ft and size"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Q5 · Outlier Detection", fontsize=14, fontweight="bold", color="white")
    for ax, col, color in zip(axes, ["Price_per_SqFt", "Size_in_SqFt"], PALETTE[:2]):
        ax.set_facecolor("#1a1d27")
        ax.boxplot(df[col].dropna(), vert=True, patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.7),
                   medianprops=dict(color="white", lw=2),
                   flierprops=dict(marker=".", color="gray", alpha=0.3))
        ax.set_title(col, color="white")
        ax.set_ylabel("Value")
        ax.tick_params(colors="#a0a8c0")
    _save(fig, "q05_outliers")


# ═══════════════════════════════════════════════════════════════
# SECTION B — LOCATION-BASED ANALYSIS  (Q6–Q10)
# ═══════════════════════════════════════════════════════════════

def q6_avg_price_per_sqft_by_state(df):
    """Q6: Average price per sq ft by state"""
    if "State" not in df.columns:
        return
    data = df.groupby("State")["Price_per_SqFt"].mean().sort_values(ascending=False).head(15)
    fig, ax = _fig("Q6 · Avg Price per Sq Ft by State (Top 15)", (12, 5))
    bars = ax.bar(data.index, data.values, color=PALETTE[0], alpha=0.85)
    ax.set_xlabel("State")
    ax.set_ylabel("Avg Price per Sq Ft (₹)")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, "q06_price_per_sqft_by_state")


def q7_avg_price_by_city(df):
    """Q7: Average property price by city"""
    if "City" not in df.columns:
        return
    data = df.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=False).head(15)
    fig, ax = _fig("Q7 · Avg Property Price by City (Top 15)", (12, 5))
    ax.barh(data.index[::-1], data.values[::-1], color=PALETTE[2], alpha=0.85)
    ax.set_xlabel("Avg Price (₹ Lakhs)")
    _save(fig, "q07_avg_price_by_city")


def q8_median_age_by_locality(df):
    """Q8: Median age of properties by locality"""
    if "Locality" not in df.columns or "Age_of_Property" not in df.columns:
        return
    data = df.groupby("Locality")["Age_of_Property"].median().sort_values().head(15)
    fig, ax = _fig("Q8 · Median Property Age by Locality (15 Youngest)", (12, 5))
    ax.bar(data.index, data.values, color=PALETTE[3], alpha=0.85)
    ax.set_xlabel("Locality")
    ax.set_ylabel("Median Age (Years)")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, "q08_median_age_by_locality")


def q9_bhk_by_city(df):
    """Q9: BHK distribution across cities"""
    if "City" not in df.columns or "BHK" not in df.columns:
        return
    top_cities = df["City"].value_counts().head(8).index
    subset = df[df["City"].isin(top_cities)]
    fig, ax = _fig("Q9 · BHK Distribution Across Top 8 Cities", (14, 6))
    bhk_city = subset.groupby(["City", "BHK"]).size().unstack(fill_value=0)
    bhk_city.plot(kind="bar", ax=ax, colormap="viridis", edgecolor="none")
    ax.set_xlabel("City")
    ax.set_ylabel("Number of Properties")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="BHK", bbox_to_anchor=(1, 1))
    _save(fig, "q09_bhk_by_city")


def q10_price_trends_top_localities(df):
    """Q10: Price trends for top 5 most expensive localities"""
    if "Locality" not in df.columns or "Year_Built" not in df.columns:
        return
    top5 = df.groupby("Locality")["Price_in_Lakhs"].mean().sort_values(ascending=False).head(5).index
    subset = df[df["Locality"].isin(top5)]
    fig, ax = _fig("Q10 · Price Trends — Top 5 Most Expensive Localities", (12, 5))
    for i, loc in enumerate(top5):
        d = subset[subset["Locality"] == loc].groupby("Year_Built")["Price_in_Lakhs"].mean()
        ax.plot(d.index, d.values, marker="o", label=loc, color=PALETTE[i % len(PALETTE)])
    ax.set_xlabel("Year Built")
    ax.set_ylabel("Avg Price (₹ Lakhs)")
    ax.legend(bbox_to_anchor=(1, 1))
    _save(fig, "q10_price_trends_localities")


# ═══════════════════════════════════════════════════════════════
# SECTION C — FEATURE RELATIONSHIPS  (Q11–Q15)
# ═══════════════════════════════════════════════════════════════

def q11_correlation_heatmap(df):
    """Q11: Correlation between numeric features"""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if not c.endswith("_enc")][:18]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.3, linecolor="#0f1117",
                ax=ax, annot_kws={"size": 7})
    ax.set_title("Q11 · Correlation Heatmap (Numeric Features)", color="white", fontsize=14, pad=12)
    ax.tick_params(colors="#a0a8c0")
    _save(fig, "q11_correlation_heatmap")


def q12_schools_vs_price(df):
    """Q12: Nearby schools vs price per sq ft"""
    if "Nearby_Schools" not in df.columns:
        return
    fig, ax = _fig("Q12 · Nearby Schools vs Price per Sq Ft", (9, 5))
    sample = df.sample(min(3000, len(df)), random_state=1)
    ax.scatter(sample["Nearby_Schools"], sample["Price_per_SqFt"],
               alpha=0.35, s=10, color=PALETTE[4])
    ax.set_xlabel("Nearby Schools (Count/Score)")
    ax.set_ylabel("Price per Sq Ft (₹)")
    _save(fig, "q12_schools_vs_price")


def q13_hospitals_vs_price(df):
    """Q13: Nearby hospitals vs price per sq ft"""
    if "Nearby_Hospitals" not in df.columns:
        return
    fig, ax = _fig("Q13 · Nearby Hospitals vs Price per Sq Ft", (9, 5))
    sample = df.sample(min(3000, len(df)), random_state=2)
    ax.scatter(sample["Nearby_Hospitals"], sample["Price_per_SqFt"],
               alpha=0.35, s=10, color=PALETTE[5])
    ax.set_xlabel("Nearby Hospitals")
    ax.set_ylabel("Price per Sq Ft (₹)")
    _save(fig, "q13_hospitals_vs_price")


def q14_price_by_furnished_status(df):
    """Q14: Price by furnished status"""
    if "Furnished_Status" not in df.columns:
        return
    fig, ax = _fig("Q14 · Price by Furnished Status", (9, 5))
    order = ["Unfurnished", "Semi-Furnished", "Semi-Furnished ", "Fully Furnished"]
    order = [o for o in order if o in df["Furnished_Status"].unique()]
    sns.violinplot(data=df, x="Furnished_Status", y="Price_in_Lakhs",
                   palette=PALETTE[:3], order=order if order else None, ax=ax)
    ax.set_xlabel("Furnished Status")
    ax.set_ylabel("Price (₹ Lakhs)")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, "q14_price_by_furnished_status")


def q15_price_by_facing(df):
    """Q15: Price per sq ft by facing direction"""
    if "Facing" not in df.columns:
        return
    data = df.groupby("Facing")["Price_per_SqFt"].mean().sort_values(ascending=False)
    fig, ax = _fig("Q15 · Avg Price per Sq Ft by Facing Direction", (9, 5))
    ax.bar(data.index, data.values, color=PALETTE[3], alpha=0.85)
    ax.set_xlabel("Facing Direction")
    ax.set_ylabel("Avg Price per Sq Ft (₹)")
    ax.tick_params(axis="x", rotation=30)
    _save(fig, "q15_price_by_facing")


# ═══════════════════════════════════════════════════════════════
# SECTION D — INVESTMENT / AMENITIES / OWNERSHIP  (Q16–Q20)
# ═══════════════════════════════════════════════════════════════

def q16_owner_type_distribution(df):
    """Q16: Properties by owner type"""
    if "Owner_Type" not in df.columns:
        return
    data = df["Owner_Type"].value_counts()
    fig, ax = _fig("Q16 · Properties by Owner Type", (8, 5))
    wedges, texts, autotexts = ax.pie(
        data.values, labels=data.index, autopct="%1.1f%%",
        colors=PALETTE, startangle=140,
        textprops={"color": "white"},
        wedgeprops={"edgecolor": "#0f1117", "linewidth": 1.5}
    )
    for at in autotexts:
        at.set_color("white")
    _save(fig, "q16_owner_type_distribution")


def q17_availability_status(df):
    """Q17: Properties by availability status"""
    if "Availability_Status" not in df.columns:
        return
    data = df["Availability_Status"].value_counts()
    fig, ax = _fig("Q17 · Properties by Availability Status", (8, 5))
    ax.bar(data.index, data.values, color=PALETTE[1], alpha=0.85)
    ax.set_xlabel("Availability Status")
    ax.set_ylabel("Number of Properties")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, "q17_availability_status")


def q18_parking_vs_price(df):
    """Q18: Parking space vs property price"""
    if "Parking_Space" not in df.columns:
        return
    fig, ax = _fig("Q18 · Parking Space vs Property Price", (9, 5))
    sns.boxplot(data=df, x="Parking_Space", y="Price_in_Lakhs",
                palette=PALETTE, ax=ax,
                flierprops=dict(marker=".", color="gray", alpha=0.3))
    ax.set_xlabel("Parking Spots")
    ax.set_ylabel("Price (₹ Lakhs)")
    _save(fig, "q18_parking_vs_price")


def q19_amenities_vs_price(df):
    """Q19: Amenity count vs price per sq ft"""
    if "Amenity_Count" not in df.columns:
        return
    fig, ax = _fig("Q19 · Amenity Count vs Price per Sq Ft", (9, 5))
    data = df.groupby("Amenity_Count")["Price_per_SqFt"].mean()
    ax.bar(data.index.astype(str), data.values, color=PALETTE[0], alpha=0.85)
    ax.set_xlabel("Number of Amenities")
    ax.set_ylabel("Avg Price per Sq Ft (₹)")
    _save(fig, "q19_amenities_vs_price")


def q20_transport_vs_price(df):
    """Q20: Public transport accessibility vs price / investment"""
    if "Public_Transport_Accessibility" not in df.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Q20 · Public Transport Accessibility vs Price & Investment",
                 fontsize=13, fontweight="bold", color="white")
    for ax in axes:
        ax.set_facecolor("#1a1d27")

    # Price per sqft
    order = df["Public_Transport_Accessibility"].unique().tolist()
    sns.boxplot(data=df, x="Public_Transport_Accessibility", y="Price_per_SqFt",
                ax=axes[0], palette=PALETTE, order=None,
                flierprops=dict(marker=".", color="gray", alpha=0.3))
    axes[0].set_title("vs Price per Sq Ft", color="white")
    axes[0].set_xlabel("Transport Accessibility")
    axes[0].set_ylabel("Price per Sq Ft (₹)")

    # Good investment rate
    if "Good_Investment" in df.columns:
        inv_rate = df.groupby("Public_Transport_Accessibility")["Good_Investment"].mean()
        axes[1].bar(inv_rate.index, inv_rate.values * 100, color=PALETTE[1], alpha=0.85)
        axes[1].set_title("vs Good Investment Rate (%)", color="white")
        axes[1].set_xlabel("Transport Accessibility")
        axes[1].set_ylabel("% Good Investment")

    _save(fig, "q20_transport_vs_price")


# ═══════════════════════════════════════════════════════════════
# BONUS: Investment classification summary
# ═══════════════════════════════════════════════════════════════
def bonus_investment_summary(df):
    if "Good_Investment" not in df.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Investment Classification Summary", fontsize=13, fontweight="bold", color="white")
    for ax in axes:
        ax.set_facecolor("#1a1d27")

    # Class balance
    data = df["Good_Investment"].value_counts()
    axes[0].pie(data.values, labels=["Not Good", "Good Investment"],
                autopct="%1.1f%%", colors=[PALETTE[2], PALETTE[1]],
                textprops={"color": "white"},
                wedgeprops={"edgecolor": "#0f1117"})
    axes[0].set_title("Good Investment Balance", color="white")

    # By city
    if "City" in df.columns:
        top_cities = df["City"].value_counts().head(8).index
        inv_by_city = (df[df["City"].isin(top_cities)]
                       .groupby("City")["Good_Investment"].mean()
                       .sort_values(ascending=False))
        axes[1].barh(inv_by_city.index[::-1], inv_by_city.values[::-1] * 100,
                     color=PALETTE[0], alpha=0.85)
        axes[1].set_title("Good Investment Rate by City (%)", color="white")
        axes[1].set_xlabel("% Good Investment")

    _save(fig, "bonus_investment_summary")


# ─────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────
def run_all_eda(df: pd.DataFrame):
    print("=" * 60)
    print("  EXPLORATORY DATA ANALYSIS — ALL 20 QUESTIONS")
    print("=" * 60)
    fns = [
        q1_price_distribution, q2_size_distribution,
        q3_price_per_sqft_by_type, q4_size_vs_price, q5_outliers,
        q6_avg_price_per_sqft_by_state, q7_avg_price_by_city,
        q8_median_age_by_locality, q9_bhk_by_city, q10_price_trends_top_localities,
        q11_correlation_heatmap, q12_schools_vs_price, q13_hospitals_vs_price,
        q14_price_by_furnished_status, q15_price_by_facing,
        q16_owner_type_distribution, q17_availability_status,
        q18_parking_vs_price, q19_amenities_vs_price, q20_transport_vs_price,
        bonus_investment_summary,
    ]
    for fn in fns:
        try:
            print(f"\n▶ {fn.__doc__}")
            fn(df)
        except Exception as e:
            print(f"  ⚠️  Skipped ({e})")

    print(f"\n✅ EDA complete. All charts saved in ./{OUT_DIR}/")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/processed_housing.csv"
    df = pd.read_csv(path)
    run_all_eda(df)
