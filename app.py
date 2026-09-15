"""
GreenMart — Supermarket Spending Predictor & Smart Recommendations
--------------------------------------------------------------------
Run with:  streamlit run app.py

Place `marketing_campaign.csv` (the same dataset used in your notebook)
in the same folder as this script. The app trains the models fresh on
startup and caches them.

Best models chosen from your notebook's comparison tables:
  - Regression : Polynomial Regression (degree=2)  -> R2 = 0.845 (best of Linear/SGD/Ridge/Lasso/Poly)
  - Clustering : K-Means (k=4)                      -> Silhouette = 0.21 (best of KMeans/KMedoids/DBSCAN/Hierarchical)
  - Free LLM   : Groq's openai/gpt-oss-20b (free tier) if an API key is supplied; otherwise
                 built-in template recommendations (always free, no key needed).
"""

import os
import random
import requests
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

# ----------------------------------------------------------------------
# PAGE CONFIG + THEME (black text only, green & white everywhere else)
# ----------------------------------------------------------------------
st.set_page_config(page_title="GreenMart", page_icon="🛒", layout="wide")

st.markdown("""
<style>
:root { --green:#1E7D32; --lightgreen:#66BB6A; --white:#FFFFFF; }
html, body, [class*="css"]  { color:#000000 !important; }
.stApp { background: linear-gradient(180deg, #FFFFFF 0%, #E9F7EC 100%); }
h1, h2, h3, h4, h5, p, span, label, div { color:#000000 !important; }

.hero {
    background: var(--green);
    padding: 28px; border-radius: 14px; margin-bottom: 20px;
    text-align:center; border: 3px solid var(--lightgreen);
}
.hero h1, .hero p { color:#FFFFFF !important; }

.product-card {
    background:#FFFFFF; border:2px solid var(--green); border-radius:12px;
    padding:10px; text-align:center; margin-bottom:10px;
}
.product-card img { border-radius:8px; }

.cart-box {
    background:#FFFFFF; border:2px solid var(--green); border-radius:12px; padding:16px;
}
.credit-card {
    background: linear-gradient(135deg, var(--green), var(--lightgreen));
    border-radius:16px; padding:22px; color:#FFFFFF !important; width:100%;
    box-shadow:0 4px 10px rgba(0,0,0,0.25);
}
.credit-card * { color:#FFFFFF !important; }

.stButton>button {
    background-color: var(--green); color:#FFFFFF !important; border-radius:8px;
    border:2px solid var(--green); font-weight:600;
}
.stButton>button:hover { background-color: var(--lightgreen); border-color:var(--lightgreen); }

.tier-badge {
    display:inline-block; background:var(--green); color:#FFFFFF !important;
    padding:6px 14px; border-radius:20px; font-weight:700;
}
.reco-box {
    background:#FFFFFF; border-left:6px solid var(--green); border-radius:10px;
    padding:16px; margin-top:10px;
}

/* Sidebar: force white/green even if the browser prefers a dark theme */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 3px solid var(--green);
}
[data-testid="stSidebar"] * { color:#000000 !important; }

/* Text inputs everywhere: white field, green border, no red focus ring */
input, textarea {
    background-color:#FFFFFF !important;
    color:#000000 !important;
    border: 2px solid var(--green) !important;
    border-radius: 8px !important;
}
input:focus, textarea:focus {
    border-color: var(--lightgreen) !important;
    box-shadow: 0 0 0 2px var(--lightgreen) !important;
    outline: none !important;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# HERO / SHOP HEADER
# ----------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🛒 GreenMart Supermarket</h1>
    <p>Fresh groceries, smart predictions, and picks made just for you.</p>
</div>
""", unsafe_allow_html=True)

PRODUCTS = [
    {"name": "Fresh Apples",     "price": 1.99, "img": "https://images.unsplash.com/photo-1560806887-1e4cd0b6cbd6?w=400"},
    {"name": "Bakery Bread",     "price": 2.49, "img": "https://images.unsplash.com/photo-1509440159596-0249088772ff?w=400"},
    {"name": "Whole Milk",       "price": 1.29, "img": "https://images.unsplash.com/photo-1550583724-b2692b85b150?w=400"},
    {"name": "Garden Veggies",   "price": 3.49, "img": "https://images.unsplash.com/photo-1540420773420-3366772f4999?w=400"},
    {"name": "Prime Meat Cuts",  "price": 8.99, "img": "https://images.unsplash.com/photo-1607623814075-e51df1bdc82f?w=400"},
    {"name": "Fresh Fish Fillet","price": 7.49, "img": "https://images.unsplash.com/photo-1519708227418-c8fd9a32b7a2?w=400"},
    {"name": "House Wine",       "price": 9.99, "img": "https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?w=400"},
    {"name": "Sweet Treats",     "price": 4.29, "img": "https://images.unsplash.com/photo-1548907040-4baa419e6e6c?w=400"},
]

if "cart" not in st.session_state:
    st.session_state.cart = {}

st.subheader("🛍️ Today's Picks")
cols = st.columns(4)
for i, p in enumerate(PRODUCTS):
    with cols[i % 4]:
        st.markdown('<div class="product-card">', unsafe_allow_html=True)
        st.image(p["img"], use_container_width=True)
        st.markdown(f"**{p['name']}**  \n${p['price']:.2f}")
        if st.button("➕ Add to cart", key=f"add_{i}"):
            st.session_state.cart[p["name"]] = st.session_state.cart.get(p["name"], 0) + 1
        st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# CART + CREDIT CARD CHECKOUT
# ----------------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.markdown('<div class="cart-box">', unsafe_allow_html=True)
    st.subheader("🛒 Your Cart")
    if not st.session_state.cart:
        st.write("Your cart is empty — add something above!")
    else:
        total = 0.0
        for name, qty in list(st.session_state.cart.items()):
            price = next(p["price"] for p in PRODUCTS if p["name"] == name)
            line = price * qty
            total += line
            c1, c2, c3 = st.columns([3, 1, 1])
            c1.write(f"{name} x{qty}")
            c2.write(f"${line:.2f}")
            if c3.button("Remove", key=f"rm_{name}"):
                del st.session_state.cart[name]
                st.rerun()
        st.markdown(f"### Total: ${total:.2f}")
        if st.button("🗑️ Clear cart"):
            st.session_state.cart = {}
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="credit-card">', unsafe_allow_html=True)
    st.subheader("💳 Checkout")
    card_name = st.text_input("Cardholder name", placeholder="Jane Doe")
    card_num = st.text_input("Card number", placeholder="1234 5678 9012 3456", max_chars=19)
    c1, c2 = st.columns(2)
    exp = c1.text_input("Expiry (MM/YY)", placeholder="08/29")
    cvv = c2.text_input("CVV", placeholder="123", max_chars=4, type="password")
    if st.button("✅ Pay Now"):
        if card_name and card_num and exp and cvv:
            st.success("Payment simulated successfully — thanks for shopping with GreenMart! (Demo only, no real charge)")
        else:
            st.warning("Please fill in all card fields.")
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# ----------------------------------------------------------------------
# MODEL TRAINING (cached) — Polynomial Regression + KMeans clustering
# ----------------------------------------------------------------------
REG_FEATURES = ["Income", "Age", "Recency", "NumWebPurchases",
                 "NumCatalogPurchases", "NumStorePurchases",
                 "NumWebVisitsMonth", "Total_Children"]
CLUSTER_FEATURES = ["Income", "Age", "Total_Spend", "Recency",
                     "Total_Purchases", "Total_Children"]
TIER_NAMES = ["Budget Shopper", "Steady Shopper", "Premium Shopper", "VIP Big Spender"]


@st.cache_resource(show_spinner="Training models on your dataset...")
def load_and_train(csv_path="marketing_campaign.csv"):
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df["Income"] = df["Income"].fillna(df["Income"].median())
    spend_cols = ["MntWines", "MntFruits", "MntMeatProducts",
                  "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    df["Total_Spend"] = df[spend_cols].sum(axis=1)
    df["Age"] = 2026 - df["Year_Birth"]
    df["Total_Children"] = df["Kidhome"] + df["Teenhome"]
    df["Total_Purchases"] = (df["NumWebPurchases"] + df["NumCatalogPurchases"]
                              + df["NumStorePurchases"])

    # --- Best regression model: Polynomial Regression (degree=2) ---
    x = df[REG_FEATURES]
    y = df["Total_Spend"]
    poly = PolynomialFeatures(degree=2, include_bias=False)
    x_poly = poly.fit_transform(x)
    reg_model = LinearRegression()
    reg_model.fit(x_poly, y)

    # --- Best clustering model: K-Means (k=4) ---
    x_cluster = df[CLUSTER_FEATURES]
    cluster_scaler = StandardScaler()
    x_cluster_scaled = cluster_scaler.fit_transform(x_cluster)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(x_cluster_scaled)

    # rank clusters by mean spend so labels are meaningful regardless of KMeans' label order
    order = df.groupby("Cluster")["Total_Spend"].mean().sort_values().index.tolist()
    tier_map = {cluster_id: TIER_NAMES[rank] for rank, cluster_id in enumerate(order)}

    return {
        "poly": poly, "reg_model": reg_model,
        "cluster_scaler": cluster_scaler, "kmeans": kmeans, "tier_map": tier_map,
    }


try:
    models = load_and_train()
    MODELS_READY = True
except FileNotFoundError:
    MODELS_READY = False
    st.error("Couldn't find `marketing_campaign.csv` next to app.py. "
              "Add it to the same folder and reload the page.")

# ----------------------------------------------------------------------
# CUSTOMER PROFILE + PREDICTION
# ----------------------------------------------------------------------
st.subheader("🔮 Predict a Customer's Spending")

if "profile" not in st.session_state:
    st.session_state.profile = dict(
        income=50000, age=40, recency=30, web=4, catalog=2, store=5, webvisits=5, children=1
    )

def randomize():
    st.session_state.profile = dict(
        income=random.randint(10000, 120000),
        age=random.randint(20, 80),
        recency=random.randint(0, 100),
        web=random.randint(0, 15),
        catalog=random.randint(0, 15),
        store=random.randint(0, 15),
        webvisits=random.randint(0, 15),
        children=random.randint(0, 3),
    )

st.button("🎲 Randomize customer", on_click=randomize)

p = st.session_state.profile
c1, c2, c3, c4 = st.columns(4)
income = c1.number_input("Income ($/yr)", 0, 500000, p["income"], step=1000)
age = c2.number_input("Age", 18, 100, p["age"])
recency = c3.number_input("Days since last purchase", 0, 365, p["recency"])
children = c4.number_input("Total children at home", 0, 5, p["children"])

c5, c6, c7 = st.columns(3)
web = c5.number_input("Web purchases (past period)", 0, 50, p["web"])
catalog = c6.number_input("Catalog purchases", 0, 50, p["catalog"])
store = c7.number_input("In-store purchases", 0, 50, p["store"])
webvisits = st.number_input("Web visits / month", 0, 30, p["webvisits"])

if st.button("📈 Predict spending & get recommendation", type="primary") and MODELS_READY:
    x_new = pd.DataFrame([{
        "Income": income, "Age": age, "Recency": recency,
        "NumWebPurchases": web, "NumCatalogPurchases": catalog,
        "NumStorePurchases": store, "NumWebVisitsMonth": webvisits,
        "Total_Children": children,
    }])[REG_FEATURES]

    x_new_poly = models["poly"].transform(x_new)
    predicted_spend = max(0, float(models["reg_model"].predict(x_new_poly)[0]))

    total_purchases = web + catalog + store
    cluster_row = pd.DataFrame([{
        "Income": income, "Age": age, "Total_Spend": predicted_spend,
        "Recency": recency, "Total_Purchases": total_purchases, "Total_Children": children,
    }])[CLUSTER_FEATURES]
    cluster_scaled = models["cluster_scaler"].transform(cluster_row)
    cluster_id = int(models["kmeans"].predict(cluster_scaled)[0])
    tier = models["tier_map"][cluster_id]

    st.session_state.result = {
        "predicted_spend": predicted_spend, "tier": tier,
        "profile": dict(income=income, age=age, recency=recency, children=children,
                         web=web, catalog=catalog, store=store, webvisits=webvisits),
    }

if "result" in st.session_state:
    r = st.session_state.result
    m1, m2 = st.columns(2)
    m1.metric("Predicted total spend", f"${r['predicted_spend']:.2f}")
    m2.markdown(f"Customer segment:<br><span class='tier-badge'>{r['tier']}</span>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # FREE LLM RECOMMENDATION
    # Uses Groq's free API if a key is provided in the sidebar; otherwise
    # falls back to a built-in template engine that needs no key at all,
    # so recommendations always work for free.
    # ------------------------------------------------------------------
    FALLBACK_TIPS = {
        "Budget Shopper": [
            "Since every dollar counts for you, our weekly deals and store-brand basics (bread, milk, veggies) "
            "will stretch your budget the furthest. Look out for our discount bundle on staples this week!",
            "You shop smart! Combining our loyalty discounts with store-brand groceries can cut your basket cost "
            "by up to 20% — check the deals aisle first.",
        ],
        "Steady Shopper": [
            "You buy consistently, so a small basket top-up of fresh produce and bakery items keeps your pantry "
            "stocked without breaking routine — try our fresh veggie box this week.",
            "A regular shopper like you might enjoy our mid-range meal kits — quick, balanced, and priced right "
            "for your usual basket.",
        ],
        "Premium Shopper": [
            "You enjoy quality — our premium meat cuts and fresh fish fillets pair perfectly with a bottle from "
            "our house wine selection for a great evening in.",
            "Treat yourself: our curated cheese and wine selection is a favorite with shoppers like you.",
        ],
        "VIP Big Spender": [
            "As one of our top shoppers, you get early access to premium imports, gourmet gift boxes, and our "
            "finest wine selection — ask about our VIP loyalty perks at checkout!",
            "You deserve the best — our exclusive gold-tier gift hampers and prime cuts are curated with you in mind.",
        ],
    }

    def fallback_recommendation(tier):
        return random.choice(FALLBACK_TIPS[tier])

    def get_recommendation(tier, spend, profile, api_key):
        prompt = (
            f"A supermarket customer belongs to the '{tier}' segment with a predicted total "
            f"spend of ${spend:.2f}. Their profile: income ${profile['income']}, age {profile['age']}, "
            f"{profile['children']} children at home, buys via web/catalog/store "
            f"{profile['web']}/{profile['catalog']}/{profile['store']} times. "
            "Write a short (3 sentences max), warm, personalized supermarket shopping recommendation "
            "with 1-2 concrete product suggestions."
        )
        if api_key:
            try:
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": "openai/gpt-oss-20b",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                        "max_tokens": 150,
                    },
                    timeout=15,
                )
                if resp.status_code == 200:
                    text = resp.json()["choices"][0]["message"]["content"].strip()
                    return text, "groq", None
                else:
                    return fallback_recommendation(tier), "fallback", f"Groq returned HTTP {resp.status_code}: {resp.text[:200]}"
            except Exception as e:
                return fallback_recommendation(tier), "fallback", f"Groq request failed: {e}"
        return fallback_recommendation(tier), "fallback", None

    with st.sidebar:
        st.header("🤖 Recommendation engine")
        # Key is pulled from Streamlit secrets or an environment variable —
        # the end user never sees or enters it. Falls back to free
        # built-in templates automatically if no key is configured.
        try:
            groq_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            groq_key = ""
        if not groq_key:
            groq_key = os.environ.get("GROQ_API_KEY", "")
        if groq_key:
            st.caption("✅ Smart LLM recommendations enabled.")
        else:
            st.caption("Using free built-in recommendation templates.")

    recommendation, source, error = get_recommendation(r["tier"], r["predicted_spend"], r["profile"], groq_key)
    source_label = "🤖 Generated by Groq LLM" if source == "groq" else "📋 Built-in free template"
    st.markdown(f'<div class="reco-box">💬 <b>Recommended for you</b> <i>({source_label})</i>:<br>{recommendation}</div>',
                unsafe_allow_html=True)
    if error:
        st.warning(f"Couldn't reach Groq, used the fallback instead. Details: {error}")
