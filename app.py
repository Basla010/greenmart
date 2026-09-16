"""
GreenMart — Supermarket Spending Predictor & Smart Recommendations
--------------------------------------------------------------------
Run with: streamlit run app.py

Place `marketing_campaign.csv` (the same dataset used in your notebook)
in the same folder as this script. The app trains the models fresh on
startup and caches them.

Home screen lets you pick "Shopping" or "Predict" — each opens its own
page, and you can always go back to Home to switch.

Free LLM recommendations: runs a small open model (flan-t5-small) locally
inside the app itself — no key, no account, no external server. See the
"FREE LLM RECOMMENDATION" section below. If the model can't load for any
reason, it silently falls back to the built-in template engine — so the
app always works either way.

Best models chosen from your notebook's comparison tables:
- Regression : Polynomial Regression (degree=2) -> R2 = 0.845 (best of Linear/SGD/Ridge/Lasso/Poly)
- Clustering : K-Means (k=4) -> Silhouette = 0.21 (best of KMeans/KMedoids/DBSCAN/Hierarchical)
"""

import random
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
html, body, [class*="css"] { color:#000000 !important; }
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
.home-card {
  background:#FFFFFF; border:3px solid var(--green); border-radius:16px;
  padding:36px 20px; text-align:center; margin-bottom:10px;
  transition: 0.15s;
}
.home-card h2 { color:#1E7D32 !important; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# SESSION STATE DEFAULTS
# ----------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"          # "home" | "shopping" | "predict" | "chat"
if "cart" not in st.session_state:
    st.session_state.cart = {}
if "profile" not in st.session_state:
    st.session_state.profile = dict(
        income=50000, age=40, recency=30, web=4, catalog=2, store=5, webvisits=5, children=1
    )
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def go_home():
    st.session_state.page = "home"

def go_shopping():
    st.session_state.page = "shopping"

def go_predict():
    st.session_state.page = "predict"

def go_chat():
    st.session_state.page = "chat"

def fill_test_card():
    """Fill the checkout fields with random fake test-card details (demo only)."""
    st.session_state.card_name = random.choice(
        ["Jane Doe", "John Smith", "Alex Green", "Sara Lee", "Sam Carter"]
    )
    st.session_state.card_num = " ".join(f"{random.randint(0, 9999):04d}" for _ in range(4))
    st.session_state.card_exp = f"{random.randint(1, 12):02d}/{random.randint(26, 31)}"
    st.session_state.card_cvv = f"{random.randint(100, 999)}"

# ----------------------------------------------------------------------
# PRODUCTS
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# FREE LLM RECOMMENDATION — a small model running inside the app itself
# ----------------------------------------------------------------------
# No key, no external server, no separate install like Ollama. This loads
# a small open-source model (google/flan-t5-small, ~300MB) straight from
# Hugging Face the first time the app runs, then caches it in memory.
# It downloads once automatically (no account/token needed for public
# models) and after that runs 100% locally and free — including on most
# hosting, since it's just a Python dependency, not a background service.
# Add to requirements.txt:  transformers  torch  sentencepiece
@st.cache_resource(show_spinner="Loading local model (first run only, ~300MB)...")
def load_llm():
    from transformers import pipeline
    return pipeline("text2text-generation", model="google/flan-t5-small")
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

def get_recommendation(tier, spend, profile):
    prompt = (
        f"Write a short, warm supermarket shopping recommendation (max 3 sentences) for a "
        f"'{tier}' customer with predicted spend ${spend:.2f}, income ${profile['income']}, "
        f"age {profile['age']}, {profile['children']} children at home. "
        "Suggest 1-2 concrete products."
    )
    try:
        llm = load_llm()
        result = llm(prompt, max_new_tokens=80, do_sample=True, temperature=0.8)
        text = result[0]["generated_text"].strip()
        if text:
            return text
    except Exception:
        pass
    return fallback_recommendation(tier)

# ----------------------------------------------------------------------
# HOME PAGE
# ----------------------------------------------------------------------
def render_home():
    st.markdown("""
    <div class="hero">
        <h1>🛒 GreenMart Supermarket</h1>
        <p>Fresh groceries, smart predictions, and picks made just for you.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="home-card">
            <h2>🛍️ Shopping</h2>
            <p>Browse products, build your cart, and check out.</p>
        </div>
        """, unsafe_allow_html=True)
        st.button("Go Shopping", use_container_width=True, on_click=go_shopping, key="home_shop_btn")
    with c2:
        st.markdown("""
        <div class="home-card">
            <h2>🔮 Predict</h2>
            <p>Enter a customer profile and predict spending + segment.</p>
        </div>
        """, unsafe_allow_html=True)
        st.button("Go Predict", use_container_width=True, on_click=go_predict, key="home_predict_btn")
    with c3:
        st.markdown("""
        <div class="home-card">
            <h2>💬 Chat</h2>
            <p>Ask our assistant for meal ideas, deals, and tips.</p>
        </div>
        """, unsafe_allow_html=True)
        st.button("Start Chat", use_container_width=True, on_click=go_chat, key="home_chat_btn")

# ----------------------------------------------------------------------
# SHOPPING PAGE
# ----------------------------------------------------------------------
def render_shopping():
    st.button("⬅️ Back to Home", on_click=go_home)
    st.subheader("🛍️ Today's Picks")

    cols = st.columns(4)
    for i, p in enumerate(PRODUCTS):
        with cols[i % 4]:
            st.markdown('<div class="product-card">', unsafe_allow_html=True)
            st.image(p["img"], use_container_width=True)
            st.markdown(f"**{p['name']}** \n${p['price']:.2f}")
            if st.button("➕ Add to cart", key=f"add_{i}"):
                st.session_state.cart[p["name"]] = st.session_state.cart.get(p["name"], 0) + 1
            st.markdown('</div>', unsafe_allow_html=True)

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
        st.button("🎲 Auto-fill test card", on_click=fill_test_card, use_container_width=True)
        card_name = st.text_input("Cardholder name", placeholder="Jane Doe", key="card_name")
        card_num = st.text_input("Card number", placeholder="1234 5678 9012 3456", max_chars=19, key="card_num")
        c1, c2 = st.columns(2)
        exp = c1.text_input("Expiry (MM/YY)", placeholder="08/29", key="card_exp")
        cvv = c2.text_input("CVV", placeholder="123", max_chars=4, type="password", key="card_cvv")
        if st.button("✅ Pay Now"):
            if card_name and card_num and exp and cvv:
                st.success("Payment simulated successfully — thanks for shopping with GreenMart! (Demo only, no real charge)")
            else:
                st.warning("Please fill in all card fields.")
        st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# PREDICT PAGE
# ----------------------------------------------------------------------
def render_predict():
    st.button("⬅️ Back to Home", on_click=go_home)

    if not MODELS_READY:
        st.error("Couldn't find `marketing_campaign.csv` next to app.py. "
                  "Add it to the same folder and reload the page.")
        return

    st.subheader("🔮 Predict a Customer's Spending")

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

    if st.button("📈 Predict spending & get recommendation", type="primary"):
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

        recommendation = get_recommendation(r["tier"], r["predicted_spend"], r["profile"])
        st.markdown(f'<div class="reco-box">💬 <b>Recommended for you:</b><br>{recommendation}</div>',
                    unsafe_allow_html=True)

# ----------------------------------------------------------------------
# CHAT PAGE — quick option buttons + free-text chat, same local LLM
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# CHAT PAGE — quick option buttons + keyword-aware free-text chat
# ----------------------------------------------------------------------
# Replace EVERYTHING from the "# CHAT PAGE" comment down to (but not
# including) the "# ROUTER" comment in your app.py with this whole block.
# It is fully self-contained — nothing else needs to exist elsewhere.

CHAT_OPTIONS = [
    "🥦 Healthy meal ideas",
    "💰 Tips to save money",
    "🎉 What's on sale today?",
    "🍽️ Help me plan dinner",
]

PRODUCT_NAMES = [p["name"] for p in PRODUCTS]


def pick_item(cart, exclude=None):
    """Pick a product to mention — prefer something already in the shopper's
    cart, otherwise a random product from the store."""
    pool = [n for n in (cart.keys() if cart else PRODUCT_NAMES) if n != exclude] or PRODUCT_NAMES
    return random.choice(pool)


# 10+ varied responses per quick option. Each is a template that gets filled
# in with a product name — pulled from the shopper's own cart when they have
# one, otherwise a random item from the store, so answers feel tailored.
CHAT_FALLBACKS = {
    "🥦 Healthy meal ideas": [
        "Try a quick stir-fry with {item} — light, fresh, and ready in under 20 minutes.",
        "A big salad topped with {item} is an easy way to keep dinner light but filling.",
        "Pair {item} with some Garden Veggies for a simple, balanced plate.",
        "Grilled {item} with a side salad is a lean, protein-packed option.",
        "For a healthy twist, try roasting {item} with a drizzle of olive oil and herbs.",
        "A veggie-forward bowl with {item} and Fresh Apples on the side makes a great light meal.",
        "Steamed {item} with a squeeze of lemon is quick, healthy, and keeps things simple.",
        "Swap heavier sides for {item} — it's a lighter way to round out any meal.",
        "A sheet-pan dinner with {item} and Garden Veggies is healthy and barely any cleanup.",
        "Try {item} in a wrap with fresh veggies for a quick, nutritious lunch.",
        "Overnight oats with a side of Fresh Apples is a great light, healthy breakfast option.",
    ],
    "💰 Tips to save money": [
        "Stock up on store-brand staples like {item} — small swaps like that add up fast.",
        "Check the deals aisle before you shop — combining loyalty discounts with basics like {item} can cut your basket cost by up to 20%.",
        "Buying {item} in bulk when it's on sale is one of the easiest ways to save over time.",
        "Plan meals around what's discounted this week — {item} is a great budget-friendly pick right now.",
        "Skip the pricier convenience items and build meals around basics like {item} instead.",
        "Our loyalty program stacks with weekly deals — worth checking before you add {item} to your cart.",
        "Buying {item} instead of a pre-made version is usually cheaper and just as good.",
        "Meal-prepping with staples like {item} at the start of the week helps avoid impulse buys.",
        "Keep an eye on the weekly flyer — items like {item} often go on rotation for discounts.",
        "A shopping list built around {item} and other staples keeps your basket predictable and affordable.",
        "Store-brand versions of {item} are usually just as good and noticeably cheaper.",
    ],
    "🎉 What's on sale today?": [
        "We've got a great price on {item} this week — worth grabbing while it lasts.",
        "{item} is one of today's featured deals — check the Shopping page for the full list.",
        "Keep an eye on {item} — it's discounted as part of our weekly picks right now.",
        "Today's standout deal is on {item} — head to the Shopping page to see it.",
        "{item} just got marked down — it's on the Shopping page along with the rest of today's picks.",
        "We're running a special on {item} this week, plus a few other staples worth checking out.",
        "If you're stocking up, now's a good time — {item} is discounted on the Shopping page.",
        "{item} is part of today's deals lineup — take a look at the Shopping page for more.",
        "There's a nice markdown on {item} right now, along with a few other weekly picks.",
        "Today's picks include a deal on {item} — swing by the Shopping page to see everything on offer.",
        "{item} and a few other essentials are discounted this week — worth adding to your cart.",
    ],
    "🍽️ Help me plan dinner": [
        "How about {item} with a side of Garden Veggies and some warm Bakery Bread?",
        "{item} paired with a fresh salad makes for an easy, satisfying dinner.",
        "Try building tonight's dinner around {item} — quick to prep and always a solid choice.",
        "{item} with roasted veggies and rice is a simple, filling dinner idea.",
        "For something a bit special, pair {item} with House Wine and Garden Veggies.",
        "A one-pan dinner with {item} and Garden Veggies keeps things easy on a weeknight.",
        "{item} grilled with a side of Bakery Bread is a quick, comforting dinner.",
        "Build tonight's plate around {item} — add a starch and a veggie and you're set.",
        "{item} with a light salad is a great no-fuss dinner for a busy night.",
        "Try pairing {item} with Sweet Treats for dessert to round out the meal.",
        "{item} is a solid base for tonight — pair it with whatever veggies you have on hand.",
    ],
}

GENERIC_FALLBACKS = [
    "Great question! Check out our Shopping page — today's picks include fresh produce, bakery, meat, and more.",
    "I'd suggest browsing our weekly picks on the Shopping page — there's usually something for every craving there.",
    "Not sure I caught all of that, but our Shopping page has fresh groceries, pantry staples, and weekly deals worth a look.",
    "Happy to help! For specifics on products or deals, the Shopping page is the best place to check.",
    "Good question — I'd start with the Shopping page, it's got everything organized by today's picks.",
    "I might not have a perfect answer for that, but our Shopping page is a great place to browse for ideas.",
    "That's a bit outside what I know, but our team keeps the Shopping page updated with fresh options daily.",
    "Let me point you to the Shopping page — you'll find fresh produce, meat, bakery, and more to choose from.",
    "I'd recommend checking out today's picks on the Shopping page for something that fits what you're after.",
    "Not 100% sure on that one, but browsing the Shopping page usually turns up something good!",
]

# ----------------------------------------------------------------------
# Keyword-based intent classifier — this is what makes free-typed
# messages get a response that actually matches what was typed.
# ----------------------------------------------------------------------
GREETING_WORDS = ["hi", "hello", "hey", "yo", "sup", "good morning", "good afternoon", "good evening"]
THANKS_WORDS = ["thanks", "thank you", "thx", "appreciate", "cheers"]
BYE_WORDS = ["bye", "goodbye", "see you", "later", "gotta go"]
CART_WORDS = ["cart", "basket", "what's in my", "what have i added", "how much do i owe"]
HOURS_WORDS = ["hours", "open", "close", "closing time", "opening time", "what time"]
DELIVERY_WORDS = ["deliver", "delivery", "shipping", "ship", "pickup", "pick up"]

KEYWORD_INTENTS = {
    "🥦 Healthy meal ideas": [
        "healthy", "diet", "light", "salad", "veg", "vegetable", "low calorie",
        "lean", "fit", "nutrition", "clean eating", "balanced",
    ],
    "💰 Tips to save money": [
        "save", "budget", "cheap", "affordable", "discount", "coupon", "money",
        "cost less", "spend less", "frugal", "cut costs",
    ],
    "🎉 What's on sale today?": [
        "sale", "deal", "offer", "promo", "special", "discounted", "bargain",
        "what's new", "today's deals",
    ],
    "🍽️ Help me plan dinner": [
        "dinner", "meal", "cook", "recipe", "lunch", "breakfast", "supper",
        "what should i eat", "plan a meal", "food ideas",
    ],
}


def classify_message(user_message):
    """Return an intent for a typed message, or None if nothing matches.
    Checked in order: greeting/thanks/bye -> cart/hours/delivery ->
    product name mention -> the 4 quick-reply categories."""
    msg = user_message.lower().strip()

    if any(w in msg for w in GREETING_WORDS):
        return "greeting"
    if any(w in msg for w in THANKS_WORDS):
        return "thanks"
    if any(w in msg for w in BYE_WORDS):
        return "bye"
    if any(w in msg for w in CART_WORDS):
        return "cart"
    if any(w in msg for w in HOURS_WORDS):
        return "hours"
    if any(w in msg for w in DELIVERY_WORDS):
        return "delivery"

    # product name / partial name mention, e.g. "how much is the milk?"
    for p in PRODUCTS:
        name = p["name"].lower()
        core_words = [w for w in name.replace("fresh ", "").replace("prime ", "").split() if len(w) > 3]
        if name in msg or any(w in msg for w in core_words):
            return ("product", p["name"])

    for intent, words in KEYWORD_INTENTS.items():
        if any(w in msg for w in words):
            return intent

    return None


GREETING_RESPONSES = [
    "Hey there! 👋 Welcome to GreenMart — ask me about deals, meal ideas, saving money, or what's in your cart.",
    "Hi! Happy to help — looking for meal ideas, today's deals, or a few money-saving tips?",
    "Hello! What can I help you find today?",
]
THANKS_RESPONSES = [
    "You're welcome! Let me know if you need anything else. 🛒",
    "Anytime! Happy shopping!",
    "No problem at all — enjoy your shop!",
]
BYE_RESPONSES = [
    "Thanks for stopping by GreenMart — see you next time! 👋",
    "Take care, and happy cooking!",
]
HOURS_RESPONSES = [
    "We're open daily from 8am to 10pm — plenty of time to grab what you need!",
]
DELIVERY_RESPONSES = [
    "We offer delivery on most orders — add items to your cart and choose delivery at checkout.",
]


def cart_summary_response(cart):
    if not cart:
        return "Your cart's empty right now — head to the Shopping page to add something!"
    lines = [f"{n} x{q}" for n, q in cart.items()]
    total = sum(next(p["price"] for p in PRODUCTS if p["name"] == n) * q for n, q in cart.items())
    return f"Here's what's in your cart: {', '.join(lines)} — total ${total:.2f}. Ready to check out?"


def product_response(name, cart):
    price = next(p["price"] for p in PRODUCTS if p["name"] == name)
    in_cart = cart.get(name, 0)
    if in_cart:
        return f"{name} is ${price:.2f} — looks like you've already got {in_cart} in your cart!"
    return f"{name} is ${price:.2f} — want to add it? Head to the Shopping page to grab it."


def get_chat_response(user_message, cart=None):
    cart = cart or {}
    intent = classify_message(user_message)

    if intent == "greeting":
        return random.choice(GREETING_RESPONSES)
    if intent == "thanks":
        return random.choice(THANKS_RESPONSES)
    if intent == "bye":
        return random.choice(BYE_RESPONSES)
    if intent == "cart":
        return cart_summary_response(cart)
    if intent == "hours":
        return random.choice(HOURS_RESPONSES)
    if intent == "delivery":
        return random.choice(DELIVERY_RESPONSES)
    if isinstance(intent, tuple) and intent[0] == "product":
        return product_response(intent[1], cart)
    if intent in CHAT_FALLBACKS:
        template = random.choice(CHAT_FALLBACKS[intent])
        return template.format(item=pick_item(cart))

    # No keyword matched — try the local LLM for a more open-ended reply.
    # If it can't load (no internet, etc.) this silently falls through.
    prompt = (
        "You are a friendly, helpful assistant for GreenMart, a supermarket. "
        f"Answer briefly and helpfully. Customer says: {user_message}"
    )
    try:
        llm = load_llm()
        result = llm(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
        text = result[0]["generated_text"].strip()
        if text:
            return text
    except Exception:
        pass

    return random.choice(GENERIC_FALLBACKS)


def send_chat_message(text):
    st.session_state.chat_history.append({"role": "user", "content": text})
    reply = get_chat_response(text, cart=st.session_state.get("cart", {}))
    st.session_state.chat_history.append({"role": "assistant", "content": reply})


def render_chat():
    st.button("⬅️ Back to Home", on_click=go_home)
    st.subheader("💬 Chat with GreenMart")

    st.write("Quick questions:")
    cols = st.columns(len(CHAT_OPTIONS))
    for i, opt in enumerate(CHAT_OPTIONS):
        if cols[i].button(opt, key=f"chatopt_{i}", use_container_width=True):
            send_chat_message(opt)
            st.rerun()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask me anything about GreenMart...")
    if user_input:
        send_chat_message(user_input)
        st.rerun()

    if st.session_state.chat_history:
        st.button("🗑️ Clear chat", on_click=lambda: st.session_state.chat_history.clear())

# ----------------------------------------------------------------------
# ROUTER
# ----------------------------------------------------------------------
if st.session_state.page == "home":
    render_home()
elif st.session_state.page == "shopping":
    render_shopping()
elif st.session_state.page == "predict":
    render_predict()
elif st.session_state.page == "chat":
    render_chat()
