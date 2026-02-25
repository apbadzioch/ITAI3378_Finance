import streamlit as st
import pandas as pd
import plotly.express as px

# --- CONFIGURATION ---
st.set_page_config(page_title="Digital Twin Sim", layout="wide")

# -- MOCK DATA -- (substitute for kaggle)
def load_data():
    data = {
        'SKU': ['Choice Ribeye', 'Sysco Imperial Chicken', 'Bulk Limes', 'Frozen Atlantic Salmon'],
        'Category': ['Meat', 'Poultry', 'Produce', 'Seafood'],
        'Unit_Cost': [15.50, 2.34, 0.86, 12.05],
        'Inventory_Level': [120, 500, 1000, 224],
        'Days_to_Expiry': [5, 12, 4, 30]
    }
    return pd.DataFrame(data)

# --- REVENUE MANAGEMENT LOGIC ---
def calculate_target_price(cost, expiry, fuel_price, base_margin):
    # Basic logic: high fuel = higher price. low expiry = aggressive discount
    fuel_overhead = (fuel_price - 3.50) * 0.05 # every dollar over $3.50 adds 5% cost
    expiry_discount = 0.25 if expiry < 5 else 0

    margin_multiplier = 1 + base_margin + fuel_overhead - expiry_discount
    return round(cost * margin_multiplier, 2)

# --- UI LAYOUT ---
st.title("Revenue Management Digital Twin")
st.markdown("---")

# --- Sidebar: The "Environment" variables for the TWIN ---
st.sidebar.header("Market Environment Variables")
fuel_price = st.sidebar.slider("Diesel Price ($/gal)", 3.00, 6.00, 3.80)
base_margin = st.sidebar.slider("Target Base Margin (%)", 10, 40, 20) / 100
competitor_pressure = st.sidebar.select_slider("Competitor Pricing Pressure", options=['Low', 'Medium', 'High'])

# --- Main Dashboard ---
inventory = load_data()

st.subheader("Inventory Info")
# Apply the pricing logic to the entire dataframe
inventory['Recommended_Price'] = inventory.apply(
    lambda row: calculate_target_price(row['Unit_Cost'], row['Days_to_Expiry'], fuel_price, base_margin), axis=1
)
