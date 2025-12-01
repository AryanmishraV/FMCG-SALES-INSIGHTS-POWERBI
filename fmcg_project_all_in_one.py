"""
FMCG Analytics Project - All in One Script

This script:
1. Generates synthetic FMCG sales data (12,000+ rows)
2. Saves: sales_data.csv
3. Computes customer churn (60 days inactivity rule)
4. Runs Prophet forecasting (if installed) and saves forecast_output.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# -----------------------------
# CONFIGURATION
# -----------------------------
NUM_ROWS = 12000
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 12, 31)
CHURN_INACTIVITY_DAYS = 60

np.random.seed(42)
random.seed(42)

# -----------------------------
# REFERENCE DATA
# -----------------------------
regions = ["North", "South", "East", "West"]

states_by_region = {
    "North": ["Delhi", "Haryana", "Punjab", "Uttar Pradesh"],
    "South": ["Tamil Nadu", "Karnataka", "Kerala", "Telangana"],
    "East": ["West Bengal", "Odisha", "Bihar"],
    "West": ["Maharashtra", "Gujarat", "Rajasthan", "Madhya Pradesh"]
}

cities_by_state = {
    "Delhi": ["New Delhi"],
    "Haryana": ["Gurugram", "Faridabad"],
    "Punjab": ["Ludhiana", "Amritsar"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Noida"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai"],
    "Karnataka": ["Bengaluru", "Mysuru"],
    "Kerala": ["Kochi", "Thiruvananthapuram"],
    "Telangana": ["Hyderabad", "Warangal"],
    "West Bengal": ["Kolkata", "Siliguri"],
    "Odisha": ["Bhubaneswar", "Cuttack"],
    "Bihar": ["Patna", "Gaya"],
    "Maharashtra": ["Mumbai", "Pune", "Nagpur"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara"],
    "Rajasthan": ["Jaipur", "Udaipur", "Jodhpur"],
    "Madhya Pradesh": ["Indore", "Bhopal"]
}

product_catalog = [
    {"Product": "Pepsi 500ml", "Category": "Beverage", "BasePrice": 35},
    {"Product": "Pepsi 1.25L", "Category": "Beverage", "BasePrice": 70},
    {"Product": "Mirinda 500ml", "Category": "Beverage", "BasePrice": 30},
    {"Product": "7UP 500ml", "Category": "Beverage", "BasePrice": 30},
    {"Product": "Lays Classic 52g", "Category": "Snack", "BasePrice": 20},
    {"Product": "Lays Masala 52g", "Category": "Snack", "BasePrice": 20},
    {"Product": "Kurkure Masala 55g", "Category": "Snack", "BasePrice": 20},
    {"Product": "Tropicana Orange 1L", "Category": "Juice", "BasePrice": 110},
    {"Product": "Tropicana Mixed Fruit 1L", "Category": "Juice", "BasePrice": 115},
    {"Product": "Quaker Oats 1kg", "Category": "Grocery", "BasePrice": 180}
]

retailer_types = ["Kirana", "Supermarket", "Hypermarket", "HoReCa"]
distributors = [f"Distributor_{i}" for i in range(1, 16)]
sales_reps = [f"Rep_{i}" for i in range(1, 21)]
customer_types = ["New", "Returning"]

# -----------------------------
# 1. GENERATE SALES DATA
# -----------------------------
def generate_sales_data():
    date_range_days = (END_DATE - START_DATE).days
    rows = []
    possible_customers = list(range(1000, 5001))

    for i in range(NUM_ROWS):
        random_day = START_DATE + timedelta(days=np.random.randint(0, date_range_days))

        region = random.choice(regions)
        state = random.choice(states_by_region[region])
        city = random.choice(cities_by_state[state])

        product_info = random.choice(product_catalog)
        product = product_info["Product"]
        category = product_info["Category"]
        base_price = product_info["BasePrice"]

        if category == "Beverage":
            units_sold = np.random.randint(10, 200)
        elif category == "Snack":
            units_sold = np.random.randint(20, 300)
        else:
            units_sold = np.random.randint(5, 80)

        discount_pct = np.random.choice([0, 2, 5, 10, 15], p=[0.3, 0.25, 0.25, 0.15, 0.05])
        unit_price = base_price * np.random.uniform(0.95, 1.05)
        revenue = units_sold * unit_price * (1 - discount_pct / 100)
        return_pct = np.random.choice([0, 1, 2, 3, 5], p=[0.6, 0.2, 0.1, 0.07, 0.03])

        rows.append({
            "InvoiceID": f"INV{i+1:06d}",
            "Date": random_day.date(),
            "Region": region,
            "State": state,
            "City": city,
            "Distributor": random.choice(distributors),
            "RetailerType": random.choice(retailer_types),
            "Product": product,
            "Category": category,
            "UnitsSold": units_sold,
            "UnitPrice": round(unit_price, 2),
            "DiscountPct": discount_pct,
            "Revenue": round(revenue, 2),
            "ReturnPct": return_pct,
            "SalesRep": random.choice(sales_reps),
            "CustomerType": random.choice(customer_types),
            "CustomerID": random.choice(possible_customers)
        })

    df = pd.DataFrame(rows)
    return df

# -----------------------------
# 2. CUSTOMER CHURN
# -----------------------------
def compute_churn(df, inactivity_days=60):
    df["Date"] = pd.to_datetime(df["Date"])
    last_date = df["Date"].max()
    cutoff = last_date - timedelta(days=inactivity_days)

    churn = (
        df.groupby("CustomerID")["Date"]
        .max()
        .reset_index()
        .rename(columns={"Date": "LastPurchaseDate"})
    )

    churn["Status"] = churn["LastPurchaseDate"].apply(
        lambda d: "Churned" if d < cutoff else "Active"
    )

    return churn

# -----------------------------
# 3. FORECASTING
# -----------------------------
def run_forecast(df):
    try:
        from prophet import Prophet
    except:
        print("Prophet not installed.")
        return None

    df_daily = df.groupby("Date", as_index=False)["Revenue"].sum()
    df_daily.columns = ["ds", "y"]

    model = Prophet()
    model.fit(df_daily)

    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)

    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    print("Generating sales data...")
    sales = generate_sales_data()
    sales.to_csv("sales_data.csv", index=False)

    print("Computing churn...")
    churn = compute_churn(sales)
    churn.to_csv("churn_customers.csv", index=False)

    print("Running forecast...")
    forecast = run_forecast(sales)
    if forecast is not None:
        forecast.to_csv("forecast_output.csv", index=False)

    print("All steps complete.")
