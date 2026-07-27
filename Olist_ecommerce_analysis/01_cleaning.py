import pandas as pd
import numpy as np
import sqlite3
import os

# ── 1. LOAD ALL DATASETS ──────────────────────────────────────────────────────
print("Loading datasets...")

orders       = pd.read_csv("olist_orders_dataset.csv")
order_items  = pd.read_csv("olist_order_items_dataset.csv")
payments     = pd.read_csv("olist_order_payments_dataset.csv")
reviews      = pd.read_csv("olist_order_reviews_dataset.csv")
products     = pd.read_csv("olist_products_dataset.csv")
customers    = pd.read_csv("olist_customers_dataset.csv")
sellers      = pd.read_csv("olist_sellers_dataset.csv")
geo          = pd.read_csv("olist_geolocation_dataset.csv")
translation  = pd.read_csv("product_category_name_translation.csv")

print("All files loaded successfully.\n")

# ── 2. INSPECT ────────────────────────────────────────────────────────────────
datasets = {
    "orders": orders,
    "order_items": order_items,
    "payments": payments,
    "reviews": reviews,
    "products": products,
    "customers": customers,
    "sellers": sellers,
}

for name, df in datasets.items():
    print(f"── {name}: {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"   Nulls: {df.isnull().sum().sum()}")
    print(f"   Columns: {df.columns.tolist()}\n")

# ── 3. CLEAN ORDERS ───────────────────────────────────────────────────────────
print("Cleaning orders...")

# Convert date columns
date_cols = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]
for col in date_cols:
    orders[col] = pd.to_datetime(orders[col], errors="coerce")

# Keep only delivered orders for revenue analysis
orders_clean = orders[orders["order_status"] == "delivered"].copy()
print(f"Delivered orders: {len(orders_clean):,} of {len(orders):,} total")

# Extract time features
orders_clean["year"]       = orders_clean["order_purchase_timestamp"].dt.year
orders_clean["month"]      = orders_clean["order_purchase_timestamp"].dt.month
orders_clean["month_name"] = orders_clean["order_purchase_timestamp"].dt.strftime("%b")
orders_clean["quarter"]    = orders_clean["order_purchase_timestamp"].dt.quarter

# Delivery time in days
orders_clean["delivery_days"] = (
    orders_clean["order_delivered_customer_date"] -
    orders_clean["order_purchase_timestamp"]
).dt.days

print(f"Date range: {orders_clean['order_purchase_timestamp'].min().date()} → "
      f"{orders_clean['order_purchase_timestamp'].max().date()}\n")

# ── 4. CLEAN ORDER ITEMS ──────────────────────────────────────────────────────
print("Cleaning order items...")

order_items_clean = order_items.copy()
order_items_clean["revenue"] = order_items_clean["price"] + order_items_clean["freight_value"]

print(f"Total revenue: ${order_items_clean['revenue'].sum():,.2f}\n")

# ── 5. CLEAN PRODUCTS ─────────────────────────────────────────────────────────
print("Cleaning products...")

# Merge English category names
products_clean = products.merge(translation, on="product_category_name", how="left")
products_clean["product_category_name_english"] = (
    products_clean["product_category_name_english"]
    .fillna(products_clean["product_category_name"])
    .fillna("unknown")
)

# Drop columns not needed for analysis
products_clean = products_clean.drop(columns=[
    "product_name_lenght", "product_description_lenght",
    "product_photos_qty", "product_weight_g",
    "product_length_cm", "product_height_cm", "product_width_cm"
], errors="ignore")

print(f"Product categories: {products_clean['product_category_name_english'].nunique()}\n")

# ── 6. CLEAN PAYMENTS ─────────────────────────────────────────────────────────
print("Cleaning payments...")

payments_clean = payments.copy()
payments_clean["payment_type"] = payments_clean["payment_type"].str.replace("_", " ").str.title()

print(f"Payment types: {payments_clean['payment_type'].unique()}\n")

# ── 7. CLEAN CUSTOMERS ────────────────────────────────────────────────────────
print("Cleaning customers...")

customers_clean = customers.copy()
customers_clean["customer_state"] = customers_clean["customer_state"].str.upper().str.strip()

print(f"States: {customers_clean['customer_state'].nunique()}\n")

# ── 8. BUILD MASTER TABLE ─────────────────────────────────────────────────────
print("Building master table...")

master = (
    orders_clean
    .merge(order_items_clean, on="order_id", how="left")
    .merge(products_clean[["product_id", "product_category_name_english"]], on="product_id", how="left")
    .merge(customers_clean[["customer_id", "customer_state", "customer_city"]], on="customer_id", how="left")
    .merge(
        payments_clean.groupby("order_id").agg(
            payment_type=("payment_type", "first"),
            payment_installments=("payment_installments", "max"),
            payment_value=("payment_value", "sum")
        ).reset_index(),
        on="order_id", how="left"
    )
)

master = master.rename(columns={"product_category_name_english": "category"})
master["category"] = master["category"].fillna("Unknown")

print(f"Master table: {master.shape[0]:,} rows × {master.shape[1]} cols")
print(f"Total revenue in master: ${master['revenue'].sum():,.2f}\n")

# ── 9. SAVE CLEANED FILES ─────────────────────────────────────────────────────
print("Saving cleaned files...")

master.to_csv("olist_master_clean.csv", index=False)
print("Saved: olist_master_clean.csv")

# Load all tables into SQLite
conn = sqlite3.connect("olist.db")

master.to_sql("master", conn, if_exists="replace", index=False)
orders_clean.to_sql("orders", conn, if_exists="replace", index=False)
order_items_clean.to_sql("order_items", conn, if_exists="replace", index=False)
products_clean.to_sql("products", conn, if_exists="replace", index=False)
customers_clean.to_sql("customers", conn, if_exists="replace", index=False)
payments_clean.to_sql("payments", conn, if_exists="replace", index=False)
sellers.to_sql("sellers", conn, if_exists="replace", index=False)

conn.close()
print("Saved: olist.db (tables: master, orders, order_items, products, customers, payments, sellers)")
print("\nCleaning complete. Run 02_sql_analysis.py next.")