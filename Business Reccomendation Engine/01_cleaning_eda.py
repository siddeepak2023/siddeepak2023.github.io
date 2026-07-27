"""
01_cleaning_eda.py
Online Retail II — Cleaning & Exploratory Data Analysis
Dataset: UCI Online Retail II (1M+ transactions, 2009-2011)
"""
import pandas as pd
import numpy as np
import sqlite3, os

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
print("Loading Online Retail II dataset (two sheets)...")
df1 = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2009-2010")
df2 = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2010-2011")
df  = pd.concat([df1, df2], ignore_index=True)
df.columns = ['Invoice','StockCode','Description','Quantity','InvoiceDate','Price','CustomerID','Country']
print(f"Raw rows: {len(df):,}  |  Columns: {df.columns.tolist()}")

# ── 2. INSPECT ────────────────────────────────────────────────────────────────
print("\n── Null counts ──")
print(df.isnull().sum())
print(f"\nDate range: {df['InvoiceDate'].min()} → {df['InvoiceDate'].max()}")
print(f"Unique customers (incl. NaN): {df['CustomerID'].nunique():,}")
print(f"Unique products: {df['StockCode'].nunique():,}")
print(f"Cancellations (C prefix): {df['Invoice'].astype(str).str.startswith('C').sum():,}")

# ── 3. CLEAN ──────────────────────────────────────────────────────────────────
print("\nCleaning...")

# Remove cancellations
df = df[~df['Invoice'].astype(str).str.startswith('C')]
# Drop rows missing CustomerID (anonymous sessions — not useful for RFM)
df = df.dropna(subset=['CustomerID'])
df['CustomerID'] = df['CustomerID'].astype(int)
# Remove zero/negative quantities and prices (returns, errors)
df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
# Drop rows with missing descriptions
df = df.dropna(subset=['Description'])
# Remove non-product StockCodes
non_products = ['M','POST','D','C2','CRUK','BANK CHARGES','AMAZONFEE']
df = df[~df['StockCode'].isin(non_products)]

print(f"Clean rows: {len(df):,}")
print(f"Unique customers: {df['CustomerID'].nunique():,}")
print(f"Unique products: {df['StockCode'].nunique():,}")

# ── 4. FEATURE ENGINEERING ───────────────────────────────────────────────────
df['Revenue']    = df['Quantity'] * df['Price']
df['Year']       = df['InvoiceDate'].dt.year
df['Month']      = df['InvoiceDate'].dt.month
df['DayOfWeek']  = df['InvoiceDate'].dt.dayofweek
df['Hour']       = df['InvoiceDate'].dt.hour

# ── 5. EDA ────────────────────────────────────────────────────────────────────
print("\n── Top 5 countries by revenue ──")
print(df.groupby('Country')['Revenue'].sum().sort_values(ascending=False).head(5)
        .apply(lambda x: f"£{x:,.0f}"))

print("\n── Monthly revenue trend ──")
monthly = df.groupby(['Year','Month'])['Revenue'].sum().reset_index()
print(monthly.tail(12).to_string(index=False))

print("\n── Top 10 products by revenue ──")
top10 = (df.groupby(['StockCode','Description'])['Revenue']
           .sum().sort_values(ascending=False).head(10))
print(top10.apply(lambda x: f"£{x:,.0f}").to_string())

print(f"\nTotal revenue: £{df['Revenue'].sum():,.2f}")
print(f"Avg order value: £{df.groupby('Invoice')['Revenue'].sum().mean():,.2f}")
print(f"Avg items per order: {df.groupby('Invoice')['Quantity'].sum().mean():.1f}")

# ── 6. RFM TABLE (SQL-style using pandas) ─────────────────────────────────────
print("\n── Building RFM table ──")
snapshot = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerID').agg(
    Recency  =('InvoiceDate', lambda x: (snapshot - x.max()).days),
    Frequency=('Invoice', 'nunique'),
    Monetary =('Revenue', 'sum'),
).reset_index()

# RFM scoring (1-5 scale)
rfm['R_Score'] = pd.qcut(rfm['Recency'],   5, labels=[5,4,3,2,1]).astype(int)
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'),  5, labels=[1,2,3,4,5]).astype(int)
rfm['RFM_Score']= rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']

print(f"RFM table shape: {rfm.shape}")
print(rfm.describe().round(1))

# ── 7. SAVE ───────────────────────────────────────────────────────────────────
df.to_csv("online_retail_clean.csv", index=False)
rfm.to_csv("rfm_raw.csv", index=False)

# SQLite for SQL analysis
conn = sqlite3.connect("retail.db")
df.to_sql("transactions", conn, if_exists="replace", index=False)
rfm.to_sql("rfm", conn, if_exists="replace", index=False)

# SQL window function example — monthly revenue with MoM growth
mom_sql = """
WITH monthly AS (
    SELECT strftime('%Y-%m', InvoiceDate) AS month,
           ROUND(SUM(Revenue), 2) AS revenue
    FROM transactions
    GROUP BY month
)
SELECT month, revenue,
       LAG(revenue) OVER (ORDER BY month) AS prev_revenue,
       ROUND((revenue - LAG(revenue) OVER (ORDER BY month))
             / LAG(revenue) OVER (ORDER BY month) * 100, 1) AS mom_growth_pct
FROM monthly
ORDER BY month;
"""
print("\n── Monthly revenue with MoM growth (SQL window function) ──")
print(pd.read_sql(mom_sql, conn).tail(8).to_string(index=False))
conn.close()

print("\nSaved: online_retail_clean.csv, rfm_raw.csv, retail.db")
print("Run 02_segmentation.py next.")
