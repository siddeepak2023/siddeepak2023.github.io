"""
05_export_dashboard.py
Compiles all analysis outputs into a single dashboard_data.json
for the live HTML dashboard.
"""
import pandas as pd
import numpy as np
import json

# ── LOAD ALL OUTPUTS ──────────────────────────────────────────────────────────
df   = pd.read_csv("online_retail_clean.csv", parse_dates=['InvoiceDate'])
cust = pd.read_csv("customer_features.csv")
df['Revenue'] = df['Quantity'] * df['Price']
desc_map = df.groupby('StockCode')['Description'].first().to_dict()

with open("model_metrics.json")           as f: metrics = json.load(f)
with open("association_rules.json")       as f: rules   = json.load(f)
with open("segment_recommendations.json") as f: recs    = json.load(f)

# ── KPIs ──────────────────────────────────────────────────────────────────────
kpis = {
    'total_revenue':       round(float(df['Revenue'].sum()), 2),
    'total_orders':        int(df['Invoice'].nunique()),
    'total_customers':     int(df['CustomerID'].nunique()),
    'total_products':      int(df['StockCode'].nunique()),
    'avg_order_value':     round(float(df.groupby('Invoice')['Revenue'].sum().mean()), 2),
    'churn_rate':          round(float(cust['Churned'].mean()) * 100, 1),
    'high_risk_customers': int((cust['ChurnRisk'] == 'High').sum()),
    'revenue_at_risk':     round(float(cust[cust['ChurnRisk'] == 'High']['Monetary'].sum()), 2),
}

# ── MONTHLY REVENUE ───────────────────────────────────────────────────────────
monthly = (df.groupby(df['InvoiceDate'].dt.to_period('M'))['Revenue']
             .sum().reset_index())
monthly['month'] = monthly['InvoiceDate'].astype(str)
monthly_list = [{'month': r['month'], 'revenue': round(float(r['Revenue']), 2)}
                for _, r in monthly.iterrows()]

# ── COUNTRY REVENUE ───────────────────────────────────────────────────────────
country_rev  = df.groupby('Country')['Revenue'].sum().sort_values(ascending=False).head(10)
country_list = [{'country': c, 'revenue': round(float(v), 2)} for c, v in country_rev.items()]

# ── TOP PRODUCTS ──────────────────────────────────────────────────────────────
prod_rev = df.groupby('StockCode')['Revenue'].sum().sort_values(ascending=False)
prod_qty = df.groupby('StockCode')['Quantity'].sum()
top_prods = []
skip = {'M','POST','D','C2','CRUK','BANK CHARGES','AMAZONFEE'}
for sc in prod_rev.index:
    if sc in skip: continue
    top_prods.append({'code': sc, 'name': desc_map.get(sc,sc),
                      'revenue': round(float(prod_rev[sc]),2),
                      'qty':     int(prod_qty.get(sc,0))})
    if len(top_prods) == 12: break

# ── SEGMENT SUMMARY ───────────────────────────────────────────────────────────
seg_summary = cust.groupby('Segment').agg(
    count          =('CustomerID','count'),
    avg_recency    =('Recency','mean'),
    avg_freq       =('Frequency','mean'),
    avg_monetary   =('Monetary','mean'),
    avg_churn_prob =('ChurnProb','mean'),
    total_revenue  =('Monetary','sum'),
).reset_index().round(1)
seg_list = seg_summary.to_dict(orient='records')

# ── PCA SCATTER ───────────────────────────────────────────────────────────────
pca_data   = cust[['CustomerID','PCA1','PCA2','Segment','ChurnProb',
                    'Monetary','Recency','Frequency']].dropna()
pca_sample = pca_data.sample(min(800, len(pca_data)), random_state=42)
pca_list = []
for _, r in pca_sample.iterrows():
    pca_list.append({
        'CustomerID': int(r['CustomerID']),
        'PCA1':       round(float(r['PCA1']),4),
        'PCA2':       round(float(r['PCA2']),4),
        'Segment':    r['Segment'],
        'ChurnProb':  round(float(r['ChurnProb']),3),
        'Monetary':   round(float(r['Monetary']),2),
        'Recency':    int(r['Recency']),
        'Frequency':  int(r['Frequency']),
    })

# ── AT-RISK CUSTOMERS ─────────────────────────────────────────────────────────
at_risk = (cust[cust['ChurnRisk']=='High']
              .sort_values('Monetary', ascending=False)
              .head(20))
at_risk_list = [{
    'CustomerID': int(r['CustomerID']),
    'Monetary':   round(float(r['Monetary']),2),
    'ChurnProb':  round(float(r['ChurnProb']),3),
    'Recency':    int(r['Recency']),
    'Frequency':  int(r['Frequency']),
    'Segment':    str(r.get('Segment','')),
} for _, r in at_risk.iterrows()]

# ── COMPILE ───────────────────────────────────────────────────────────────────
dashboard_data = {
    'kpis':               kpis,
    'monthly_revenue':    monthly_list,
    'country_revenue':    country_list,
    'top_products':       top_prods,
    'segment_summary':    seg_list,
    'pca_scatter':        pca_list,
    'risk_distribution':  {k: int(v) for k,v in cust['ChurnRisk'].value_counts().items()},
    'churn_by_segment':   cust.groupby('Segment')['ChurnProb'].mean().round(3).to_dict(),
    'at_risk_customers':  at_risk_list,
    'association_rules':  rules[:30],
    'recommendations':    recs,
    'model_metrics':      metrics,
}

with open("dashboard_data.json","w") as f:
    json.dump(dashboard_data, f, indent=2)

print("dashboard_data.json written.")
print(f"KPIs: {kpis}")
print(f"Total data points: {len(pca_list)} PCA, {len(monthly_list)} monthly, {len(rules)} rules")
