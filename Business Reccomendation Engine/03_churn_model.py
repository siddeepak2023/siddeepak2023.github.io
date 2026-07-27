"""
03_churn_model.py
Churn Prediction — Random Forest with time-based train/test split
Predicts which customers will NOT purchase in the next 90 days
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, roc_auc_score,
                              confusion_matrix, roc_curve)
import json, warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
df     = pd.read_csv("online_retail_clean.csv", parse_dates=['InvoiceDate'])
rfm    = pd.read_csv("rfm_segmented.csv")
df['Revenue'] = df['Quantity'] * df['Price']
print(f"Transactions: {len(df):,}  |  Customers: {df['CustomerID'].nunique():,}")

# ── 2. TIME-BASED SPLIT (no data leakage) ─────────────────────────────────────
# Train on pre-Sep 2011 data; churn label = did NOT buy in Sep-Dec 2011
cutoff   = pd.Timestamp('2011-09-01')
train_df = df[df['InvoiceDate'] < cutoff].copy()
future_df= df[df['InvoiceDate'] >= cutoff].copy()
active_in_future = set(future_df['CustomerID'].unique())
print(f"\nTraining period: {train_df['InvoiceDate'].min().date()} → {cutoff.date()}")
print(f"Prediction window: {cutoff.date()} → {df['InvoiceDate'].max().date()}")
print(f"Customers active in future window: {len(active_in_future):,}")

# ── 3. FEATURE ENGINEERING ────────────────────────────────────────────────────
snapshot = cutoff

# Invoice-level revenue for avg order value
inv_rev  = train_df.groupby(['CustomerID','Invoice'])['Revenue'].sum().reset_index()
avg_order= inv_rev.groupby('CustomerID')['Revenue'].mean().reset_index()
avg_order.columns = ['CustomerID','AvgOrderVal']

# Tenure
dates = train_df.groupby('CustomerID').agg(
    first_purchase=('InvoiceDate','min'),
    last_purchase =('InvoiceDate','max'),
).reset_index()
dates['Recency'] = (snapshot - dates['last_purchase']).dt.days
dates['Tenure']  = (dates['last_purchase'] - dates['first_purchase']).dt.days

# Main features
cust = train_df.groupby('CustomerID').agg(
    Frequency   =('Invoice','nunique'),
    Monetary    =('Revenue','sum'),
    UniqueProds =('StockCode','nunique'),
    AvgQuantity =('Quantity','mean'),
    AvgPrice    =('Price','mean'),
    ActiveMonths=('InvoiceDate', lambda x: x.dt.to_period('M').nunique()),
).reset_index()

cust = (cust
        .merge(dates[['CustomerID','Recency','Tenure']], on='CustomerID')
        .merge(avg_order, on='CustomerID', how='left'))
cust['TxPerMonth'] = cust['Frequency'] / cust['ActiveMonths'].clip(lower=1)
cust['Churned']    = (~cust['CustomerID'].isin(active_in_future)).astype(int)

print(f"\nFeature table: {len(cust):,} customers")
print(f"Churn rate: {cust['Churned'].mean()*100:.1f}%")

# ── 4. TRAIN RANDOM FOREST ────────────────────────────────────────────────────
features = ['Recency','Frequency','Monetary','AvgOrderVal','UniqueProds',
            'AvgQuantity','AvgPrice','Tenure','ActiveMonths','TxPerMonth']
X = cust[features].fillna(0)
y = cust['Churned']

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=300, max_depth=10,
                             min_samples_leaf=5, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)

pred = rf.predict(X_te)
prob = rf.predict_proba(X_te)[:,1]
cm   = confusion_matrix(y_te, pred)
fi   = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

print(f"\n── Model Performance ──")
print(f"Accuracy:  {(pred==y_te).mean()*100:.1f}%")
print(f"ROC-AUC:   {roc_auc_score(y_te, prob):.3f}")
print(f"\n{classification_report(y_te, pred, target_names=['Retained','Churned'])}")
print(f"Confusion matrix:\n{cm}")
print(f"\nTop churn drivers:\n{fi.round(4)}")

# ── 5. SCORE ALL CUSTOMERS ────────────────────────────────────────────────────
cust['ChurnProb'] = rf.predict_proba(X)[:,1]
cust['ChurnRisk'] = pd.cut(cust['ChurnProb'],
                            bins=[0,.35,.65,1.0],
                            labels=['Low','Medium','High'])
print(f"\nRisk tier distribution:\n{cust['ChurnRisk'].value_counts()}")
print(f"\nRevenue at risk (High tier): £{cust[cust['ChurnRisk']=='High']['Monetary'].sum():,.0f}")

# ── 6. BUSINESS IMPACT ────────────────────────────────────────────────────────
high_risk = cust[cust['ChurnRisk']=='High']
print(f"\n── Business Impact ──")
print(f"High-risk customers: {len(high_risk):,}")
print(f"Avg monetary value:  £{high_risk['Monetary'].mean():,.0f}")
print(f"Total revenue at risk: £{high_risk['Monetary'].sum():,.0f}")
print(f"If model prevents 20% churn: £{high_risk['Monetary'].sum()*0.20:,.0f} saved")

# ── 7. SAVE ───────────────────────────────────────────────────────────────────
cust = cust.merge(rfm[['CustomerID','Segment','PCA1','PCA2']], on='CustomerID', how='left')
cust.to_csv("customer_features.csv", index=False)

metrics = {
    'accuracy':    round(float((pred==y_te).mean()),4),
    'roc_auc':     round(float(roc_auc_score(y_te, prob)),3),
    'churn_rate':  round(float(y.mean()),4),
    'n_customers': int(len(cust)),
    'train_size':  int(len(X_tr)),
    'test_size':   int(len(X_te)),
    'cm_tn':int(cm[0,0]),'cm_fp':int(cm[0,1]),
    'cm_fn':int(cm[1,0]),'cm_tp':int(cm[1,1]),
    'feature_importance': {k: round(float(v),4) for k,v in fi.items()}
}
with open("model_metrics.json","w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved: customer_features.csv, model_metrics.json")
print("Run 04_recommendations.py next.")
