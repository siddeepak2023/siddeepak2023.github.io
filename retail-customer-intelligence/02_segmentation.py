"""
02_segmentation.py
Customer Segmentation via RFM + K-Means Clustering
Produces 4 business-labeled customer personas
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
rfm = pd.read_csv("rfm_raw.csv")
print(f"Loaded RFM table: {len(rfm):,} customers")

# ── 2. LOG TRANSFORM + SCALE ──────────────────────────────────────────────────
# RFM distributions are heavily right-skewed — log transform before clustering
rfm_log = rfm[['Recency','Frequency','Monetary']].copy()
rfm_log['Recency']   = np.log1p(rfm_log['Recency'])
rfm_log['Frequency'] = np.log1p(rfm_log['Frequency'])
rfm_log['Monetary']  = np.log1p(rfm_log['Monetary'])

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)

# ── 3. ELBOW + SILHOUETTE — FIND BEST K ──────────────────────────────────────
print("\nFinding optimal K (2-8)...")
inertias, silhouettes = [], []
for k in range(2, 9):
    km     = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(rfm_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(rfm_scaled, labels)
    silhouettes.append(sil)
    print(f"  K={k}  inertia={km.inertia_:,.0f}  silhouette={sil:.4f}")

# Use K=4 — best business interpretability with reasonable silhouette
K = 4
print(f"\nSelected K={K} (business-meaningful segments)")

# ── 4. FIT FINAL MODEL ────────────────────────────────────────────────────────
km_final = KMeans(n_clusters=K, random_state=42, n_init=20)
rfm['Cluster'] = km_final.fit_predict(rfm_scaled)

# ── 5. LABEL CLUSTERS ─────────────────────────────────────────────────────────
# Rank clusters by composite RFM score (low recency=good, high freq+monetary=good)
profile = rfm.groupby('Cluster').agg(
    Recency  =('Recency',  'mean'),
    Frequency=('Frequency','mean'),
    Monetary =('Monetary', 'mean'),
    Count    =('CustomerID','count'),
).round(1)

profile['r_rank'] = profile['Recency'].rank()           # lower recency = better → higher rank
profile['f_rank'] = profile['Frequency'].rank(ascending=False)
profile['m_rank'] = profile['Monetary'].rank(ascending=False)
profile['score']  = profile['r_rank'] + profile['f_rank'] + profile['m_rank']
sorted_clusters   = profile['score'].sort_values().index.tolist()

label_map = {
    sorted_clusters[0]: 'Champions',       # best RFM overall
    sorted_clusters[1]: 'Loyal Customers', # good freq+monetary, moderate recency
    sorted_clusters[2]: 'At-Risk',         # decent history but fading
    sorted_clusters[3]: 'Lost',            # high recency, low everything
}
rfm['Segment'] = rfm['Cluster'].map(label_map)

print("\n── Segment profiles ──")
for seg in ['Champions','Loyal Customers','At-Risk','Lost']:
    seg_data = rfm[rfm['Segment']==seg]
    print(f"\n{seg} (n={len(seg_data):,})")
    print(f"  Avg Recency:   {seg_data['Recency'].mean():.0f} days")
    print(f"  Avg Frequency: {seg_data['Frequency'].mean():.1f} orders")
    print(f"  Avg Monetary:  £{seg_data['Monetary'].mean():,.0f}")
    print(f"  Avg RFM Score: {seg_data['RFM_Score'].mean():.1f}")

# ── 6. PCA FOR VISUALIZATION ──────────────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(rfm_scaled)
rfm['PCA1'] = pca_coords[:,0]
rfm['PCA2'] = pca_coords[:,1]
print(f"\nPCA variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")
print(f"  PC1: {pca.explained_variance_ratio_[0]*100:.1f}%  PC2: {pca.explained_variance_ratio_[1]*100:.1f}%")

# ── 7. BUSINESS INSIGHTS ──────────────────────────────────────────────────────
print("\n── Revenue distribution by segment ──")
rev_by_seg = rfm.groupby('Segment')['Monetary'].agg(['sum','mean','count'])
rev_by_seg['revenue_pct'] = rev_by_seg['sum'] / rev_by_seg['sum'].sum() * 100
rev_by_seg.columns = ['Total Revenue','Avg Revenue','Customers','Revenue %']
print(rev_by_seg.round(1))

# ── 8. SAVE ───────────────────────────────────────────────────────────────────
rfm.to_csv("rfm_segmented.csv", index=False)
print("\nSaved: rfm_segmented.csv")
print("Run 03_churn_model.py next.")
