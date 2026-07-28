"""
04_recommendations.py
Product Recommendation Engine
- Market Basket Analysis (Apriori / Association Rules)
- Segment-specific product recommendations
"""
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import json, warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
df   = pd.read_csv("online_retail_clean.csv", parse_dates=['InvoiceDate'])
cust = pd.read_csv("customer_features.csv")
df['Revenue'] = df['Quantity'] * df['Price']
desc_map = df.groupby('StockCode')['Description'].first().to_dict()

# ── 2. MARKET BASKET ANALYSIS ─────────────────────────────────────────────────
# UK only — largest, most consistent market
print("Building market baskets (UK transactions)...")
uk = df[df['Country'] == 'United Kingdom'].copy()

# Top 150 products by frequency to keep matrix manageable
top_prods = uk['StockCode'].value_counts().head(150).index.tolist()
uk_top    = uk[uk['StockCode'].isin(top_prods)]

basket = (uk_top.groupby(['Invoice','StockCode'])['Quantity']
               .sum().unstack(fill_value=0))
basket = (basket > 0).astype(bool)
print(f"Basket matrix: {basket.shape[0]:,} transactions × {basket.shape[1]} products")

# Apriori
freq_items = apriori(basket, min_support=0.02, use_colnames=True, verbose=0)
print(f"Frequent itemsets: {len(freq_items):,}")

rules = association_rules(freq_items, metric='lift', min_threshold=1.5,
                           num_itemsets=len(freq_items))
rules = rules[rules['confidence'] >= 0.30].sort_values('lift', ascending=False)
print(f"Association rules: {len(rules):,}")

print("\n── Top 10 Rules by Lift ──")
for _, row in rules.head(10).iterrows():
    ant = [desc_map.get(c,c) for c in list(row['antecedents'])]
    con = [desc_map.get(c,c) for c in list(row['consequents'])]
    print(f"  {ant} => {con}")
    print(f"    support={row['support']:.3f}  confidence={row['confidence']:.3f}  lift={row['lift']:.2f}")

# ── 3. SEGMENT-BASED RECOMMENDATIONS ─────────────────────────────────────────
print("\n── Segment-based product recommendations ──")
seg_recs = {}
for seg in ['Champions','Loyal Customers','At-Risk','Lost']:
    seg_custs = cust[cust['Segment']==seg]['CustomerID'].tolist()
    seg_txns  = uk[uk['CustomerID'].isin(seg_custs)]
    top_items = (seg_txns.groupby('StockCode')
                         .agg(qty=('Quantity','sum'), rev=('Revenue','sum'),
                              orders=('Invoice','nunique'))
                         .sort_values('rev', ascending=False).head(8))
    seg_recs[seg] = []
    for sc, row in top_items.iterrows():
        seg_recs[seg].append({
            'code':   sc,
            'name':   desc_map.get(sc, sc),
            'qty':    int(row['qty']),
            'revenue':round(float(row['rev']),2),
            'orders': int(row['orders']),
        })
    print(f"\n{seg} top 3 recommendations:")
    for r in seg_recs[seg][:3]:
        print(f"  {r['name']} — £{r['revenue']:,.0f} revenue")

# ── 4. SERIALIZE RULES ────────────────────────────────────────────────────────
rules_out = []
for _, row in rules.head(50).iterrows():
    ant = list(row['antecedents'])
    con = list(row['consequents'])
    rules_out.append({
        'antecedents':      [desc_map.get(c,c) for c in ant],
        'consequents':      [desc_map.get(c,c) for c in con],
        'antecedent_codes': ant,
        'consequent_codes': con,
        'support':    round(float(row['support']),4),
        'confidence': round(float(row['confidence']),4),
        'lift':       round(float(row['lift']),4),
    })

# ── 5. SAVE ───────────────────────────────────────────────────────────────────
with open("association_rules.json","w") as f:
    json.dump(rules_out, f, indent=2)
with open("segment_recommendations.json","w") as f:
    json.dump(seg_recs, f, indent=2)

print("\nSaved: association_rules.json, segment_recommendations.json")
print("Run 05_export_dashboard.py next.")
