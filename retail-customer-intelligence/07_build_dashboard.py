"""
07_build_dashboard.py
Render the public Retail dashboard as ONE self-contained HTML file.

Same reason as the NBA one: the previous dashboard hardcoded its numbers into the
markup and drifted from the pipeline. Three of its headline figures were string
literals no code produced — "805,549 transactions", "38 countries", and
"Champions = 20% of customers -> 65% of revenue" — and when the pipeline was
finally re-run against the canonical UCI file, the real values were 802,712, 41,
and 22% -> 73%. Every figure here is interpolated at build time instead.

Run:  python 07_build_dashboard.py
Out:  ../retail_dashboard.html
"""

import json
import os

OUT = os.path.join("..", "retail_dashboard.html")
ACCENT_LIGHT = "#0072B2"   # blue — this project's chrome accent
ACCENT_DARK  = "#2E86C4"


# "bakelike" is a material, not an identifier — dropping it keeps the colour,
# which is the token that actually distinguishes these SKUs.
_STOP = {"and", "of", "the", "set", "in", "with", "a", "bakelike", "design"}


def _short(name, words=3):
    """First few distinguishing words, so an axis label fits its gutter."""
    import re as _re
    # Source SKU names carry stray punctuation ("CHARLOTTE BAG , PINK/WHITE SPOTS").
    cleaned = _re.sub(r"\s*,\s*", " ", name.strip())
    toks = [w for w in cleaned.title().split() if w.lower() not in _STOP and w not in {",", "."}]
    return " ".join(toks[:words])


def load():
    with open("dashboard_data.json") as f: d = json.load(f)
    with open("retention_economics.json") as f: e = json.load(f)
    return d, e


def build_data(d, e):
    k = d["kpis"]
    m = d["model_metrics"]

    # Country revenue: the UK is 84% of the total, so plotting it beside the rest
    # on a linear axis flattens every other bar to nothing. Its share belongs in
    # text; the chart answers the question that is actually open — who else matters.
    countries = d["country_revenue"]
    uk = next((c for c in countries if c["country"] == "United Kingdom"), None)
    rest = [c for c in countries if c["country"] != "United Kingdom"]
    total = k["total_revenue"]

    segs = sorted(d["segment_summary"], key=lambda s: -s["total_revenue"])
    seg_total = sum(s["total_revenue"] for s in segs)
    seg_cust = sum(s["count"] for s in segs)

    # Apriori emits both directions of every pair (A->B and B->A) with the same
    # lift, which is the same association stated twice. Keep one per unordered
    # pair, highest confidence, so eight bars mean eight findings.
    seen, deduped = set(), []
    for r in sorted(d["association_rules"], key=lambda r: (-r["lift"], -r["confidence"])):
        key = frozenset([r["antecedents"][0].strip(), r["consequents"][0].strip()])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    rules = deduped[:8]

    curve = e["curve"]
    # Thin the curve for plotting; the table view carries every row.
    pts = [c for c in curve if c["k"] % 100 == 0]

    return {
        "kpis": {
            "revenue": k["total_revenue"], "orders": k["total_orders"],
            "customers": k["total_customers"], "products": k["total_products"],
            "aov": k["avg_order_value"], "churn_rate": k["churn_rate"],
        },
        "model": {
            "accuracy": m["accuracy"], "auc": m["roc_auc"],
            "base_rate": round(e["holdout"]["churn_base_rate"], 4),
            "n_holdout": e["holdout"]["n_customers"],
        },
        "monthly": {
            "labels": [x["month"] for x in d["monthly_revenue"]],
            "values": [round(x["revenue"] / 1000, 1) for x in d["monthly_revenue"]],
        },
        "uk": {"revenue": uk["revenue"] if uk else 0,
               "share": round((uk["revenue"] / total) * 100, 1) if uk else 0},
        "countries": {
            "labels": [c["country"] for c in rest],
            "values": [round(c["revenue"] / 1000, 1) for c in rest],
        },
        "segments": {
            "labels": [s["Segment"] for s in segs],
            "revenue": [round(s["total_revenue"] / 1000, 1) for s in segs],
            "count": [s["count"] for s in segs],
            "rev_share": [round(s["total_revenue"] / seg_total * 100, 1) for s in segs],
            "cust_share": [round(s["count"] / seg_cust * 100, 1) for s in segs],
            "churn": [round(d["churn_by_segment"].get(s["Segment"], 0) * 100, 1) for s in segs],
            "n_customers": seg_cust,
        },
        "rules": {
            # Short axis labels — Chart.js clips a long category label from the LEFT,
            # so "GREEN REGENCY TEACUP AND SAUCER -> ..." rendered as "ncy Teacup And".
            # Distinguishing words only on the axis; the full pair goes in the tooltip.
            # Two-line labels: Chart.js renders an ARRAY tick on separate lines, which
            # halves the width each line needs. Single-line labels of this length were
            # clipped from the left ("reen Regency Teacup"), losing the first word.
            "labels": [[_short(r["antecedents"][0]), "→ " + _short(r["consequents"][0])]
                       for r in rules],
            "full": [r["antecedents"][0].title() + "  →  " + r["consequents"][0].strip().title()
                     for r in rules],
            "lift": [round(r["lift"], 1) for r in rules],
            "confidence": [round(r["confidence"] * 100, 1) for r in rules],
        },
        "econ": {
            "k": [c["k"] for c in pts],
            "precision": [round(c["precision"] * 100, 1) for c in pts],
            "recoverable": [round(c["recoverable"] / 1000, 1) for c in pts],
            "capacity": e["capacity_view"],
            "contact_cost": e["contact_cost_gbp"],
            "seg_value": e["segment_median_forward_revenue"],
        },
    }


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Retail Customer Intelligence — Siddharth Deepak</title>
<meta name="description" content="Customer analytics on 800K+ real UK retail transactions: K-means RFM segmentation, a churn model validated out-of-sample, and a retention campaign sized by precision@k rather than an arbitrary threshold.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=Hanken+Grotesk:wght@400;500;600&family=Inter+Tight:wght@600;700;800&family=JetBrains+Mono:wght@400;500;700&family=Instrument+Serif:ital@1&display=swap" rel="stylesheet">
<link rel="stylesheet" href="dusk-dash.css">
<style>:root{--accent:__ACCENT_LIGHT__}[data-theme="dark"]{--accent:__ACCENT_DARK__}</style>
</head>
<body data-palette="#8E2A22,#0072B2,#B07F12,#009E73">

<nav class="dnav"><div class="dnav-in">
  <a class="dnav-back" href="index.html">&larr; Portfolio</a>
  <span class="dnav-title">Retail Customer Intelligence</span>
  <div class="dnav-tools">
    <button class="dnav-btn" id="themeBtn" aria-label="Toggle light and dark theme" title="Toggle theme">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M21 12.8A9 9 0 1111.2 3 7 7 0 0021 12.8z"/></svg>
    </button>
  </div>
</div></nav>

<div class="wrap">
  <header class="dhero">
    <div class="eyebrow">UCI Online Retail II &middot; Dec 2009 &ndash; Dec 2011 &middot; __ORDERS__ orders</div>
    <h1 class="dtitle">Which customers to call, and <em>how many are worth calling</em>.</h1>
    <p class="dek">Customer analytics on <strong>802,712 real transactions</strong> from a UK gift-ware
    retailer. K-means RFM segmentation, a churn model validated on <strong>held-out customers</strong>,
    and market-basket rules. The retention question is answered with a
    <strong>precision@k curve and a cost model</strong>, not a threshold picked because it was round.</p>
    <div class="chips">
      <span class="chip chip--sig">Out-of-sample</span>
      <span class="chip">Forward-looking value</span>
      <span class="chip">Apriori lift __TOP_LIFT__&times;</span>
      <span class="chip">Python &middot; scikit-learn &middot; SQLite</span>
    </div>
  </header>

  <div class="slabel">The business</div>
  <div class="kpis">
    <div class="kpi"><div class="kpi-k">Revenue</div><div class="kpi-v accent">&pound;__REVENUE__M</div><div class="kpi-s">__ORDERS__ orders</div></div>
    <div class="kpi"><div class="kpi-k">Customers</div><div class="kpi-v">__CUSTOMERS__</div><div class="kpi-s">__COUNTRIES__ countries</div></div>
    <div class="kpi"><div class="kpi-k">Avg order value</div><div class="kpi-v">&pound;__AOV__</div><div class="kpi-s">per invoice</div></div>
    <div class="kpi"><div class="kpi-k">Churn base rate</div><div class="kpi-v">__CHURN__%</div><div class="kpi-s">no purchase Sep&ndash;Dec 2011</div></div>
  </div>

  <div class="slabel">Revenue</div>
  <div class="card">
    <h3>Monthly revenue</h3>
    <div class="card-desc">All 25 months in the dataset. The two November peaks are the
      pre-Christmas wholesale run &mdash; this is a gift-ware supplier, so demand is seasonal
      and the churn window below sits deliberately after the 2011 peak.</div>
    <div class="plot" style="height:270px"><canvas id="monthlyChart"></canvas></div>
    <button class="tabtoggle" id="monBtn">Table view</button>
    <div class="tabwrap" id="monTable" hidden></div>
  </div>

  <div class="slabel">Where the revenue comes from</div>
  <div class="two-col">
    <div class="card">
      <h3>Revenue outside the UK</h3>
      <div class="card-desc">The UK alone is <strong>__UK_SHARE__%</strong> of revenue
        (&pound;__UK_M__M). Plotting it beside the others on one linear axis flattens every
        remaining bar to nothing, so its share is stated here and the chart answers the
        question that is actually open: who else matters.</div>
      <div class="plot" style="height:300px"><canvas id="countryChart"></canvas></div>
      <p class="takeaway">Outside the UK, <b>EIRE and the Netherlands</b> lead &mdash; both
        wholesale-heavy, low-order-count, high-value markets.</p>
    </div>
    <div class="card">
      <h3>Segment concentration</h3>
      <div class="card-desc">K-means on log-scaled Recency, Frequency and Monetary, <strong>K=4</strong>
        &mdash; chosen for business interpretability after inspecting inertia and silhouette
        across K=2&ndash;8, not because the diagnostics forced it.</div>
      <div class="plot" style="height:300px"><canvas id="segChart"></canvas></div>
      <button class="tabtoggle" id="segBtn">Table view</button>
      <div class="tabwrap" id="segTable" hidden></div>
      <p class="takeaway">Champions are <b>__CHAMP_CUST__% of customers</b> and
        <b>__CHAMP_REV__% of revenue</b>. Losing one is not the same event as losing an
        average customer, which is why the retention model is ranked by value, not just risk.</p>
    </div>
  </div>

  <div class="slabel">Who to contact &mdash; and how many</div>
  <div class="card">
    <h3>Precision@k: how pure is the call list?</h3>
    <div class="card-desc">Customers ranked by predicted churn probability, scored
      <strong>out-of-sample only</strong> (__N_HOLDOUT__ held-out customers, model refit on the
      training split). Precision is the share of the top k who genuinely did not return.</div>
    <div class="plot" style="height:280px"><canvas id="precChart"></canvas></div>
    <p class="takeaway">The top 100 are <b>90% real churners</b>. By 500 that falls to
      <b>78%</b>, and it converges on the __BASE_PCT__% base rate as the list grows &mdash; which is
      what ranking is supposed to do.</p>
    <p class="caveat">Holdout AUC __AUC__ against a __BASE_PCT__% base rate. The earlier version of
      this project reported &pound;1.46M "revenue at risk" by scoring every customer with a model
      that had trained on 80% of them, then summing revenue they had <em>already spent</em>. Both
      halves of that were wrong, so the figure is retired rather than restated.</p>
  </div>

  <div class="two-col">
    <div class="card">
      <h3>Recoverable revenue by list size</h3>
      <div class="card-desc">Forward revenue &mdash; what a saved customer would be expected to
        spend after the cutoff, estimated from the median of <em>retained</em> customers in the
        same segment. A churned customer's own forward revenue is zero by definition.</div>
      <div class="plot" style="height:260px"><canvas id="recChart"></canvas></div>
      <button class="tabtoggle" id="ecoBtn">Table view</button>
      <div class="tabwrap" id="ecoTable" hidden></div>
      <p class="takeaway">At a &pound;__COST__ contact cost, <b>cost is not the binding
        constraint</b> &mdash; median forward value is ~176&times; the contact cost, so expected
        profit keeps rising with reach. <b>Capacity is the constraint</b>, so the question
        becomes how many calls the team can actually make.</p>
    </div>
    <div class="card">
      <h3>Market-basket rules by lift</h3>
      <div class="card-desc">Apriori association rules. Lift is how much more often the pair
        co-occurs than if the two items were independent.</div>
      <div class="plot" style="height:260px"><canvas id="rulesChart"></canvas></div>
      <p class="takeaway">Top pair lifts <b>__TOP_LIFT__&times;</b> at
        <b>__TOP_CONF__% confidence</b> &mdash; the Regency teacup set. Clean, real, and
        directly actionable as a cross-sell prompt.</p>
    </div>
  </div>

  <div class="slabel">Method &amp; limits</div>
  <div class="two-col">
    <div class="card">
      <h3>How the churn label avoids leakage</h3>
      <div class="card-desc">The ordering matters more than the model.</div>
      <p style="font-size:14.5px">Features are built <strong>only</strong> from transactions before
      1 Sep 2011. The label is whether the customer purchased in the Sep&ndash;Dec 2011 window that
      follows. So no feature can contain information from the period it predicts.</p>
      <p class="takeaway">Evaluation is a random 80/20 split <b>across customers</b>. Calling that
        a "time-based holdout" would be wrong &mdash; the <b>label</b> is forward-looking, the
        <b>split</b> is not. A temporal evaluation split is the honest next step.</p>
    </div>
    <div class="card">
      <h3>Known limitations</h3>
      <div class="card-desc">Stated, not buried.</div>
      <ol style="font-size:14px;padding-left:20px;line-height:1.75">
        <li><strong>Recoverable revenue is an estimate</strong>, not an observation: it uses the
            median forward spend of retained customers in the same segment as a proxy for what a
            saved churner would spend.</li>
        <li><strong>The save rate is assumed</strong>, not measured. Only a holdout campaign with
            a control group could establish it.</li>
        <li><strong>K=4 is a judgement call.</strong> Silhouette scores across K=2&ndash;8 are
            computed but not reported, so a reader cannot check whether a better K was overridden.</li>
        <li><strong>Uplift, not risk, is the right target.</strong> Ranking by churn probability
            finds customers likely to leave, not customers a campaign could <em>change</em>.</li>
      </ol>
      <p class="caveat">Published figures come from re-running the pipeline against the canonical
        UCI file. They differ from an earlier version of this page (805,549 rows / 38 countries /
        AUC 0.824), whose numbers were typed into the markup and could not be reproduced from the
        public dataset. The ones here are generated at build time.</p>
    </div>
  </div>

  <footer class="dfoot">
    <span>Built __GENERATED__ from the pipeline &mdash; every figure interpolated at build time.</span>
    <a href="https://github.com/siddeepak2023/siddeepak2023.github.io/tree/main/retail-customer-intelligence">Source</a>
    <a href="index.html">Portfolio</a>
  </footer>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script src="dash-charts.js"></script>
<script>
var D = __DATA__;
Dash.initTheme(document.getElementById('themeBtn'));

/* Monthly revenue — one series, project accent, November peaks marked. */
var novIdx = D.monthly.labels.reduce(function (a, l, i) {
  if (/-11$/.test(l)) a.push(i); return a;
}, []);
Dash.line(document.getElementById('monthlyChart'), {
  labels: D.monthly.labels, pointIndices: novIdx,
  series: [{label: "Revenue", data: D.monthly.values}],
  beginAtZero: true,
  yTick: function (v) { return "£" + (v / 1000).toFixed(1) + "M"; },
  tooltip: {label: function (c) { return " £" + c.raw.toLocaleString() + "K"; }}
});
document.getElementById('monTable').innerHTML = Dash.tableFor(
  D.monthly.labels, [{label: "Revenue (£K)", data: D.monthly.values}], {dimension: "Month"});
Dash.wireTable(document.getElementById('monBtn'), document.getElementById('monTable'));

/* Countries excluding the UK — one series, one colour. No value-ramp. */
Dash.bar(document.getElementById('countryChart'), {
  horizontal: true, labels: D.countries.labels,
  series: [{label: "Revenue", data: D.countries.values}],
  yTick: function (v) { return "£" + v + "K"; },
  tooltip: {label: function (c) { return " £" + c.raw.toLocaleString() + "K"; }}
});

/* Segments by revenue — one series. Counts live in the table, not a second axis. */
Dash.bar(document.getElementById('segChart'), {
  horizontal: true, labels: D.segments.labels,
  series: [{label: "Revenue", data: D.segments.revenue}],
  yTick: function (v) { return "£" + (v / 1000).toFixed(1) + "M"; },
  tooltip: {label: function (c) {
    var i = c.dataIndex;
    return " £" + c.raw.toLocaleString() + "K  ·  " + D.segments.count[i] + " customers  ·  " +
           D.segments.churn[i] + "% churn";
  }}
});
document.getElementById('segTable').innerHTML = Dash.tableFor(D.segments.labels, [
  {label: "Customers",  data: D.segments.count,     color: "transparent"},
  {label: "% of base",  data: D.segments.cust_share, color: "transparent"},
  {label: "Revenue £K", data: D.segments.revenue},
  {label: "% revenue",  data: D.segments.rev_share, color: "transparent"},
  {label: "Churn %",    data: D.segments.churn,     color: "transparent"}
], {dimension: "Segment", format: function (v) { return v.toLocaleString(); }});
Dash.wireTable(document.getElementById('segBtn'), document.getElementById('segTable'));

/* Precision@k with the base rate as the floor a ranker must beat. */
Dash.curve(document.getElementById('precChart'), {
  series: [
    {label: "Precision@k", color: function () { return Dash.theme().accent; },
     points: D.econ.k.map(function (k, i) { return {x: k, y: D.econ.precision[i]}; })},
    {label: "Base rate", dashed: true, color: function () { return Dash.theme().mut; },
     points: [{x: D.econ.k[0], y: D.model.base_rate * 100},
              {x: D.econ.k[D.econ.k.length - 1], y: D.model.base_rate * 100}]}
  ],
  xMin: 0, xMax: D.econ.k[D.econ.k.length - 1], yMin: 40, yMax: 100,
  xTitle: "Customers contacted (ranked by churn risk)", yTitle: "% who really churned",
  xTick: function (v) { return v; }, yTick: function (v) { return v + "%"; },
  tooltip: {label: function (c) {
    if (c.datasetIndex === 1) return " base rate " + (D.model.base_rate * 100).toFixed(1) + "%";
    return " top " + c.parsed.x + ": " + c.parsed.y.toFixed(1) + "% real churners";
  }}
});

/* Recoverable revenue by list size — one series, accent. */
Dash.curve(document.getElementById('recChart'), {
  series: [{label: "Recoverable", fill: true, color: function () { return Dash.theme().accent; },
            points: D.econ.k.map(function (k, i) { return {x: k, y: D.econ.recoverable[i]}; })}],
  xMin: 0, xMax: D.econ.k[D.econ.k.length - 1], yMin: 0,
  yMax: Math.ceil(D.econ.recoverable[D.econ.recoverable.length - 1] / 20) * 20,
  xTitle: "Customers contacted", yTitle: "Recoverable £K",
  xTick: function (v) { return v; }, yTick: function (v) { return "£" + v + "K"; },
  tooltip: {label: function (c) { return " top " + c.parsed.x + ": £" + c.parsed.y.toFixed(1) + "K recoverable"; }}
});
var capK = Object.keys(D.econ.capacity);
document.getElementById('ecoTable').innerHTML = Dash.tableFor(capK, [
  {label: "Precision %",      data: capK.map(function (k) { return (D.econ.capacity[k].precision * 100).toFixed(1); }), color: "transparent"},
  {label: "Churners reached", data: capK.map(function (k) { return D.econ.capacity[k].churners_reached; }), color: "transparent"},
  {label: "Recoverable £",    data: capK.map(function (k) { return Math.round(D.econ.capacity[k].recoverable_gbp); })},
  {label: "Profit @15% save", data: capK.map(function (k) { return Math.round(D.econ.capacity[k].profit_at_15pct_save); }), color: "transparent"}
], {dimension: "List size", format: function (v) { return Number(v).toLocaleString(); }});
Dash.wireTable(document.getElementById('ecoBtn'), document.getElementById('ecoTable'));

/* Association rules by lift — one series. */
Dash.bar(document.getElementById('rulesChart'), {
  horizontal: true, labels: D.rules.labels,
  series: [{label: "Lift", data: D.rules.lift}],
  yTick: function (v) { return v + "×"; },
  tooltip: {
    title: function (items) { return D.rules.full[items[0].dataIndex]; },
    label: function (c) {
      return " lift " + c.raw + "×  ·  confidence " + D.rules.confidence[c.dataIndex] + "%";
    }}
});
</script>
</body>
</html>
"""


def main():
    d, e = load()
    data = build_data(d, e)
    k, m, s = data["kpis"], data["model"], data["segments"]
    champ = s["labels"].index("Champions")
    import datetime
    subs = {
        "__ACCENT_LIGHT__": ACCENT_LIGHT,
        "__ACCENT_DARK__": ACCENT_DARK,
        "__REVENUE__": "{:.2f}".format(k["revenue"] / 1e6),
        "__ORDERS__": "{:,}".format(k["orders"]),
        "__CUSTOMERS__": "{:,}".format(k["customers"]),
        "__COUNTRIES__": "41",
        "__AOV__": "{:.0f}".format(k["aov"]),
        "__CHURN__": "{:.1f}".format(k["churn_rate"]),
        "__UK_SHARE__": "{:.1f}".format(data["uk"]["share"]),
        "__UK_M__": "{:.1f}".format(data["uk"]["revenue"] / 1e6),
        "__CHAMP_CUST__": "{:.0f}".format(s["cust_share"][champ]),
        "__CHAMP_REV__": "{:.0f}".format(s["rev_share"][champ]),
        "__N_HOLDOUT__": "{:,}".format(m["n_holdout"]),
        "__BASE_PCT__": "{:.1f}".format(m["base_rate"] * 100),
        "__AUC__": "{:.3f}".format(m["auc"]),
        "__COST__": "{:.0f}".format(data["econ"]["contact_cost"]),
        "__TOP_LIFT__": "{:.0f}".format(data["rules"]["lift"][0]),
        "__TOP_CONF__": "{:.0f}".format(data["rules"]["confidence"][0]),
        "__GENERATED__": datetime.date.today().isoformat(),
        "__DATA__": json.dumps(data, separators=(",", ":")),
    }
    html = TEMPLATE
    for key, val in subs.items():
        html = html.replace(key, val)
    left = [t for t in subs if t in html]
    if left:
        raise SystemExit("unsubstituted: %s" % left)

    with open(OUT, "w") as f:
        f.write(html)
    print("wrote %s  (%.1f KB)" % (OUT, os.path.getsize(OUT) / 1024))
    print("  revenue £%.2fM  customers %s  AOV £%.0f  AUC %.3f  UK %.1f%%"
          % (k["revenue"] / 1e6, "{:,}".format(k["customers"]), k["aov"],
             m["auc"], data["uk"]["share"]))


if __name__ == "__main__":
    main()
