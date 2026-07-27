/* ============================================================================
   dash-charts.js — shared Chart.js 4 theme for the project dashboards
   ----------------------------------------------------------------------------
   Encodes the mark specs and interaction rules ONCE so every chart inherits
   them rather than restating them per page:

     bars   <=24px thick, 4px rounded data-end, square at the baseline
     lines  2px, round join/cap, area fill at ~10% opacity
     points >=8px diameter with a 2px surface ring (also the hover target)
     grid   hairline 1px SOLID, recessive, one step off surface
     text   always a text token (fg/mut) — never the series colour
     legend present for >=2 series, absent for 1 (the title names it)

   Palette comes from CSS custom properties (--c1..--c5), so light and dark each
   use their own validated steps and a theme flip needs no JS palette table.
   Assign slots in fixed order; colour follows the ENTITY, never its rank, so a
   filter must never repaint the survivors.
   ========================================================================== */
(function (global) {
  "use strict";

  var FONT = '"Hanken Grotesk", system-ui, sans-serif';
  var MONO = '"JetBrains Mono", monospace';
  var charts = [];

  function tok(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  /** Theme-aware token bundle. Re-read on every build and on theme change. */
  function theme() {
    return {
      fg: tok("--fg"),
      mut: tok("--mut"),
      grid: tok("--grid"),
      surface: tok("--chart-surface"),
      accent: tok("--accent"),
      series: [tok("--c1"), tok("--c2"), tok("--c3"), tok("--c4"), tok("--c5")]
    };
  }

  /** Series colour by fixed slot index. Past slot 5, fold into "Other". */
  function seriesColor(i) {
    var s = theme().series;
    if (i < s.length) return s[i];
      throw new Error(
        "dash-charts: no 6th categorical slot. Fold the tail into 'Other', " +
        "facet into small multiples, or use composite encoding."
      );
  }

  /** Hex -> rgba, for the ~10% area wash. */
  function wash(hex, a) {
    var h = hex.replace("#", "");
    if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
    var n = parseInt(h, 16);
    return "rgba(" + ((n >> 16) & 255) + "," + ((n >> 8) & 255) + "," + (n & 255) + "," + (a == null ? 0.1 : a) + ")";
  }

  function fmt(n, opts) {
    opts = opts || {};
    var v = Number(n);
    if (!isFinite(v)) return String(n);
    if (opts.pct) return (v * (opts.alreadyPct ? 1 : 100)).toFixed(opts.dp == null ? 1 : opts.dp) + "%";
    var s = v.toLocaleString("en-US", {
      minimumFractionDigits: opts.dp == null ? 0 : opts.dp,
      maximumFractionDigits: opts.dp == null ? 0 : opts.dp
    });
    return (opts.prefix || "") + s + (opts.suffix || "");
  }

  /* ── Base options ─────────────────────────────────────────────────────────
     axes: single y only. A dual-axis chart is never produced by this theme —
     two measures of different scale get two charts or a common index base. */
  function baseOptions(t, cfg) {
    cfg = cfg || {};
    var showLegend = cfg.legend === true;
    return {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: cfg.mode || "index", intersect: false },
      layout: { padding: { top: 6, right: 10, bottom: 0, left: 0 } },
      plugins: {
        legend: showLegend
          ? {
              display: true,
              position: "top",
              align: "start",
              labels: {
                boxWidth: 9, boxHeight: 9, usePointStyle: false,
                color: t.mut, font: { family: MONO, size: 10.5 },
                padding: 16
              }
            }
          : { display: false },
        tooltip: {
          backgroundColor: t.surface,
          titleColor: t.fg,
          bodyColor: t.fg,
          borderColor: t.grid,
          borderWidth: 1,
          cornerRadius: 0,
          padding: 11,
          displayColors: true,
          boxWidth: 9,
          boxHeight: 9,
          titleFont: { family: MONO, size: 10.5, weight: "400" },
          bodyFont: { family: FONT, size: 12.5 },
          callbacks: cfg.tooltip || {}
        }
      },
      scales: {
        x: {
          grid: { display: false, drawBorder: false },
          border: { color: t.grid },
          ticks: {
            color: t.mut, font: { family: MONO, size: 9.5 },
            maxRotation: 0, autoSkipPadding: 14
          }
        },
        y: {
          beginAtZero: cfg.beginAtZero !== false,
          grid: { color: t.grid, drawTicks: false, lineWidth: 1 /* hairline, solid */ },
          border: { display: false },
          ticks: {
            color: t.mut, font: { family: MONO, size: 9.5 },
            padding: 8, maxTicksLimit: 6,
            callback: cfg.yTick || function (v) { return fmt(v); }
          }
        }
      },
      animation: { duration: cfg.animate === false ? 0 : 900 }
    };
  }

  /* ── Builders ─────────────────────────────────────────────────────────── */

  /** Line / area. One series -> no legend. Endpoint marker only, per spec. */
  function line(canvas, spec) {
    var t = theme();
    // Label selectively: mark the points the story is about, never every point.
    // spec.pointIndices marks several; s.markIndex marks one; default is the endpoint.
    var marked = spec.pointIndices || (spec.series.length === 1 && spec.markIndices) || null;
    var sets = spec.series.map(function (s, i) {
      // One series has no identity problem, so it wears the PROJECT accent — that
      // is what gives each dashboard its own character. Two or more series switch
      // to the shared validated ramp so chart grammar stays consistent everywhere.
      var c = s.color || (spec.series.length === 1 ? t.accent : seriesColor(i));
      var markAt = s.markIndex;
      return {
        label: s.label,
        data: s.data,
        borderColor: c,
        borderWidth: 2,                       // 2px line
        borderJoinStyle: "round",
        borderCapStyle: "round",
        tension: s.tension == null ? 0.35 : s.tension,
        fill: spec.series.length === 1 && s.fill !== false,
        backgroundColor: wash(c, 0.1),        // ~10% wash, never a solid block
        pointRadius: function (ctx) {
          if (marked) return marked.indexOf(ctx.dataIndex) !== -1 ? 4.5 : 0;
          if (markAt == null) return ctx.dataIndex === s.data.length - 1 ? 4 : 0;
          return ctx.dataIndex === markAt ? 4.5 : 0;   // >=8px diameter
        },
        pointBackgroundColor: c,
        pointBorderColor: t.surface,
        pointBorderWidth: 2,                  // 2px surface ring
        pointHoverRadius: 5,
        pointHoverBorderWidth: 2,
        pointHitRadius: 14                    // hit target > mark
      };
    });
    return mk(canvas, "line", spec.labels, sets, t, spec);
  }

  /** Bar / column. Capped thickness leaves air in the band. */
  function bar(canvas, spec) {
    var t = theme();
    var horiz = spec.horizontal === true;
    var stacked = spec.stacked === true;
    var sets = spec.series.map(function (s, i) {
      var c = s.color || (spec.series.length === 1 ? t.accent : seriesColor(i));
      return {
        label: s.label,
        data: s.data,
        backgroundColor: Array.isArray(s.color) ? s.color : c,
        borderColor: stacked ? t.surface : "transparent",
        borderWidth: stacked ? 2 : 0,         // 2px surface gap between segments
        borderRadius: 4,                      // 4px rounded data-end
        borderSkipped: "start",               // square at the baseline
        maxBarThickness: 24,
        categoryPercentage: 0.74,
        barPercentage: 0.9
      };
    });
    var o = mk(canvas, "bar", spec.labels, sets, t, spec, function (opt) {
      if (horiz) opt.indexAxis = "y";
      if (stacked) { opt.scales.x.stacked = true; opt.scales.y.stacked = true; }
      if (horiz) {
        // Horizontal: the VALUE axis becomes x and the CATEGORY axis becomes y.
        // Both the grid and the tick formatter have to move with them — leaving
        // the value formatter on y renders category names as numbers.
        opt.scales.x.grid = { color: t.grid, drawTicks: false, lineWidth: 1 };
        opt.scales.x.ticks.callback = spec.yTick || function (v) { return fmt(v); };
        opt.scales.x.beginAtZero = spec.beginAtZero !== false;
        opt.scales.y.grid = { display: false };
        opt.scales.y.ticks.font = { family: FONT, size: 11.5 };
        opt.scales.y.ticks.callback = function (v) { return this.getLabelForValue(v); };
        opt.scales.y.ticks.maxTicksLimit = undefined;   // never drop a category
      }
      return opt;
    });
    return o;
  }

  /**
   * Horizontal dot plot — the right form when values sit in a narrow band above a
   * meaningful floor (AUC over 0.5, scores over a baseline). A bar would have to
   * start at zero, which compresses the differences that matter into nothing;
   * dots carry no area, so a clipped axis is honest here where it would not be
   * for bars. Pass refLine to draw the floor.
   */
  function dots(canvas, spec) {
    var t = theme();
    var n = spec.labels.length;
    var sets = spec.series.map(function (s, i) {
      var c = s.color || (spec.series.length === 1 ? t.accent : seriesColor(i));
      return {
        label: s.label,
        data: s.data,
        showLine: false,
        pointRadius: 6,                       // >=8px diameter
        pointHoverRadius: 7,
        pointBackgroundColor: c,
        pointBorderColor: t.surface,
        pointBorderWidth: 2,                  // 2px surface ring
        pointHitRadius: 16
      };
    });
    if (spec.refLine != null) {
      sets.push({
        label: spec.refLabel || "Reference",
        data: new Array(n).fill(spec.refLine),
        showLine: true,
        borderColor: t.mut,
        borderWidth: 1.5,
        borderDash: [4, 4],
        pointRadius: 0,
        pointHitRadius: 0,
        fill: false
      });
    }
    var opt = baseOptions(t, {
      legend: spec.legend != null ? spec.legend : (sets.length > 1),
      mode: "nearest", tooltip: spec.tooltip, animate: spec.animate
    });
    opt.indexAxis = "y";
    opt.scales.x = {
      min: spec.xMin, max: spec.xMax,
      grid: { color: t.grid, drawTicks: false, lineWidth: 1 },
      border: { display: false },
      ticks: {
        color: t.mut, font: { family: MONO, size: 9.5 }, maxTicksLimit: 6,
        callback: spec.xTick || function (v) { return fmt(v); }
      }
    };
    opt.scales.y = {
      grid: { display: false },
      border: { display: false },
      ticks: {
        color: t.mut, font: { family: FONT, size: 11.5 }, padding: 8,
        // A dot plot's whole point is per-entity comparison, so never let
        // autoSkip drop half the categories — an unlabelled dot says nothing.
        autoSkip: false,
        callback: function (v) { return this.getLabelForValue(v); }
      }
    };
    var chart = new global.Chart(canvas, {
      type: "line", data: { labels: spec.labels, datasets: sets }, options: opt
    });
    charts.push({ chart: chart, spec: spec, kind: "dots", canvas: canvas });
    return chart;
  }

  /** Scatter / curve (ROC, precision@k) — no category axis. */
  function curve(canvas, spec) {
    var t = theme();
    var sets = spec.series.map(function (s, i) {
      var c = s.color || seriesColor(i);
      return {
        label: s.label,
        data: s.points,
        borderColor: c,
        borderWidth: 2,
        borderDash: s.dashed ? [4, 4] : undefined,
        pointRadius: 0,
        pointHitRadius: 10,
        tension: 0,
        fill: s.fill === true ? "origin" : false,
        backgroundColor: wash(c, 0.09)
      };
    });
    var chart = new global.Chart(canvas, {
      type: "line",
      data: { datasets: sets },
      options: (function () {
        var o = baseOptions(t, { legend: spec.series.length > 1, mode: "nearest", tooltip: spec.tooltip });
        o.scales.x = {
          type: "linear", min: spec.xMin == null ? 0 : spec.xMin, max: spec.xMax == null ? 1 : spec.xMax,
          grid: { color: t.grid, drawTicks: false, lineWidth: 1 },
          border: { display: false },
          title: spec.xTitle ? { display: true, text: spec.xTitle, color: t.mut, font: { family: MONO, size: 10 } } : undefined,
          ticks: { color: t.mut, font: { family: MONO, size: 9.5 }, maxTicksLimit: 6, callback: spec.xTick || function (v) { return v; } }
        };
        o.scales.y.min = spec.yMin == null ? 0 : spec.yMin;
        o.scales.y.max = spec.yMax == null ? 1 : spec.yMax;
        if (spec.yTitle) o.scales.y.title = { display: true, text: spec.yTitle, color: t.mut, font: { family: MONO, size: 10 } };
        if (spec.yTick) o.scales.y.ticks.callback = spec.yTick;
        return o;
      })()
    });
    charts.push({ chart: chart, spec: spec, kind: "curve", canvas: canvas });
    return chart;
  }

  function mk(canvas, type, labels, sets, t, spec, tweak) {
    var opt = baseOptions(t, {
      legend: spec.legend != null ? spec.legend : sets.length > 1,
      tooltip: spec.tooltip,
      yTick: spec.yTick,
      beginAtZero: spec.beginAtZero,
      animate: spec.animate,
      mode: spec.mode
    });
    if (tweak) opt = tweak(opt);
    var chart = new global.Chart(canvas, { type: type, data: { labels: labels, datasets: sets }, options: opt });
    charts.push({ chart: chart, spec: spec, kind: type, canvas: canvas });
    return chart;
  }

  /* ── Table view ───────────────────────────────────────────────────────────
     Not optional. The ochre slot carries a sub-3:1 contrast WARN, and that
     obligates relief: every plotted value stays reachable as text. */
  function tableFor(labels, series, opts) {
    opts = opts || {};
    var t = theme();
    var head = "<thead><tr><th>" + (opts.dimension || "") + "</th>" +
      series.map(function (s, i) {
        var c = s.color || t.series[i];
        return '<th><span class="sw" style="background:' + c + '"></span>' + s.label + "</th>";
      }).join("") + "</tr></thead>";
    var body = "<tbody>" + labels.map(function (l, r) {
      return "<tr><td>" + l + "</td>" + series.map(function (s) {
        var v = s.data[r];
        return "<td>" + (opts.format ? opts.format(v) : fmt(v)) + "</td>";
      }).join("") + "</tr>";
    }).join("") + "</tbody>";
    return '<table class="dt">' + head + body + "</table>";
  }

  /** Wire a "Table" button to a container holding the markup. */
  function wireTable(btn, wrap) {
    btn.addEventListener("click", function () {
      var hidden = wrap.hasAttribute("hidden");
      if (hidden) wrap.removeAttribute("hidden"); else wrap.setAttribute("hidden", "");
      btn.textContent = hidden ? "Hide table" : "Table view";
    });
  }

  /* ── Theme ────────────────────────────────────────────────────────────────
     Dark is its own set of validated steps, so a flip must rebuild the charts
     rather than recolour them in place. */
  function applyTheme(mode) {
    if (mode === "dark") document.documentElement.setAttribute("data-theme", "dark");
    else document.documentElement.removeAttribute("data-theme");
    try { localStorage.setItem("dusk-theme", mode); } catch (e) {}
    rebuild();
    document.querySelectorAll("[data-swatch-slot]").forEach(function (el) {
      el.style.background = seriesColor(Number(el.getAttribute("data-swatch-slot")));
    });
  }

  function rebuild() {
    var t = theme();
    charts.forEach(function (rec) {
      var c = rec.chart, i = 0;
      c.data.datasets.forEach(function (ds) {
        var col = (rec.spec.series[i] && rec.spec.series[i].color) ||
                  (rec.spec.series.length === 1 ? t.accent : seriesColor(i));
        if (!Array.isArray(ds.backgroundColor) || rec.kind !== "bar") {
          ds.borderColor = col;
          ds.backgroundColor = rec.kind === "bar" ? col : wash(col, 0.1);
        }
        if (rec.kind === "bar" && rec.spec.stacked) ds.borderColor = t.surface;
        if (ds.pointBorderColor) ds.pointBorderColor = t.surface;
        i++;
      });
      var o = c.options;
      if (o.scales.y) { if (o.scales.y.grid) o.scales.y.grid.color = t.grid; o.scales.y.ticks.color = t.mut; }
      if (o.scales.x) { if (o.scales.x.grid && o.scales.x.grid.color) o.scales.x.grid.color = t.grid; o.scales.x.ticks.color = t.mut; if (o.scales.x.border) o.scales.x.border.color = t.grid; }
      if (o.plugins.legend.labels) o.plugins.legend.labels.color = t.mut;
      var tp = o.plugins.tooltip;
      tp.backgroundColor = t.surface; tp.titleColor = t.fg; tp.bodyColor = t.fg; tp.borderColor = t.grid;
      c.update("none");
    });
  }

  function initTheme(btn) {
    var saved = null;
    try { saved = localStorage.getItem("dusk-theme"); } catch (e) {}
    if (saved) applyThemeSilent(saved);
    else if (global.matchMedia && global.matchMedia("(prefers-color-scheme: dark)").matches) applyThemeSilent("dark");
    if (btn) btn.addEventListener("click", function () {
      applyTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark");
    });
  }
  function applyThemeSilent(mode) {
    if (mode === "dark") document.documentElement.setAttribute("data-theme", "dark");
    else document.documentElement.removeAttribute("data-theme");
  }

  global.Dash = {
    theme: theme, seriesColor: seriesColor, wash: wash, fmt: fmt,
    line: line, bar: bar, curve: curve, dots: dots,
    tableFor: tableFor, wireTable: wireTable,
    initTheme: initTheme, applyTheme: applyTheme, rebuild: rebuild,
    _charts: charts
  };
})(window);
