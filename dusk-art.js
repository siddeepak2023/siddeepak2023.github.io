/* ============================================================
   DUSK INSTRUMENT — shared art + behavior
   Theme toggle · animated hero terrain · still massif ·
   Bayer-dithered art (sphere / dome / image) · scroll reveals ·
   demo typing/count-up helpers.
   Markup contract:
     <canvas id="terrain">            animated hero (optional)
     <canvas data-art="sphere|dome|massif">   stills
     <canvas data-art="image" data-src="x.png">  dithered image
     <button id="themeToggle">        theme switch (optional)
     .rv                              scroll-reveal elements
   ============================================================ */
(function(){
"use strict";
var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
var root = document.documentElement;

/* ---------- THEME ----------
   Portfolio owns its theme toggle (sd-theme key, explicit light/dark attrs)
   in the page script; it calls window.DUSK.sync() after switching. */

/* ---------- SHARED MATH ---------- */
function hash2(a, b){
  var s = Math.sin(a * 127.1 + b * 311.7) * 43758.5453;
  return s - Math.floor(s);
}
function n2(x, y){
  return Math.sin(x * 1.7 + Math.sin(y * 0.8) * 1.3) * Math.cos(y * 1.1 + Math.sin(x * 0.6));
}
function fbm(x, y, t){
  var v = 0, a = 0.55, f = 1;
  for (var i = 0; i < 4; i++) {
    v += a * n2(x * f + t * 0.25, y * f - t * 0.18);
    f *= 2.05; a *= 0.5;
  }
  return v * 0.5 + 0.5;
}
function cssVar(name){ return getComputedStyle(root).getPropertyValue(name).trim(); }

/* ---------- ANIMATED HERO TERRAIN + FALCON MORPH/DOCK (talon parity) ----------
   Wave identical at rest. On scroll the dust morphs into the falcon, rides
   the viewport, and docks beside the [data-dock] sec-head — claws on its
   bottom text line. Docked bird is painted by a document-positioned canvas
   (native compositor scroll, no redraw lag). */
var canvas = document.getElementById("terrain");
var ctx = canvas ? canvas.getContext("2d") : null;
var W = 0, H = 0, dotColor = "#EDECE8", sigColor = "#B0453B", last = 0, heroVisible = true;
var gmx = -1e4, gmy = -1e4, ggx = -1e4, ggy = -1e4, gAmp = 0;
var DPRc = Math.min(window.devicePixelRatio || 1, 2);
var follow = !reduceMotion && !!canvas;
var falconPts = null, falconAspect = 1.336;
var mT = 0, mE = 0, sc = 0, txr = 0, wsr = 0;
var waveTopDoc = 0, waveH = 0, dockBotDoc = 0, dockSecTop = 0, dockShown = false;
var dockSec = document.querySelector("[data-dock]");
var dockSection = dockSec ? dockSec.closest("section") : null;
var dcv = null;
if (follow && dockSection) {
  dcv = document.createElement("canvas");
  dcv.setAttribute("aria-hidden", "true");
  dcv.style.cssText = "position:absolute;z-index:-1;pointer-events:none;opacity:0";
  dockSection.appendChild(dcv);
}
window.addEventListener("pointermove", function(e){
  if (!canvas) return;
  var r = canvas.getBoundingClientRect();
  gmx = e.clientX - r.left; gmy = e.clientY - r.top;
}, { passive: true });
document.addEventListener("pointerleave", function(){ gmx = -1e4; gmy = -1e4; });
/* Morph target: "SD" initials sampled from the display face
   (portfolio swap — statics load the falcon mark here). */
(function loadBird(){
  if (!follow) return;
  function sampleGlyph(){
    var SW = 320, SH = 200;
    var oc = document.createElement("canvas");
    oc.width = SW; oc.height = SH;
    var o2 = oc.getContext("2d"); if (!o2) return;
    o2.font = '700 150px "Instrument Sans", sans-serif';
    o2.textAlign = "center"; o2.textBaseline = "middle";
    o2.fillText("SD", SW / 2, SH / 2 + 6);
    var px = o2.getImageData(0, 0, SW, SH).data;
    var minX = SW, maxX = 0, minY = SH, maxY = 0, raw = [];
    for (var y = 0; y < SH; y += 2) for (var x = 0; x < SW; x += 2) {
      if (px[(y * SW + x) * 4 + 3] > 60) {
        raw.push(x, y);
        if (x < minX) minX = x; if (x > maxX) maxX = x;
        if (y < minY) minY = y; if (y > maxY) maxY = y;
      }
    }
    if (raw.length < 40) return;
    var bw = Math.max(1, maxX - minX), bh = Math.max(1, maxY - minY);
    falconAspect = bw / bh;
    var pts = [];
    for (var i = 0; i < raw.length; i += 2) {
      pts.push((raw[i] - minX) / bw, (raw[i + 1] - minY) / bh);
    }
    falconPts = new Float32Array(pts);
  }
  sampleGlyph();
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(sampleGlyph);
})();
function docTopOf(el){
  var y = 0, n = el;
  while (n) { y += n.offsetTop; n = n.offsetParent; }
  return y;
}
function measureDock(){
  if (!dockSec) return;
  var next = docTopOf(dockSec) + dockSec.offsetHeight;
  if (dockSection) dockSecTop = docTopOf(dockSection);
  if (Math.abs(next - dockBotDoc) > 1) {
    dockBotDoc = next;
    if (dockShown) drawDockFalcon();
  }
}
function heroScroll(){
  sc = window.scrollY;
  mT = Math.min(1, Math.max(0, sc / (Math.max(1, waveH) * 0.25)));
  measureDock();
}
function falconGeom(){
  var minL = Math.max(txr, wsr) > 0 ? Math.max(txr, wsr) + 24 : 0;
  var avail = W - minL - 12;
  /* SD glyph is wider than the falcon mark — cap the docked size lower */
  var fH = Math.max(120, Math.min(H * 0.26, avail / falconAspect));
  var fW = fH * falconAspect;
  var fX = Math.min(Math.max(W * 0.74 - fW / 2, minL), W - fW - 8);
  return { minL: minL, avail: avail, fH: fH, fW: fW, fX: fX };
}
function drawDockFalcon(){
  if (!falconPts || !dcv) return;
  var g = falconGeom();
  var fYdoc = dockBotDoc - g.fH, pad = 8;
  var bw = Math.ceil(g.fW + pad * 2), bh = Math.ceil(g.fH + pad * 2);
  dcv.width = bw * DPRc; dcv.height = bh * DPRc;
  dcv.style.width = bw + "px"; dcv.style.height = bh + "px";
  dcv.style.left = Math.round(g.fX - pad) + "px";
  dcv.style.top = Math.round(fYdoc - pad - dockSecTop) + "px";
  var d2 = dcv.getContext("2d"); if (!d2) return;
  d2.setTransform(DPRc, 0, 0, DPRc, 0, 0);
  d2.clearRect(0, 0, bw, bh);
  var nPts = falconPts.length / 2;
  var ROWS = 92, COLS = Math.min(340, Math.max(150, Math.floor(W / 4)));
  var ox = g.fX - pad, oyDoc = fYdoc - pad;
  d2.fillStyle = dotColor;
  for (var r = 0; r < ROWS; r++) for (var c = 0; c <= COLS; c++) {
    var ti = (Math.floor(hash2(c + 13, r + 7) * nPts)) % nPts;
    var tx = g.fX + falconPts[ti * 2] * g.fW + (hash2(c + 3, r + 5) - 0.5) * 5;
    var ty = fYdoc + falconPts[ti * 2 + 1] * g.fH + (hash2(c + 9, r + 1) - 0.5) * 5;
    d2.globalAlpha = 0.55 + hash2(c, r + 41) * 0.4;
    d2.fillRect(tx - ox, ty - oyDoc, 1.4, 1.4);
  }
  d2.globalAlpha = 1;
}
function drawFrame(t){
  if (follow) {
    sc = window.scrollY;
    mT = Math.min(1, Math.max(0, sc / (Math.max(1, waveH) * 0.25)));
  }
  var Hw = follow ? waveH : H;
  var yOff = follow ? waveTopDoc - sc : 0;
  ctx.clearRect(0, 0, W, H);
  var ROWS = 92, COLS = Math.min(340, Math.max(150, Math.floor(W / 4)));
  var horizon = Hw * 0.24, spread = Hw * 0.68, maxH = Hw * 0.78;
  var stepX = W / COLS;
  ggx += (gmx - ggx) * 0.07; ggy += (gmy - ggy) * 0.07;
  gAmp += (((gmx > -1e3 && gmy > -1e3) ? 1 : 0) - gAmp) * 0.045;
  var gR2 = Math.pow(Math.min(W, Hw) * 0.16, 2);
  var pin = window.__morphPin;
  if (typeof pin === "number") { mT = pin; mE = pin; }
  mE += (mT - mE) * 0.09;
  var g = falconGeom();
  var morph = falconPts && mE > 0.004 && W >= 640 && g.avail >= 220;
  var k = morph ? (mE < 0.5 ? 2 * mE * mE : 1 - Math.pow(-2 * mE + 2, 2) / 2) : 0;
  var followCY = H * 0.45;
  var dockCY = follow && dockBotDoc > 0 ? dockBotDoc - sc - g.fH / 2 : 1e9;
  var fCY = Math.min(followCY, dockCY);
  var fY = fCY - g.fH / 2;
  var docked = !!(morph && mE > 0.97 && dockCY < followCY && dcv);
  if (docked !== dockShown) {
    dockShown = docked;
    if (docked) drawDockFalcon();
    if (dcv) dcv.style.opacity = docked ? "1" : "0";
  }
  if (docked) { ctx.clearRect(0, 0, W, H); return; }
  if (follow && waveTopDoc + Hw - sc < -60 && fY + g.fH < -60) return;
  var nPts = falconPts ? falconPts.length / 2 : 0;
  for (var r = 0; r < ROWS; r++) {
    var d = r / (ROWS - 1);
    var base = horizon + d * spread;
    for (var c = 0; c <= COLS; c++) {
      var jx = (hash2(c, r) - 0.5) * stepX * 1.8;
      var jy = (hash2(c + 57, r + 91) - 0.5) * 7;
      var u = (c + jx / stepX) / COLS;
      var e1 = Math.exp(-Math.pow((u - 0.62) / 0.16, 2));
      var e2 = 0.55 * Math.exp(-Math.pow((u - 0.86) / 0.10, 2));
      var env = Math.min(1, e1 + e2);
      var h = fbm(u * 4.2, d * 3.1, t) * env;
      var hidden = h < 0.02;
      if (hidden && !morph) continue;
      var y = base - (hidden ? 0 : h * maxH * (0.35 + 0.65 * d)) + jy + yOff;
      var alpha = hidden ? 0 : Math.min(0.95, 0.05 + Math.pow(h, 1.25) * 1.5) * (0.3 + 0.7 * d);
      if (gAmp > 0.01 && !hidden) {
        var gpx = u * W - ggx, gpy = y - ggy;
        var gd2 = gpx * gpx + gpy * gpy;
        if (gd2 < gR2) {
          var gg = Math.exp((-gd2 / gR2) * 3.2) * gAmp;
          y -= gg * 26;
          alpha = Math.min(0.95, alpha + gg * 0.35);
        }
      }
      var x = u * W;
      if (k > 0) {
        var ti = (Math.floor(hash2(c + 13, r + 7) * nPts)) % nPts;
        var tx = g.fX + falconPts[ti * 2] * g.fW + (hash2(c + 3, r + 5) - 0.5) * 5;
        var ty = fY + falconPts[ti * 2 + 1] * g.fH + (hash2(c + 9, r + 1) - 0.5) * 5;
        x += (tx - x) * k; y += (ty - y) * k;
        alpha += ((0.55 + hash2(c, r + 41) * 0.4) - alpha) * k;
      }
      if (y < -8 || y > H + 8) continue;
      if (!hidden && h > 0.62 && ((c * 7 + r * 13) % 97) === 0) {
        ctx.fillStyle = sigColor; ctx.globalAlpha = 0.9; ctx.fillRect(x, y, 2, 2);
      } else {
        ctx.fillStyle = dotColor; ctx.globalAlpha = alpha; ctx.fillRect(x, y, 1.4, 1.4);
      }
    }
  }
  ctx.globalAlpha = 1;
}
function resizeTerrain(){
  if (!canvas) return;
  if (follow) canvas.classList.add("hero-canvas--follow");
  DPRc = Math.min(window.devicePixelRatio || 1, 2);
  W = canvas.clientWidth; H = canvas.clientHeight;
  canvas.width = W * DPRc; canvas.height = H * DPRc;
  ctx.setTransform(DPRc, 0, 0, DPRc, 0, 0);
  var scY = window.scrollY;
  var hero = document.querySelector(".hero");
  if (hero) {
    var hr = hero.getBoundingClientRect();
    waveTopDoc = hr.top + scY + hr.height * 0.18;
    waveH = hr.height * 0.82;
  } else { waveTopDoc = 0; waveH = H; }
  var crL = follow ? 0 : canvas.getBoundingClientRect().left;
  txr = 0; wsr = 0;
  function glyphRight(el){
    var rg = document.createRange();
    rg.selectNodeContents(el);
    var m = 0, rects = rg.getClientRects();
    for (var i = 0; i < rects.length; i++) m = Math.max(m, rects[i].right - crL);
    return m;
  }
  var pb = document.querySelector(".hero .promptbox");
  if (pb) txr = Math.max(txr, pb.getBoundingClientRect().right - crL);
  document.querySelectorAll(".hero .hero-sub, .hero .trust").forEach(function(el){
    txr = Math.max(txr, glyphRight(el));
  });
  if (dockSec) wsr = glyphRight(dockSec);
  measureDock();
  heroScroll();
  if (dockShown) drawDockFalcon(); // resize invalidates the docked bird's geometry
  drawFrame(last / 1000);
}
function loop(ts){
  requestAnimationFrame(loop);
  if (document.hidden) return;
  if (!heroVisible && !dockShown && mE < 0.01) { /* resting + offscreen */ }
  if (mE < 0.01 && ts - last < 33) return;
  last = ts;
  drawFrame(ts / 1000);
}
if (follow) window.addEventListener("scroll", heroScroll, { passive: true });

/* ---------- STILL MASSIF ---------- */
function drawMassifStill(cv){
  var w = cv.clientWidth, h = cv.clientHeight;
  if (!w || !h) return;
  var dpr = Math.min(window.devicePixelRatio || 1, 2);
  cv.width = w * dpr; cv.height = h * dpr;
  var c2 = cv.getContext("2d");
  c2.setTransform(dpr, 0, 0, dpr, 0, 0);
  var fg = cssVar("--fg"), sg = cssVar("--signal");
  var ROWS = 88, COLS = Math.max(140, Math.floor(w / 4));
  var horizon = h * 0.18, spread = h * 0.74, maxH = h * 0.95, stepX = w / COLS;
  var SEED = parseFloat(cv.getAttribute("data-seed")) || 7.31;
  var twin = cv.getAttribute("data-variant") === "twin";
  for (var r = 0; r < ROWS; r++) {
    var d = r / (ROWS - 1);
    var base = horizon + d * spread;
    for (var q = 0; q <= COLS; q++) {
      var jx = (hash2(q, r) - 0.5) * stepX * 1.8;
      var jy = (hash2(q + 57, r + 91) - 0.5) * 6;
      var u = (q + jx / stepX) / COLS;
      var env = twin
        ? Math.min(1, 0.92 * Math.exp(-Math.pow((u - 0.38) / 0.15, 2))
                    + 0.85 * Math.exp(-Math.pow((u - 0.66) / 0.12, 2)))
        : Math.min(1, Math.exp(-Math.pow((u - 0.5) / 0.13, 2))
                    + 0.4 * Math.exp(-Math.pow((u - 0.31) / 0.09, 2))
                    + 0.34 * Math.exp(-Math.pow((u - 0.69) / 0.08, 2)));
      var hh = Math.pow(fbm(u * 6.5, d * 4.2, SEED), 1.15) * env;
      if (hh < 0.02) continue;
      var y = base - hh * maxH * (0.3 + 0.7 * d) + jy;
      if (y < 0) continue;
      var alpha = Math.min(0.95, 0.04 + Math.pow(hh, 1.3) * 1.7) * (0.28 + 0.72 * d);
      if (hh > 0.6 && ((q * 7 + r * 13) % 131) === 0) {
        c2.fillStyle = sg; c2.globalAlpha = 0.85; c2.fillRect(u * w, y, 2, 2);
      } else {
        c2.fillStyle = fg; c2.globalAlpha = alpha; c2.fillRect(u * w, y, 1.3, 1.3);
      }
    }
  }
  c2.globalAlpha = 1;
}

/* ---------- BAYER DITHER ---------- */
var BAYER = [[0,8,2,10],[12,4,14,6],[3,11,1,9],[15,7,13,5]];
function bayer(x, y){ return (BAYER[y % 4][x % 4] + 0.5) / 16; }
function ditherField(cv, field, cell){
  var w = cv.clientWidth, h = cv.clientHeight;
  if (!w || !h) return;
  cv.width = w; cv.height = h;
  var c2 = cv.getContext("2d");
  c2.clearRect(0, 0, w, h);
  c2.fillStyle = cssVar("--fg");
  for (var y = 0; y < h; y += cell) {
    for (var x = 0; x < w; x += cell) {
      var b = field(x / w, y / h);
      if (b > bayer((x / cell) | 0, (y / cell) | 0)) c2.fillRect(x, y, cell - 1, cell - 1);
    }
  }
}
function sphereField(u, v){
  var dx = u - 0.5, dy = v - 0.5;
  var r = Math.sqrt(dx * dx + dy * dy);
  if (r > 0.47) return 0;
  var z = Math.sqrt(Math.max(0, 0.47 * 0.47 - r * r)) / 0.47;
  var lum = Math.max(0, (-dx / 0.47) * -0.45 + (-dy / 0.47) * 0.55 + z * 0.72);
  return Math.pow(lum, 1.6) * 0.95;
}
function diamondField(u, v){
  /* brilliant-cut rhombus, same dithered register as the old dome */
  var dx = (u - 0.5) * 2.15, dy = (v - 0.5) * 1.95;
  var m = Math.abs(dx) + Math.abs(dy);
  if (m > 0.95) return 0;
  var b = 0.10 + 0.45 * Math.pow(m / 0.95, 3.0);          /* rim glow */
  b += 0.38 * Math.exp(-Math.pow((Math.abs(dx) - Math.abs(dy)) / 0.05, 2)); /* facet ridges */
  b += 0.30 * Math.exp(-Math.pow(dx / 0.035, 2));          /* vertical axis */
  b += 0.26 * Math.exp(-Math.pow((dy + 0.12) / 0.035, 2)); /* girdle line */
  b += 0.45 * Math.exp(-Math.pow((dx + 0.18) / 0.10, 2)) * Math.exp(-Math.pow((dy + 0.30) / 0.10, 2)); /* crown highlight */
  b *= dy < -0.12 ? 1.18 : 0.82;                           /* crown brighter than pavilion */
  return Math.min(1, b);
}
/* dithered image (e.g. falcon logo) */
var imgCache = {};
function imageField(src){
  var rec = imgCache[src];
  if (!rec) return function(){ return 0; };
  return function(u, v){
    var x = Math.min(rec.w - 1, (u * rec.w) | 0);
    var y = Math.min(rec.h - 1, (v * rec.h) | 0);
    var a = rec.d.data[(y * rec.w + x) * 4 + 3] / 255;
    return a * (0.85 - v * 0.25);
  };
}
function loadArtImage(src){
  if (imgCache[src]) return;
  imgCache[src] = null;
  var img = new Image();
  img.onload = function(){
    var off = document.createElement("canvas");
    off.width = 128; off.height = Math.round(128 * img.height / img.width);
    off.getContext("2d").drawImage(img, 0, 0, off.width, off.height);
    imgCache[src] = { d: off.getContext("2d").getImageData(0, 0, off.width, off.height), w: off.width, h: off.height };
    renderArts();
  };
  img.src = src;
}

/* ---------- DATA OBSERVATORY KIT (ported from Talon landing) ---------- */
function prepArt(cv){
  var w = cv.clientWidth, h = cv.clientHeight; if (!w || !h) return null;
  var dpr = Math.min(window.devicePixelRatio || 1, 2);
  cv.width = w * dpr; cv.height = h * dpr;
  var c2 = cv.getContext("2d"); if (!c2) return null;
  c2.setTransform(dpr, 0, 0, dpr, 0, 0);
  c2.clearRect(0, 0, w, h);
  return { c2: c2, w: w, h: h, fg: cssVar("--fg"), sg: cssVar("--signal") };
}
function dottedVGrid(c2, w, h, step, fg){
  c2.fillStyle = fg;
  for (var x = step / 2; x < w; x += step) {
    c2.globalAlpha = 0.10;
    for (var y = 6; y < h - 6; y += 7) c2.fillRect(x, y, 1, 1);
    c2.globalAlpha = 0.5; c2.fillRect(x - 1, 2, 2, 2); c2.fillRect(x - 1, h - 4, 2, 2);
  }
  c2.globalAlpha = 1;
}
function drawGhostChart(cv){
  var p = prepArt(cv); if (!p) return;
  var c2 = p.c2, w = p.w, h = p.h, fg = p.fg, sg = p.sg;
  dottedVGrid(c2, w, h, w / 9, fg);
  var SEED = 4.7, COLSN = Math.max(200, Math.floor(w / 1.6));
  var curves = [{off:0,amp:1,a:1},{off:0.13,amp:0.82,a:0.55},{off:0.24,amp:0.66,a:0.3}];
  function line(u, off, amp){
    return 0.72 + off * 0.85 - amp * (0.40 * u + 0.13 * Math.sin(u * 5.2 + 1.2) * (1 - u * 0.4) + 0.09 * fbm(u * 3.2, off * 9, SEED));
  }
  curves.forEach(function(cu, ci){
    for (var q = 0; q <= COLSN; q++) {
      var u = q / COLSN, cy = line(u, cu.off, cu.amp);
      var thick = 0.015 + 0.05 * fbm(u * 4.6, ci * 3.1, SEED + 2) + 0.02 * (1 - u);
      for (var s = 0; s < 22; s++) {
        var g = (hash2(q + ci * 971, s) + hash2(q + 13, s + ci * 57) - 1);
        var v = cy + g * thick * 2.2;
        var a = Math.max(0, 0.65 - Math.abs(g) * 0.9) * cu.a * (0.4 + 0.6 * hash2(q, s + 91));
        if (a < 0.03) continue;
        c2.fillStyle = fg; c2.globalAlpha = Math.min(0.85, a);
        var big = hash2(q + s, 7) > 0.93;
        c2.fillRect(u * w, v * h, big ? 1.8 : 1.15, big ? 1.8 : 1.15);
      }
      if (ci === 0 && (q * 7) % 149 === 0) { c2.fillStyle = sg; c2.globalAlpha = 0.85; c2.fillRect(u * w, cy * h, 2, 2); }
    }
  });
  var au = 0.66, ax = au * w, ay = (line(au, 0, 1) - 0.16) * h;
  c2.globalAlpha = 0.2; c2.fillStyle = sg; c2.beginPath(); c2.arc(ax, ay, 8, 0, 7); c2.fill();
  c2.globalAlpha = 0.95; c2.fillRect(ax - 1.5, ay - 1.5, 3, 3);
  c2.globalAlpha = 0.3;
  for (var yy = ay + 7; yy < line(au, 0, 1) * h; yy += 5) c2.fillRect(ax - 0.5, yy, 1, 2);
  c2.globalAlpha = 1;
}
function drawGlitchSphere(cv){
  var w = cv.clientWidth, h = cv.clientHeight; if (!w || !h) return;
  ditherField(cv, sphereField, 3);
  var c2 = cv.getContext("2d"); if (!c2) return;
  var fg = cssVar("--fg"), sg = cssVar("--signal");
  for (var i = 0; i < 3; i++) {
    var sy = Math.round(h * (0.3 + 0.4 * hash2(i, 3)));
    var dx = Math.round((hash2(i, 9) > 0.5 ? 1 : -1) * (6 + hash2(i, 9) * 8));
    var band = c2.getImageData(0, sy, cv.width, 2);
    c2.clearRect(0, sy, cv.width, 2);
    c2.putImageData(band, dx, sy);
  }
  c2.strokeStyle = fg; c2.globalAlpha = 0.45; c2.lineWidth = 1;
  c2.setLineDash([2, 4]);
  c2.beginPath(); c2.ellipse(w / 2, h / 2, w * 0.60, h * 0.20, -0.35, 0, 7); c2.stroke();
  c2.setLineDash([]);
  c2.fillStyle = sg; c2.globalAlpha = 0.95;
  c2.beginPath(); c2.arc(w / 2 + w * 0.5, h / 2 - h * 0.155, 2.4, 0, 7); c2.fill();
  c2.globalAlpha = 1;
}
function drawAscent(cv){
  var p = prepArt(cv); if (!p) return;
  var c2 = p.c2, w = p.w, h = p.h, fg = p.fg, sg = p.sg;
  var SEED = 8.13, ROWS = 88, COLSN = Math.max(150, Math.floor(w / 2.6));
  var horizon = h * 0.16, spread = h * 0.76, maxHh = h * 0.95, stepX = w / COLSN;
  function env(u){ return Math.min(1, Math.exp(-Math.pow((u - 0.68) / 0.22, 2)) + 0.38 * Math.exp(-Math.pow((u - 0.3) / 0.13, 2))); }
  for (var r = 0; r < ROWS; r++) {
    var d = r / (ROWS - 1), base = horizon + d * spread;
    for (var q = 0; q <= COLSN; q++) {
      var jx = (hash2(q, r) - 0.5) * stepX * 1.8;
      var jy = (hash2(q + 57, r + 91) - 0.5) * 6;
      var u = (q + jx / stepX) / COLSN;
      var hh = Math.pow(fbm(u * 6.5, d * 4.2, SEED), 1.15) * env(u);
      if (hh < 0.02) continue;
      var y = base - hh * maxHh * (0.3 + 0.7 * d) + jy;
      if (y < 0) continue;
      var alpha = Math.min(0.95, 0.04 + Math.pow(hh, 1.3) * 1.7) * (0.28 + 0.72 * d);
      if (hh > 0.6 && ((q * 7 + r * 13) % 131) === 0) { c2.fillStyle = sg; c2.globalAlpha = 0.85; c2.fillRect(u * w, y, 2, 2); }
      else { c2.fillStyle = fg; c2.globalAlpha = alpha; c2.fillRect(u * w, y, 1.3, 1.3); }
    }
  }
  function route(u){ return h * (0.84 - 0.58 * Math.exp(-Math.pow((u - 0.68) / 0.30, 2)) + 0.015 * Math.sin(u * 21)); }
  for (var uu = 0.08; uu < 0.72; uu += 0.008) { c2.fillStyle = fg; c2.globalAlpha = 0.6; c2.fillRect(uu * w, route(uu), 1.6, 1.6); }
  [0.12, 0.42, 0.68].forEach(function(u, i, arr){
    var y = route(u), last = i === arr.length - 1;
    c2.fillStyle = last ? sg : fg; c2.globalAlpha = last ? 0.95 : 0.85;
    c2.fillRect(u * w - 2.5, y - 2.5, 5, 5);
    c2.globalAlpha = 0.35; c2.fillStyle = fg; c2.fillRect(u * w - 0.5, y + 6, 1, 9);
  });
  c2.globalAlpha = 1;
}
function drawCurveFam(cv){
  var p = prepArt(cv); if (!p) return;
  var c2 = p.c2, w = p.w, h = p.h, fg = p.fg, sg = p.sg;
  dottedVGrid(c2, w, h, w / 8, fg);
  var SEED = 6.02, COLSN = Math.max(200, Math.floor(w / 1.6));
  var fam = [{k:2.3,a:1},{k:1.7,a:0.5},{k:1.15,a:0.28}];
  fam.forEach(function(cu, ci){
    function f(u){ return 0.82 - (0.68 - ci * 0.06) * (Math.exp(u * cu.k) - 1) / (Math.exp(cu.k) - 1); }
    for (var q = 0; q <= COLSN; q++) {
      var u = q / COLSN, cy = f(u);
      var thick = 0.014 + 0.055 * fbm(u * 4.2, ci * 2.7, SEED) + 0.03 * (1 - u);
      for (var s = 0; s < 20; s++) {
        var g = (hash2(q + ci * 733, s) + hash2(q + 29, s + ci * 41) - 1);
        var v = cy + g * thick * 2.2;
        var a = Math.max(0, 0.62 - Math.abs(g) * 0.85) * cu.a * (0.4 + 0.6 * hash2(q, s + 63));
        if (a < 0.03) continue;
        c2.fillStyle = fg; c2.globalAlpha = Math.min(0.85, a);
        var big = hash2(q + s, 11) > 0.93;
        c2.fillRect(u * w, v * h, big ? 1.8 : 1.15, big ? 1.8 : 1.15);
      }
      if (ci === 0 && (q * 11) % 163 === 0) { c2.fillStyle = sg; c2.globalAlpha = 0.85; c2.fillRect(u * w, cy * h, 2, 2); }
    }
  });
  function f0(u){ return 0.82 - 0.68 * (Math.exp(u * 2.3) - 1) / (Math.exp(2.3) - 1); }
  var ex = 0.95 * w, ey = f0(0.95) * h;
  var grad = c2.createRadialGradient(ex, ey, 0, ex, ey, 16);
  grad.addColorStop(0, sg); grad.addColorStop(1, "rgba(0,0,0,0)");
  c2.globalAlpha = 0.55; c2.fillStyle = grad;
  c2.beginPath(); c2.arc(ex, ey, 16, 0, 7); c2.fill();
  c2.globalAlpha = 1; c2.fillStyle = sg; c2.fillRect(ex - 1.5, ey - 1.5, 3, 3);
}
function drawEclipse(cv){
  var p = prepArt(cv); if (!p) return;
  var c2 = p.c2, w = p.w, h = p.h, fg = p.fg, sg = p.sg;
  var cxp = w / 2, cyp = h * 0.5, R = Math.min(w, h) * 0.16, SEED = 2.44;
  function dot(x, y, a, i){
    if (a < 0.03 || y < 0 || y > h) return;
    c2.fillStyle = hash2(i, 73) > 0.995 ? sg : fg;
    c2.globalAlpha = Math.min(0.85, a);
    var big = hash2(i, 77) > 0.92;
    c2.fillRect(x, y, big ? 1.8 : 1.15, big ? 1.8 : 1.15);
  }
  var ROWS = 46, COLSN = Math.max(200, Math.floor(w / 2.4)), stepX = w / COLSN;
  for (var r = 0; r < ROWS; r++) {
    var d = r / (ROWS - 1);
    for (var q = 0; q <= COLSN; q++) {
      var jx = (hash2(q, r) - 0.5) * stepX * 1.8;
      var u = (q + jx / stepX) / COLSN;
      var cx = Math.abs(u - 0.5);
      var env = Math.min(1, Math.exp(-Math.pow(cx / 0.34, 2)) + 0.3);
      var hh = Math.pow(fbm(u * 5.8, d * 3.6, SEED), 1.2) * env;
      if (hh < 0.02) continue;
      var reach = h * (0.34 - 0.16 * Math.exp(-Math.pow(cx / 0.2, 2)));
      var y = d * reach * (0.35 + 0.65 * hh) + (hash2(q + 57, r + 91) - 0.5) * 5;
      dot(u * w, y, (0.05 + Math.pow(hh, 1.25) * 1.3) * (0.3 + 0.7 * d), q * 31 + r);
    }
  }
  for (r = 0; r < ROWS + 14; r++) {
    d = r / (ROWS + 13);
    for (q = 0; q <= COLSN; q++) {
      jx = (hash2(q + 401, r) - 0.5) * stepX * 1.8;
      u = (q + jx / stepX) / COLSN;
      cx = Math.abs(u - 0.5);
      env = Math.min(1, 0.5 + Math.exp(-Math.pow(cx / 0.3, 2)));
      hh = Math.pow(fbm(u * 6.2, d * 4.0, SEED + 5), 1.15) * env;
      if (hh < 0.02) continue;
      var base = h * (0.62 + d * 0.36);
      var rise = h * (0.2 + 0.1 * Math.exp(-Math.pow(cx / 0.22, 2)));
      y = base - hh * rise + (hash2(q + 91, r + 17) - 0.5) * 5;
      if (Math.sqrt((u * w - cxp) * (u * w - cxp) + (y - cyp) * (y - cyp)) < R * 1.5) continue;
      dot(u * w, y, (0.05 + Math.pow(hh, 1.25) * 1.5) * (0.3 + 0.7 * d), q * 17 + r * 3);
    }
  }
  for (var i = 0; i < 700; i++) {
    var x = hash2(i, 61) * w, yv = hash2(i, 67) * h;
    if (Math.sqrt((x - cxp) * (x - cxp) + (yv - cyp) * (yv - cyp)) < R * 1.6) continue;
    dot(x, yv, 0.05 + 0.3 * Math.pow(hash2(i, 71), 3), i);
  }
  c2.globalAlpha = 1; c2.strokeStyle = fg; c2.lineWidth = 2;
  c2.shadowColor = fg; c2.shadowBlur = 16;
  c2.beginPath(); c2.arc(cxp, cyp, R, 0, 7); c2.stroke();
  c2.shadowBlur = 30; c2.globalAlpha = 0.5;
  c2.beginPath(); c2.arc(cxp, cyp, R, 0, 7); c2.stroke();
  c2.shadowBlur = 0; c2.globalAlpha = 1;
}


function drawStarfield(cv){
  var p = prepArt(cv); if (!p) return;
  var c2 = p.c2, w = p.w, h = p.h, fg = p.fg, sg = p.sg;
  var SEED = 3.91, step = 3;
  for (var y = 0; y < h; y += step) for (var x = 0; x < w; x += step) {
    var u = x / w, v = y / h;
    var n = Math.pow(fbm(u * 3.4, v * 3.4, SEED), 2.6);
    var a = n * 0.55 * (0.4 + 0.6 * hash2(x, y));
    if (a < 0.03) continue;
    var jx = (hash2(x + 7, y) - 0.5) * step * 1.6;
    var jy = (hash2(x, y + 7) - 0.5) * step * 1.6;
    c2.fillStyle = fg; c2.globalAlpha = Math.min(0.6, a);
    c2.fillRect(x + jx, y + jy, 1.2, 1.2);
  }
  for (var i = 0; i < 130; i++) {
    var sx = hash2(i, 51) * w, sy = hash2(i, 53) * h;
    var s = hash2(i, 57);
    c2.fillStyle = s > 0.96 ? sg : fg;
    c2.globalAlpha = 0.25 + 0.65 * s * s;
    var sz = s > 0.88 ? 2.2 : 1.4;
    c2.fillRect(sx, sy, sz, sz);
    if (s > 0.93) {
      c2.globalAlpha = 0.25;
      c2.fillRect(sx - 3, sy + 0.5, 8, 1); c2.fillRect(sx + 0.5, sy - 3, 1, 8);
    }
  }
  var cons = [[0.26,0.22],[0.42,0.3],[0.52,0.17],[0.68,0.26],[0.78,0.4]];
  c2.strokeStyle = fg; c2.globalAlpha = 0.2; c2.lineWidth = 1;
  c2.beginPath();
  cons.forEach(function(uv, i){ if (i === 0) c2.moveTo(uv[0]*w, uv[1]*h); else c2.lineTo(uv[0]*w, uv[1]*h); });
  c2.stroke();
  cons.forEach(function(uv, i){
    c2.globalAlpha = 0.75; c2.fillStyle = i === 2 ? sg : fg;
    c2.fillRect(uv[0]*w - 1.2, uv[1]*h - 1.2, 2.4, 2.4);
  });
  c2.globalAlpha = 1;
}

function renderArts(){
  document.querySelectorAll("[data-art]").forEach(function(cv){
    var kind = cv.getAttribute("data-art");
    if (kind === "sphere") ditherField(cv, sphereField, 3);
    else if (kind === "glitchsphere") drawGlitchSphere(cv);
    else if (kind === "ghostchart") drawGhostChart(cv);
    else if (kind === "ascent") drawAscent(cv);
    else if (kind === "curve") drawCurveFam(cv);
    else if (kind === "eclipse") drawEclipse(cv);
    else if (kind === "starfield") drawStarfield(cv);
    else if (kind === "dome" || kind === "diamond") ditherField(cv, diamondField, 2);
    else if (kind === "massif") drawMassifStill(cv);
    else if (kind === "image") {
      var src = cv.getAttribute("data-src");
      if (imgCache[src]) ditherField(cv, imageField(src), 3);
      else loadArtImage(src);
    }
  });
}

/* ---------- COLOR SYNC ---------- */
function syncColors(){
  dotColor = cssVar("--fg") || dotColor;
  sigColor = cssVar("--signal") || sigColor;
  if (canvas) drawFrame(last / 1000);
  renderArts();
}

/* ---------- INIT ---------- */
window.addEventListener("resize", function(){ resizeTerrain(); renderArts(); });
resizeTerrain();
syncColors();
if (canvas && !reduceMotion) {
  new IntersectionObserver(function(en){ heroVisible = en[0].isIntersecting; })
    .observe(canvas.closest("section") || canvas);
  requestAnimationFrame(loop);
}

/* ---------- SCROLL REVEALS ---------- */
var io = new IntersectionObserver(function(entries){
  entries.forEach(function(en){
    if (en.isIntersecting) { en.target.classList.add("on"); io.unobserve(en.target); }
  });
}, { threshold: 0.15 });
document.querySelectorAll(".rv").forEach(function(el){ io.observe(el); });

/* ---------- DEMO HELPERS (typing + count-up), exported ---------- */
window.DUSK = {
  reduceMotion: reduceMotion,
  renderArts: renderArts,
  sync: syncColors,
  typeInto: function(el, text, speed, done, runRef){
    var i = 0, run = runRef.v;
    el.innerHTML = '<span class="caret"></span>';
    (function tick(){
      if (run !== runRef.v) return;
      if (i <= text.length) {
        el.innerHTML = text.slice(0, i).replace(/\n/g, "<br>") + '<span class="caret"></span>';
        i++;
        runRef.t = setTimeout(tick, speed);
      } else {
        el.innerHTML = text.replace(/\n/g, "<br>");
        done && done();
      }
    })();
  },
  countUp: function(container, runRef){
    container.classList.add("show");
    container.querySelectorAll("b[data-count]").forEach(function(b){
      var target = parseInt(b.getAttribute("data-count"), 10);
      var prefix = b.getAttribute("data-prefix") || "";
      var suffix = b.getAttribute("data-suffix") || "";
      if (reduceMotion) { b.textContent = prefix + target.toLocaleString() + suffix; return; }
      var t0 = performance.now(), run = runRef.v;
      (function step(now){
        if (run !== runRef.v) return;
        var p = Math.min(1, (now - t0) / 900);
        var eased = 1 - Math.pow(1 - p, 3);
        b.textContent = prefix + Math.round(target * eased).toLocaleString() + suffix;
        if (p < 1) requestAnimationFrame(step);
      })(t0);
    });
  },

  /* Auto-advancing scripted product demo ("demo video").
     cfg: { root, tabsEl, qEl, codeEl, tilesEl, capEl, progEl, scenarios }
     scenario: { label, cap, q, code, tiles:[{v,prefix,suffix,label,neg,static}] } */
  demoRunner: function(cfg){
    var D = window.DUSK;
    var runRef = { v: 0, t: null };
    var active = 0, HOLD = 4600;
    cfg.scenarios.forEach(function(sc, i){
      var b = document.createElement("button");
      b.type = "button";
      b.className = "demo-tab";
      b.textContent = "0" + (i + 1) + " " + sc.label;
      b.addEventListener("click", function(){ play(i); });
      cfg.tabsEl.appendChild(b);
    });
    var tabs = cfg.tabsEl.querySelectorAll(".demo-tab");
    function renderTiles(sc){
      cfg.tilesEl.classList.remove("show");
      cfg.tilesEl.innerHTML = sc.tiles.map(function(t){
        var b = t.static !== undefined
          ? "<b>" + t.static + "</b>"
          : '<b data-count="' + t.v + '"' +
            (t.prefix ? ' data-prefix="' + t.prefix + '"' : "") +
            (t.suffix ? ' data-suffix="' + t.suffix + '"' : "") + ">0</b>";
        return '<div class="tile' + (t.neg ? " neg" : "") + '">' + b + "<s>" + t.label + "</s></div>";
      }).join("");
    }
    function setProg(ms){
      if (!cfg.progEl) return;
      var bar = cfg.progEl.firstElementChild;
      bar.classList.remove("run");
      bar.style.transitionDuration = "0ms";
      bar.offsetWidth; /* reflow — restart the bar */
      if (ms) { bar.style.transitionDuration = ms + "ms"; bar.classList.add("run"); }
    }
    function setIns(sc, show){
      if (!cfg.insEl) return;
      cfg.insEl.innerHTML = sc.ins ? "<span>" + sc.ins + "</span>" : "";
      cfg.insEl.classList.toggle("show", !!show && !!sc.ins);
    }
    function play(i){
      runRef.v++;
      clearTimeout(runRef.t);
      active = i;
      var sc = cfg.scenarios[i];
      tabs.forEach(function(tb, j){ tb.classList.toggle("on", j === i); });
      if (cfg.capEl) cfg.capEl.innerHTML = "FEATURE 0" + (i + 1) + " — <b>" + sc.cap + "</b>";
      renderTiles(sc);
      setIns(sc, false);
      cfg.qEl.textContent = ""; cfg.codeEl.textContent = "";
      if (D.reduceMotion) {
        cfg.qEl.textContent = sc.q;
        cfg.codeEl.innerHTML = sc.code.replace(/\n/g, "<br>");
        D.countUp(cfg.tilesEl, runRef);
        setIns(sc, true);
        setProg(0);
        return;
      }
      var total = sc.q.length * 30 + 420 + sc.code.length * 8 + 300 + 900 + HOLD;
      setProg(total);
      D.typeInto(cfg.qEl, sc.q, 30, function(){
        runRef.t = setTimeout(function(){
          D.typeInto(cfg.codeEl, sc.code, 8, function(){
            runRef.t = setTimeout(function(){
              D.countUp(cfg.tilesEl, runRef);
              (function(){ var v = runRef.v; setTimeout(function(){ if (v === runRef.v) setIns(sc, true); }, 500); })();
              runRef.t = setTimeout(function(){
                play((active + 1) % cfg.scenarios.length);
              }, HOLD);
            }, 300);
          }, runRef);
        }, 420);
      }, runRef);
    }
    /* static first frame — demo never renders blank (hidden tabs, no-IO, reduced motion) */
    (function primeScene(){
      var sc = cfg.scenarios[0];
      tabs[0].classList.add("on");
      if (cfg.capEl) cfg.capEl.innerHTML = "FEATURE 01 — <b>" + sc.cap + "</b>";
      renderTiles(sc);
      cfg.qEl.textContent = sc.q;
      cfg.codeEl.innerHTML = sc.code.replace(/\n/g, "<br>");
      cfg.tilesEl.classList.add("show");
      cfg.tilesEl.querySelectorAll("b[data-count]").forEach(function(b){
        b.textContent = (b.getAttribute("data-prefix") || "")
          + parseInt(b.getAttribute("data-count"), 10).toLocaleString()
          + (b.getAttribute("data-suffix") || "");
      });
      setIns(sc, true);
    })();
    new IntersectionObserver(function(en){
      if (en[0].isIntersecting) play(active);
      else { runRef.v++; clearTimeout(runRef.t); setProg(0); }
    }, { threshold: 0.3 }).observe(cfg.root);
    return { play: play };
  }
};
})();
