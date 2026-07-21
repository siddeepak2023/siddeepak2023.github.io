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
function routeMassif(cv, SEED){
  var p = prepArt(cv); if (!p) return;
  var c2 = p.c2, w = p.w, h = p.h, fg = p.fg, sg = p.sg;
  var ROWS = 88, COLSN = Math.max(150, Math.floor(w / 2.6));
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
function drawGhostChart(cv){ routeMassif(cv, 5.62); }
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


/* ── ported from PF/CF static engine: sentinel face (bust) + ledger dense particle sphere (oort) ── */
var HEAD_V = [69, 277, 431, 88, 276, 425, 51, 282, 431, 50, 294, 429, 67, 300, 429, 85, 301, 422, 85, 317, 423, 95, 297, 408, 106, 294, 406, 95, 282, 411, 107, 283, 412, 52, 312, 426, 87, 312, 413, 127, 280, 412, 127, 295, 408, 148, 278, 412, 149, 297, 411, 148, 319, 417, 165, 317, 410, 108, 317, 411, 33, 296, 426, 33, 282, 429, 73, 260, 434, 89, 262, 430, 152, 262, 414, 106, 261, 426, 114, 269, 420, 126, 259, 423, 52, 261, 432, 12, 297, 424, 13, 278, 426, 30, 260, 432, 101, 243, 433, 14, 317, 423, 58, 241, 437, 80, 236, 437, 144, 243, 422, 164, 242, 411, 125, 237, 430, 103, 224, 436, 33, 238, 433, 126, 222, 431, 145, 222, 424, 164, 221, 414, -12, 183, 412, 4, 201, 422, 28, 179, 429, 66, 183, 435, 55, 160, 428, 92, 168, 432, 86, 151, 429, -5, 317, 414, -9, 294, 412, 0, 289, 419, 14, 257, 426, -10, 275, 411, -40, 233, 386, -31, 240, 394, -12, 224, 411, -26, 211, 402, -44, 195, 389, 14, 213, 427, -53, 256, 371, -49, 227, 379, -92, 58, 264, -53, 54, 322, -27, 28, 313, -74, 31, 247, 170, 354, 431, 147, 357, 433, 131, 358, 425, 149, 339, 424, 164, 337, 424, 159, 363, 438, 124, 354, 418, 115, 375, 417, 128, 375, 427, 146, 382, 430, 149, 388, 424, 165, 391, 426, 133, 387, 422, 172, 379, 436, 170, 394, 409, 146, 396, 415, 129, 395, 413, 166, 372, 439, 103, 375, 421, 87, 377, 425, 92, 397, 421, 167, 413, 409, 110, 432, 420, 93, 439, 408, 107, 449, 409, 112, 469, 409, 94, 447, 410, 168, 433, 411, 162, 427, 419, 107, 439, 411, 149, 414, 407, 148, 419, 422, 128, 434, 422, 127, 418, 421, 149, 431, 422, 148, 438, 404, 148, 449, 409, 163, 451, 404, 128, 473, 408, 133, 450, 410, 146, 473, 404, 131, 438, 409, 156, 436, 398, 103, 454, 394, 89, 450, 395, 86, 436, 398, 132, 415, 407, 110, 416, 402, 131, 472, 392, 111, 471, 389, 156, 412, 397, 151, 470, 396, 37, 508, 235, 56, 520, 233, 60, 505, 218, 99, 408, 399, 115, 397, 408, 82, 483, 309, 97, 486, 301, 109, 484, 374, 92, 479, 376, 92, 466, 387, 111, 491, 337, 84, 415, 395, 72, 436, 391, 149, 399, 395, 129, 488, 366, 146, 380, 374, 150, 478, 377, 126, 379, 385, 77, 467, 380, 92, 392, 391, 104, 385, 389, 73, 476, 355, 89, 484, 348, 155, 452, 396, 71, 416, 391, 54, 408, 374, 50, 430, 372, 128, 361, 348, 123, 366, 351, 112, 360, 357, 102, 361, 362, 61, 468, 362, 75, 378, 373, 88, 368, 367, 67, 480, 318, 50, 474, 326, 26, 510, 259, 4, 510, 270, 29, 393, 347, 23, 424, 345, 92, 353, 316, 103, 354, 310, 81, 349, 322, 67, 353, 329, 38, 466, 336, 51, 362, 339, 40, 371, 344, 72, 359, 276, 57, 355, 282, 47, 358, 288, 14, 463, 299, 16, 486, 292, 25, 370, 300, 17, 377, 304, 34, 363, 295, 24, 494, 233, 14, 485, 240, 161, 471, 405, 1, 404, 308, 2, 424, 305, 14, 388, 260, -2, 399, 266, 33, 381, 252, 5, 472, 245, -13, 413, 269, -9, 432, 258, 55, 382, 241, 47, 380, 244, 4, 452, 247, 7, 439, 237, 15, 447, 232, 27, 449, 226, 51, 449, 212, 33, 452, 221, 11, 415, 236, -4, 423, 242, 0, 431, 241, 36, 411, 223, 51, 411, 213, 26, 413, 228, 117, 163, 426, 115, 146, 422, 148, 488, 394, 92, 470, 407, 126, 542, 360, 93, 536, 359, -5, 476, 349, -22, 458, 345, -22, 466, 334, -43, 461, 273, -47, 448, 285, -61, 435, 271, -57, 452, 259, -41, 485, 256, -56, 474, 247, -54, 509, 238, -53, 553, 233, -71, 542, 216, -76, 586, 214, -90, 577, 203, -92, 620, 213, -111, 613, 195, -16, 486, 287, -17, 477, 305, -32, 463, 297, -29, 475, 281, -17, 502, 266, -29, 491, 260, -29, 527, 250, -28, 566, 246, -57, 594, 230, -73, 626, 226, -40, 438, 336, -40, 445, 323, 90, 535, 326, 118, 541, 331, 63, 529, 304, 96, 535, 296, 67, 536, 267, 62, 555, 241, 32, 584, 240, 28, 552, 251, 32, 619, 237, 58, 587, 232, 52, 621, 229, 2, 613, 246, 28, 653, 243, 62, 654, 221, 34, 530, 272, 7, 649, 252, -58, 632, 236, -30, 604, 245, -3, 541, 250, 8, 490, 350, 11, 485, 361, 10, 155, 419, -75, 440, 240, -75, 461, 230, -108, 564, 183, -126, 606, 183, -84, 532, 202, -75, 490, 219, -91, 431, 214, -90, 453, 206, -140, 593, 169, -88, 480, 197, -98, 520, 185, -103, 426, 184, -103, 446, 179, -123, 549, 156, -145, 584, 151, -105, 472, 173, -111, 510, 165, -111, 414, 157, -111, 440, 153, -124, 541, 135, -139, 575, 116, -112, 466, 148, -115, 500, 143, -107, 417, 133, -105, 435, 130, -112, 539, 112, -105, 461, 127, -107, 497, 122, -91, 412, 112, -88, 430, 112, -89, 536, 87, -115, 571, 85, -86, 455, 110, -88, 492, 105, -50, 395, 83, -41, 423, 84, -51, 423, 89, -53, 535, 72, -56, 572, 50, -83, 571, 67, -38, 450, 87, -48, 450, 91, -45, 487, 84, -131, 103, 138, -113, 60, 186, -90, 46, 163, -111, 93, 116, -66, 38, 144, -87, 87, 98, -146, 145, 194, -134, 112, 233, -125, 84, 207, -144, 120, 162, -137, 220, 261, -129, 206, 281, -134, 176, 269, -143, 195, 243, -134, 146, 252, -146, 170, 220, -65, 334, 353, -61, 310, 362, -71, 311, 349, -53, 285, 371, -68, 282, 354, -93, 124, 328, -115, 128, 292, -118, 163, 301, -101, 161, 327, -64, 358, 349, -118, 339, 205, -125, 316, 204, -130, 313, 179, -123, 338, 180, -106, 91, 280, -75, 86, 327, -131, 308, 151, -125, 335, 153, -18, 447, 360, -35, 428, 352, -75, 423, 250, -92, 411, 224, -105, 409, 188, -108, 395, 132, -93, 397, 112, 28, 350, 428, 31, 337, 428, 10, 335, 422, 32, 318, 427, 46, 330, 429, -58, 413, 333, -61, 416, 316, -48, 405, 349, 120, 533, 384, 89, 527, 381, 63, 507, 382, 61, 523, 356, 33, 506, 358, 23, 506, 332, 27, 513, 304, 40, 519, 305, 46, 519, 330, -9, 648, 254, 10, 579, 243, 63, 526, 329, 17, 478, 371, 0, 464, 368, 6, 501, 296, 8, 496, 319, -4, 487, 313, -7, 574, 246, -34, 640, 245, -5, 483, 328, -20, 471, 320, -52, 435, 299, -64, 414, 289, 8, 493, 335, -36, 454, 310, -74, 409, 264, -128, 241, 274, -121, 231, 290, -120, 259, 283, -114, 260, 296, -76, 360, 333, -75, 336, 339, -33, 31, 125, -59, 83, 79, -64, 384, 338, -75, 385, 323, -149, 214, 155, -150, 181, 170, -148, 164, 138, -146, 205, 125, -86, 397, 248, -60, 371, 82, -49, 371, 75, -83, 354, 91, -92, 333, 86, -73, 333, 72, -65, 354, 79, -102, 355, 108, -109, 334, 105, -97, 378, 111, -114, 391, 158, -109, 393, 187, -99, 393, 217, -72, 407, 99, -75, 392, 98, -139, 285, 177, -139, 279, 149, -86, 304, 64, -95, 270, 58, -74, 267, 43, -68, 302, 50, -136, 291, 203, -127, 305, 125, -136, 273, 122, -128, 271, 97, -119, 304, 101, -119, 320, 234, -131, 299, 226, -138, 269, 230, -139, 250, 239, -145, 239, 212, -143, 262, 204, -146, 246, 147, -143, 239, 119, -146, 254, 177, -149, 226, 183, -97, 230, 57, -99, 186, 62, -75, 183, 46, -75, 227, 43, -136, 197, 100, -133, 235, 96, -138, 153, 112, -139, 235, 248, -147, 216, 224, -150, 198, 198, -98, 141, 73, -71, 134, 56, 67, 28, 337, 90, 24, 323, 57, 2, 271, 27, 7, 288, -20, 65, 364, 10, 45, 359, 44, 139, 423, 32, 120, 413, -2, 138, 409, -31, 163, 399, 3, 13, 301, -49, 15, 230, -120, 333, 127, 135, 144, 413, 145, 160, 414, 148, 143, 407, 76, 129, 422, 65, 107, 412, -41, 151, 389, -13, 124, 399, -60, 180, 376, -105, 304, 80, -85, 164, 348, -71, 130, 356, -49, 94, 363, 21, 103, 404, -74, 397, 276, -83, 387, 263, -93, 386, 240, -122, 305, 244, -129, 276, 250, -129, 263, 258, -126, 258, 271, -54, 140, 376, -27, 109, 387, -72, 171, 365, -59, 332, 62, 5, 85, 391, 108, 75, 383, 126, 94, 389, 125, 73, 374, -68, 453, 99, -69, 490, 94, 11, 0, 197, -24, 6, 216, -107, 357, 230, -102, 350, 253, -108, 337, 251, -113, 340, 227, 43, 33, 349, 38, 69, 388, 72, 62, 379, 129, 122, 405, 147, 121, 394, -115, 270, 75, -69, 426, 100, -120, 145, 90, 95, 59, 369, 114, 55, 359, -117, 232, 74, -120, 190, 79, 53, 89, 402, 151, 537, 363, 126, 316, 410, -112, 322, 258, -117, 308, 259, -122, 280, 267, 98, 99, 403, 109, 125, 415, 86, 81, 392, 77, 298, 417, 99, 286, 393, 99, 291, 393, 97, 297, 387, 89, 265, 404, 68, 258, 412, 69, 319, 411, 50, 318, 409, 97, 277, 388, 87, 314, 388, 48, 259, 408, 85, 261, 389, 32, 278, 409, 31, 297, 407, 41, 269, 416, 68, 323, 388, 38, 311, 406, 69, 253, 391, 88, 296, 371, 88, 280, 371, 85, 310, 375, 38, 265, 405, 48, 320, 392, 51, 255, 390, 72, 314, 371, 84, 266, 375, 70, 261, 373, 36, 311, 390, 29, 296, 390, 30, 278, 389, 36, 264, 393, 52, 314, 373, 49, 263, 373, 70, 272, 363, 69, 302, 362, 36, 295, 372, 49, 273, 365, 39, 279, 370, 48, 300, 365, 38, 267, 379, 68, 294, 359, 53, 290, 360, 54, 285, 360, 70, 282, 359, -56, 379, 351, -87, 387, 305, -203, 662, 303, -264, 649, 354, -230, 667, 377, -177, 673, 326, -328, 668, 399, -291, 681, 420, -371, 715, 421, -346, 741, 440, -399, 749, 431, -425, 787, 437, -391, 796, 448, -447, 827, 442, -401, 840, 448, -472, 875, 455, -435, 880, 464, -196, 687, 389, -149, 685, 341, -247, 702, 420, -286, 734, 435, -318, 768, 438, -342, 794, 440, -363, 824, 440, -389, 875, 451, -391, 908, 451, -115, 731, 381, -164, 743, 409, -111, 779, 399, -61, 756, 366, -205, 742, 421, -162, 796, 416, -259, 783, 431, -212, 809, 420, -258, 832, 411, -303, 836, 414, -296, 869, 364, -319, 862, 394, -344, 873, 416, -331, 891, 351, -360, 920, 406, -355, 924, 362, 3, 780, 337, 47, 783, 308, -41, 788, 370, -207, 843, 410, -254, 881, 367, -306, 894, 321, -310, 904, 300, -350, 917, 318, 105, 785, 270, 73, 786, 290, -63, 829, 400, -15, 827, 375, -127, 832, 415, -173, 854, 416, -206, 874, 403, -228, 887, 381, -268, 918, 324, -349, 918, 272, -109, 856, 421, -171, 888, 413, -234, 922, 352, -199, 924, 382, -309, 930, 277, -31, 886, 408, 20, 879, 373, -94, 885, 427, -139, 908, 428, -167, 925, 410, -203, 972, 354, -247, 960, 323, -296, 966, 273, 111, 870, 297, 92, 876, 315, 108, 927, 325, -78, 917, 433, -118, 940, 436, -142, 961, 420, -163, 973, 393, 127, 970, 330, 113, 822, 278, 66, 832, 313, -83, 664, 249, -226, 656, 282, -290, 640, 326, -355, 663, 366, -401, 710, 391, -427, 745, 403, -449, 781, 411, -482, 826, 413, -497, 875, 433, 76, 883, 330, -35, 726, 338, 23, 740, 308, -77, 714, 344, 21, 696, 277, -36, 691, 279, -167, 659, 259, -119, 681, 281, -113, 652, 236, -129, 645, 229, -184, 606, 153, -178, 600, 127, -112, 596, 61, -85, 597, 49, -178, 618, 183, -60, 599, 42, -139, 596, 74, -147, 635, 220, -163, 626, 207, -160, 599, 95, -213, 638, 222, -223, 633, 203, -188, 624, 91, -160, 623, 68, -197, 645, 239, -211, 622, 127, -223, 628, 157, -225, 633, 181, -131, 623, 50, -98, 625, 38, -69, 627, 24, -110, 653, 27, -78, 654, 9, -257, 645, 158, -242, 644, 129, -214, 648, 94, -183, 649, 65, -249, 644, 264, -263, 642, 240, -150, 650, 41, -271, 640, 215, -268, 643, 187, -16, 706, 301, 67, 698, 238, 59, 718, 268, 76, 746, 274, 20, 831, 348, 56, 927, 366, -120, 698, 15, -84, 700, -5, -325, 637, 265, -324, 647, 229, -165, 694, 35, -207, 690, 65, -311, 656, 192, -289, 664, 159, -263, 669, 126, -241, 687, 92, -313, 635, 299, -301, 691, 141, -382, 660, 335, -394, 659, 294, -379, 667, 245, -352, 697, 182, -296, 725, 116, -296, 759, 103, -337, 757, 139, -254, 930, 82, -255, 967, 100, -301, 918, 103, -293, 794, 89, -340, 794, 140, -269, 882, 79, -311, 865, 111, -253, 748, 77, -213, 925, 66, -207, 961, 66, -256, 797, 58, -208, 877, 54, -205, 739, 62, -197, 788, 57, -185, 923, 42, -166, 958, 34, -169, 741, 28, -167, 775, 27, -155, 921, 23, -166, 807, 22, -159, 877, 21, -127, 744, 7, -129, 794, 3, -123, 917, 5, -120, 956, 8, -124, 875, 3, -88, 746, -14, -88, 793, -17, -81, 914, -14, -77, 955, -11, -84, 872, -16, -425, 707, 355, -430, 701, 309, -422, 698, 261, -394, 702, 213, -437, 835, 236, -406, 849, 214, -398, 881, 223, -451, 740, 368, -457, 734, 323, -326, 906, 145, -313, 939, 146, -326, 949, 185, -341, 915, 187, -451, 730, 277, -432, 750, 235, -412, 805, 213, -384, 835, 195, -350, 920, 227, -357, 874, 185, -466, 764, 377, -476, 761, 334, -326, 965, 222, -467, 757, 291, -395, 740, 196, -381, 784, 182, -486, 800, 374, -484, 789, 317, -459, 783, 264, -436, 791, 235, -458, 843, 254, -389, 917, 240, -503, 830, 354, -496, 832, 316, -469, 810, 274, -479, 836, 281, -481, 871, 280, -451, 879, 253, -374, 933, 265, -515, 873, 401, -523, 872, 362, -508, 860, 321, -508, 902, 301, -481, 913, 275, -404, 963, 277, -437, 930, 258, -373, 939, 298, -523, 895, 330, -431, 887, 246, -334, 868, 144, -446, 976, 275, -539, 920, 414, -547, 922, 378, -496, 925, 469, -445, 938, 478, -395, 976, 329, -488, 961, 280, -535, 935, 329, -522, 921, 444, -514, 950, 294, -388, 966, 423, -381, 969, 383, -572, 969, 411, -482, 970, 494, -454, 989, 488, -566, 967, 358, -555, 979, 330, -540, 955, 459, -541, 989, 309, -523, 976, 494, -567, 988, 462, -9, 973, 428, -46, 971, 444, -34, 996, 450, 24, 974, 411, 58, 976, 388, -76, 992, 451, -64, 944, 440, -98, 966, 447, -14, 926, 414, 22, 929, 394, 105, 988, 352, 153, 987, 321, -121, 997, 433, -49, 673, 259, 12, 522, 270, 33, 492, 373, 39, 488, 381, 72, 451, 384, 56, 449, 367, 31, 445, 341, 10, 442, 301, 9, 390, 306, 63, 394, 374, 153, 372, 439, -78, 695, 287, -117, 699, 345, -159, 708, 388, -243, 746, 431, -301, 803, 432, -330, 841, 423, -406, 955, 455, -22, 296, 402, -7, 257, 413, 9, 243, 424, 51, 220, 437, 78, 218, 438, 100, 202, 436, 124, 199, 432, 145, 197, 424, 163, 203, 414, -432, 1000, 469, -287, 831, 81, -317, 827, 116, -249, 830, 54, -195, 834, 48, -164, 835, 21, -126, 837, 1, -86, 833, -16, -361, 837, 173, -437, 930, 252, -342, 830, 147, -296, 974, 148, 87, 349, 429, 103, 340, 420, 128, 335, 416, 114, 333, 415, 87, 333, 425, 155, 184, 416, 74, 203, 437, 107, 415, 420, 22, 222, 430, -33, 300, 392, 125, 489, 402, 105, 391, 420, 108, 483, 403, 115, 352, 418, 107, 361, 421, 71, 472, 404, 71, 452, 411, 73, 434, 415, 92, 416, 420, 70, 410, 421, 2, 236, 419, -26, 269, 398, -23, 379, 387, -11, 395, 392, -32, 322, 391, -30, 339, 392, -28, 358, 390, -19, 351, 400, -6, 361, 409, 7, 374, 413, 32, 470, 386, 146, 526, 385, 153, 511, 389, 54, 453, 405, 122, 509, 400, 68, 490, 396, 94, 505, 397, 51, 472, 396, -42, 369, 372, -10, 337, 410, -19, 317, 404, -51, 329, 371, -51, 308, 374, -43, 287, 383, 13, 354, 421, -48, 349, 370, 142, 179, 421, 121, 179, 430, 96, 184, 434, 41, 200, 435, -35, 266, 391, -42, 262, 382, -23, 246, 401, -37, 394, 366, -25, 414, 370, 1, 416, 392, 13, 396, 409, 50, 434, 412, 90, 488, 402, 72, 373, 429, 69, 423, 418, 46, 484, 388, 32, 450, 396, 13, 449, 384, -8, 432, 377, 133, 343, 420, -52, 354, 71, -79, 373, 95, -114, 356, 129, -111, 373, 131, -120, 358, 154, -117, 376, 155, -118, 359, 182, -113, 378, 184, -111, 360, 209, -106, 378, 212, -100, 372, 234, -91, 377, 255, -95, 365, 249, 49, 419, 416, 50, 394, 424, 72, 396, 424, 35, 433, 406, 26, 412, 410, 34, 398, 419, 52, 374, 428, 64, 362, 429, 50, 352, 430, 31, 373, 423, 18, 428, 399, 62, 334, 430, 68, 315, 426, 111, 382, 387, -64, 215, 367, -79, 203, 353, -91, 197, 340, -103, 194, 324, -102, 222, 322, -92, 225, 333, -73, 239, 352, -65, 249, 360, -82, 231, 344, -103, 245, 314, -93, 248, 328, -78, 265, 343, -89, 260, 331, -90, 371, 272, -98, 357, 274, -90, 377, 290, -111, 223, 308, -115, 197, 307, -124, 280, 291, -119, 300, 277, -112, 321, 273, -83, 357, 324, -84, 338, 327, -133, 292, 288, -145, 296, 296, -148, 282, 298, -89, 373, 317, -87, 298, 330, -86, 318, 328, -112, 355, 296, -122, 354, 293, -90, 279, 329, -109, 279, 315, -99, 388, 309, -105, 374, 308, -109, 376, 296, -107, 337, 274, -92, 351, 315, -106, 333, 297, -132, 339, 280, -124, 334, 295, -99, 298, 323, -106, 283, 323, -104, 356, 308, -93, 336, 308, -143, 279, 305, -141, 317, 292, -145, 315, 281, -137, 334, 287, -127, 279, 309, -129, 267, 311, -118, 278, 322, -95, 321, 320, -100, 338, 313, -104, 319, 309, -105, 299, 314, -125, 300, 307, -123, 319, 298, -125, 309, 303, -111, 318, 299, -135, 297, 303, -93, 340, 301, -126, 315, 282, -102, 263, 314, 206, 273, 349, 207, 262, 350, 217, 273, 342, 221, 280, 328, 223, 279, 314, 218, 294, 330, 223, 295, 313, 186, 296, 354, 205, 298, 346, 187, 279, 356, 182, 278, 365, 173, 292, 367, 174, 284, 370, 178, 300, 361, 194, 316, 356, 205, 317, 347, 220, 263, 333, 169, 276, 390, 166, 295, 390, 161, 278, 405, 170, 315, 389, 161, 295, 405, 182, 315, 366, 218, 318, 329, 190, 261, 371, 189, 264, 352, 164, 262, 405, 180, 262, 385, 201, 251, 364, 228, 260, 313, 232, 279, 294, 230, 296, 295, 221, 238, 335, 199, 239, 372, 185, 241, 390, 174, 259, 395, 233, 296, 277, 232, 258, 292, 211, 239, 354, 228, 237, 315, 211, 222, 355, 199, 220, 376, 184, 220, 395, 226, 180, 271, 222, 163, 302, 223, 179, 313, 229, 198, 288, 209, 153, 332, 210, 170, 341, 195, 146, 361, 196, 163, 366, 232, 276, 272, 234, 315, 275, 231, 240, 292, 231, 259, 273, 219, 225, 231, 225, 205, 251, 221, 189, 232, 224, 166, 257, 229, 222, 273, 212, 246, 210, 208, 240, 196, 215, 215, 215, 120, 49, 132, 95, 24, 142, 133, 21, 213, 154, 44, 193, 182, 359, 424, 185, 356, 408, 185, 356, 389, 178, 341, 402, 173, 337, 413, 188, 375, 373, 188, 356, 371, 184, 381, 406, 181, 387, 415, 187, 378, 391, 183, 375, 428, 178, 387, 421, 181, 390, 388, 170, 395, 393, 189, 393, 372, 204, 376, 353, 200, 418, 354, 197, 429, 360, 198, 436, 348, 191, 448, 351, 187, 452, 368, 181, 468, 372, 188, 437, 368, 190, 439, 351, 181, 420, 405, 186, 433, 387, 183, 414, 393, 191, 414, 373, 179, 430, 403, 168, 437, 389, 181, 464, 381, 181, 451, 385, 168, 449, 393, 166, 470, 389, 174, 454, 350, 171, 450, 374, 185, 435, 336, 165, 415, 389, 181, 450, 334, 181, 417, 352, 175, 415, 371, 173, 471, 370, 167, 472, 353, 174, 394, 350, 169, 398, 371, 111, 482, 291, 168, 473, 333, 157, 482, 344, 128, 487, 324, 184, 415, 336, 148, 488, 356, 158, 378, 363, 175, 463, 319, 176, 391, 337, 164, 381, 353, 147, 479, 310, 184, 411, 320, 185, 429, 315, 162, 466, 298, 177, 403, 297, 177, 424, 293, 144, 361, 334, 164, 374, 318, 156, 367, 324, 130, 476, 277, 92, 507, 217, 116, 506, 211, 144, 467, 268, 165, 394, 261, 165, 418, 257, 114, 348, 301, 127, 350, 292, 153, 461, 265, 146, 360, 279, 153, 366, 274, 140, 356, 283, 83, 356, 267, 93, 354, 260, 108, 359, 249, 130, 458, 227, 124, 481, 224, 128, 374, 234, 68, 492, 205, 79, 482, 200, 141, 398, 221, 138, 418, 220, 94, 386, 210, 104, 395, 200, 85, 382, 218, 69, 379, 229, 86, 468, 195, 109, 414, 189, 87, 449, 195, 74, 438, 195, 65, 447, 200, 71, 413, 198, 85, 419, 187, 89, 431, 189, 168, 143, 391, 180, 160, 387, 164, 487, 383, 206, 436, 331, 198, 447, 342, 201, 452, 332, 188, 472, 353, 152, 536, 332, 184, 468, 231, 182, 453, 212, 131, 453, 164, 125, 444, 146, 136, 427, 148, 150, 433, 167, 117, 478, 158, 115, 467, 140, 111, 513, 151, 107, 496, 129, 101, 538, 119, 102, 578, 103, 97, 569, 86, 109, 611, 87, 101, 604, 63, 134, 479, 194, 134, 468, 180, 148, 455, 184, 150, 470, 201, 114, 501, 187, 116, 485, 170, 107, 525, 169, 108, 553, 149, 108, 592, 143, 107, 584, 119, 114, 621, 118, 175, 435, 190, 185, 428, 196, 125, 534, 304, 117, 526, 270, 83, 535, 258, 96, 530, 240, 80, 581, 208, 88, 563, 204, 82, 553, 224, 69, 620, 217, 85, 613, 201, 71, 585, 221, 93, 544, 200, 106, 524, 220, 103, 645, 192, 93, 650, 204, 101, 568, 176, 100, 534, 183, 180, 483, 242, 192, 474, 256, 220, 146, 289, 107, 454, 117, 115, 432, 122, 93, 590, 35, 85, 555, 62, 90, 524, 93, 99, 482, 110, 91, 445, 93, 98, 423, 96, 85, 472, 89, 81, 512, 73, 72, 439, 70, 76, 418, 73, 64, 572, 9, 58, 537, 32, 70, 465, 64, 68, 502, 54, 53, 433, 51, 54, 414, 53, 50, 459, 47, 49, 493, 40, 30, 429, 46, 31, 411, 46, 27, 566, 6, 22, 533, 28, 28, 455, 44, 26, 491, 39, 5, 426, 54, 6, 407, 52, 2, 566, 12, 0, 533, 35, 5, 451, 54, 3, 488, 48, -34, 395, 73, -32, 423, 77, -34, 533, 57, -28, 569, 33, -30, 449, 80, -31, 486, 75, 27, 96, 40, -3, 88, 48, 28, 41, 89, 59, 52, 78, -29, 84, 62, 0, 35, 103, 86, 134, 49, 55, 111, 38, 85, 74, 76, 114, 101, 78, 147, 207, 84, 132, 183, 71, 150, 164, 92, 161, 194, 100, 111, 159, 60, 133, 135, 85, 210, 320, 195, 200, 323, 174, 202, 299, 179, 212, 297, 198, 209, 272, 196, 182, 112, 157, 186, 149, 147, 171, 150, 120, 159, 116, 120, 208, 343, 195, 199, 347, 177, 96, 329, 72, 75, 330, 56, 76, 304, 50, 97, 306, 66, 142, 80, 125, 171, 75, 174, 52, 328, 43, 51, 300, 37, 198, 438, 226, 197, 417, 207, 123, 414, 127, 106, 402, 100, 80, 401, 73, 55, 399, 55, 29, 395, 47, 6, 393, 51, 233, 336, 294, 231, 316, 293, 230, 333, 310, 226, 319, 313, 175, 406, 169, 189, 402, 179, 198, 394, 195, 173, 524, 325, 181, 507, 297, 157, 527, 303, 175, 505, 273, 153, 503, 247, 154, 515, 273, 131, 508, 238, 98, 609, 186, 112, 642, 179, 140, 523, 282, 198, 455, 244, 134, 489, 206, 152, 481, 216, 153, 489, 229, 111, 601, 165, 114, 634, 153, 165, 469, 213, 147, 412, 152, 167, 486, 236, 162, 446, 187, 134, 400, 134, 156, 229, 97, 167, 218, 111, 162, 247, 107, 170, 240, 118, 189, 356, 158, 186, 374, 160, 194, 373, 177, 56, 205, 27, 28, 197, 17, 38, 156, 22, 69, 171, 34, 124, 388, 117, -36, 376, 69, -19, 358, 54, -39, 359, 65, -44, 331, 53, -23, 330, 42, 3, 351, 43, 2, 329, 36, 29, 375, 45, 5, 373, 47, 54, 376, 51, 79, 385, 69, 102, 385, 92, -15, 397, 62, 51, 270, 30, 77, 275, 43, -47, 302, 39, -50, 268, 30, 99, 281, 57, 26, 298, 29, 1, 299, 26, -1, 265, 18, 25, 266, 21, 115, 309, 78, 118, 288, 71, 122, 257, 67, 101, 252, 52, 108, 228, 54, 130, 239, 71, 51, 237, 24, 24, 232, 15, 77, 244, 37, 83, 216, 38, -53, 228, 30, -50, 184, 32, -2, 229, 14, 1, 191, 15, 10, 146, 21, 137, 223, 76, 118, 205, 59, 95, 187, 46, -44, 139, 39, 113, 26, 308, 86, 4, 251, 179, 57, 240, 159, 39, 265, 216, 128, 274, 203, 112, 306, 208, 132, 320, 221, 152, 243, 218, 179, 218, 109, 8, 236, 68, 10, 157, 27, 327, 36, 163, 159, 402, 187, 102, 336, 192, 124, 350, 215, 140, 229, 211, 114, 260, 214, 168, 205, -24, 300, 29, 199, 152, 171, 197, 118, 188, 192, 84, 213, 199, 95, 293, 157, 396, 156, 145, 388, 140, 136, 377, 127, 125, 372, 112, 127, 311, 91, 131, 294, 87, 137, 265, 84, 151, 256, 96, 149, 238, 91, 205, 99, 242, 209, 128, 212, 208, 159, 190, 193, 77, 274, 139, 74, 363, 145, 94, 377, -14, 450, 66, -15, 487, 61, 44, 3, 174, 115, 355, 95, 114, 330, 86, 126, 335, 100, 135, 29, 291, 152, 59, 329, 175, 63, 302, -27, 267, 20, -15, 423, 65, -18, 141, 27, 132, 57, 346, -20, 532, 41, -28, 228, 19, -25, 186, 20, 181, 83, 322, 176, 316, 374, 184, 509, 327, 171, 527, 352, 149, 295, 101, 139, 318, 103, 147, 274, 96, 171, 123, 376, 164, 96, 362, 158, 77, 348, 213, 275, 333, 170, 295, 350, 169, 282, 352, 211, 296, 334, 215, 282, 314, 209, 262, 332, 189, 251, 334, 169, 263, 346, 187, 315, 334, 212, 300, 308, 203, 311, 313, 175, 309, 343, 206, 259, 310, 166, 312, 329, 207, 279, 294, 156, 280, 343, 161, 305, 338, 205, 296, 295, 157, 292, 343, 181, 247, 324, 186, 248, 314, 153, 297, 331, 164, 255, 330, 185, 317, 311, 150, 278, 331, 201, 262, 297, 197, 310, 300, 191, 250, 307, 155, 263, 330, 166, 313, 311, 174, 248, 315, 184, 310, 296, 167, 253, 309, 187, 296, 287, 187, 260, 293, 153, 299, 311, 149, 278, 314, 190, 278, 285, 155, 262, 312, 168, 310, 300, 169, 277, 288, 165, 295, 292, 168, 262, 294, 156, 280, 296, 203, 368, 190, 170, 372, 140, 242, 642, 26, 251, 654, 59, 320, 643, 34, 313, 623, -5, 386, 651, -2, 382, 637, -44, 422, 690, -30, 436, 696, -86, 437, 742, -51, 461, 742, -95, 465, 756, -140, 472, 786, -99, 484, 786, -138, 473, 826, -91, 506, 836, -139, 510, 846, -178, 253, 668, 91, 317, 665, 69, 368, 675, 37, 401, 704, 7, 420, 736, -22, 429, 795, -49, 471, 870, -87, 276, 714, 137, 241, 742, 177, 301, 748, 135, 341, 741, 92, 359, 784, 64, 378, 741, 43, 368, 805, 32, 387, 796, 4, 364, 821, 5, 346, 842, -7, 390, 834, -45, 344, 849, -55, 341, 868, -89, 378, 892, -94, 419, 886, -80, 186, 757, 217, 144, 779, 247, 193, 786, 225, 238, 776, 196, 287, 776, 158, 326, 788, 118, 347, 809, 80, 350, 839, 53, 320, 863, -46, 296, 884, -89, 344, 898, -117, 236, 802, 213, 279, 801, 178, 318, 825, 146, 341, 832, 95, 333, 875, 36, 322, 873, -3, 307, 881, -37, 293, 890, -128, 232, 843, 232, 275, 827, 197, 340, 866, 94, 315, 898, 10, 290, 918, -53, 274, 856, 211, 319, 883, 178, 340, 887, 129, 338, 903, 96, 325, 916, 64, 310, 929, 26, 297, 936, -16, 143, 876, 285, 234, 891, 252, 275, 889, 225, 340, 930, 135, 324, 952, 89, 303, 970, 47, 142, 945, 311, 137, 662, 133, 142, 645, 88, 232, 636, -3, 298, 615, -41, 365, 632, -82, 422, 691, -125, 449, 731, -147, 477, 782, -170, 494, 817, -182, 153, 833, 262, 162, 929, 296, 203, 716, 189, 151, 735, 228, 235, 692, 138, 109, 698, 227, 127, 675, 191, 131, 672, 168, 187, 650, 52, 188, 663, 79, 141, 633, 61, 95, 593, -21, 68, 589, -27, -20, 592, 3, -42, 595, 22, 113, 601, -10, 4, 590, -14, 126, 609, 4, 141, 622, 40, 136, 613, 21, 32, 590, -25, 172, 622, -18, 159, 617, -35, 43, 614, -52, 9, 615, -38, 180, 629, 4, 85, 609, -58, 117, 613, -55, 140, 617, -47, -20, 618, -19, -45, 622, 5, 134, 628, -85, 101, 629, -85, 58, 637, -76, 18, 640, -60, 225, 623, -31, 210, 622, -55, -18, 644, -42, 191, 620, -73, -48, 650, -12, 164, 624, -83, 183, 673, 102, 145, 705, 215, 160, 699, 191, 170, 689, 145, 185, 637, 27, 110, 744, 254, 194, 824, 240, 195, 883, 264, -31, 691, -44, 259, 611, -99, 227, 623, -115, 29, 666, -78, -13, 672, -56, 189, 634, -119, 150, 645, -115, 110, 653, -106, 69, 660, -94, 284, 609, -73, 166, 665, -143, 117, 692, -134, 72, 687, -109, 33, 694, -90, 348, 628, -120, 317, 628, -149, -53, 710, -31, 267, 638, -158, 216, 650, -156, 184, 695, -170, 159, 735, -170, 109, 742, -149, 82, 915, -128, 120, 916, -153, 120, 952, -130, 65, 950, -104, 150, 796, -169, 97, 777, -154, 61, 839, -130, 129, 846, -163, 121, 884, -168, 82, 883, -136, 69, 717, -119, 63, 751, -125, 50, 914, -97, 53, 784, -135, 51, 882, -100, 30, 745, -83, 16, 914, -83, 17, 950, -80, 24, 794, -84, 19, 839, -86, 16, 882, -85, -15, 734, -67, -15, 768, -68, -30, 914, -54, -27, 952, -52, -17, 800, -70, -18, 856, -68, -38, 876, -51, -52, 741, -39, -52, 790, -45, 388, 672, -152, 348, 667, -176, 302, 667, -190, 241, 693, -192, 294, 794, -219, 293, 837, -213, 243, 832, -188, 413, 702, -171, 375, 698, -196, 169, 885, -164, 214, 891, -159, 208, 926, -148, 166, 918, -152, 331, 695, -210, 282, 698, -209, 261, 776, -207, 240, 795, -196, 252, 871, -171, 264, 897, -143, 245, 928, -135, 204, 830, -176, 435, 736, -185, 381, 733, -218, 339, 735, -223, 299, 742, -219, 268, 734, -210, 236, 754, -199, 203, 740, -188, 408, 744, -214, 338, 794, -234, 285, 883, -184, 464, 789, -207, 424, 778, -224, 452, 767, -197, 385, 780, -235, 330, 842, -232, 299, 903, -155, 471, 835, -228, 441, 815, -238, 387, 825, -247, 389, 863, -258, 363, 873, -252, 338, 884, -233, 311, 912, -193, 328, 936, -183, 343, 927, -149, 420, 853, -258, 265, 942, -103, 270, 917, -108, 200, 777, -185, 169, 846, -170, 347, 951, -181, 514, 875, -222, 476, 878, -260, 530, 889, -138, 535, 880, -173, 348, 935, -231, 380, 915, -262, 427, 893, -271, 513, 903, -112, 430, 932, -105, 379, 940, -136, 528, 908, -249, 515, 928, -280, 537, 945, -130, 554, 925, -152, 469, 922, -287, 428, 942, -289, 560, 922, -223, 572, 929, -189, 555, 940, -258, 293, 973, 247, 319, 968, 223, 279, 930, 237, 259, 964, 265, 223, 969, 285, 339, 974, 193, 332, 937, 186, 241, 919, 258, 207, 940, 280, 343, 976, 144, 105, 606, 176, 167, 496, 250, 179, 494, 257, 188, 488, 270, 194, 481, 284, 180, 446, 317, 172, 444, 294, 159, 440, 261, 133, 437, 225, 159, 380, 267, 172, 390, 306, 180, 401, 328, 301, 688, 101, 347, 697, 72, 399, 747, 13, 409, 772, -10, 428, 834, -57, 485, 916, -100, 229, 278, 254, 225, 277, 237, 228, 263, 257, 229, 247, 267, 228, 222, 307, 222, 217, 332, 211, 202, 353, 199, 198, 375, 185, 203, 395, 512, 959, -120, 89, 814, -154, -18, 829, -70, -52, 833, -45, 272, 851, -195, 315, 897, -217, 163, 954, -138, 210, 336, 349, 187, 333, 369, 208, 354, 352, 175, 328, 389, 166, 184, 407, 184, 183, 392, 223, 204, 327, 229, 213, 301, 166, 505, 382, 224, 294, 232, 230, 298, 255, 176, 487, 370, 199, 465, 330, 202, 397, 352, 208, 414, 336, 225, 238, 252, 223, 257, 236, 224, 356, 238, 223, 380, 256, 217, 391, 241, 217, 373, 228, 225, 316, 234, 222, 330, 227, 230, 337, 251, 229, 355, 255, 229, 369, 268, 199, 464, 274, 210, 437, 309, 179, 506, 360, 204, 456, 312, 195, 481, 321, 190, 488, 336, 193, 489, 309, 199, 471, 309, 203, 460, 294, 232, 353, 273, 218, 355, 219, 232, 317, 256, 178, 338, 390, 217, 316, 212, 219, 296, 214, 215, 274, 209, 219, 276, 219, 234, 333, 272, 233, 350, 290, 217, 336, 213, 198, 179, 371, 212, 184, 348, 223, 192, 321, 217, 251, 221, 210, 362, 205, 208, 383, 213, 208, 404, 225, 223, 393, 271, 217, 407, 257, 196, 473, 328, 183, 493, 353, 218, 379, 330, 197, 477, 294, 212, 434, 290, 206, 441, 264, 208, 423, 242, 228, 375, 289, -17, 377, 59, 27, 350, 41, 52, 351, 47, 76, 351, 61, 77, 370, 65, 98, 351, 79, 100, 369, 84, 128, 356, 111, 215, 414, 318, 219, 410, 304, 220, 397, 315, 214, 396, 332, 223, 397, 294, 224, 372, 316, 228, 354, 311, 219, 355, 333, 215, 419, 276, 222, 335, 330, 209, 313, 333, 175, 430, 374, 209, 203, 196, 203, 190, 177, 196, 184, 161, 188, 181, 143, 193, 212, 156, 186, 210, 142, 200, 223, 175, 192, 239, 157, 182, 233, 137, 201, 270, 178, 198, 253, 171, 147, 352, 120, 149, 336, 114, 148, 365, 128, 157, 372, 136, 178, 211, 128, 177, 184, 124, 169, 261, 114, 153, 316, 109, 188, 344, 153, 190, 318, 155, 168, 279, 100, 187, 264, 97, 186, 266, 87, 186, 279, 83, 183, 373, 134, 160, 359, 128, 192, 298, 156, 170, 357, 118, 168, 331, 116, 172, 335, 96, 172, 352, 106, 190, 259, 152, 187, 259, 135, 174, 371, 124, 181, 367, 120, 191, 276, 155, 185, 354, 136, 169, 320, 97, 188, 279, 135, 183, 315, 137, 177, 348, 106, 184, 337, 136, 181, 352, 119, 183, 296, 81, 179, 308, 83, 174, 313, 85, 176, 294, 80, 191, 260, 114, 197, 265, 126, 180, 317, 98, 185, 299, 136, 181, 338, 115, 179, 329, 100, 188, 278, 98, 174, 332, 136, 185, 298, 97, 173, 319, 134, 188, 278, 118, 174, 318, 120, 182, 297, 116, 180, 314, 116, 177, 303, 127, 175, 303, 119, 167, 295, 98, 196, 285, 167];
var HEAD_F = [11, 4, 3, 7, 5, 12, 1, 9, 10, 13, 10, 8, 13, 8, 14, 15, 13, 14, 15, 14, 16, 17, 18, 16, 19, 12, 6, 22, 0, 1, 22, 1, 23, 25, 26, 27, 28, 2, 22, 13, 27, 26, 29, 20, 21, 29, 21, 30, 2, 28, 21, 28, 31, 21, 32, 23, 25, 32, 25, 27, 22, 23, 32, 30, 21, 31, 13, 15, 27, 15, 24, 27, 31, 28, 34, 34, 28, 22, 34, 22, 35, 35, 22, 32, 36, 24, 37, 38, 27, 36, 35, 32, 39, 27, 24, 36, 40, 31, 34, 41, 38, 36, 41, 36, 42, 42, 36, 37, 42, 37, 43, 44, 45, 46, 46, 47, 48, 48, 47, 49, 48, 49, 50, 51, 29, 52, 53, 29, 30, 52, 29, 53, 51, 33, 29, 54, 30, 31, 55, 30, 54, 52, 53, 30, 52, 30, 55, 31, 40, 54, 56, 57, 58, 56, 58, 59, 44, 60, 59, 61, 45, 58, 62, 56, 63, 64, 65, 66, 64, 66, 67, 71, 70, 69, 72, 71, 69, 72, 69, 68, 69, 73, 68, 74, 75, 76, 74, 76, 70, 77, 78, 79, 80, 78, 77, 80, 77, 76, 86, 87, 88, 95, 89, 96, 78, 84, 83, 80, 84, 78, 90, 91, 97, 95, 102, 103, 100, 104, 102, 103, 102, 104, 95, 103, 104, 95, 104, 105, 102, 96, 99, 106, 107, 92, 106, 108, 104, 106, 104, 107, 96, 102, 95, 104, 109, 103, 100, 109, 104, 104, 103, 105, 107, 111, 92, 112, 94, 92, 112, 92, 111, 103, 98, 110, 109, 114, 98, 109, 98, 103, 107, 116, 117, 107, 117, 111, 116, 107, 104, 116, 104, 119, 120, 121, 122, 125, 126, 121, 125, 121, 120, 114, 84, 83, 114, 83, 98, 117, 127, 128, 129, 111, 117, 126, 125, 130, 131, 113, 132, 98, 133, 118, 98, 83, 133, 117, 116, 119, 129, 112, 111, 134, 116, 136, 83, 137, 133, 133, 137, 135, 127, 117, 116, 127, 116, 134, 138, 129, 128, 139, 115, 123, 115, 139, 140, 115, 140, 124, 119, 136, 116, 84, 137, 83, 141, 128, 127, 141, 127, 142, 128, 129, 117, 119, 104, 143, 144, 131, 132, 145, 144, 132, 145, 132, 146, 135, 147, 148, 137, 149, 147, 137, 147, 135, 142, 127, 134, 142, 134, 130, 151, 138, 128, 151, 128, 141, 139, 152, 153, 139, 153, 140, 104, 105, 143, 137, 150, 149, 141, 142, 154, 141, 154, 155, 156, 157, 155, 156, 155, 154, 158, 145, 146, 158, 146, 159, 147, 160, 161, 147, 161, 148, 149, 162, 160, 149, 160, 147, 154, 142, 130, 154, 130, 125, 164, 151, 141, 164, 141, 155, 165, 152, 166, 152, 165, 153, 150, 163, 162, 150, 162, 149, 160, 167, 161, 162, 168, 167, 162, 167, 160, 156, 154, 125, 156, 125, 120, 170, 164, 155, 170, 155, 171, 172, 165, 166, 172, 166, 173, 165, 172, 174, 163, 169, 168, 163, 168, 162, 175, 176, 157, 175, 157, 156, 171, 155, 157, 177, 105, 104, 177, 104, 108, 178, 158, 159, 178, 159, 179, 180, 172, 173, 180, 173, 181, 172, 180, 174, 169, 182, 168, 183, 171, 157, 183, 157, 176, 184, 178, 179, 184, 179, 185, 168, 182, 187, 168, 187, 167, 167, 187, 186, 175, 156, 120, 188, 170, 171, 188, 171, 183, 189, 188, 183, 189, 183, 190, 191, 190, 176, 191, 176, 175, 120, 122, 192, 120, 192, 193, 194, 180, 181, 182, 180, 194, 190, 183, 176, 195, 184, 185, 195, 185, 196, 186, 197, 198, 182, 197, 187, 187, 197, 186, 175, 120, 193, 175, 193, 191, 199, 197, 182, 189, 190, 194, 50, 49, 200, 50, 200, 201, 202, 108, 106, 94, 203, 92, 203, 93, 92, 206, 207, 208, 209, 210, 211, 209, 211, 212, 213, 209, 212, 213, 212, 214, 215, 213, 214, 216, 215, 217, 218, 216, 217, 218, 217, 219, 220, 218, 219, 220, 219, 221, 222, 223, 224, 222, 224, 225, 226, 222, 225, 226, 225, 227, 228, 226, 227, 229, 228, 216, 230, 229, 216, 231, 230, 220, 208, 207, 232, 208, 232, 233, 205, 234, 235, 234, 236, 237, 236, 238, 237, 239, 240, 243, 240, 242, 244, 240, 244, 243, 245, 242, 240, 240, 239, 241, 242, 246, 247, 242, 247, 244, 249, 246, 242, 249, 242, 245, 230, 231, 250, 230, 250, 251, 229, 230, 251, 228, 229, 252, 253, 254, 206, 44, 46, 255, 210, 209, 225, 210, 225, 224, 209, 213, 227, 209, 227, 225, 213, 215, 228, 213, 228, 227, 215, 216, 228, 216, 218, 230, 218, 220, 230, 212, 256, 257, 212, 257, 214, 219, 258, 259, 219, 259, 221, 217, 260, 258, 217, 258, 219, 215, 261, 260, 215, 260, 217, 214, 257, 261, 214, 261, 215, 256, 262, 263, 256, 263, 257, 258, 264, 259, 257, 263, 265, 257, 265, 261, 261, 265, 266, 261, 266, 260, 260, 266, 258, 262, 267, 268, 262, 268, 263, 258, 269, 270, 258, 270, 264, 263, 268, 271, 263, 271, 265, 265, 271, 272, 265, 272, 266, 266, 272, 269, 266, 269, 258, 267, 273, 274, 267, 274, 268, 269, 275, 276, 269, 276, 270, 268, 274, 277, 268, 277, 271, 271, 277, 278, 271, 278, 272, 272, 278, 275, 272, 275, 269, 273, 279, 280, 273, 280, 274, 275, 281, 276, 274, 280, 282, 274, 282, 277, 277, 282, 283, 277, 283, 278, 278, 283, 281, 278, 281, 275, 279, 284, 285, 279, 285, 280, 281, 286, 287, 281, 287, 276, 280, 285, 288, 280, 288, 282, 282, 288, 289, 282, 289, 283, 283, 289, 286, 283, 286, 281, 290, 291, 292, 293, 294, 295, 292, 291, 296, 292, 296, 297, 297, 296, 298, 299, 300, 301, 299, 301, 302, 302, 301, 303, 302, 303, 304, 305, 306, 307, 305, 307, 308, 309, 310, 311, 309, 311, 312, 308, 307, 300, 308, 300, 299, 312, 311, 313, 312, 313, 314, 315, 316, 317, 316, 318, 319, 316, 319, 317, 320, 321, 322, 320, 322, 323, 325, 326, 327, 325, 327, 328, 318, 62, 319, 329, 330, 65, 329, 65, 64, 321, 320, 330, 321, 330, 329, 327, 331, 332, 327, 332, 328, 333, 334, 232, 333, 232, 207, 212, 211, 335, 212, 335, 256, 262, 256, 335, 262, 335, 336, 267, 262, 336, 267, 336, 337, 273, 267, 337, 279, 273, 338, 284, 279, 338, 284, 338, 339, 340, 341, 342, 11, 343, 344, 20, 29, 33, 341, 344, 343, 1, 10, 26, 1, 26, 25, 10, 13, 26, 23, 1, 25, 2, 0, 22, 233, 232, 345, 233, 345, 346, 334, 347, 345, 334, 345, 232, 205, 204, 348, 349, 350, 351, 351, 205, 349, 352, 351, 350, 353, 354, 355, 353, 355, 356, 239, 238, 248, 239, 248, 241, 245, 357, 249, 358, 245, 240, 240, 241, 358, 356, 355, 236, 356, 236, 359, 351, 352, 353, 351, 353, 356, 360, 361, 206, 360, 206, 254, 362, 363, 364, 252, 229, 365, 229, 251, 365, 251, 250, 366, 367, 368, 223, 367, 223, 364, 369, 370, 211, 369, 211, 210, 371, 367, 364, 371, 364, 363, 369, 372, 233, 369, 233, 346, 368, 372, 224, 368, 224, 223, 373, 335, 211, 373, 211, 370, 374, 375, 310, 374, 310, 309, 376, 377, 375, 376, 375, 374, 378, 324, 315, 378, 315, 379, 304, 303, 380, 304, 380, 381, 314, 313, 306, 314, 306, 305, 346, 345, 382, 346, 382, 383, 384, 385, 386, 384, 386, 387, 388, 336, 335, 388, 335, 373, 389, 390, 290, 391, 392, 393, 391, 393, 394, 395, 396, 392, 395, 392, 391, 338, 397, 339, 398, 338, 273, 399, 398, 273, 399, 273, 337, 400, 399, 337, 400, 337, 336, 401, 284, 339, 401, 339, 402, 327, 403, 404, 327, 404, 331, 405, 406, 407, 405, 407, 408, 326, 409, 403, 326, 403, 327, 410, 411, 412, 410, 412, 413, 414, 415, 409, 414, 409, 326, 331, 404, 411, 331, 411, 410, 416, 417, 418, 416, 418, 419, 420, 384, 387, 420, 387, 421, 422, 423, 384, 422, 384, 420, 424, 425, 426, 424, 426, 427, 419, 418, 423, 419, 423, 422, 421, 387, 428, 421, 428, 429, 386, 308, 299, 386, 299, 430, 431, 309, 312, 431, 312, 432, 385, 305, 308, 385, 308, 386, 433, 314, 305, 433, 305, 385, 434, 304, 381, 434, 381, 435, 432, 312, 314, 432, 314, 433, 436, 437, 438, 436, 438, 439, 440, 441, 66, 440, 66, 65, 255, 442, 443, 255, 443, 444, 445, 60, 44, 445, 44, 255, 445, 255, 444, 446, 447, 67, 446, 67, 66, 331, 410, 448, 331, 448, 332, 410, 413, 396, 410, 396, 448, 449, 450, 451, 443, 442, 452, 443, 452, 453, 201, 200, 450, 201, 450, 449, 454, 445, 444, 454, 444, 455, 454, 456, 60, 454, 60, 445, 396, 413, 457, 396, 457, 392, 458, 459, 320, 458, 320, 323, 459, 460, 330, 459, 330, 320, 455, 444, 443, 455, 443, 461, 205, 235, 204, 392, 457, 405, 392, 405, 393, 356, 359, 351, 206, 208, 368, 206, 368, 367, 346, 370, 369, 253, 206, 367, 253, 367, 371, 210, 224, 372, 210, 372, 369, 208, 233, 372, 208, 372, 368, 462, 373, 370, 463, 388, 373, 463, 373, 462, 464, 400, 336, 464, 336, 388, 414, 465, 415, 466, 467, 417, 466, 417, 416, 468, 374, 309, 468, 309, 431, 459, 469, 470, 459, 470, 460, 459, 458, 471, 459, 471, 469, 393, 405, 408, 393, 408, 472, 460, 470, 473, 460, 473, 440, 474, 475, 476, 477, 297, 298, 477, 298, 478, 438, 479, 480, 438, 480, 439, 481, 482, 483, 481, 483, 484, 485, 441, 486, 485, 486, 487, 39, 32, 38, 39, 38, 41, 475, 488, 489, 457, 490, 406, 457, 406, 405, 441, 440, 473, 441, 473, 486, 491, 292, 297, 491, 297, 477, 492, 302, 304, 492, 304, 434, 460, 440, 65, 460, 65, 330, 493, 494, 437, 493, 437, 436, 286, 293, 295, 401, 290, 292, 401, 292, 491, 495, 496, 425, 495, 425, 424, 478, 298, 293, 478, 293, 286, 485, 436, 439, 485, 439, 446, 461, 497, 486, 461, 486, 473, 32, 27, 38, 108, 202, 119, 349, 205, 348, 351, 359, 234, 351, 234, 205, 236, 234, 359, 234, 237, 235, 223, 222, 362, 223, 362, 364, 222, 226, 157, 222, 157, 362, 355, 354, 248, 236, 355, 248, 238, 236, 248, 226, 228, 252, 226, 252, 157, 465, 414, 500, 465, 500, 501, 502, 467, 466, 417, 431, 432, 417, 432, 418, 488, 475, 503, 488, 503, 504, 504, 503, 453, 504, 453, 452, 290, 401, 402, 439, 480, 447, 439, 447, 446, 288, 477, 478, 288, 478, 289, 436, 485, 487, 436, 487, 493, 413, 412, 490, 413, 490, 457, 285, 491, 477, 285, 477, 288, 430, 299, 302, 430, 302, 492, 286, 295, 287, 284, 401, 491, 284, 491, 285, 429, 428, 496, 429, 496, 495, 289, 478, 286, 441, 485, 446, 441, 446, 66, 474, 493, 487, 474, 487, 505, 505, 487, 486, 505, 486, 497, 506, 5, 7, 9, 1, 0, 12, 506, 7, 9, 507, 7, 508, 7, 507, 509, 7, 508, 510, 9, 0, 510, 0, 511, 21, 2, 3, 511, 0, 28, 514, 9, 510, 12, 7, 515, 21, 3, 20, 28, 0, 2, 9, 514, 507, 515, 7, 509, 511, 28, 516, 514, 510, 517, 518, 21, 20, 518, 20, 519, 516, 28, 2, 516, 2, 520, 509, 508, 507, 509, 507, 514, 519, 20, 3, 520, 2, 518, 517, 510, 511, 518, 2, 21, 521, 512, 12, 521, 12, 515, 522, 11, 513, 511, 516, 523, 525, 514, 517, 516, 520, 527, 524, 509, 514, 524, 514, 525, 527, 520, 518, 523, 517, 511, 522, 513, 528, 521, 515, 530, 525, 517, 531, 529, 523, 516, 11, 522, 519, 11, 519, 3, 523, 532, 517, 517, 532, 531, 516, 527, 529, 528, 533, 522, 522, 533, 534, 522, 534, 519, 518, 535, 536, 518, 536, 527, 519, 534, 535, 519, 535, 518, 527, 536, 529, 530, 537, 521, 523, 529, 532, 531, 532, 539, 531, 539, 525, 529, 538, 532, 532, 542, 539, 535, 543, 542, 535, 542, 545, 544, 546, 547, 542, 532, 538, 547, 546, 548, 544, 540, 546, 548, 546, 549, 539, 542, 549, 534, 541, 543, 534, 543, 535, 548, 549, 542, 544, 530, 540, 538, 529, 536, 538, 536, 545, 537, 530, 544, 539, 549, 525, 541, 534, 533, 546, 524, 525, 546, 525, 549, 33, 342, 341, 550, 382, 345, 550, 345, 347, 383, 551, 370, 383, 370, 346, 343, 33, 341, 3, 343, 11, 552, 553, 554, 552, 554, 555, 553, 556, 557, 553, 557, 554, 556, 558, 559, 556, 559, 557, 558, 560, 559, 560, 561, 562, 560, 562, 559, 561, 563, 564, 561, 564, 562, 563, 565, 566, 563, 566, 564, 555, 554, 567, 555, 567, 568, 554, 557, 569, 554, 569, 567, 557, 559, 570, 557, 570, 569, 559, 571, 570, 559, 562, 572, 559, 572, 571, 562, 564, 573, 564, 566, 574, 566, 575, 574, 576, 577, 578, 576, 578, 579, 577, 580, 581, 577, 581, 578, 580, 582, 583, 580, 583, 581, 582, 584, 583, 585, 586, 584, 585, 587, 586, 587, 588, 589, 587, 589, 586, 588, 590, 591, 588, 591, 589, 579, 578, 594, 581, 583, 595, 583, 584, 595, 584, 586, 596, 586, 597, 596, 586, 597, 598, 589, 591, 599, 594, 578, 602, 594, 602, 603, 578, 581, 604, 578, 604, 602, 581, 595, 605, 581, 605, 604, 595, 606, 605, 596, 597, 608, 599, 609, 597, 602, 604, 610, 604, 605, 611, 604, 611, 610, 605, 606, 611, 607, 596, 612, 607, 612, 613, 596, 608, 612, 579, 594, 592, 603, 602, 615, 603, 615, 616, 602, 610, 617, 602, 617, 615, 610, 611, 618, 610, 618, 617, 611, 619, 618, 613, 612, 620, 612, 608, 621, 612, 621, 620, 608, 614, 622, 608, 622, 621, 623, 624, 625, 615, 617, 626, 617, 618, 627, 617, 627, 626, 618, 619, 628, 618, 628, 627, 613, 620, 629, 631, 632, 624, 631, 624, 623, 600, 601, 632, 600, 632, 631, 633, 366, 250, 634, 635, 553, 634, 553, 552, 635, 636, 556, 635, 556, 553, 636, 637, 558, 636, 558, 556, 637, 638, 560, 637, 560, 558, 638, 639, 561, 638, 561, 560, 639, 640, 563, 639, 563, 561, 640, 641, 565, 640, 565, 563, 601, 593, 632, 624, 642, 625, 643, 579, 592, 643, 592, 644, 624, 632, 642, 644, 592, 593, 645, 576, 579, 645, 579, 643, 646, 246, 249, 646, 249, 357, 646, 357, 647, 648, 552, 555, 648, 555, 649, 650, 231, 220, 650, 220, 651, 652, 270, 276, 652, 276, 653, 654, 295, 655, 656, 264, 270, 656, 270, 652, 655, 295, 294, 655, 294, 657, 295, 654, 658, 295, 658, 287, 656, 259, 264, 659, 220, 221, 659, 221, 660, 661, 276, 287, 661, 287, 658, 660, 221, 259, 660, 259, 656, 653, 276, 661, 662, 660, 656, 662, 656, 663, 664, 661, 658, 664, 658, 665, 666, 659, 660, 666, 660, 662, 667, 653, 661, 667, 661, 664, 668, 652, 653, 668, 653, 667, 669, 656, 652, 669, 652, 668, 670, 654, 655, 670, 655, 671, 663, 656, 669, 654, 670, 665, 654, 665, 658, 671, 655, 657, 671, 657, 672, 673, 671, 672, 673, 672, 674, 675, 668, 667, 675, 667, 676, 676, 667, 664, 676, 664, 677, 677, 664, 665, 677, 665, 678, 679, 666, 662, 679, 662, 680, 670, 681, 678, 670, 678, 665, 680, 662, 663, 680, 663, 682, 681, 670, 671, 681, 671, 673, 682, 663, 669, 682, 669, 683, 683, 669, 668, 683, 668, 675, 684, 646, 647, 231, 650, 633, 231, 633, 250, 648, 650, 651, 568, 649, 555, 659, 666, 648, 659, 648, 651, 552, 648, 634, 643, 684, 647, 643, 647, 645, 650, 648, 649, 650, 649, 633, 644, 646, 684, 644, 684, 643, 666, 679, 634, 666, 634, 648, 659, 651, 220, 686, 687, 685, 687, 601, 600, 632, 688, 616, 632, 616, 642, 642, 616, 689, 642, 689, 625, 593, 592, 688, 593, 688, 632, 646, 685, 247, 646, 247, 246, 686, 685, 646, 646, 644, 687, 646, 687, 686, 644, 593, 601, 644, 601, 687, 592, 594, 603, 592, 603, 688, 688, 603, 616, 690, 673, 674, 690, 674, 691, 692, 680, 682, 692, 682, 693, 681, 694, 695, 681, 695, 678, 693, 682, 683, 693, 683, 696, 694, 681, 673, 694, 673, 690, 696, 683, 675, 696, 675, 697, 697, 675, 676, 697, 676, 698, 698, 676, 677, 698, 677, 699, 699, 677, 678, 699, 678, 695, 700, 679, 680, 700, 680, 692, 701, 697, 698, 701, 698, 699, 702, 700, 692, 702, 692, 703, 703, 692, 693, 703, 693, 704, 704, 693, 696, 704, 696, 705, 705, 696, 697, 705, 697, 701, 706, 707, 708, 706, 708, 705, 709, 710, 711, 707, 712, 713, 707, 713, 708, 714, 711, 715, 714, 709, 711, 716, 707, 706, 717, 718, 710, 717, 710, 709, 716, 719, 712, 716, 712, 707, 720, 717, 709, 720, 709, 714, 721, 722, 716, 723, 724, 718, 723, 718, 717, 722, 719, 716, 720, 723, 717, 725, 726, 722, 725, 722, 721, 727, 724, 723, 726, 728, 722, 729, 727, 723, 729, 723, 720, 730, 731, 726, 730, 726, 725, 732, 733, 724, 732, 724, 727, 731, 728, 726, 734, 732, 727, 734, 727, 729, 735, 736, 731, 735, 731, 730, 737, 738, 733, 737, 733, 732, 739, 737, 732, 739, 732, 734, 705, 701, 706, 706, 701, 699, 706, 699, 716, 716, 699, 695, 716, 695, 721, 694, 725, 721, 694, 721, 695, 725, 694, 690, 725, 690, 730, 730, 690, 691, 730, 691, 735, 636, 635, 700, 636, 700, 702, 635, 634, 679, 635, 679, 700, 740, 702, 703, 740, 703, 741, 741, 703, 704, 741, 704, 742, 742, 704, 705, 742, 705, 743, 744, 745, 746, 747, 740, 741, 747, 741, 748, 749, 750, 751, 749, 751, 752, 748, 741, 742, 748, 742, 753, 753, 742, 743, 753, 743, 754, 755, 756, 745, 755, 745, 744, 758, 752, 757, 759, 747, 748, 759, 748, 760, 752, 751, 761, 752, 761, 757, 760, 748, 753, 760, 753, 762, 762, 753, 754, 754, 763, 764, 758, 757, 746, 765, 759, 760, 765, 760, 766, 766, 760, 762, 766, 762, 754, 766, 754, 767, 767, 754, 768, 768, 754, 764, 768, 764, 755, 746, 757, 770, 757, 609, 770, 640, 765, 771, 614, 598, 609, 771, 765, 766, 771, 766, 772, 772, 766, 773, 772, 773, 774, 774, 773, 744, 774, 744, 769, 775, 769, 744, 775, 744, 776, 770, 609, 777, 778, 640, 771, 778, 771, 779, 779, 771, 772, 779, 772, 780, 589, 599, 597, 589, 597, 586, 780, 772, 774, 780, 774, 775, 775, 774, 769, 781, 775, 782, 782, 775, 776, 770, 783, 784, 770, 777, 783, 777, 609, 785, 777, 785, 783, 779, 780, 786, 609, 599, 785, 786, 780, 775, 786, 775, 781, 598, 597, 609, 598, 614, 608, 598, 608, 597, 764, 756, 755, 757, 614, 609, 766, 767, 773, 773, 767, 768, 773, 768, 744, 744, 768, 755, 606, 607, 613, 606, 613, 611, 584, 596, 595, 595, 596, 607, 595, 607, 606, 562, 573, 572, 619, 613, 629, 619, 629, 628, 611, 613, 619, 637, 636, 702, 637, 702, 740, 638, 637, 740, 638, 740, 747, 747, 759, 639, 747, 639, 638, 639, 759, 765, 765, 640, 639, 641, 640, 778, 614, 757, 761, 614, 761, 622, 744, 746, 787, 744, 787, 776, 776, 787, 784, 776, 784, 782, 788, 749, 752, 788, 752, 758, 790, 778, 779, 790, 779, 791, 565, 792, 793, 565, 793, 566, 783, 785, 794, 795, 782, 784, 791, 779, 786, 791, 786, 796, 566, 793, 575, 796, 786, 781, 792, 565, 641, 792, 641, 797, 796, 781, 782, 796, 782, 798, 798, 782, 795, 797, 641, 778, 797, 778, 790, 590, 799, 800, 590, 800, 591, 784, 783, 789, 591, 800, 794, 591, 794, 599, 785, 599, 794, 801, 790, 791, 793, 802, 803, 801, 791, 796, 801, 796, 804, 804, 796, 805, 806, 797, 790, 806, 790, 801, 805, 796, 798, 805, 798, 807, 807, 798, 795, 808, 792, 797, 808, 797, 806, 792, 808, 802, 792, 802, 793, 809, 806, 801, 808, 806, 809, 810, 811, 812, 815, 811, 816, 815, 816, 817, 810, 818, 816, 810, 816, 811, 810, 813, 819, 810, 819, 818, 813, 814, 689, 813, 689, 819, 814, 820, 689, 816, 818, 615, 816, 615, 626, 817, 816, 626, 817, 626, 627, 818, 819, 616, 818, 616, 615, 689, 616, 819, 625, 689, 820, 625, 820, 630, 627, 628, 822, 627, 822, 817, 823, 366, 633, 366, 245, 251, 245, 365, 251, 252, 824, 157, 824, 362, 157, 362, 353, 363, 353, 352, 253, 353, 253, 371, 352, 825, 360, 352, 360, 254, 826, 825, 350, 68, 85, 81, 85, 68, 73, 91, 94, 92, 112, 132, 113, 112, 827, 132, 828, 827, 138, 828, 138, 151, 829, 828, 151, 829, 151, 164, 829, 830, 179, 829, 179, 159, 830, 185, 179, 185, 189, 196, 194, 181, 184, 194, 184, 195, 181, 831, 178, 181, 178, 184, 158, 831, 173, 158, 173, 166, 158, 832, 145, 131, 832, 152, 131, 152, 139, 131, 139, 123, 131, 91, 113, 91, 94, 97, 69, 833, 85, 69, 85, 73, 69, 77, 833, 834, 823, 633, 834, 633, 649, 834, 835, 645, 834, 645, 647, 836, 835, 568, 836, 568, 567, 580, 836, 567, 580, 567, 569, 837, 580, 569, 837, 569, 570, 582, 837, 570, 582, 570, 571, 582, 838, 584, 585, 838, 572, 585, 572, 573, 839, 588, 587, 839, 587, 585, 588, 564, 574, 588, 575, 590, 575, 840, 799, 575, 799, 590, 647, 357, 366, 647, 366, 823, 357, 245, 366, 245, 358, 365, 358, 241, 252, 358, 252, 365, 241, 248, 824, 241, 824, 252, 248, 354, 362, 248, 362, 824, 354, 353, 362, 371, 363, 353, 254, 253, 352, 352, 350, 825, 76, 77, 69, 94, 112, 91, 113, 91, 112, 129, 138, 827, 129, 827, 112, 146, 132, 827, 146, 827, 828, 159, 146, 828, 159, 828, 829, 164, 170, 830, 164, 830, 829, 170, 188, 185, 170, 185, 830, 188, 189, 185, 195, 196, 189, 195, 189, 194, 181, 173, 831, 158, 178, 831, 166, 152, 832, 166, 832, 158, 144, 145, 832, 144, 832, 131, 123, 115, 91, 123, 91, 131, 115, 97, 91, 52, 55, 841, 842, 54, 843, 845, 39, 846, 39, 41, 847, 39, 847, 846, 41, 42, 848, 41, 848, 847, 42, 43, 849, 42, 849, 848, 647, 823, 834, 649, 568, 835, 649, 835, 834, 576, 645, 835, 576, 835, 836, 577, 576, 836, 577, 836, 580, 582, 580, 837, 571, 572, 838, 571, 838, 582, 585, 584, 838, 573, 839, 585, 573, 564, 588, 573, 588, 839, 574, 575, 588, 575, 793, 840, 793, 803, 850, 793, 850, 840, 712, 851, 852, 712, 852, 713, 719, 853, 851, 719, 851, 712, 722, 854, 853, 722, 853, 719, 728, 855, 854, 728, 854, 722, 731, 856, 855, 731, 855, 728, 736, 857, 856, 736, 856, 731, 764, 858, 756, 756, 746, 745, 787, 746, 859, 787, 859, 784, 713, 860, 858, 713, 858, 764, 784, 859, 789, 784, 789, 795, 851, 714, 715, 851, 715, 852, 853, 720, 714, 853, 714, 851, 854, 720, 853, 855, 729, 720, 855, 720, 854, 856, 734, 729, 856, 729, 855, 857, 739, 734, 857, 734, 856, 858, 758, 756, 756, 758, 746, 746, 770, 784, 746, 784, 859, 860, 788, 758, 860, 758, 858, 743, 705, 763, 743, 763, 754, 705, 708, 764, 705, 764, 763, 708, 713, 764, 860, 713, 852, 788, 860, 852, 788, 852, 715, 715, 711, 749, 715, 749, 788, 711, 750, 749, 711, 710, 861, 711, 861, 750, 19, 7, 12, 20, 3, 2, 20, 2, 21, 5, 6, 12, 8, 7, 19, 865, 864, 499, 72, 18, 17, 72, 17, 71, 76, 84, 80, 866, 19, 6, 848, 849, 867, 846, 868, 845, 78, 83, 82, 79, 78, 82, 79, 81, 77, 77, 81, 85, 77, 85, 833, 90, 869, 91, 342, 33, 51, 868, 844, 845, 69, 70, 76, 550, 324, 382, 3, 20, 343, 343, 20, 33, 866, 862, 863, 383, 382, 324, 383, 324, 378, 530, 515, 526, 526, 524, 530, 524, 540, 530, 546, 540, 524, 541, 533, 537, 541, 537, 544, 844, 870, 40, 871, 841, 52, 124, 84, 76, 124, 76, 75, 83, 98, 89, 83, 89, 82, 202, 106, 872, 75, 873, 124, 874, 872, 106, 874, 106, 93, 93, 203, 874, 875, 876, 75, 875, 75, 74, 877, 203, 94, 94, 878, 877, 879, 878, 94, 879, 94, 91, 91, 869, 880, 75, 86, 88, 75, 88, 873, 88, 881, 880, 75, 876, 86, 499, 16, 14, 843, 40, 870, 843, 870, 882, 882, 842, 843, 842, 883, 55, 883, 871, 841, 883, 841, 55, 19, 865, 499, 887, 888, 889, 884, 890, 889, 884, 889, 888, 891, 890, 884, 891, 884, 885, 826, 892, 360, 826, 360, 825, 879, 895, 878, 864, 17, 499, 896, 348, 893, 898, 348, 896, 877, 895, 899, 877, 899, 897, 898, 349, 348, 888, 900, 884, 890, 901, 889, 901, 887, 889, 901, 902, 886, 901, 886, 887, 902, 841, 871, 902, 871, 886, 904, 905, 318, 51, 901, 342, 901, 906, 342, 906, 340, 342, 450, 908, 867, 200, 909, 908, 200, 908, 450, 47, 910, 49, 45, 61, 911, 45, 911, 46, 46, 911, 47, 49, 910, 909, 49, 909, 200, 318, 905, 912, 318, 912, 913, 913, 912, 914, 913, 914, 57, 57, 914, 58, 900, 915, 884, 915, 916, 885, 915, 885, 884, 885, 917, 918, 919, 895, 879, 896, 893, 894, 877, 897, 898, 877, 898, 920, 877, 878, 895, 898, 350, 349, 875, 864, 865, 875, 865, 863, 876, 875, 863, 876, 863, 862, 921, 87, 876, 921, 876, 862, 897, 350, 898, 899, 923, 350, 899, 350, 897, 899, 892, 923, 348, 498, 893, 899, 895, 924, 899, 924, 892, 925, 892, 924, 917, 885, 916, 917, 916, 926, 255, 46, 48, 255, 48, 442, 442, 48, 50, 442, 50, 452, 63, 56, 59, 63, 59, 60, 45, 44, 59, 45, 59, 58, 913, 57, 56, 913, 56, 62, 452, 50, 201, 452, 201, 504, 903, 904, 316, 903, 316, 315, 904, 318, 316, 907, 903, 315, 907, 315, 324, 318, 913, 62, 926, 916, 334, 926, 334, 333, 916, 915, 347, 916, 347, 334, 488, 449, 451, 488, 451, 489, 504, 201, 449, 504, 449, 488, 900, 550, 347, 900, 347, 915, 900, 907, 324, 900, 324, 550, 333, 207, 206, 333, 206, 361, 19, 499, 8, 863, 865, 19, 863, 19, 866, 17, 16, 499, 891, 885, 918, 880, 881, 922, 880, 922, 879, 87, 86, 876, 875, 74, 864, 70, 74, 875, 74, 927, 864, 203, 877, 920, 70, 927, 74, 71, 927, 70, 71, 864, 927, 864, 71, 17, 874, 203, 920, 874, 920, 898, 872, 874, 898, 872, 898, 896, 896, 894, 202, 896, 202, 872, 88, 880, 869, 40, 843, 54, 394, 928, 390, 394, 390, 389, 930, 395, 397, 930, 397, 931, 932, 930, 931, 932, 931, 933, 934, 932, 933, 934, 933, 935, 936, 934, 935, 936, 935, 937, 481, 936, 937, 481, 937, 938, 464, 939, 940, 464, 940, 938, 931, 397, 338, 937, 935, 399, 937, 399, 400, 938, 937, 400, 938, 400, 464, 402, 929, 389, 402, 389, 290, 933, 931, 338, 933, 338, 398, 935, 933, 398, 935, 398, 399, 339, 397, 929, 339, 929, 402, 388, 463, 939, 388, 939, 464, 880, 879, 91, 941, 881, 942, 879, 922, 941, 879, 941, 919, 881, 941, 922, 88, 943, 881, 941, 944, 919, 944, 924, 895, 944, 895, 919, 923, 826, 350, 923, 892, 826, 204, 498, 348, 361, 360, 892, 361, 892, 925, 918, 945, 946, 947, 948, 949, 947, 949, 950, 925, 924, 944, 925, 944, 951, 948, 862, 949, 925, 951, 917, 925, 917, 926, 921, 948, 947, 862, 948, 921, 951, 945, 918, 951, 918, 917, 361, 925, 926, 361, 926, 333, 941, 945, 951, 941, 951, 944, 945, 941, 942, 945, 942, 946, 950, 949, 340, 950, 340, 906, 341, 340, 949, 949, 952, 344, 949, 344, 341, 11, 344, 952, 11, 952, 953, 953, 4, 11, 887, 907, 900, 887, 900, 888, 887, 886, 903, 887, 903, 907, 886, 871, 904, 886, 904, 903, 905, 904, 871, 912, 905, 871, 912, 871, 883, 914, 912, 883, 914, 883, 842, 58, 914, 842, 58, 842, 882, 533, 528, 537, 513, 11, 953, 513, 953, 512, 882, 870, 61, 882, 61, 58, 911, 61, 870, 911, 870, 844, 47, 911, 844, 47, 844, 868, 910, 47, 868, 910, 868, 846, 846, 847, 909, 846, 909, 910, 847, 848, 908, 847, 908, 909, 848, 867, 908, 536, 535, 545, 952, 949, 862, 952, 862, 866, 953, 952, 866, 953, 866, 6, 8, 499, 14, 542, 538, 545, 548, 542, 543, 547, 548, 543, 547, 543, 541, 544, 547, 541, 528, 521, 537, 521, 528, 513, 513, 512, 521, 512, 953, 12, 506, 12, 953, 506, 953, 4, 5, 506, 4, 4, 6, 5, 6, 4, 953, 509, 524, 526, 509, 526, 515, 918, 950, 891, 881, 943, 942, 946, 942, 950, 946, 950, 918, 99, 96, 89, 101, 869, 90, 92, 93, 106, 108, 119, 177, 97, 109, 100, 100, 90, 97, 90, 100, 101, 869, 84, 124, 869, 101, 84, 124, 873, 88, 124, 88, 869, 88, 87, 943, 87, 921, 942, 87, 942, 943, 921, 947, 942, 950, 942, 947, 950, 906, 891, 99, 89, 98, 114, 101, 99, 114, 99, 98, 99, 101, 100, 99, 100, 102, 84, 954, 137, 954, 153, 150, 954, 150, 137, 153, 163, 150, 163, 174, 169, 174, 182, 169, 182, 194, 199, 84, 101, 114, 98, 83, 84, 98, 84, 114, 190, 199, 194, 190, 191, 199, 191, 193, 197, 191, 197, 199, 193, 192, 198, 193, 198, 197, 174, 180, 182, 174, 163, 165, 163, 153, 165, 153, 954, 140, 954, 84, 124, 954, 124, 140, 115, 124, 84, 115, 84, 114, 114, 109, 97, 114, 97, 115, 890, 891, 906, 901, 890, 906, 902, 901, 51, 841, 902, 51, 841, 51, 52, 55, 54, 842, 40, 34, 844, 34, 35, 845, 34, 845, 844, 35, 39, 845, 387, 386, 430, 387, 430, 428, 423, 433, 385, 423, 385, 384, 425, 434, 435, 425, 435, 426, 418, 432, 433, 418, 433, 423, 467, 468, 431, 467, 431, 417, 496, 492, 434, 496, 434, 425, 428, 430, 492, 428, 492, 496, 376, 374, 468, 468, 467, 502, 484, 414, 326, 484, 326, 325, 483, 414, 484, 456, 955, 60, 956, 471, 458, 956, 458, 957, 323, 958, 957, 323, 957, 458, 958, 959, 960, 958, 960, 957, 955, 961, 962, 963, 956, 957, 963, 957, 960, 959, 964, 965, 959, 965, 960, 961, 966, 319, 961, 319, 962, 967, 963, 960, 967, 960, 965, 939, 968, 969, 939, 969, 940, 462, 970, 968, 462, 968, 463, 968, 939, 463, 462, 370, 551, 462, 551, 970, 500, 414, 483, 319, 62, 962, 955, 962, 62, 955, 62, 63, 964, 959, 971, 323, 322, 972, 323, 972, 958, 972, 971, 959, 972, 959, 958, 63, 60, 955, 379, 315, 317, 377, 376, 973, 974, 501, 500, 974, 500, 975, 976, 378, 379, 976, 379, 977, 978, 979, 980, 978, 980, 973, 551, 383, 981, 969, 968, 970, 986, 967, 965, 977, 379, 317, 977, 317, 983, 988, 989, 990, 975, 500, 483, 975, 483, 991, 973, 376, 468, 976, 992, 981, 993, 994, 995, 996, 982, 986, 996, 986, 997, 989, 981, 998, 999, 977, 983, 987, 997, 986, 992, 977, 999, 1001, 1002, 994, 1001, 994, 1003, 992, 998, 981, 987, 1004, 997, 1004, 1005, 980, 1004, 980, 1000, 1004, 1006, 997, 985, 990, 984, 1008, 999, 983, 1008, 983, 1007, 998, 992, 999, 998, 999, 1008, 982, 996, 1007, 982, 1007, 983, 989, 998, 984, 989, 984, 990, 999, 1008, 1009, 993, 1009, 1014, 1011, 1010, 1009, 1001, 1013, 1011, 1001, 1011, 1015, 993, 999, 1009, 987, 997, 1006, 987, 1010, 996, 987, 996, 997, 996, 1010, 1009, 1009, 1007, 996, 1008, 1007, 1009, 929, 391, 394, 929, 394, 389, 397, 395, 391, 397, 391, 929, 469, 454, 455, 469, 455, 470, 469, 471, 456, 469, 456, 454, 470, 455, 461, 470, 461, 473, 493, 474, 476, 493, 476, 494, 938, 940, 482, 938, 482, 481, 443, 453, 497, 443, 497, 461, 475, 474, 505, 475, 505, 503, 503, 505, 497, 503, 497, 453, 393, 472, 928, 393, 928, 394, 448, 396, 395, 448, 395, 930, 332, 448, 930, 332, 930, 932, 328, 332, 932, 328, 932, 934, 325, 328, 934, 325, 934, 936, 484, 325, 936, 484, 936, 481, 471, 956, 955, 471, 955, 456, 956, 963, 961, 956, 961, 955, 963, 967, 966, 963, 966, 961, 940, 969, 482, 984, 985, 994, 984, 994, 993, 985, 1003, 994, 985, 995, 1003, 998, 985, 984, 1001, 1003, 995, 1004, 1000, 979, 1000, 980, 979, 1012, 1011, 1013, 1011, 1013, 1009, 1011, 1010, 987, 1011, 987, 1004, 1009, 1014, 1012, 1009, 1013, 1014, 1013, 1012, 1014, 1014, 1012, 995, 995, 993, 1014, 1011, 1004, 979, 1011, 979, 1015, 1001, 995, 1012, 995, 985, 984, 1008, 999, 1016, 999, 993, 1016, 1008, 1016, 993, 993, 998, 1008, 993, 995, 985, 993, 985, 998, 1001, 1015, 979, 1005, 1004, 973, 1005, 973, 980, 992, 976, 977, 989, 988, 981, 1004, 987, 1006, 551, 988, 990, 551, 990, 970, 1018, 987, 986, 1018, 986, 967, 1002, 1017, 995, 1002, 995, 994, 974, 978, 973, 991, 993, 995, 969, 984, 993, 969, 993, 991, 1004, 377, 973, 1017, 974, 975, 988, 551, 981, 984, 969, 970, 984, 970, 990, 1017, 975, 991, 1017, 991, 995, 502, 973, 468, 319, 982, 983, 319, 983, 317, 964, 1018, 967, 964, 967, 965, 966, 986, 982, 966, 982, 319, 502, 974, 973, 378, 976, 981, 378, 981, 383, 482, 991, 483, 967, 986, 966, 969, 991, 482, 1001, 1012, 1013, 979, 1002, 1001, 1017, 1002, 979, 1017, 979, 978, 974, 1017, 978, 974, 502, 501, 502, 466, 465, 502, 465, 501, 415, 465, 466, 415, 466, 416, 416, 419, 409, 416, 409, 415, 419, 422, 403, 419, 403, 409, 1005, 1004, 987, 377, 1004, 987, 377, 987, 1018, 1018, 964, 377, 375, 377, 964, 375, 964, 971, 310, 375, 971, 310, 971, 972, 311, 310, 972, 311, 972, 322, 313, 311, 322, 313, 322, 321, 306, 313, 321, 306, 321, 329, 307, 306, 329, 307, 329, 64, 300, 307, 64, 300, 64, 67, 301, 300, 67, 301, 67, 447, 303, 301, 447, 303, 447, 480, 380, 303, 480, 380, 480, 479, 422, 420, 404, 422, 404, 403, 420, 421, 411, 420, 411, 404, 421, 429, 412, 421, 412, 411, 490, 412, 429, 490, 429, 495, 495, 424, 406, 495, 406, 490, 424, 427, 407, 424, 407, 406, 7, 8, 10, 7, 10, 9, 1021, 1022, 1020, 1021, 1020, 1019, 1024, 1025, 1023, 1024, 1023, 1022, 1026, 1033, 1034, 1026, 1034, 1027, 1020, 1028, 1019, 1035, 1020, 1022, 1022, 1023, 1035, 1039, 1037, 1040, 1039, 1040, 18, 1038, 1040, 1037, 1038, 1037, 1036, 18, 1040, 16, 15, 1040, 1038, 15, 16, 1040, 1041, 1033, 1026, 1041, 1026, 1032, 1042, 1025, 1024, 1043, 1029, 1044, 1020, 1043, 1044, 1036, 1045, 1038, 1038, 1045, 24, 1038, 24, 15, 1047, 1043, 1020, 1036, 1043, 1046, 1049, 1048, 1023, 1048, 1051, 1035, 1052, 1053, 1043, 1052, 1043, 1047, 1053, 1054, 1046, 1053, 1046, 1043, 1055, 1049, 1050, 1057, 1052, 1047, 1057, 1047, 1020, 1056, 1048, 1049, 1051, 1057, 1020, 1051, 1020, 1035, 1036, 1046, 1054, 1036, 1054, 1045, 1058, 1051, 1048, 1059, 1060, 1052, 1059, 1052, 1057, 1053, 37, 1045, 1053, 1045, 1054, 1051, 1059, 1057, 1061, 43, 37, 1061, 37, 1053, 1062, 1063, 1064, 1062, 1064, 1065, 1063, 1066, 1067, 1063, 1067, 1064, 1066, 1068, 1069, 1066, 1069, 1067, 1055, 1070, 1049, 1070, 1056, 1049, 1073, 1056, 1070, 1058, 1048, 1056, 1058, 1056, 1072, 1062, 1075, 1076, 1062, 1076, 1077, 1065, 1078, 1075, 1079, 1080, 1081, 1079, 1081, 1074, 1082, 1083, 1084, 1082, 1084, 1085, 1089, 1087, 1088, 1087, 68, 1086, 68, 1090, 72, 1090, 68, 1087, 1088, 1091, 1092, 1090, 1087, 1089, 1096, 79, 1097, 79, 1096, 81, 1097, 79, 82, 1100, 1095, 1098, 1094, 82, 1093, 1102, 1103, 1104, 1095, 82, 1099, 1095, 1099, 1098, 1093, 82, 1095, 1105, 1106, 1107, 1103, 1108, 1104, 1109, 1104, 1108, 89, 1112, 1099, 89, 1099, 82, 1112, 1111, 1113, 1110, 1114, 1111, 1110, 1111, 1112, 95, 1114, 1110, 1114, 1115, 1111, 95, 1115, 1114, 89, 95, 1110, 1116, 1106, 1117, 1117, 1106, 1108, 1108, 1111, 1117, 105, 1118, 1117, 1116, 1117, 1118, 1116, 1118, 1119, 1118, 1117, 1111, 1108, 1106, 1105, 1108, 1105, 1109, 1111, 1108, 1115, 1115, 1118, 1111, 1115, 1108, 1109, 95, 105, 103, 95, 103, 1115, 1115, 1109, 1120, 1115, 1120, 1121, 1121, 1118, 1115, 1105, 1120, 1109, 1115, 110, 98, 1115, 98, 1123, 1124, 1120, 1105, 1109, 1108, 1126, 1109, 1126, 1125, 1121, 1120, 1127, 1128, 1127, 1120, 1123, 98, 118, 1127, 1119, 1118, 1127, 1118, 1121, 1125, 1126, 1130, 1125, 1130, 1129, 1131, 122, 121, 1131, 121, 126, 1128, 1132, 1133, 1132, 1128, 1120, 126, 130, 1134, 126, 1134, 1131, 130, 134, 1136, 130, 1136, 1134, 1123, 118, 133, 1123, 133, 1099, 1128, 136, 1119, 1128, 1119, 1127, 1123, 1099, 1130, 1124, 1132, 1120, 1136, 134, 136, 1099, 133, 135, 1133, 1136, 136, 1133, 136, 1128, 1130, 1099, 135, 1130, 135, 1137, 1138, 1132, 1124, 1129, 1139, 1125, 1129, 1130, 1140, 119, 1119, 136, 1141, 1133, 1132, 1119, 119, 143, 1119, 143, 1118, 1142, 1143, 1122, 1142, 1122, 1135, 1144, 1141, 1132, 1145, 1146, 1143, 1145, 1143, 1142, 1141, 1134, 1136, 1141, 1136, 1133, 1137, 135, 1147, 1144, 1132, 1138, 1148, 1139, 1129, 1129, 1140, 1149, 1129, 1149, 1148, 135, 147, 1147, 1151, 1150, 1152, 1153, 1150, 1141, 1153, 1141, 1144, 1154, 1155, 1146, 1154, 1146, 1145, 147, 161, 1156, 1150, 1131, 1134, 1150, 1134, 1141, 1158, 1153, 1144, 1159, 1160, 1148, 1148, 1149, 1161, 1148, 1161, 1159, 1147, 147, 1156, 1147, 1156, 1157, 1156, 161, 1162, 1156, 1162, 1163, 161, 167, 1162, 1151, 122, 1131, 1151, 1131, 1150, 1157, 1163, 1164, 1165, 1166, 1153, 1165, 1153, 1158, 1167, 1160, 1159, 1159, 1161, 1164, 1159, 1164, 1167, 1157, 1156, 1163, 1168, 1151, 1152, 1168, 1152, 1169, 1166, 1152, 1150, 1166, 1150, 1153, 1170, 1171, 1155, 1170, 1155, 1154, 1172, 1173, 1167, 1167, 1164, 1174, 1167, 1174, 1172, 1176, 1169, 1152, 1176, 1152, 1166, 1177, 1171, 1170, 1163, 1162, 1175, 1162, 167, 186, 1162, 186, 1175, 1168, 122, 1151, 1164, 1163, 1175, 1178, 1176, 1166, 1178, 1166, 1165, 1179, 1176, 1178, 192, 1168, 1169, 192, 1169, 1180, 1181, 1173, 1172, 1172, 1174, 1181, 1179, 1180, 1169, 1179, 1169, 1176, 1182, 1183, 1177, 1175, 186, 198, 1168, 192, 122, 1068, 1184, 1185, 1068, 1185, 1069, 1186, 1127, 1107, 1186, 1107, 1119, 1187, 1104, 1188, 1187, 1188, 1189, 1190, 1105, 1107, 1194, 1195, 1196, 1194, 1196, 1197, 1198, 1199, 1195, 1198, 1195, 1194, 1200, 1201, 1199, 1200, 1199, 1198, 1202, 1201, 1200, 1203, 1204, 1202, 1205, 1206, 1204, 1205, 1204, 1203, 1207, 1208, 1209, 1207, 1209, 1210, 1211, 1212, 1208, 1211, 1208, 1207, 1213, 1200, 1212, 1213, 1212, 1211, 1214, 1200, 1213, 1215, 1216, 1214, 1217, 1205, 1216, 1217, 1216, 1215, 1193, 1218, 1219, 1191, 235, 1220, 1220, 237, 1221, 1221, 237, 1222, 1221, 1222, 1223, 1223, 1222, 239, 1224, 1225, 1226, 1227, 1228, 1224, 1227, 1224, 1229, 239, 243, 1229, 1229, 243, 1227, 1229, 1224, 1226, 1229, 1226, 239, 1225, 1230, 1231, 1225, 1231, 1223, 243, 244, 1227, 1232, 1228, 1233, 1214, 1234, 1215, 1213, 1235, 1234, 1213, 1234, 1214, 237, 238, 1222, 1236, 1192, 1237, 1077, 1238, 1063, 1077, 1063, 1062, 1197, 1209, 1208, 1197, 1208, 1194, 1194, 1208, 1212, 1194, 1212, 1198, 1198, 1212, 1200, 1200, 1214, 1202, 1202, 1214, 1216, 1202, 1216, 1203, 1203, 1216, 1205, 1195, 1199, 1239, 1195, 1239, 1240, 1204, 1206, 1241, 1204, 1241, 1242, 1202, 1204, 1242, 1202, 1242, 1243, 1201, 1202, 1243, 1201, 1243, 1244, 1199, 1201, 1244, 1199, 1244, 1239, 1240, 1239, 1245, 1240, 1245, 1246, 1239, 1244, 1247, 1239, 1247, 1245, 1244, 1243, 1248, 1244, 1248, 1247, 1243, 1242, 1248, 1246, 1245, 1249, 1246, 1249, 1250, 1242, 1241, 1251, 1242, 1251, 1252, 1245, 1247, 1253, 1245, 1253, 1249, 1247, 1248, 1254, 1247, 1254, 1253, 1248, 1242, 1252, 1248, 1252, 1254, 1250, 1249, 1255, 1250, 1255, 1256, 1249, 1253, 1257, 1249, 1257, 1255, 1253, 1254, 1258, 1253, 1258, 1257, 1254, 1252, 1258, 1256, 1255, 1259, 1256, 1259, 1260, 1252, 1251, 1261, 1252, 1261, 1262, 1255, 1257, 1263, 1255, 1263, 1259, 1257, 1258, 1264, 1257, 1264, 1263, 1258, 1252, 1262, 1258, 1262, 1264, 1260, 1259, 1265, 1260, 1265, 1266, 1262, 1261, 1267, 1262, 1267, 1268, 1259, 1263, 1269, 1259, 1269, 1265, 1263, 1264, 1270, 1263, 1270, 1269, 1264, 1262, 1268, 1264, 1268, 1270, 1271, 1272, 291, 1271, 291, 290, 1273, 1274, 294, 1273, 294, 293, 1272, 1275, 296, 1272, 296, 291, 1275, 1276, 298, 1275, 298, 296, 1276, 1273, 293, 1276, 293, 298, 1277, 1278, 1279, 1277, 1279, 1280, 1278, 1281, 1282, 1278, 1282, 1279, 1283, 1284, 1285, 1283, 1285, 1286, 1287, 1288, 1289, 1287, 1289, 1290, 1284, 1277, 1280, 1284, 1280, 1285, 1288, 1291, 1292, 1288, 1292, 1289, 1293, 1294, 1295, 1293, 1295, 1296, 1296, 1295, 1297, 1298, 1299, 1300, 1298, 1300, 1301, 1302, 1303, 1294, 1302, 1294, 1293, 1304, 1305, 1306, 1304, 1306, 1307, 1297, 1080, 1079, 1308, 1082, 1085, 1308, 1085, 1309, 1301, 1308, 1309, 1301, 1309, 1298, 1306, 1305, 1310, 1306, 1310, 1311, 1312, 1193, 1219, 1312, 1219, 1313, 1195, 1240, 1314, 1195, 1314, 1196, 1246, 1315, 1314, 1246, 1314, 1240, 1250, 1316, 1315, 1250, 1315, 1246, 1256, 1317, 1316, 1256, 1316, 1250, 1260, 1318, 1317, 1260, 1317, 1256, 1266, 1319, 1318, 1266, 1318, 1260, 1320, 1321, 1322, 1042, 1323, 1025, 1321, 1323, 1322, 1023, 1048, 1035, 1025, 1050, 1049, 1025, 1049, 1023, 1031, 1036, 1037, 1031, 1037, 1030, 1029, 1043, 1036, 1036, 1031, 1029, 1218, 1324, 1325, 1218, 1325, 1219, 1313, 1219, 1325, 1313, 1325, 1326, 1222, 238, 239, 1227, 244, 247, 1328, 1327, 1329, 1331, 1332, 1333, 1226, 1225, 1223, 239, 1226, 1223, 1227, 247, 1233, 1227, 1233, 1228, 1334, 1228, 1232, 1334, 1232, 1335, 1225, 1224, 1228, 1225, 1228, 1334, 1224, 1225, 1230, 1332, 1336, 1221, 1332, 1221, 1333, 1332, 1328, 1329, 1332, 1331, 1330, 1237, 1192, 1337, 1152, 1338, 1339, 1152, 1339, 1340, 1234, 1341, 1215, 1215, 1341, 1342, 1215, 1342, 1217, 1343, 1339, 1210, 1197, 1196, 1344, 1345, 1340, 1339, 1345, 1339, 1343, 1197, 1324, 1218, 1197, 1218, 1346, 1343, 1210, 1209, 1343, 1209, 1346, 1347, 1344, 1196, 1347, 1196, 1314, 1348, 1287, 1290, 1348, 1290, 1349, 1350, 1348, 1349, 1350, 1349, 1351, 1352, 1294, 1303, 1281, 381, 380, 1281, 380, 1282, 1291, 1283, 1286, 1291, 1286, 1292, 1324, 1353, 1354, 1324, 1354, 1325, 1355, 1356, 1357, 1355, 1357, 1358, 1359, 1347, 1314, 1359, 1314, 1315, 1360, 1271, 290, 1360, 290, 390, 1361, 1362, 1363, 1361, 1363, 1364, 1365, 1361, 1364, 1365, 1364, 1366, 1367, 1318, 1319, 1367, 1319, 1368, 1369, 1317, 1318, 1369, 1318, 1367, 1370, 1316, 1317, 1370, 1317, 1369, 1371, 1315, 1316, 1371, 1316, 1370, 1372, 1319, 1266, 1306, 1311, 1373, 1306, 1373, 1374, 1375, 408, 407, 1375, 407, 1376, 1307, 1306, 1374, 1307, 1374, 1377, 1378, 1379, 1380, 1378, 1380, 1381, 1382, 1307, 1377, 1382, 1377, 1383, 1311, 1378, 1381, 1311, 1381, 1373, 1384, 1385, 1386, 1384, 1386, 1387, 1388, 1389, 1356, 1388, 1356, 1355, 1390, 1388, 1355, 1390, 1355, 1391, 1392, 427, 426, 1392, 426, 1393, 1385, 1390, 1391, 1385, 1391, 1386, 1389, 1394, 1395, 1389, 1395, 1356, 1357, 1396, 1277, 1357, 1277, 1284, 1397, 1398, 1288, 1397, 1288, 1287, 1358, 1357, 1284, 1358, 1284, 1283, 1399, 1358, 1283, 1399, 1283, 1291, 1400, 435, 381, 1400, 381, 1281, 1398, 1399, 1291, 1398, 1291, 1288, 1401, 1402, 438, 1401, 438, 437, 1403, 1085, 1084, 1403, 1084, 1404, 1238, 1405, 1406, 1238, 1406, 1407, 1408, 1077, 1076, 1408, 1076, 1409, 1408, 1405, 1238, 1408, 1238, 1077, 1410, 1084, 1083, 1410, 1083, 1411, 1311, 1310, 1412, 1311, 1412, 1378, 1378, 1412, 1366, 1378, 1366, 1379, 1184, 451, 450, 1184, 450, 1413, 1406, 1414, 1415, 1406, 1415, 1407, 1184, 1413, 1185, 1416, 1417, 1405, 1416, 1405, 1408, 1416, 1408, 1409, 1416, 1409, 1418, 1366, 1364, 1419, 1366, 1419, 1379, 1420, 1299, 1298, 1420, 1298, 1421, 1421, 1298, 1309, 1421, 1309, 1422, 1417, 1423, 1406, 1417, 1406, 1405, 1364, 1363, 1375, 1364, 1375, 1419, 1332, 1329, 1336, 1328, 1332, 1330, 1192, 1343, 1193, 1324, 1197, 1344, 1324, 1344, 1424, 1236, 1345, 1343, 1236, 1343, 1192, 1197, 1346, 1209, 1193, 1343, 1346, 1193, 1346, 1218, 1425, 1424, 1344, 1425, 1344, 1347, 1426, 1425, 1347, 1426, 1347, 1359, 1427, 1359, 1315, 1427, 1315, 1371, 1428, 1382, 1383, 1428, 1383, 1429, 1430, 1384, 1387, 1430, 1387, 1431, 1432, 1397, 1287, 1432, 1287, 1348, 1421, 1422, 1433, 1421, 1433, 1434, 1421, 1434, 1435, 1421, 1435, 1420, 1363, 472, 408, 1363, 408, 1375, 1422, 1403, 1436, 1422, 1436, 1433, 1437, 476, 475, 1437, 475, 1438, 1439, 1440, 1276, 1439, 1276, 1275, 438, 1402, 1441, 438, 1441, 479, 1442, 1443, 1444, 1445, 1446, 1447, 1445, 1447, 1404, 1060, 1061, 1053, 1060, 1053, 1052, 1438, 475, 489, 1419, 1375, 1376, 1419, 1376, 1448, 1404, 1447, 1436, 1404, 1436, 1403, 1449, 1439, 1275, 1449, 1275, 1272, 1450, 1400, 1281, 1450, 1281, 1278, 1422, 1309, 1085, 1422, 1085, 1403, 1451, 1401, 437, 1451, 437, 494, 1452, 1274, 1273, 1372, 1449, 1272, 1372, 1272, 1271, 1453, 1392, 1393, 1453, 1393, 1454, 1440, 1452, 1273, 1440, 1273, 1276, 1445, 1410, 1402, 1445, 1402, 1401, 1423, 1436, 1447, 1423, 1447, 1455, 1127, 1190, 1107, 1327, 1191, 1329, 1329, 1191, 1220, 1329, 1220, 1336, 1221, 1336, 1220, 1220, 235, 237, 235, 1191, 204, 204, 1191, 498, 1456, 1030, 1039, 1457, 1327, 1328, 1327, 1458, 1191, 1458, 498, 1191, 1088, 1095, 1091, 1152, 1211, 1338, 1210, 1339, 1338, 1210, 1338, 1207, 1207, 1338, 1211, 1333, 1223, 1231, 1221, 1223, 1333, 1152, 1235, 1211, 1211, 1235, 1213, 1429, 1459, 1460, 1429, 1460, 1428, 1461, 1430, 1431, 1387, 1386, 1398, 1387, 1398, 1397, 489, 1462, 1463, 489, 1463, 1438, 1462, 1415, 1414, 1462, 1414, 1463, 1402, 1410, 1411, 1402, 1411, 1441, 1269, 1270, 1440, 1269, 1440, 1439, 1401, 1451, 1446, 1401, 1446, 1445, 1379, 1419, 1448, 1379, 1448, 1380, 1265, 1269, 1439, 1265, 1439, 1449, 1396, 1450, 1278, 1396, 1278, 1277, 1268, 1267, 1274, 1268, 1274, 1452, 1266, 1265, 1449, 1266, 1449, 1372, 1394, 1453, 1454, 1394, 1454, 1395, 1270, 1268, 1452, 1270, 1452, 1440, 1404, 1084, 1410, 1404, 1410, 1445, 1437, 1464, 1446, 1437, 1446, 1451, 1464, 1455, 1447, 1464, 1447, 1446, 1465, 1022, 1021, 1030, 1467, 1031, 1019, 1465, 1021, 1023, 1469, 1022, 1465, 1023, 1022, 1466, 1467, 1028, 1466, 1028, 1026, 1470, 1022, 1465, 1044, 1020, 1019, 1044, 1019, 1028, 1465, 1019, 1020, 1465, 1020, 1470, 1044, 1471, 1020, 1470, 1020, 1471, 1472, 1044, 1028, 1473, 1026, 1027, 1474, 1475, 1468, 1474, 1468, 1024, 1467, 1472, 1028, 1473, 1476, 1026, 1477, 1022, 1470, 1477, 1023, 1022, 1471, 1477, 1470, 1478, 1476, 1473, 1469, 1479, 1023, 1478, 1481, 1466, 1482, 1474, 1025, 1483, 1480, 1467, 1483, 1467, 1466, 1477, 1469, 1023, 1472, 1471, 1044, 1471, 1484, 1485, 1486, 1483, 1466, 1487, 1471, 1472, 1488, 1478, 1473, 1482, 1475, 1474, 1485, 1477, 1471, 1480, 1487, 1472, 1480, 1472, 1467, 1486, 1489, 1480, 1486, 1480, 1483, 1487, 1484, 1471, 1477, 1479, 1469, 1477, 1490, 1479, 1482, 1491, 1475, 1485, 1492, 1477, 1478, 1486, 1481, 1480, 1489, 1493, 1480, 1493, 1487, 1025, 1474, 1024, 1478, 1488, 1494, 1484, 1487, 1495, 1491, 1482, 1496, 1485, 1484, 1495, 1496, 1482, 1498, 1477, 1492, 1499, 1478, 1494, 1500, 1490, 1477, 1499, 1495, 1487, 1497, 1489, 1486, 1501, 1479, 1490, 1499, 1479, 1499, 1502, 1485, 1495, 1497, 1493, 1489, 1501, 1493, 1501, 1503, 1487, 1493, 1503, 1492, 1485, 1497, 1492, 1497, 1499, 1482, 1479, 1502, 1482, 1502, 1498, 1497, 1487, 1503, 1486, 1500, 1501, 1505, 1506, 1498, 1503, 1505, 1507, 1507, 1505, 1499, 1505, 1502, 1499, 1506, 1508, 1500, 1507, 1497, 1503, 1508, 1501, 1500, 1505, 1498, 1502, 1501, 1508, 1505, 1503, 1501, 1505, 1507, 1499, 1497, 1506, 1500, 1494, 1506, 1494, 1504, 1504, 1496, 1506, 1321, 1320, 1071, 1509, 1326, 1325, 1509, 1325, 1354, 1353, 1324, 1424, 1353, 1424, 1510, 1323, 1321, 1025, 1024, 1022, 1469, 1024, 1469, 1468, 1468, 1469, 1025, 1468, 1025, 1024, 1511, 1512, 1513, 1511, 1513, 1514, 1514, 1513, 1515, 1514, 1515, 1516, 1516, 1515, 1517, 1516, 1517, 1518, 1518, 1517, 1519, 1518, 1519, 1520, 1521, 1522, 1523, 1523, 1522, 1524, 1523, 1524, 1525, 1512, 1527, 1528, 1512, 1528, 1513, 1513, 1528, 1529, 1513, 1529, 1515, 1515, 1529, 1530, 1515, 1530, 1517, 1517, 1530, 1531, 1517, 1531, 1519, 1522, 1532, 1524, 1524, 1533, 1525, 1534, 1535, 1536, 1534, 1536, 1537, 1537, 1538, 1539, 1539, 1538, 1540, 1539, 1540, 1541, 1541, 1542, 1543, 1541, 1543, 1544, 1544, 1543, 1545, 1544, 1545, 1546, 1544, 1546, 1547, 1544, 1547, 1548, 1549, 1550, 1551, 1535, 1552, 1553, 1535, 1553, 1536, 1536, 1553, 1554, 1536, 1554, 1537, 1537, 1554, 1555, 1537, 1555, 1538, 1538, 1555, 1556, 1538, 1556, 1540, 1543, 1557, 1545, 1545, 1557, 1558, 1546, 1559, 1547, 1552, 1560, 1561, 1552, 1561, 1553, 1553, 1561, 1562, 1553, 1562, 1554, 1554, 1562, 1563, 1554, 1563, 1555, 1555, 1563, 1556, 1543, 1564, 1565, 1543, 1565, 1566, 1543, 1566, 1557, 1546, 1558, 1567, 1560, 1568, 1569, 1560, 1569, 1561, 1561, 1569, 1562, 1562, 1570, 1563, 1563, 1570, 1556, 1564, 1571, 1565, 1565, 1571, 1572, 1565, 1572, 1566, 1566, 1572, 1558, 1535, 1549, 1551, 1535, 1551, 1552, 1568, 1573, 1569, 1569, 1573, 1574, 1569, 1574, 1562, 1562, 1574, 1575, 1562, 1575, 1570, 1570, 1575, 1576, 1564, 1577, 1578, 1564, 1578, 1571, 1571, 1578, 1579, 1571, 1579, 1572, 623, 625, 1580, 1568, 1581, 1582, 1568, 1582, 1573, 1573, 1582, 1574, 1574, 1583, 1575, 1575, 1583, 1576, 1577, 1584, 1585, 1577, 1585, 1578, 625, 630, 1586, 625, 1586, 1580, 631, 623, 1580, 1587, 1588, 1217, 1587, 1217, 1342, 1589, 1511, 1514, 1589, 1514, 1590, 1590, 1514, 1516, 1590, 1516, 1591, 1591, 1516, 1518, 1591, 1518, 1592, 1592, 1518, 1520, 1592, 1520, 1593, 1521, 1523, 1594, 1594, 1523, 1525, 1594, 1525, 1595, 1595, 1525, 1526, 600, 631, 1596, 600, 1596, 1550, 1580, 1586, 1597, 1598, 1599, 1549, 1598, 1549, 1535, 631, 1580, 1596, 1599, 1550, 1549, 1600, 1598, 1535, 1600, 1535, 1534, 1601, 1602, 1232, 1601, 1232, 1233, 1601, 1233, 247, 1602, 1603, 1335, 1602, 1335, 1232, 1604, 1605, 1512, 1604, 1512, 1511, 1588, 1606, 1205, 1588, 1205, 1217, 1607, 1608, 1251, 1609, 1610, 1274, 1611, 1607, 1251, 1611, 1251, 1241, 1610, 657, 294, 1610, 294, 1274, 1274, 1267, 1612, 1274, 1612, 1609, 1613, 1611, 1241, 1614, 1615, 1206, 1614, 1206, 1205, 1616, 1612, 1267, 1616, 1267, 1261, 1615, 1613, 1241, 1615, 1241, 1206, 1608, 1616, 1261, 1608, 1261, 1251, 1617, 1618, 1613, 1617, 1613, 1615, 1619, 1620, 1612, 1619, 1612, 1616, 1621, 1617, 1615, 1621, 1615, 1614, 1622, 1619, 1616, 1622, 1616, 1608, 1623, 1622, 1608, 1623, 1608, 1607, 1624, 1623, 1607, 1624, 1607, 1611, 1625, 1626, 1610, 1625, 1610, 1609, 1618, 1624, 1611, 1618, 1611, 1613, 1609, 1612, 1620, 1609, 1620, 1625, 1626, 672, 657, 1626, 657, 1610, 1627, 1628, 1622, 1627, 1622, 1623, 1628, 1629, 1619, 1628, 1619, 1622, 1629, 1630, 1620, 1629, 1620, 1619, 1631, 1632, 1617, 1631, 1617, 1621, 1625, 1620, 1630, 1625, 1630, 1633, 1632, 1634, 1618, 1632, 1618, 1617, 1633, 1635, 1626, 1633, 1626, 1625, 1634, 1636, 1624, 1634, 1624, 1618, 1636, 1627, 1623, 1636, 1623, 1624, 1637, 1605, 1588, 1637, 1588, 1587, 1638, 1639, 1602, 1638, 1602, 1601, 1639, 1640, 1603, 1639, 1603, 1602, 1604, 1641, 1606, 1604, 1606, 1588, 1527, 1512, 1605, 1527, 1605, 1637, 1614, 1606, 1641, 1614, 1641, 1621, 1511, 1589, 1641, 1511, 1641, 1604, 1598, 1600, 1640, 1598, 1640, 1639, 1588, 1605, 1604, 1599, 1638, 1601, 1599, 1598, 1639, 1599, 1639, 1638, 1621, 1641, 1589, 1621, 1589, 1631, 1614, 1205, 1606, 1642, 687, 600, 1596, 1580, 1644, 1596, 1644, 1643, 1580, 1597, 1644, 1550, 1596, 1643, 1550, 1643, 1551, 1601, 247, 685, 1601, 1642, 1599, 1599, 1642, 600, 1599, 600, 1550, 1551, 1643, 1560, 1551, 1560, 1552, 1644, 1581, 1568, 1643, 1644, 1568, 1643, 1568, 1560, 1645, 691, 674, 1645, 674, 1635, 1646, 1647, 1634, 1646, 1634, 1632, 1633, 1630, 1648, 1633, 1648, 1649, 1647, 1650, 1636, 1647, 1636, 1634, 1649, 1645, 1635, 1649, 1635, 1633, 1650, 1651, 1627, 1650, 1627, 1636, 1651, 1652, 1628, 1651, 1628, 1627, 1652, 1653, 1629, 1652, 1629, 1628, 1653, 1648, 1630, 1653, 1630, 1629, 1654, 1646, 1632, 1654, 1632, 1631, 1655, 1656, 1652, 1655, 1652, 1651, 1656, 1657, 1653, 1656, 1653, 1652, 1657, 1658, 1648, 1657, 1648, 1653, 1659, 1660, 1646, 1659, 1646, 1654, 1661, 691, 1645, 1660, 1662, 1647, 1660, 1647, 1646, 1649, 1648, 1658, 1649, 1658, 1645, 1662, 1663, 1650, 1662, 1650, 1647, 1663, 1655, 1651, 1663, 1651, 1650, 1656, 1664, 1665, 1656, 1665, 1666, 1667, 1668, 1669, 1667, 1669, 1670, 1666, 1665, 1671, 1666, 1671, 1672, 1673, 1674, 1675, 1673, 1675, 1676, 1676, 1675, 1668, 1676, 1668, 1667, 1677, 1656, 1666, 1677, 1666, 1678, 1679, 1667, 1670, 1678, 1666, 1672, 1678, 1672, 1680, 1673, 1676, 1681, 1681, 1676, 1667, 1681, 1667, 1679, 1682, 1677, 1678, 1683, 1679, 1670, 1683, 1670, 1684, 1682, 1678, 1680, 1682, 1680, 1685, 1686, 1673, 1681, 1686, 1681, 1687, 1687, 1681, 1679, 1687, 1679, 1683, 1688, 1682, 1689, 1690, 1683, 1684, 1690, 1684, 1691, 1689, 1682, 1685, 1689, 1685, 1692, 1693, 1686, 1687, 1693, 1687, 1694, 1694, 1687, 1683, 1694, 1683, 1690, 1695, 1688, 1689, 1695, 1689, 1696, 1696, 1689, 1692, 735, 1695, 1696, 735, 1696, 736, 737, 1690, 1691, 737, 1691, 738, 739, 1694, 1690, 739, 1690, 737, 1664, 1656, 1655, 1656, 1677, 1657, 1677, 1682, 1658, 1677, 1658, 1657, 1645, 1658, 1682, 1645, 1682, 1688, 1688, 1695, 1661, 1688, 1661, 1645, 1695, 735, 691, 1695, 691, 1661, 1591, 1659, 1654, 1591, 1654, 1590, 1590, 1654, 1631, 1590, 1631, 1589, 1697, 1698, 1660, 1697, 1660, 1659, 1698, 1699, 1662, 1698, 1662, 1660, 1699, 1700, 1663, 1699, 1663, 1662, 1701, 1702, 1703, 1704, 1705, 1698, 1704, 1698, 1697, 1706, 1707, 1708, 1706, 1708, 1709, 1705, 1710, 1699, 1705, 1699, 1698, 1710, 1711, 1700, 1710, 1700, 1699, 1712, 1701, 1703, 1712, 1703, 1713, 1707, 1714, 1715, 1707, 1715, 1716, 1717, 1703, 1714, 1717, 1714, 1707, 1718, 1719, 1705, 1718, 1705, 1704, 1707, 1716, 1708, 1719, 1720, 1710, 1719, 1710, 1705, 1720, 1721, 1711, 1720, 1711, 1710, 1721, 1722, 1700, 1721, 1700, 1711, 1722, 1723, 1724, 1722, 1724, 1700, 1718, 1725, 1719, 1725, 1719, 1720, 1719, 1720, 1721, 1720, 1721, 1722, 1721, 1712, 1723, 1721, 1723, 1722, 1726, 1702, 1701, 1714, 1727, 1567, 1714, 1567, 1715, 1728, 1729, 1730, 1732, 1702, 1726, 1727, 1733, 1567, 1734, 1735, 1729, 1734, 1729, 1728, 1735, 1736, 1731, 1735, 1731, 1729, 1546, 1545, 1558, 1736, 1726, 1731, 1736, 1732, 1726, 1737, 1738, 1732, 1737, 1732, 1736, 1738, 1739, 1732, 1727, 1740, 1741, 1727, 1741, 1733, 1733, 1741, 1742, 1733, 1742, 1567, 1734, 1743, 1736, 1734, 1736, 1735, 1567, 1742, 1559, 1567, 1559, 1546, 1743, 1737, 1736, 1572, 1744, 1745, 1558, 1557, 1566, 1558, 1572, 1745, 1723, 1712, 1713, 1723, 1713, 1746, 1730, 1729, 1725, 1730, 1725, 1718, 1729, 1731, 1719, 1729, 1719, 1725, 1715, 1567, 1558, 1731, 1726, 1720, 1731, 1720, 1719, 1726, 1701, 1721, 1726, 1721, 1720, 1701, 1712, 1721, 1556, 1570, 1564, 1540, 1556, 1543, 1540, 1543, 1542, 1520, 1522, 1521, 1593, 1520, 1521, 1556, 1564, 1543, 1520, 1519, 1532, 1520, 1532, 1522, 1541, 1540, 1542, 1576, 1583, 1584, 1576, 1584, 1577, 1570, 1576, 1577, 1570, 1577, 1564, 1716, 1715, 1558, 1716, 1558, 1745, 1592, 1697, 1659, 1592, 1659, 1591, 1592, 1704, 1697, 1704, 1592, 1593, 1704, 1593, 1718, 1521, 1718, 1593, 1718, 1521, 1594, 1718, 1594, 1730, 1730, 1594, 1728, 1595, 1734, 1728, 1595, 1728, 1594, 1734, 1595, 1526, 1745, 1744, 1716, 1732, 1739, 1702, 1747, 1717, 1707, 1747, 1707, 1706, 1746, 1724, 1723, 1740, 1741, 1748, 1749, 1750, 1734, 1526, 1525, 1751, 1526, 1751, 1752, 1741, 1748, 1742, 1753, 1739, 1754, 1750, 1755, 1743, 1750, 1743, 1734, 1525, 1533, 1756, 1525, 1756, 1751, 1755, 1737, 1743, 1752, 1749, 1526, 1755, 1754, 1738, 1755, 1738, 1737, 1754, 1739, 1738, 1749, 1734, 1526, 1548, 1547, 1757, 1740, 1753, 1741, 1547, 1559, 1758, 1547, 1758, 1757, 1742, 1758, 1559, 1759, 1760, 1750, 1759, 1750, 1749, 1751, 1756, 1761, 1751, 1761, 1762, 1760, 1763, 1755, 1760, 1755, 1750, 1763, 1764, 1755, 1765, 1759, 1749, 1764, 1754, 1755, 1766, 1765, 1749, 1766, 1749, 1752, 1752, 1751, 1762, 1752, 1762, 1766, 1765, 1767, 1759, 1767, 1760, 1759, 1768, 1769, 1770, 1768, 1770, 1771, 1773, 1774, 1769, 1770, 1769, 1774, 1770, 1775, 1771, 1771, 1775, 1776, 1771, 1776, 1772, 1772, 1776, 821, 1774, 1574, 1582, 1774, 1582, 1770, 1774, 1583, 1574, 1770, 1582, 1581, 1770, 1581, 1775, 1644, 1776, 1775, 1644, 1775, 1581, 1597, 821, 1776, 1597, 1776, 1644, 630, 821, 1586, 1586, 821, 1597, 1583, 1774, 1777, 1342, 1341, 1778, 1778, 1341, 1234, 1234, 1235, 1230, 1230, 1235, 1152, 1333, 1152, 1340, 1333, 1340, 1331, 1779, 1330, 1331, 1779, 1345, 1236, 1779, 1236, 1780, 1781, 1330, 1780, 1782, 1328, 1781, 68, 81, 1096, 68, 1096, 1086, 1188, 1105, 1189, 1122, 1124, 1105, 1124, 1122, 1143, 1124, 1143, 1783, 1784, 1144, 1138, 1784, 1138, 1783, 1785, 1158, 1144, 1785, 1144, 1784, 1785, 1155, 1171, 1785, 1171, 1786, 1786, 1171, 1177, 1786, 1177, 1183, 1181, 1182, 1177, 1181, 1177, 1173, 1173, 1177, 1170, 1173, 1170, 1167, 1787, 1160, 1167, 1787, 1154, 1145, 1787, 1145, 1788, 1789, 1139, 1148, 1789, 1148, 1788, 1135, 1125, 1139, 1135, 1139, 1789, 1135, 1122, 1125, 1125, 1122, 1109, 1087, 1086, 1096, 1087, 1096, 1093, 1640, 1637, 1587, 1790, 1528, 1527, 1790, 1527, 1600, 1791, 1529, 1528, 1791, 1528, 1790, 1539, 1530, 1529, 1539, 1529, 1791, 1792, 1531, 1530, 1792, 1530, 1539, 1792, 1539, 1541, 1792, 1541, 1793, 1532, 1519, 1793, 1532, 1541, 1544, 1532, 1544, 1794, 1794, 1544, 1548, 1794, 1548, 1533, 1533, 1548, 1757, 1533, 1757, 1795, 1603, 1587, 1342, 1603, 1342, 1335, 1335, 1342, 1778, 1335, 1778, 1334, 1334, 1778, 1234, 1334, 1234, 1225, 1225, 1234, 1230, 1230, 1152, 1231, 1231, 1152, 1333, 1345, 1779, 1331, 1345, 1331, 1340, 1330, 1779, 1780, 1237, 1781, 1780, 1237, 1780, 1236, 1330, 1781, 1328, 1095, 1088, 1087, 1095, 1087, 1093, 1104, 1105, 1188, 1109, 1122, 1105, 1124, 1783, 1138, 1146, 1784, 1783, 1146, 1783, 1143, 1155, 1785, 1784, 1155, 1784, 1146, 1158, 1785, 1786, 1158, 1786, 1165, 1165, 1786, 1183, 1165, 1183, 1178, 1178, 1183, 1179, 1182, 1181, 1179, 1182, 1179, 1183, 1154, 1787, 1167, 1154, 1167, 1170, 1160, 1787, 1788, 1160, 1788, 1148, 1142, 1789, 1788, 1142, 1788, 1145, 1135, 1789, 1142, 1796, 1797, 1798, 1798, 1799, 1073, 1073, 1799, 1072, 1058, 1800, 1801, 1801, 1802, 1059, 1059, 1802, 1803, 1059, 1803, 1060, 1060, 1803, 1804, 1060, 1804, 1061, 1061, 1804, 849, 1061, 849, 43, 1640, 1587, 1603, 1637, 1640, 1600, 1637, 1600, 1527, 1534, 1790, 1600, 1537, 1791, 1790, 1537, 1790, 1534, 1537, 1539, 1791, 1531, 1792, 1793, 1531, 1793, 1519, 1541, 1532, 1793, 1532, 1794, 1524, 1524, 1794, 1533, 1533, 1795, 1756, 1756, 1795, 1805, 1756, 1805, 1761, 1672, 1671, 1806, 1680, 1672, 1806, 1680, 1806, 1673, 1685, 1680, 1673, 1685, 1673, 1686, 1692, 1685, 1686, 1692, 1686, 1807, 1696, 1692, 1807, 1696, 1807, 1808, 736, 1696, 1808, 736, 1808, 857, 1746, 1713, 1703, 1746, 1703, 1717, 1703, 1702, 1809, 1702, 1727, 1809, 1702, 1739, 1810, 1702, 1810, 1727, 1746, 1717, 1671, 1739, 1753, 1810, 1806, 1671, 1674, 1806, 1674, 1673, 1807, 1686, 1693, 1808, 1807, 1693, 1808, 1693, 1694, 857, 1808, 1694, 857, 1694, 739, 1703, 1809, 1714, 1809, 1727, 1714, 1727, 1810, 1740, 1671, 1717, 1747, 1810, 1753, 1740, 1663, 1700, 1664, 1663, 1664, 1655, 1664, 1700, 1724, 1664, 1724, 1665, 1665, 1724, 1746, 1665, 1746, 1671, 1747, 1674, 1671, 1674, 1747, 1706, 1674, 1706, 1675, 1675, 1706, 1709, 1675, 1709, 1668, 1668, 1709, 1811, 1668, 1811, 1669, 1020, 1044, 1028, 1044, 1029, 1028, 1812, 1813, 1814, 1456, 1039, 1815, 1813, 1456, 1815, 72, 1090, 18, 1095, 1100, 1091, 1813, 1033, 1041, 849, 1816, 867, 1804, 1817, 1816, 1804, 1816, 849, 1803, 1817, 1804, 1094, 1097, 82, 1097, 1094, 1093, 1097, 1093, 1096, 1032, 1030, 1026, 1030, 1041, 1032, 1321, 1055, 1050, 1071, 1055, 1321, 1802, 1801, 1818, 1509, 1354, 1303, 1509, 1303, 1302, 1025, 1321, 1050, 1353, 1352, 1303, 1353, 1303, 1354, 1023, 1025, 1469, 1476, 1466, 1026, 1478, 1466, 1476, 1479, 1482, 1025, 1479, 1025, 1023, 1486, 1478, 1500, 1498, 1506, 1496, 1818, 1801, 1800, 1818, 1800, 1819, 1034, 1033, 1813, 1034, 1813, 1812, 1821, 1797, 1796, 1821, 1796, 1822, 1186, 1823, 1127, 1190, 1127, 1823, 1824, 1105, 1190, 1189, 1105, 1824, 1104, 1187, 1102, 1091, 1100, 1101, 1825, 1102, 1826, 1072, 1078, 1819, 1072, 1819, 1800, 1078, 1072, 1799, 1078, 1799, 1827, 1827, 1799, 1798, 1827, 1798, 1828, 1828, 1798, 1797, 1041, 1456, 1813, 1829, 1830, 1831, 1829, 1831, 1832, 1833, 1835, 1829, 1833, 1829, 1834, 1829, 1835, 1836, 1837, 1830, 1829, 1837, 1829, 1836, 1782, 1781, 1237, 1782, 1237, 1838, 1815, 1039, 18, 1840, 894, 498, 1840, 498, 1458, 1839, 1187, 1189, 1839, 1189, 1841, 1842, 1843, 1457, 1842, 1457, 1844, 1840, 1458, 1457, 1843, 1840, 1457, 1845, 1846, 1841, 1845, 1842, 1844, 1847, 1837, 1836, 1829, 1832, 1848, 1847, 1836, 1835, 1835, 1833, 1849, 1850, 1815, 1039, 1815, 1850, 1813, 1849, 1833, 1822, 1822, 1833, 1821, 1851, 1293, 1296, 1851, 1296, 1852, 1852, 1296, 1853, 1852, 1853, 1854, 1055, 1822, 1070, 1071, 1849, 1822, 1071, 1822, 1055, 1849, 1071, 1855, 1855, 1071, 1320, 1855, 1320, 1856, 1857, 1302, 1293, 1857, 1293, 1851, 1413, 450, 867, 1413, 867, 1816, 1185, 1413, 1816, 1185, 1816, 1817, 1067, 1069, 1858, 1067, 1858, 1859, 1065, 1064, 1860, 1064, 1067, 1859, 1064, 1859, 1860, 1069, 1185, 1817, 1069, 1817, 1858, 1853, 1861, 1828, 1853, 1828, 1854, 1861, 1074, 1827, 1861, 1827, 1828, 1074, 1075, 1078, 1074, 1078, 1827, 1302, 1857, 1848, 1302, 1848, 1862, 1862, 1848, 1832, 1862, 1832, 1863, 1863, 1832, 1831, 1863, 1831, 1864, 1830, 1865, 1866, 1830, 1866, 1831, 1840, 1820, 894, 1839, 1841, 1846, 1867, 1843, 1842, 1843, 1868, 1840, 1824, 1845, 1841, 1824, 1841, 1189, 1824, 1867, 1842, 1824, 1842, 1845, 1092, 1814, 1813, 1814, 1092, 1101, 1814, 1101, 1869, 1845, 1844, 1870, 1845, 1870, 1846, 1846, 1871, 1839, 1872, 1871, 1846, 1872, 1846, 1838, 1866, 1873, 1864, 1866, 1864, 1831, 1238, 1407, 1066, 1238, 1066, 1063, 1407, 1415, 1068, 1407, 1068, 1066, 1081, 1076, 1075, 1081, 1075, 1074, 1065, 1075, 1062, 1861, 1079, 1074, 1415, 1462, 1184, 1415, 1184, 1068, 1296, 1297, 1853, 1853, 1297, 1079, 1853, 1079, 1861, 1873, 1312, 1313, 1873, 1313, 1864, 1864, 1313, 1326, 1864, 1326, 1863, 489, 451, 1184, 1462, 489, 1184, 1862, 1863, 1326, 1862, 1326, 1509, 1862, 1509, 1302, 1312, 1337, 1192, 1312, 1192, 1193, 1844, 1457, 1328, 1041, 1030, 1456, 1837, 1874, 1865, 1837, 1865, 1830, 1101, 1092, 1091, 1102, 1187, 1826, 1092, 1813, 1088, 1088, 1813, 1850, 1190, 1843, 1867, 1190, 1867, 1824, 1089, 1088, 1850, 1089, 1850, 1815, 1815, 18, 1090, 1815, 1090, 1089, 1190, 1868, 1843, 1823, 1840, 1868, 1823, 1868, 1190, 1840, 1823, 1186, 1840, 1186, 1820, 1825, 1100, 1102, 1800, 1058, 1072, 1362, 390, 928, 1875, 1360, 1362, 1875, 1362, 1361, 1368, 1875, 1361, 1876, 1367, 1368, 1876, 1368, 1365, 1877, 1369, 1367, 1877, 1367, 1876, 1878, 1879, 1369, 1878, 1369, 1877, 1880, 1881, 1879, 1880, 1879, 1878, 1442, 1881, 1880, 1427, 1442, 1882, 1362, 1360, 390, 1881, 1371, 1370, 1881, 1370, 1879, 1442, 1427, 1371, 1442, 1371, 1881, 1372, 1271, 1360, 1372, 1360, 1875, 1879, 1370, 1369, 1319, 1372, 1875, 1319, 1875, 1368, 1359, 1427, 1426, 1883, 1884, 1885, 1883, 1885, 1886, 1187, 1839, 1883, 1187, 1883, 1826, 1886, 1826, 1883, 1825, 1826, 1886, 1883, 1839, 1871, 1883, 1871, 1884, 1457, 1458, 1327, 1870, 1844, 1328, 1870, 1328, 1782, 1870, 1782, 1838, 1870, 1838, 1846, 1337, 1872, 1838, 1337, 1838, 1237, 1874, 1887, 1865, 1888, 1889, 1890, 1872, 1891, 1871, 1890, 1889, 1892, 1890, 1892, 1812, 1872, 1873, 1866, 1872, 1866, 1891, 1814, 1869, 1888, 1814, 1888, 1890, 1812, 1814, 1890, 1891, 1866, 1865, 1891, 1865, 1887, 1337, 1312, 1873, 1337, 1873, 1872, 1884, 1871, 1891, 1884, 1891, 1887, 1887, 1885, 1884, 1889, 1856, 1320, 1322, 1892, 1889, 1322, 1889, 1320, 1892, 1322, 1323, 1892, 1323, 1042, 1893, 1042, 1024, 1024, 1468, 1893, 1468, 1024, 1027, 1834, 1829, 1848, 1834, 1848, 1857, 1834, 1857, 1851, 1834, 1851, 1833, 1833, 1851, 1852, 1833, 1852, 1821, 1854, 1797, 1821, 1854, 1821, 1852, 1828, 1797, 1854, 1498, 1496, 1488, 1488, 1491, 1496, 1488, 1475, 1491, 1475, 1893, 1468, 1078, 1065, 1819, 1860, 1818, 1819, 1860, 1819, 1065, 1859, 1802, 1818, 1859, 1818, 1860, 1858, 1803, 1802, 1858, 1802, 1859, 1803, 1858, 1817, 1496, 1504, 1494, 1812, 1892, 1042, 1812, 1042, 1034, 1893, 1034, 1042, 1030, 1037, 1039, 1506, 1505, 1508, 1496, 1494, 1488, 1473, 1475, 1488, 1475, 1473, 1893, 1893, 1473, 1027, 1893, 1027, 1468, 1468, 1027, 1034, 1468, 1034, 1893, 1466, 1481, 1486, 1874, 1837, 1847, 1886, 1885, 1869, 1887, 1888, 1869, 1887, 1869, 1885, 1887, 1874, 1888, 1100, 1113, 1102, 1113, 1108, 1103, 1110, 1112, 89, 1119, 1107, 1116, 1106, 1116, 1107, 1109, 1105, 1104, 1113, 1103, 1102, 1113, 1111, 1108, 1100, 1825, 1101, 1101, 1825, 1886, 1101, 1886, 1869, 1874, 1889, 1888, 1889, 1874, 1847, 1889, 1847, 1856, 1108, 1115, 1894, 1894, 1115, 1123, 1894, 1123, 1126, 1126, 1123, 1130, 1130, 1137, 1140, 1140, 1137, 1147, 1164, 1175, 1174, 1174, 1175, 198, 1174, 198, 1181, 1100, 1098, 1112, 1100, 1112, 1113, 1112, 1098, 1099, 1179, 1181, 198, 1179, 198, 1180, 1180, 198, 192, 1164, 1161, 1157, 1157, 1161, 1149, 1157, 1149, 1147, 1147, 1149, 1140, 1126, 1108, 1894, 1847, 1855, 1856, 1835, 1855, 1847, 1849, 1855, 1835, 1796, 1070, 1822, 1070, 1796, 1798, 1070, 1798, 1073, 1056, 1073, 1072, 1058, 1801, 1051, 1051, 1801, 1059, 1356, 1395, 1396, 1356, 1396, 1357, 1391, 1355, 1358, 1391, 1358, 1399, 1393, 426, 435, 1393, 435, 1400, 1386, 1391, 1399, 1386, 1399, 1398, 1431, 1387, 1397, 1431, 1397, 1432, 1454, 1393, 1400, 1454, 1400, 1450, 1395, 1454, 1450, 1395, 1450, 1396, 1350, 1431, 1432, 1350, 1432, 1348, 1443, 1304, 1307, 1443, 1307, 1382, 1444, 1443, 1382, 1444, 1382, 1428, 1418, 1409, 1895, 1896, 1897, 1420, 1896, 1420, 1435, 1299, 1420, 1897, 1299, 1897, 1898, 1898, 1897, 1899, 1898, 1899, 1900, 1895, 1080, 1901, 1901, 1899, 1897, 1901, 1897, 1896, 1900, 1899, 1902, 1900, 1902, 1903, 1901, 1080, 1904, 1901, 1904, 1905, 1902, 1899, 1901, 1427, 1882, 1906, 1444, 1460, 1907, 1425, 1426, 1908, 1425, 1908, 1909, 1906, 1908, 1426, 1906, 1426, 1427, 1425, 1909, 1510, 1425, 1510, 1424, 1460, 1444, 1428, 1297, 1904, 1080, 1895, 1081, 1080, 1903, 1910, 1900, 1299, 1898, 1911, 1299, 1911, 1300, 1911, 1898, 1900, 1911, 1900, 1910, 1081, 1895, 1409, 1081, 1409, 1076, 1295, 1904, 1297, 1912, 1350, 1351, 1459, 1913, 1460, 1914, 1915, 1294, 1914, 1294, 1352, 1916, 1917, 1918, 1916, 1918, 1919, 1920, 1352, 1353, 1920, 1353, 1510, 1906, 1921, 1908, 1923, 1924, 1925, 1923, 1925, 1926, 1915, 1922, 1295, 1915, 1295, 1294, 1920, 1929, 1930, 1928, 1927, 1931, 1929, 1923, 1930, 1912, 1431, 1350, 1924, 1933, 1925, 1926, 1925, 1936, 1928, 1931, 1934, 1937, 1935, 1915, 1937, 1915, 1914, 1920, 1930, 1938, 1920, 1938, 1932, 1917, 1919, 1918, 1939, 1940, 1941, 1939, 1941, 1942, 1940, 1941, 1933, 1930, 1923, 1938, 1914, 1932, 1937, 1925, 1933, 1941, 1936, 1938, 1923, 1936, 1923, 1926, 1943, 1928, 1934, 1943, 1934, 1944, 1933, 1941, 1945, 1922, 1915, 1935, 1922, 1935, 1946, 1947, 1938, 1936, 1936, 1925, 1948, 1936, 1948, 1947, 1917, 1949, 1919, 1940, 1945, 1941, 1947, 1948, 1945, 1932, 1938, 1947, 1932, 1947, 1937, 1952, 1935, 1937, 1953, 1949, 1943, 1951, 1956, 1945, 1953, 1946, 1955, 1955, 1949, 1953, 1952, 1924, 1954, 1956, 1954, 1947, 1953, 1934, 1946, 1928, 1944, 1934, 1368, 1361, 1365, 1434, 1433, 1417, 1434, 1417, 1416, 1434, 1416, 1418, 1434, 1418, 1435, 1433, 1436, 1423, 1433, 1423, 1417, 1451, 494, 476, 1451, 476, 1437, 1442, 1444, 1882, 1406, 1423, 1455, 1406, 1455, 1414, 1438, 1463, 1464, 1438, 1464, 1437, 1463, 1414, 1455, 1463, 1455, 1464, 1363, 1362, 928, 1363, 928, 472, 1412, 1876, 1365, 1412, 1365, 1366, 1310, 1877, 1876, 1310, 1876, 1412, 1305, 1878, 1877, 1305, 1877, 1310, 1304, 1880, 1878, 1304, 1878, 1305, 1443, 1442, 1880, 1443, 1880, 1304, 1435, 1418, 1895, 1435, 1895, 1896, 1896, 1895, 1901, 1901, 1905, 1902, 1882, 1444, 1907, 1882, 1907, 1906, 1931, 1922, 1934, 1925, 1933, 1945, 1925, 1945, 1948, 1934, 1922, 1946, 1949, 1955, 1951, 1924, 1947, 1954, 1937, 1947, 1950, 1939, 1951, 1945, 1939, 1945, 1940, 1943, 1949, 1917, 1951, 1919, 1949, 1946, 1955, 1957, 1952, 1956, 1935, 1956, 1952, 1954, 1951, 1955, 1956, 1954, 1956, 1955, 1954, 1955, 1958, 1958, 1955, 1957, 1955, 1956, 1958, 1946, 1956, 1955, 1935, 1956, 1946, 1956, 1954, 1958, 1937, 1950, 1952, 1952, 1950, 1947, 1952, 1947, 1924, 1956, 1947, 1945, 1942, 1941, 1933, 1942, 1933, 1959, 1916, 1912, 1917, 1943, 1944, 1928, 1960, 1922, 1915, 1906, 1907, 1924, 1906, 1924, 1923, 1920, 1510, 1929, 1510, 1921, 1923, 1510, 1923, 1929, 1931, 1960, 1922, 1914, 1352, 1932, 1907, 1913, 1933, 1907, 1933, 1924, 1943, 1917, 1912, 1959, 1933, 1913, 1959, 1913, 1459, 1920, 1932, 1352, 1923, 1921, 1906, 1431, 1912, 1916, 1953, 1943, 1934, 1904, 1295, 1922, 1904, 1922, 1960, 1903, 1902, 1927, 1903, 1927, 1928, 1909, 1908, 1921, 1909, 1921, 1510, 1905, 1904, 1960, 1905, 1960, 1931, 1461, 1431, 1916, 1907, 1460, 1913, 1902, 1905, 1931, 1902, 1931, 1927, 1939, 1919, 1951, 1919, 1939, 1942, 1959, 1916, 1919, 1959, 1919, 1942, 1459, 1916, 1959, 1916, 1459, 1461, 1461, 1459, 1429, 1461, 1429, 1430, 1383, 1384, 1430, 1383, 1430, 1429, 1384, 1383, 1377, 1384, 1377, 1385, 1385, 1377, 1374, 1385, 1374, 1390, 1912, 1928, 1943, 1928, 1912, 1351, 1928, 1351, 1903, 1349, 1910, 1903, 1349, 1903, 1351, 1290, 1911, 1910, 1290, 1910, 1349, 1289, 1300, 1911, 1289, 1911, 1290, 1292, 1301, 1300, 1292, 1300, 1289, 1286, 1308, 1301, 1286, 1301, 1292, 1285, 1082, 1308, 1285, 1308, 1286, 1280, 1083, 1082, 1280, 1082, 1285, 1279, 1411, 1083, 1279, 1083, 1280, 1282, 1441, 1411, 1282, 1411, 1279, 380, 479, 1441, 380, 1441, 1282, 1390, 1374, 1373, 1390, 1373, 1388, 1388, 1373, 1381, 1388, 1381, 1389, 1389, 1381, 1380, 1389, 1380, 1394, 1448, 1453, 1394, 1448, 1394, 1380, 1453, 1448, 1376, 1453, 1376, 1392, 1392, 1376, 407, 1392, 407, 427, 1029, 1031, 1467, 1029, 1467, 1028, 1026, 1028, 1030, 1026, 1030, 1466, 1028, 1467, 1030, 37, 24, 1045, 1118, 143, 105, 177, 1119, 1118, 177, 1118, 105, 1119, 177, 119, 1119, 119, 202, 1119, 202, 1186, 1820, 1186, 202, 1820, 202, 894, 498, 894, 893, 1601, 685, 687, 1601, 687, 1642, 1635, 674, 672, 1635, 672, 1626, 1115, 103, 110];
function drawBust(cv){
  /* Featureless head. The mesh is used ONLY as a head-shaped coverage mask
     (head crop, shoulders dropped); it is heavily blurred to a smooth blob
     (ear/nose/brow rounded off) and shaded by a SYNTHETIC volume normal
     derived from the mask gradient — never from mesh face normals — so no
     eyes/nose/mouth/ear can appear. Even sphere-like light + bright rim, then
     rendered as dense fine grain (hero-terrain language). Silver-surfer. */
  var p = prepArt(cv); if (!p) return;
  var c2 = p.c2, w = p.w, h = p.h, fg = p.fg, sg = p.sg;
  var V = HEAD_V, F = HEAD_F, i, x, y;

  /* mask raster (coverage only, head crop) */
  var RW = Math.max(80, Math.min(150, w * 0.42 | 0));
  var RH = Math.max(80, Math.min(150, h * 0.42 | 0));
  var cov = new Float32Array(RW * RH);

  /* head crop: include cranium→jaw→chin, exclude neck/shoulders */
  var HEAD_MAX_Y = 600;
  /* placement: head centered + large; keep the cranium-to-jaw taper */
  var S = h * 1.42, oy = h * 0.04;
  var ox = w * 0.5 + 0.037 * S;
  var rx = RW / w, ry = RH / h;
  for (i = 0; i < F.length; i += 3) {
    var a=F[i]*3, b=F[i+1]*3, c=F[i+2]*3;
    if (V[a+1] > HEAD_MAX_Y || V[b+1] > HEAD_MAX_Y || V[c+1] > HEAD_MAX_Y) continue;
    var ax=(ox-(V[a]/1000)*S)*rx, ay=(oy+(V[a+1]/1000)*S)*ry;
    var bx=(ox-(V[b]/1000)*S)*rx, by=(oy+(V[b+1]/1000)*S)*ry;
    var cx2=(ox-(V[c]/1000)*S)*rx, cy2=(oy+(V[c+1]/1000)*S)*ry;
    var x0=Math.max(0, Math.min(ax,bx,cx2)|0), x1=Math.min(RW-1, Math.ceil(Math.max(ax,bx,cx2)));
    var y0=Math.max(0, Math.min(ay,by,cy2)|0), y1=Math.min(RH-1, Math.ceil(Math.max(ay,by,cy2)));
    var d = (bx-ax)*(cy2-ay)-(cx2-ax)*(by-ay);
    if (!d) continue;
    for (y = y0; y <= y1; y++) for (x = x0; x <= x1; x++) {
      var w0=((bx-x)*(cy2-y)-(cx2-x)*(by-y))/d;
      var w1=((cx2-x)*(ay-y)-(ax-x)*(cy2-y))/d;
      if (w0<-0.03||w1<-0.03||(1-w0-w1)<-0.03) continue;
      cov[y*RW+x]=1;
    }
  }

  /* heavy blur → smooth head blob, ear/nose absorbed */
  var tmp = new Float32Array(RW * RH), pass, RAD=1;
  for (pass = 0; pass < 4; pass++) {
    for (y = 0; y < RH; y++) for (x = 0; x < RW; x++) {
      var s=0, cnt=0;
      for (var dy=-RAD; dy<=RAD; dy++) for (var dx=-RAD; dx<=RAD; dx++) {
        var yy=y+dy, xx=x+dx;
        if (yy<0||yy>=RH||xx<0||xx>=RW) continue;
        s+=cov[yy*RW+xx]; cnt++;
      }
      tmp[y*RW+x]=s/cnt;
    }
    cov.set(tmp);
  }

  /* synthetic shading from mask: volume normal (n = -grad(cov), nz from cov),
     one key light, even + smooth. rim from cov falloff. NO mesh normals. */
  var lum = new Float32Array(RW * RH);
  var L=[-0.42,-0.5,0.76], ll=Math.sqrt(L[0]*L[0]+L[1]*L[1]+L[2]*L[2]);
  L=[L[0]/ll,L[1]/ll,L[2]/ll];
  for (y = 0; y < RH; y++) for (x = 0; x < RW; x++) {
    var idc=y*RW+x, cvg=cov[idc];
    if (cvg < 0.05) continue;
    var gx2 = (cov[idc+(x<RW-1?1:0)] - cov[idc-(x>0?1:0)]);
    var gy2 = (cov[idc+(y<RH-1?RW:0)] - cov[idc-(y>0?RW:0)]);
    var nz = Math.sqrt(Math.max(0.02, cvg));      /* center bulges toward viewer */
    var nx2=-gx2*3.2, ny2=-gy2*3.2, nl=Math.sqrt(nx2*nx2+ny2*ny2+nz*nz)||1;
    nx2/=nl; ny2/=nl; nz/=nl;
    var lit = Math.max(0, nx2*L[0]+ny2*L[1]+nz*L[2]);
    var edge = 1 - Math.min(1, cvg*1.1);
    var rim = Math.pow(edge, 1.5) * (1 - (y/RH)*0.3);
    lum[idc] = Math.min(1.3, 0.34 + 0.62*Math.pow(lit,0.9) + 1.15*rim);
  }
  /* soften shading once so no synthetic banding */
  for (y = 0; y < RH; y++) for (x = 0; x < RW; x++) {
    var s3=0, c3=0;
    for (var dy3=-1; dy3<=1; dy3++) for (var dx3=-1; dx3<=1; dx3++) {
      var yy3=y+dy3, xx3=x+dx3;
      if (yy3<0||yy3>=RH||xx3<0||xx3>=RW) continue;
      s3+=lum[yy3*RW+xx3]; c3++;
    }
    tmp[y*RW+x]=s3/c3;
  }
  lum.set(tmp);

  /* glow behind crown */
  var grad = c2.createRadialGradient(w*0.5, h*0.34, 0, w*0.5, h*0.34, h*0.5);
  grad.addColorStop(0, fg); grad.addColorStop(1, "rgba(0,0,0,0)");
  c2.globalAlpha = 0.05; c2.fillStyle = grad; c2.fillRect(0, 0, w, h);

  /* stipple */
  var A = Math.min(150000, (w * h * 0.6) | 0);
  for (i = 0; i < A; i++) {
    var ux = hash2(i, 3), uy = hash2(i, 7);
    var sx = Math.min(RW-1.001, ux*RW), sy = Math.min(RH-1.001, uy*RH);
    var xi = sx|0, yi = sy|0, fx=sx-xi, fy=sy-yi;
    var i00=yi*RW+xi, i10=i00+1, i01=i00+RW, i11=i01+1;
    if (i11 >= lum.length) continue;
    var Cv = cov[i00]*(1-fx)*(1-fy)+cov[i10]*fx*(1-fy)+cov[i01]*(1-fx)*fy+cov[i11]*fx*fy;
    if (Cv < 0.06) continue;
    var Lv = lum[i00]*(1-fx)*(1-fy)+lum[i10]*fx*(1-fy)+lum[i01]*(1-fx)*fy+lum[i11]*fx*fy;
    var grain = 0.64 + 0.68 * fbm(ux*7, uy*7, 2.6);
    var keep = Math.min(1, Cv*1.3) * (0.08 + 0.92*Math.pow(Math.min(1,Lv),1.3)) * grain;
    if (hash2(i, 11) > keep) continue;
    /* dissolve the chin/neck base into dust — no hard bottom line */
    var fd = uy < 0.70 ? 1 : Math.max(0, 1 - (uy - 0.70) / 0.22);
    if (fd <= 0.02 && hash2(i, 23) > fd + 0.15) continue;
    var spark = hash2(i, 31) > 0.9986;
    c2.fillStyle = spark ? sg : fg;
    c2.globalAlpha = (spark ? 0.95 : Math.min(1, 0.42 + 0.6*Lv) * (0.75 + 0.25*hash2(i,13))) * (0.3 + 0.7*fd);
    var szb = Lv > 0.85 && hash2(i, 17) > 0.965 ? 1.7 : 1.05;
    c2.fillRect(ux*w, uy*h, szb, szb);
  }

  /* ambient dust */
  for (i = 0; i < 560; i++) {
    c2.fillStyle = fg;
    c2.globalAlpha = 0.03 + 0.09 * Math.pow(hash2(i, 41), 2.2);
    c2.fillRect(hash2(i, 31) * w, hash2(i, 37) * h, 1.1, 1.1);
  }
  c2.globalAlpha = 1;
}

/* ---------- OORT (ledger) — dense volumetric particle sphere ----------
   Not a lattice: a mottled, clumped swarm. Dense jittered surface shell +
   inner shell for thickness + interior fill + bright polar caps + a tight
   dust skin. Clump value is baked per point (object space) so rotation keeps
   the mottling stable. */
function oortPoints(R){
  var pts = [], i, GA = Math.PI * (3 - Math.sqrt(5));
  function push(x, y, z, k){
    pts.push({ x: x, y: y, z: z, k: k, c: fbm((x / R + 1) * 2.2, (y / R + 1) * 2.2, 3.3 + z / R) });
  }
  var NS = 4600;
  for (i = 0; i < NS; i++) { /* main shell */
    var yy = 1 - (i / (NS - 1)) * 2, rad = Math.sqrt(1 - yy * yy);
    var th = GA * i + (hash2(i, 21) - 0.5) * 0.5;
    var rr = R * (1 + (hash2(i, 3) - 0.5) * 0.06);
    push(Math.cos(th) * rad * rr, yy * rr, Math.sin(th) * rad * rr, 0);
  }
  for (i = 0; i < 1300; i++) { /* inner shell — thickness */
    var a = hash2(i, 5) * Math.PI * 2, b = Math.acos(2 * hash2(i, 7) - 1);
    var r2 = R * (0.86 + hash2(i, 9) * 0.12);
    push(Math.sin(b) * Math.cos(a) * r2, Math.cos(b) * r2, Math.sin(b) * Math.sin(a) * r2, 1);
  }
  for (i = 0; i < 850; i++) { /* interior swarm */
    var a3 = hash2(i, 25) * Math.PI * 2, b3 = Math.acos(2 * hash2(i, 27) - 1);
    var r3 = R * 0.84 * Math.cbrt(hash2(i, 29));
    push(Math.sin(b3) * Math.cos(a3) * r3, Math.cos(b3) * r3, Math.sin(b3) * Math.sin(a3) * r3, 1);
  }
  for (i = 0; i < 700; i++) { /* polar caps — the refs' bright crowns */
    var sgn = i % 2 ? 1 : -1;
    var yy2 = sgn * (0.72 + 0.27 * hash2(i, 11)), rad2 = Math.sqrt(Math.max(0, 1 - yy2 * yy2));
    var th2 = hash2(i, 13) * Math.PI * 2;
    var rr2 = R * (1 + (hash2(i, 15) - 0.5) * 0.05);
    push(Math.cos(th2) * rad2 * rr2, yy2 * rr2, Math.sin(th2) * rad2 * rr2, 3);
  }
  for (i = 0; i < 240; i++) { /* tight dust skin */
    var a2 = hash2(i, 31) * Math.PI * 2, b2 = Math.acos(2 * hash2(i, 33) - 1);
    var r4 = R * (1.04 + Math.pow(hash2(i, 17), 2.4) * 0.18);
    push(Math.sin(b2) * Math.cos(a2) * r4, Math.cos(b2) * r4, Math.sin(b2) * Math.sin(a2) * r4, 2);
  }
  return pts;
}
function drawOortFrame(cv, rot){
  var p = prepArt(cv); if (!p) return;
  var c2 = p.c2, w = p.w, h = p.h, fg = p.fg, sg = p.sg;
  var R = Math.min(w, h) * 0.40, cx = w / 2, cy = h * 0.5;
  if (!cv.__oort || cv.__oortR !== R) { cv.__oort = oortPoints(R); cv.__oortR = R; }
  var pts = cv.__oort, tilt = 0.35, ct = Math.cos(tilt), st = Math.sin(tilt);
  var cr = Math.cos(rot), sr = Math.sin(rot);
  for (var i = 0; i < pts.length; i++) {
    var q = pts[i];
    var xr = q.x * cr + q.z * sr;
    var zr = -q.x * sr + q.z * cr;
    var yr = q.y * ct - zr * st, zz = q.y * st + zr * ct;
    var depth = Math.max(0, Math.min(1, (zz / R + 1) / 2)); /* 0 back -> 1 front */
    var clump = 0.35 + 1.15 * q.c;
    var al, sz;
    if (q.k === 2) { al = 0.05 + 0.13 * depth; sz = 1; }
    else if (q.k === 3) { al = (0.22 + 0.68 * depth) * clump; sz = depth > 0.6 ? 2 : 1.5; }
    else if (q.k === 1) { al = (0.05 + 0.38 * Math.pow(depth, 1.7)) * clump; sz = 1.2; }
    else { al = (0.09 + 0.62 * Math.pow(depth, 1.6)) * clump; sz = depth > 0.75 ? 1.8 : 1.3; }
    c2.fillStyle = (i % 137 === 0) ? sg : fg;
    c2.globalAlpha = Math.max(0.02, Math.min(0.95, al));
    c2.fillRect(cx + xr - sz / 2, cy + yr - sz / 2, sz, sz);
  }
  c2.globalAlpha = 1;
}
function drawOort(cv){ drawOortFrame(cv, cv.__rot || 0.6); }
function startOortIdle(cv){
  if (reduceMotion) return;
  cv.__rot = cv.__rot || 0.6;
  /* Constant slow one-direction drift, non-interactive, ~12fps redraw — light
     on any device and steadier to look at than the reversing orbit. */
  var SPEED = 0.010, FRAME = 83;
  var visible = false, running = false;
  function spin(){
    if (!visible) { running = false; return; }
    cv.__rot += SPEED;
    drawOortFrame(cv, cv.__rot);
    running = true;
    setTimeout(function(){ requestAnimationFrame(spin); }, FRAME);
  }
  new IntersectionObserver(function(en){
    visible = en[0].isIntersecting;
    if (visible && !running && cv.dataset.formed === "1") requestAnimationFrame(spin);
  }, { threshold: 0.1 }).observe(cv);
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
    else if (kind === "bust") drawBust(cv);
    else if (kind === "oort") drawOort(cv);
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
