/* Recapture the project-card minis from the real dashboards.
 *
 * The previous pass shot the bare <canvas>. That produced a wide, unlabelled strip:
 * the NBA one is six dots with no title and no axis name, and the retail one is a
 * line with no statement of what it proves. Olist shipped its whole .card — title,
 * subtitle, figures, footnote — and is the only one that reads at a glance.
 *
 * So: shoot the enclosing .card for all of them. Nothing is redrawn or re-typed;
 * it is still a screenshot of the artifact that backs the claim.
 */
import { createRequire } from "node:module";
import { mkdirSync } from "node:fs";
const require = createRequire("/Users/siddharthdeepak/.npm/_npx/e41f203b7505f1fb/node_modules/");
const { chromium } = require("playwright-core");

const ROOT = "/Users/siddharthdeepak/Source";
const EXE = process.env.HOME +
  "/Library/Caches/ms-playwright/chromium-1234/chrome-mac-x64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing";

const JOBS = [
  { out: "retail", page: "retail_dashboard.html", h3: "Precision@k", themes: ["light", "dark"] },
  { out: "nba", page: "nba_dashboard.html", h3: "Holdout ROC-AUC", themes: ["light", "dark"] },
  // Olist's panel carries a four-line method footnote. Shot at 880 it shrinks past
  // reading size, so this one is framed narrower — the panel reflows, nothing is cut.
  { out: "olist", page: "olist_dashboard.html", h3: "Delivery Speed vs Customer Rating", themes: ["light"], width: 620 },
];

const browser = await chromium.launch({ executablePath: EXE });
mkdirSync(`${ROOT}/cards`, { recursive: true });

for (const job of JOBS) {
  for (const theme of job.themes) {
    const ctx = await browser.newContext({
      viewport: { width: job.width || 880, height: 1500 },
      deviceScaleFactor: 2,
      colorScheme: theme,
    });
    const page = await ctx.newPage();
    await page.goto(`file://${ROOT}/${job.page}`, { waitUntil: "networkidle" });
    await page.waitForTimeout(1800);

    // colorScheme alone is not enough: these dashboards own their theme through
    // Dash, and a flip has to REBUILD the charts (dark is its own validated palette,
    // not a recolour). Without this the dark capture is a light chart.
    await page.evaluate((mode) => {
      if (window.Dash && window.Dash.applyTheme) window.Dash.applyTheme(mode);
    }, theme);
    await page.waitForTimeout(900);

    // Sticky/fixed chrome would bleed a nav bar across the shot.
    await page.evaluate(() => {
      for (const el of document.querySelectorAll("*")) {
        const p = getComputedStyle(el).position;
        if (p === "fixed" || p === "sticky") el.style.display = "none";
      }
    });

    const card = await page.evaluateHandle((needle) => {
      for (const c of document.querySelectorAll(".card")) {
        const h = c.querySelector("h3");
        if (h && h.textContent.includes(needle)) return c;
      }
      return null;
    }, job.h3);

    const el = card.asElement();
    if (!el) throw new Error(`no .card matching "${job.h3}" in ${job.page}`);

    // The caveat paragraph is the longest block on these cards and is unreadable at
    // thumbnail size. Drop it from the frame only; it stays on the dashboard.
    await el.evaluate((c) => {
      c.querySelectorAll(".caveat, .tabtoggle, .tabwrap").forEach((n) => n.remove());
      c.style.margin = "0";
    });
    await page.waitForTimeout(400);

    const file = `${ROOT}/cards/${job.out}-${theme}.png`;
    await el.screenshot({ path: file });
    console.log(`wrote ${file}`);
    await ctx.close();
  }
}
await browser.close();
