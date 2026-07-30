/* Shoot each project card at real portfolio width, both themes, so the minis are
 * judged at the size a recruiter actually sees rather than at capture resolution. */
import { createRequire } from "node:module";
const require = createRequire("/Users/siddharthdeepak/.npm/_npx/e41f203b7505f1fb/node_modules/");
const { chromium } = require("playwright-core");

const ROOT = "/Users/siddharthdeepak/Source";
const OUT = process.argv[2] || "/tmp";
const EXE = process.env.HOME +
  "/Library/Caches/ms-playwright/chromium-1234/chrome-mac-x64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing";

const browser = await chromium.launch({ executablePath: EXE });
for (const theme of ["light", "dark"]) {
  const ctx = await browser.newContext({
    viewport: { width: 1440, height: 1200 },
    deviceScaleFactor: 2,
    colorScheme: theme,
  });
  const page = await ctx.newPage();
  await page.goto(`file://${ROOT}/index.html`, { waitUntil: "networkidle" });
  // The portfolio picks its theme from the sd-theme key, not from the OS, so a dark
  // browser context alone leaves it light — and light is what .shot--light selects.
  await page.evaluate((t) => document.documentElement.setAttribute("data-theme", t), theme);
  await page.waitForTimeout(1200);
  const cards = await page.$$(".prod");
  for (let i = 0; i < cards.length; i++) {
    const h3 = await cards[i].$eval("h3", (n) => n.textContent.trim()).catch(() => null);
    if (!h3) continue;
    const slug = h3.toLowerCase().replace(/[^a-z0-9]+/g, "-").slice(0, 24);
    await cards[i].scrollIntoViewIfNeeded();
    await page.waitForTimeout(300);
    await cards[i].screenshot({ path: `${OUT}/card-${slug}-${theme}.png` });
    console.log(`${slug} ${theme}`);
  }
  await ctx.close();
}
await browser.close();
