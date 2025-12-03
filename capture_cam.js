// node capture_cam.js --cam_id cam09 --url "https://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=58b5510817139d0010f35d4e&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8"
const puppeteer = require("puppeteer");
const fs = require("fs");
const path = require("path");

function parseArgs() {
  const args = {};
  const argv = process.argv.slice(2);
  for (let i = 0; i < argv.length; i++) {
    if (argv[i].startsWith("--")) {
      args[argv[i].replace("--", "")] =
        argv[i + 1] && !argv[i + 1].startsWith("--") ? argv[i + 1] : true;
      if (argv[i + 1] && !argv[i + 1].startsWith("--")) i++;
    }
  }
  return args;
}

const args = parseArgs();
const URL = args.url;
const CAMERA_ID = args.cam_id;
const SAVE_DIR = path.resolve(args.save_dir || path.join(__dirname, "dataset_raw/screenshots"));
const INTERVAL_MS = Number(args.interval_ms || 5 * 60 * 1000);
const NAV_TIMEOUT = Number(args.nav_timeout || 45000);
const MAX_RETRY = Number(args.max_retry || 3);
const MAX_CONSECUTIVE_FAILURES_TO_EXIT = Number(args.max_failures || 8);
const WAIT_AFTER_RELOAD_MS = Number(args.wait_after_reload_ms || 3000);
const TZ = "Asia/Ho_Chi_Minh";

if (!URL || !CAMERA_ID) {
  console.error("Usage: node capture_cam.js --cam_id <cam> --url <link>");
  process.exit(1);
}

fs.mkdirSync(SAVE_DIR, { recursive: true });

function partsFor(dt) {
  const f = new Intl.DateTimeFormat("en-GB", {
    timeZone: TZ,
    hour12: false,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  const parts = {};
  for (const p of f.formatToParts(dt)) {
    if (p.type !== "literal") parts[p.type] = p.value;
  }
  return parts;
}

function tsFileFormat(dt) {
  const p = partsFor(dt);
  return `${p.year}${p.month}${p.day}_${p.hour}${p.minute}${p.second}`;
}

let browser = null;
let page = null;
let consecutiveFailures = 0;
let stopped = false;

async function startBrowser() {
  if (browser) {
    try { await browser.close(); } catch (err) {}
    browser = null;
    page = null;
  }
  browser = await puppeteer.launch({
    headless: "new",
    args: [
      "--no-sandbox",
      "--disable-setuid-sandbox",
      "--autoplay-policy=no-user-gesture-required",
    ],
  });
  page = await browser.newPage();
  page.setDefaultNavigationTimeout(NAV_TIMEOUT);
  await page.setViewport({ width: 1280, height: 800 });
  console.log("Browser started");
}

async function safeGoto(url) {
  for (let i = 0; i < Math.max(1, MAX_RETRY); i++) {
    try {
      await page.goto(url, { waitUntil: "domcontentloaded", timeout: NAV_TIMEOUT });
      return;
    } catch (err) {
      console.warn(`goto attempt ${i + 1} failed:`, err.message || err);
      await new Promise(r => setTimeout(r, 1000 * (i + 1)));
    }
  }
  throw new Error("safeGoto: all retries failed");
}

async function takeShotOnce() {
  for (let attempt = 1; attempt <= MAX_RETRY; attempt++) {
    try {
      await page.reload({ waitUntil: "domcontentloaded", timeout: NAV_TIMEOUT });
      await page.waitForTimeout(WAIT_AFTER_RELOAD_MS);

      const now = new Date();
      const fname = `${CAMERA_ID}_${tsFileFormat(now)}.png`;
      const fpath = path.join(SAVE_DIR, fname);

      await page.screenshot({ path: fpath, type: "png", fullPage: false });
      console.log(`[OK] Saved ${fpath}`);
      consecutiveFailures = 0;
      return;
    } catch (err) {
      console.error(`[WARN] Screenshot attempt ${attempt} failed:`, err.message || err);
      if (attempt === MAX_RETRY) throw err;
      await new Promise((res) => setTimeout(res, 2000 * attempt));
    }
  }
}

async function ensureAndRun() {
  try {
    if (!browser || !page) await startBrowser();
    try {
      const currentUrl = page.url();
      if (!currentUrl || currentUrl === "about:blank") {
        await safeGoto(URL);
      }
    } catch(err) {}

    try {
      try {
        await page.waitForSelector("video", { timeout: 5000 }).catch(() => {});
      } catch (err2) {}
      await takeShotOnce();
    } catch (err) {
      console.error("[ERROR] takeShotOnce failed:", err.message || err);
      consecutiveFailures++;
      console.warn("Attempting to restart browser/page to recover...");
      try {
        await startBrowser();
        await safeGoto(URL);
      } catch (err2) {
        console.error("Restart browser failed:", err2.message || err2);
      }
      if (consecutiveFailures >= MAX_CONSECUTIVE_FAILURES_TO_EXIT) {
        console.error(`Too many consecutive failures (${consecutiveFailures}). Exiting to allow external supervisor to restart.`);
        try { await browser.close(); } catch(err2) {}
        process.exit(1);
      }
    }
  } catch (outerErr) {
    console.error("[FATAL] ensureAndRun outer error:", outerErr.message || outerErr);
    consecutiveFailures++;
    if (consecutiveFailures >= MAX_CONSECUTIVE_FAILURES_TO_EXIT) {
      try { await browser.close(); } catch (err) {}
      process.exit(1);
    }
  }
}

async function runLoop() {
  while (!stopped) {
    const start = Date.now();
    await ensureAndRun().catch((e) => {
      console.error("runLoop ensureAndRun error:", e && e.message ? e.message : e);
    });
    const elapsed = Date.now() - start;
    const wait = Math.max(0, INTERVAL_MS - elapsed);
    if (wait === 0) {
      await new Promise((res) => setTimeout(res, 100));
    } else {
      await new Promise((res) => setTimeout(res, wait));
    }
  }
}

function setupSignalHandlers() {
  const shutdown = async () => {
    if (stopped) return;
    stopped = true;
    console.log("Shutting down...");
    try {
      if (browser) await browser.close();
    } catch (e) {
      /* ignore */
    }
    process.exit(0);
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

(async () => {
  try {
    setupSignalHandlers();
    await startBrowser();
    await safeGoto(URL);
    await ensureAndRun();
    // setInterval(ensureAndRun, INTERVAL_MS);
    runLoop();
  } catch (err) {
    console.error("Startup error:", err && err.message ? err.message : err);
    try { if (browser) await browser.close(); } catch (err) {}
    process.exit(1);
  }
})();
