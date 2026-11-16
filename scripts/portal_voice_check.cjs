// Real-provider check of the portal's AI voice call, with a fake microphone (no person needed).
//
//   PLAYWRIGHT_DIR=../frontend.voice-labs node scripts/portal_voice_check.cjs \
//     --portal http://127.0.0.1:3000 --platform http://127.0.0.1:18000 --admin-key "$PLATFORM_API_KEYS" \
//     --wav q1.wav [--wav q2.wav …] --runs 10 [--chrome /path/to/chrome] [--out results.json]
//
// Each run: open the portal in a fresh headless Chromium whose microphone plays one WAV (silence, the question,
// then silence), press "Start AI Voice Call", wait for the assistant's reply in the transcript, press "End call".
// Afterwards it reads the session status from Platform (admin key), so cleanup is checked too, and the reply latency:
// `turn_to_first_audio` when the Runtime emits it, otherwise the gap between the stored SpeechStopped (the turn
// detector decided the user finished) and the next AgentSpeechStarted event (first reply audio). In agent mode the
// Runtime currently emits only the latter pair; both measure from end-of-turn detection, not from the last syllable.
// BILLABLE: every run spends ElevenLabs STT + TTS. Log credits before/after in docs/status.md.
const path = require("path");
const fs = require("fs");

function arg(name, fallback) {
  const values = [];
  process.argv.forEach((a, i) => { if (a === `--${name}`) values.push(process.argv[i + 1]); });
  return values.length ? values : fallback;
}
const portal = arg("portal", ["http://127.0.0.1:3000"])[0];
const platform = arg("platform", ["http://127.0.0.1:18000"])[0];
const adminKey = arg("admin-key", [process.env.PLATFORM_API_KEYS || ""])[0].split(",")[0];
const wavs = arg("wav", []);
const runs = Number(arg("runs", ["1"])[0]);
const chrome = arg("chrome", [process.env.CHROME_PATH || ""])[0];
const out = arg("out", [""])[0];
const { chromium } = require(path.resolve(process.env.PLAYWRIGHT_DIR || ".", "node_modules/playwright"));

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function platformGet(p) {
  const res = await fetch(`${platform}${p}`, { headers: adminKey ? { authorization: `Bearer ${adminKey}` } : {} });
  if (!res.ok) throw new Error(`${p} → ${res.status}`);
  return res.json();
}

async function oneRun(n, wav) {
  const browser = await chromium.launch({
    executablePath: chrome || undefined,
    args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", `--use-file-for-fake-audio-capture=${path.resolve(wav)}`, "--autoplay-policy=no-user-gesture-required"],
  });
  const page = await (await browser.newContext({ permissions: ["microphone"] })).newPage();
  let session = null;
  page.on("response", async (r) => {
    if (/\/demo\/agents\/[^/]+\/sessions$/.test(r.url()) && r.request().method() === "POST") session = await r.json().catch(() => null);
  });
  const result = { run: n, wav: path.basename(wav), ok: false };
  const t0 = Date.now();
  try {
    await page.goto(portal, { waitUntil: "networkidle" });
    await page.getByRole("button", { name: /Start AI Voice Call/ }).click();
    await page.getByRole("button", { name: "End call" }).waitFor({ timeout: 20_000 });
    result.connected_ms = Date.now() - t0;
    await page.locator('.portal-turn[data-role="assistant"]').first().waitFor({ timeout: 45_000 });
    await sleep(6_000); // let the reply finish and the transcript settle
    result.user = await page.locator('.portal-turn[data-role="user"]').allInnerTexts();
    result.assistant = await page.locator('.portal-turn[data-role="assistant"]').allInnerTexts();
    result.ok = result.assistant.length > 0;
  } catch (e) {
    result.error = String(e.message || e).split("\n")[0];
  }
  try {
    const end = page.getByRole("button", { name: "End call" });
    if (await end.isVisible().catch(() => false)) await end.click();
    await sleep(2_500);
  } finally {
    await browser.close();
  }
  if (session) {
    result.session_id = session.session_id;
    result.conversation_id = session.conversation_id;
    try {
      const s = await platformGet(`/voice/sessions/${session.session_id}`);
      result.session_status_after_end = s.status;
      const rs = await platformGet(`/conversations/${session.conversation_id}/runtime-state`);
      result.turn_to_first_audio_ms = rs.events.filter((e) => e.event_type === "LatencyMeasured" && e.payload.metric === "turn_to_first_audio").map((e) => e.payload.value_ms);
      if (!result.turn_to_first_audio_ms.length) {
        result.latency_source = "SpeechStopped→AgentSpeechStarted";
        let stopped = null;
        for (const e of rs.events) {
          if (e.event_type === "SpeechStopped") stopped = Date.parse(e.received_at);
          else if (e.event_type === "AgentSpeechStarted" && stopped !== null) { result.turn_to_first_audio_ms.push(Date.parse(e.received_at) - stopped); stopped = null; }
          else if (e.event_type === "SpeechStarted") stopped = null;
        }
      }
    } catch (e) {
      result.platform_error = String(e.message || e);
    }
  }
  return result;
}

(async () => {
  if (!wavs.length) throw new Error("give at least one --wav");
  const results = [];
  for (let n = 1; n <= runs; n++) {
    const r = await oneRun(n, wavs[(n - 1) % wavs.length]);
    results.push(r);
    console.log(JSON.stringify(r));
    await sleep(3_000);
  }
  const lat = results.flatMap((r) => r.turn_to_first_audio_ms || []).sort((a, b) => a - b);
  const median = lat.length ? (lat.length % 2 ? lat[(lat.length - 1) / 2] : (lat[lat.length / 2 - 1] + lat[lat.length / 2]) / 2) : null;
  const summary = {
    runs, ok: results.filter((r) => r.ok).length,
    cleaned_up: results.filter((r) => ["ended", "completed"].includes(r.session_status_after_end)).length,
    turn_to_first_audio_ms: { n: lat.length, median, min: lat[0] ?? null, max: lat.at(-1) ?? null },
  };
  console.log("SUMMARY", JSON.stringify(summary));
  if (out) fs.writeFileSync(out, JSON.stringify({ summary, results }, null, 2));
})();
