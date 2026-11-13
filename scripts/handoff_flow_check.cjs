// Real-provider check of the whole UNAI demo journey with two fake-microphone browsers (no people needed):
// student opens the portal → AI voice call → "Connect to Admissions Officer" → officer opens /staff, picks up →
// translated EN↔KO call → the officer hangs up → both ended. Prints the translated call's transcript.
//
//   STAFF_KEY=… PLAYWRIGHT_DIR=../frontend.voice-labs node scripts/handoff_flow_check.cjs \
//     --portal http://127.0.0.1:3000 --platform http://127.0.0.1:18000 --admin-key "$PLATFORM_API_KEYS" \
//     --student-wav student_en.wav --staff-wav staff_ko.wav [--talk-s 110] [--chrome /path/to/chrome] [--out r.json]
//
// The WAVs hold silence-separated sentences (English for the student, Korean for the officer); Chromium plays each
// file as that browser's microphone. BILLABLE: STT + translation + TTS for every sentence. Log credits in docs/status.md.
const path = require("path");
const fs = require("fs");
const arg = (n, d) => { const i = process.argv.indexOf(`--${n}`); return i > 0 ? process.argv[i + 1] : d; };
const portal = arg("portal", "http://127.0.0.1:3000");
const platform = arg("platform", "http://127.0.0.1:18000");
const adminKey = (arg("admin-key", process.env.PLATFORM_API_KEYS || "")).split(",")[0];
const staffKey = process.env.STAFF_KEY || "";
const talkMs = Number(arg("talk-s", "110")) * 1000;
const chrome = arg("chrome", process.env.CHROME_PATH || "");
const out = arg("out", "");
const { chromium } = require(path.resolve(process.env.PLAYWRIGHT_DIR || ".", "node_modules/playwright"));
const t0 = Date.now();
const log = (...a) => console.log(`[${((Date.now() - t0) / 1000).toFixed(1)}s]`, ...a);
const get = async (p) => (await fetch(`${platform}${p}`, { headers: { authorization: `Bearer ${adminKey}` } })).json();
let failures = 0;
const check = (name, ok, extra = "") => { log(`${ok ? "PASS" : "FAIL"}  ${name}${extra ? "  " + extra : ""}`); if (!ok) failures += 1; };

async function person(wav, label) {
  const browser = await chromium.launch({ executablePath: chrome || undefined,
    args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", `--use-file-for-fake-audio-capture=${path.resolve(wav)}`, "--autoplay-policy=no-user-gesture-required"] });
  const page = await (await browser.newContext({ permissions: ["microphone"], viewport: { width: 1280, height: 900 } })).newPage();
  page.on("pageerror", (e) => log(`${label} PAGE ERROR`, String(e).slice(0, 200)));
  return { browser, page };
}

(async () => {
  if (!staffKey || !adminKey) throw new Error("set STAFF_KEY and --admin-key");
  const result = {};
  const student = await person(arg("student-wav"), "student");
  let handoff = null, aiSession = null;
  student.page.on("response", async (r) => {
    if (r.request().method() !== "POST") return;
    if (/\/handoffs$/.test(r.url())) handoff = await r.json().catch(() => null);
    if (/\/demo\/agents\/[^/]+\/sessions$/.test(r.url())) aiSession = await r.json().catch(() => null);
  });
  await student.page.goto(portal, { waitUntil: "networkidle" });
  await student.page.getByRole("button", { name: /Start AI Voice Call/ }).click();
  await student.page.getByRole("button", { name: "End call" }).waitFor({ timeout: 20_000 });
  check("student is in the AI voice call", true);
  await student.page.waitForTimeout(2_000);
  await student.page.locator(".portal-controls").getByRole("button", { name: "Connect to Admissions Officer" }).click();
  await student.page.getByRole("alertdialog").getByRole("button", { name: "Connect to Admissions Officer" }).click();
  await student.page.getByText(/Waiting for an admissions officer/).first().waitFor({ timeout: 30_000 });
  check("student waits for an officer in the translated call", !!handoff, handoff ? `handoff ${handoff.handoff_id}` : "");
  if (aiSession) check("the AI session was ended by the handoff", (await get(`/voice/sessions/${aiSession.session_id}`)).status === "ended");

  const staff = await person(arg("staff-wav"), "staff");
  await staff.page.goto(`${portal}/staff`, { waitUntil: "networkidle" });
  await staff.page.getByLabel("Staff key").fill(staffKey);
  await staff.page.getByRole("button", { name: "Open the queue" }).click();
  await staff.page.getByRole("button", { name: "Pick up" }).first().waitFor({ timeout: 15_000 });
  check("the waiting student is in the staff queue", true);
  await staff.page.getByRole("button", { name: "Pick up" }).first().click();
  await staff.page.getByText(/invited you to a call/).waitFor({ timeout: 20_000 });
  await staff.page.getByRole("button", { name: /Check microphone/ }).click();
  await staff.page.getByText("Say something to test it").waitFor({ timeout: 15_000 });
  await staff.page.getByRole("button", { name: /Join call/ }).click();
  await staff.page.getByRole("toolbar", { name: "Call controls" }).waitFor({ timeout: 30_000 });
  const joinedAt = Date.now();
  check("the officer joined the translated call", true, `${((joinedAt - t0) / 1000).toFixed(1)}s after start`);
  await student.page.getByText(/joined the call/).first().waitFor({ timeout: 20_000 }).then(() => check("the student sees the officer join", true), () => check("the student sees the officer join", false));

  log(`talking for ${talkMs / 1000}s…`);
  await staff.page.waitForTimeout(talkMs);
  await staff.page.getByRole("button", { name: /End call/ }).first().click();
  await staff.page.getByRole("dialog").getByRole("button", { name: /End call/ }).last().click();
  await staff.page.getByText(/Call ended/).first().waitFor({ timeout: 15_000 }).then(() => check("officer sees the call ended", true), () => check("officer sees the call ended", false));
  await student.page.getByText(/ended the call|Call ended/).first().waitFor({ timeout: 15_000 }).then(() => check("student sees the call ended", true), () => check("student sees the call ended", false));
  if (handoff) {
    const callId = handoff.session_id;
    await new Promise((r) => setTimeout(r, 2_500));
    check("the call session is ended on the Platform", (await get(`/voice/sessions/${callId}`)).status === "ended");
    const conv = handoff.conversation_id;
    const messages = conv ? await get(`/conversations/${conv}/messages`) : [];
    result.messages = Array.isArray(messages) ? messages.map((m) => ({ role: m.role, lang: m.primary_language, content: m.content, at: m.created_at })) : messages;
    for (const m of result.messages || []) log("  ", m.role, m.lang || "", "|", m.content);
  }
  await staff.browser.close();
  await student.browser.close();
  result.failures = failures;
  if (out) fs.writeFileSync(out, JSON.stringify({ handoff_id: handoff?.handoff_id, ...result }, null, 2));
  log(failures ? `${failures} check(s) failed` : "all checks passed");
  process.exit(failures ? 1 : 0);
})().catch((e) => { console.error("FLOW ERROR", String(e.message || e).slice(0, 600)); process.exit(2); });
