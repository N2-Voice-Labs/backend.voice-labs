#!/usr/bin/env node
/**
 * Validates the browser-side audio timing (frontend `lib/audio-latency.ts`) with real browsers, real LiveKit and a SYNTHETIC
 * publisher standing in for the Runtime: no providers, no credits. Needs a running Platform (public calls on, no Runtime) and the
 * built frontend (see docs/calls.md), and LiveKit settings in the environment (python publisher):
 *
 *   FRONTEND_DIR=../frontend.voice-labs WEB_URL=http://127.0.0.1:3000 PYTHON=python3 node scripts/browser_latency_check.cjs
 *
 * Two people join a call (desktop creator, phone-sized guest) with diagnostics on. The publisher plays, on each person's private
 * listen track, tones with silence between them: a SHORT translation (0.5 s, twice, 3 s apart) for the creator and one LONG
 * translation (3 s) for the guest, starting together. Then it reads each page's `window.__voiceLabsLatency.export()` and checks:
 * per-listener numbering, silence not counted, one listener's long audio not delaying the other's, and track/listener tagging.
 * The latency figures printed compare the browser's onset with the publisher's hand-over time on the SAME host
 * (performance.timeOrigin + at, time.time()); they validate the instrumentation, they are not a Runtime measurement.
 */
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');
const os = require('os');
const dir = path.resolve(process.env.FRONTEND_DIR || '../frontend.voice-labs');
const { chromium, devices } = require(path.join(dir, 'node_modules/@playwright/test'));
const WEB = process.env.WEB_URL || 'http://127.0.0.1:3000';
const PY = process.env.PYTHON || 'python3';
const args = ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream', '--autoplay-policy=no-user-gesture-required'];
let failures = 0;
const check = (name, ok, extra = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${extra ? '  ' + extra : ''}`); if (!ok) failures += 1; };

async function person(browser, device) {
  const ctx = await browser.newContext({ ...(device || { viewport: { width: 1280, height: 900 } }), permissions: ['microphone'] });
  await ctx.addInitScript(() => { try { localStorage.setItem('vl-latency-diag', '1'); } catch {} });
  const page = await ctx.newPage();
  page.on('pageerror', (e) => console.log('  page error:', String(e).slice(0, 160)));
  return page;
}
async function checkMic(page) {
  await page.getByRole('button', { name: /Check microphone/ }).click();
  await page.getByText('Say something to test it').waitFor({ timeout: 15000 });
}

(async () => {
  const browser = await chromium.launch({ channel: process.env.BROWSER_CHANNEL || 'chrome', args });
  const creator = await person(browser);
  const guest = await person(browser, devices['iPhone 13']);
  let creatorListen = null, guestListen = null, creatorId = null, guestId = null;
  creator.on('response', async (r) => { if (r.url().endsWith('/calls') && r.request().method() === 'POST') { const b = await r.json(); creatorListen = b.connection.listen; creatorId = `browser-${b.session_id}`; } });
  guest.on('response', async (r) => { if (/\/join$/.test(r.url()) && r.request().method() === 'POST') { const b = await r.json(); guestListen = b.connection.listen; guestId = b.participant_identity; } });

  await creator.goto(`${WEB}/call`);
  await checkMic(creator);
  await creator.getByRole('button', { name: /Start call/ }).click();
  await creator.getByRole('heading', { name: 'Invite the other person' }).waitFor({ timeout: 30000 });
  await creator.getByRole('button', { name: /Create invite link/ }).click();
  const link = await creator.getByLabel('Invite link').inputValue();
  await guest.goto(link);
  await guest.getByText(/invited you to a call/).waitFor({ timeout: 15000 });
  await checkMic(guest);
  await guest.getByRole('button', { name: /Join call/ }).click();
  await guest.getByRole('toolbar', { name: 'Call controls' }).waitFor({ timeout: 30000 });
  await creator.getByText(/joined the call/).first().waitFor({ timeout: 20000 });
  check('both people joined, each with a private listen connection', Boolean(creatorListen && guestListen), `${creatorListen && creatorListen.track} | ${guestListen && guestListen.track}`);

  const spec = [
    { room: creatorListen.room_name, track: creatorListen.track, tones: [[1.0, 0.5], [4.0, 0.5]] },
    { room: guestListen.room_name, track: guestListen.track, tones: [[1.0, 3.0]] },
  ];
  const specFile = path.join(os.tmpdir(), `latency-spec-${process.pid}.json`);
  fs.writeFileSync(specFile, JSON.stringify(spec));
  const out = await new Promise((resolve, reject) => {
    const child = spawn(PY, [path.join(__dirname, 'synthetic_translation_publisher.py'), specFile], { env: process.env });
    let text = '', err = '';
    child.stdout.on('data', (d) => (text += d)); child.stderr.on('data', (d) => (err += d));
    child.on('close', (code) => (code === 0 ? resolve(text) : reject(new Error(`publisher exited ${code}: ${err.slice(-300)}`))));
  });
  fs.unlinkSync(specFile);
  await creator.waitForTimeout(1500);
  const sent = out.trim().split('\n').map((l) => JSON.parse(l));
  const get = async (page) => { const t = await page.evaluate(() => window.__voiceLabsLatency && window.__voiceLabsLatency.export()); return t ? JSON.parse(t) : null; };
  const A = await get(creator), B = await get(guest);
  check('diagnostics were on and exported by both pages', Boolean(A && B));
  const of = (x, name) => x.events.filter((e) => e.name === name);

  for (const [who, x, id, listen] of [['creator', A, creatorId, creatorListen], ['guest', B, guestId, guestListen]]) {
    check(`${who}: listener room connected and translated track subscribed`, of(x, 'listener_room_connected').length === 1 && of(x, 'translated_track_subscribed').length >= 1);
    const sub = of(x, 'translated_track_subscribed')[0];
    check(`${who}: events carry this listener and their own track`, sub && sub.listener === id && sub.track === listen.track, sub ? `${sub.listener} / ${sub.track}` : '');
    check(`${who}: no event names another listener's track`, x.events.every((e) => !e.track || e.track === listen.track));
  }
  const onA = of(A, 'browser_first_voiced_audio'), endA = of(A, 'browser_voiced_audio_ended');
  const onB = of(B, 'browser_first_voiced_audio'), endB = of(B, 'browser_voiced_audio_ended');
  check('creator heard two separate utterances, numbered 1 and 2', onA.length === 2 && onA[0].seq === 1 && onA[1].seq === 2, `n=${onA.length}`);
  check('guest heard one utterance', onB.length === 1 && onB[0].seq === 1, `n=${onB.length}`);

  const epoch = (x, e) => x.time_origin_epoch_ms + e.at;
  const toneOf = (room, start) => sent.find((s) => s.room === room && s.tone_start_s === start);
  const rows = [
    ['creator #1 (0.5 s)', epoch(A, onA[0]) - toneOf(creatorListen.room_name, 1.0).handed_to_livekit_epoch_s * 1000],
    ['creator #2 (0.5 s)', epoch(A, onA[1]) - toneOf(creatorListen.room_name, 4.0).handed_to_livekit_epoch_s * 1000],
    ['guest   #1 (3.0 s)', epoch(B, onB[0]) - toneOf(guestListen.room_name, 1.0).handed_to_livekit_epoch_s * 1000],
  ];
  console.log('\n  publisher hand-over -> browser_first_voiced_audio (same host clock; validation only):');
  for (const [name, ms] of rows) console.log(`    ${name}: ${Math.round(ms)} ms`);
  check('silence before the first tone was not counted (onsets come after the hand-over)', rows.every(([, ms]) => ms > 0));
  check('onsets are plausible (under 2 s after hand-over)', rows.every(([, ms]) => ms < 2000));
  check("the guest's long translation did not delay the creator's first audio", Math.abs(rows[0][1] - rows[2][1]) < 500, `difference ${Math.round(Math.abs(rows[0][1] - rows[2][1]))} ms`);
  check("the gap between the creator's two onsets matches the publisher (3.0 s)", Math.abs((onA[1].at - onA[0].at) / 1000 - 3.0) < 0.4, `${((onA[1].at - onA[0].at) / 1000).toFixed(2)} s`);
  check("the guest's utterance lasted about 3 s", endB.length === 1 && Math.abs((endB[0].at - onB[0].at) / 1000 - 3.0) < 0.5, endB[0] ? `${((endB[0].at - onB[0].at) / 1000).toFixed(2)} s` : 'no end');
  check("the creator's utterances lasted about 0.5 s", endA.length === 2 && endA.every((e, i) => Math.abs((e.at - onA[i].at) / 1000 - 0.5) < 0.35));
  check('playback started on both pages', of(A, 'playback_started').length >= 1 && of(B, 'playback_started').length >= 1);
  check('RTP arrival was seen on both tracks', of(A, 'first_rtp_audio_received').length >= 1 && of(B, 'first_rtp_audio_received').length >= 1);
  check('no failure events (analyser, playback, element, listener connect)', [A, B].every((x) => x.events.every((e) => !/failed|error|unavailable|blocked|stalled/.test(e.name))), JSON.stringify([A, B].flatMap((x) => x.events.filter((e) => /failed|error|unavailable|blocked|stalled/.test(e.name)).map((e) => [e.name, e.detail || null]))));
  console.log('  informational: element_waiting events (browser buffering at startup):', [A, B].map((x) => of(x, 'element_waiting').length).join(' / '));
  check('the export holds no credentials or speech', !/vls_|token|eyJ/i.test(JSON.stringify([A, B])));

  await browser.close();
  console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
  process.exit(failures ? 1 : 0);
})().catch((e) => { console.error('ERROR', e.message.slice(0, 600)); process.exit(2); });
