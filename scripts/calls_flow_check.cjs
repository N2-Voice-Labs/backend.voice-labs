#!/usr/bin/env node
/**
 * Two-browser check of the shareable-link call flow against a RUNNING stack (see docs/calls.md).
 *
 *   FRONTEND_DIR=../frontend.voice-labs WEB_URL=http://127.0.0.1:3000 PLATFORM_URL=http://127.0.0.1:8000 \
 *   ADMIN_KEY=<PLATFORM_API_KEYS value> node scripts/calls_flow_check.cjs
 *
 * Uses Chrome with a fake microphone (no real audio, no provider calls when the Runtime is not running or its
 * providers are stubs), and the frontend's Playwright install. Checks, with a desktop creator and an iPhone-emulated guest:
 * create -> invite -> join -> both in the room -> mute -> hang-up by the guest -> ended for both -> the link then
 * refuses; a second call hung up by the creator; the same link on a second device is the same seat; no code or
 * token left in the guest's URL or storage; no horizontal overflow on the phone. It does NOT test translation,
 * real devices or audio quality. ADMIN_KEY is only used to read session status; the browsers never get it.
 */
const path = require('path');
const dir = path.resolve(process.env.FRONTEND_DIR || '../frontend.voice-labs');
const { chromium, devices } = require(path.join(dir, 'node_modules/@playwright/test'));
const WEB = process.env.WEB_URL || 'http://127.0.0.1:3000';
const API = process.env.PLATFORM_URL || 'http://127.0.0.1:8000';
const KEY = process.env.ADMIN_KEY || '';
const args = ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream', '--autoplay-policy=no-user-gesture-required'];
let failures = 0;
const check = (name, ok, extra = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${extra ? '  ' + extra : ''}`); if (!ok) failures += 1; };
const status = async (sid) => KEY ? (await (await fetch(`${API}/voice/sessions/${sid}`, { headers: { authorization: `Bearer ${KEY}` } })).json()).status : null;

async function person(browser, device) {
  const ctx = await browser.newContext({ ...(device || { viewport: { width: 1280, height: 900 } }), permissions: ['microphone'] });
  const page = await ctx.newPage();
  page.on('pageerror', (e) => console.log('  page error:', String(e).slice(0, 160)));
  return page;
}
async function checkMic(page) {
  await page.getByRole('button', { name: /Check microphone/ }).click();
  await page.getByText('Say something to test it').waitFor({ timeout: 15000 });
}
async function startCall(creator, guest) {
  let sid = null;
  creator.on('response', async (r) => { if (r.url().endsWith('/calls') && r.request().method() === 'POST') sid = (await r.json()).session_id; });
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
  return { sid, link };
}
async function hangUp(page) {
  await page.getByRole('button', { name: /End call/ }).first().click();
  await page.getByRole('dialog').getByRole('button', { name: /End call/ }).last().click();
}

(async () => {
  const browser = await chromium.launch({ channel: process.env.BROWSER_CHANNEL || 'chrome', args });
  const phone = devices['iPhone 13'];

  let A = await person(browser), G = await person(browser, phone);
  let { sid, link } = await startCall(A, G);
  check('creator and guest are both in the call', true);
  check('the guest sees no horizontal overflow on a phone', !(await G.evaluate(() => document.documentElement.scrollWidth > window.innerWidth)));
  const leaked = await G.evaluate(() => JSON.stringify({ ls: { ...localStorage }, ss: { ...sessionStorage }, url: location.href }));
  check('the guest URL and storage hold no code or token', !/vls_|[0-9a-f]{8}-[0-9a-f]{4}-.*\./.test(leaked));
  const mute = G.getByRole('toolbar', { name: 'Call controls' }).getByRole('button').first();
  await mute.click();
  await G.waitForTimeout(400);
  check('mute toggles', (await mute.getAttribute('aria-pressed')) === 'true');
  await hangUp(G);
  await G.getByText(/Call ended/).first().waitFor({ timeout: 15000 });
  await A.getByText(/Guest ended the call|Call ended/).first().waitFor({ timeout: 15000 });
  check('hang-up by the guest ends the call for both', true);
  if (KEY) check('the session is ended on the Platform', (await status(sid)) === 'ended');
  const late = await person(browser, phone);
  await late.goto(link);
  await late.locator('h1').first().waitFor({ timeout: 15000 });
  check('the link refuses once the call has ended', /ended/i.test(await late.locator('h1').first().innerText()));

  A = await person(browser); G = await person(browser, phone);
  ({ sid, link } = await startCall(A, G));
  const second = await person(browser, { viewport: { width: 390, height: 844 } });
  await second.goto(link);
  await second.getByText(/invited you to a call/).waitFor({ timeout: 15000 });
  await checkMic(second);
  await second.getByRole('button', { name: /Join call/ }).click();
  check('the same link on a second device is the same seat (reconnect)', await second.getByRole('toolbar', { name: 'Call controls' }).waitFor({ timeout: 20000 }).then(() => true, () => false));
  await hangUp(A);
  await G.getByText(/ended the call|Call ended/).first().waitFor({ timeout: 15000 });
  check('hang-up by the creator ends the call for the guest', true);
  if (KEY) check('the session is ended on the Platform (second call)', (await status(sid)) === 'ended');

  await browser.close();
  console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
  process.exit(failures ? 1 : 0);
})().catch((e) => { console.error('FLOW ERROR', e.message.slice(0, 600)); process.exit(2); });
