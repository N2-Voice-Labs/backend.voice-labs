# Goal: Enable Real Human-to-Human Translated Calls Through Shareable Links

We are moving from infrastructure development to real-world product testing.

Our immediate objective is to make our existing real-time translation system usable by two real people through a simple browser-based voice call.

This is a prototyping and validation platform, not our final distribution channel. We may eventually integrate carrier calls, messengers, or native applications, but those are out of scope for now.

## Primary objective

Two people who speak different languages should be able to establish a real-time translated voice call without installing an application or creating accounts.

The experience should be approximately as simple as joining a Google Meet call.

### Intended user experience

**Person A:**
1. Opens our website.
2. Selects their language and the other participant's language.
3. Clicks Start Call.
4. Receives a shareable invitation link.
5. Sends the link through KakaoTalk, WhatsApp, SMS, or another messenger.

**Person B:**
1. Opens the invitation link.
2. Confirms their language.
3. Grants microphone access.
4. Joins the existing call.
5. Immediately begins communicating through our translation system.

Both participants should hear translations in their respective languages.

## Technical objectives

### 1. Reuse existing infrastructure

We already have:

- LiveKit.
- Bidirectional real-time translation.
- ElevenLabs STT/TTS.
- Google Cloud Translation with Gemini fallback.
- Per-call session isolation.
- Audio routing.
- Optional cinematic dubbing.
- Latency instrumentation.

Audit the existing application before implementing anything.

Reuse existing components and avoid unnecessary architectural changes.

### 2. Implement browser-based calls

Build the minimum frontend and backend functionality required to:

- Create a call.
- Generate a secure invitation link.
- Join an existing call.
- Connect both participants to the same translation session.
- Display connection status.
- Mute/unmute the microphone.
- Hang up.
- End and clean up abandoned sessions.

Support desktop and mobile browsers.

### 3. Make the experience simple

Prioritize a minimal, functional interface.

Avoid unnecessary onboarding, registration, complex settings, or additional features.

The recipient should be able to join with as few interactions as possible.

Handle microphone permissions, audio playback restrictions, connection failures, and reconnections clearly.

### 4. Translation and dubbing

Both participants must receive the correct translations.

For testing, support two audio experiences:

- AI translation only.
- Live original voice with cinematic ducking.

Use the existing dubbing implementation.

Make the original voice quieter when the AI translation starts. Start by testing approximately −24 dB during translated playback.

Allow the experience to be selected for testing, but preserve existing production defaults.

Do not introduce additional translation latency.

### 5. Security and reliability

Even though this is a prototype, implement basic safeguards:

- Unpredictable invitation links.
- Expiring invitations.
- Short-lived LiveKit access tokens.
- Maximum two human participants per room.
- No API credentials exposed to browsers.
- Session cleanup after calls.
- Protection against unauthorized joins.
- Basic limits against resource abuse.

### 6. Real-world testing

The application should allow us to test:

- Korean ↔ English translation.
- End-to-end conversational latency.
- Translation accuracy.
- Sentence splitting.
- Interruptions and overlapping speech.
- AI-only versus cinematic dubbing.
- Audio echo with headphones and speakers.
- Mobile versus desktop behavior.
- Connection stability.

Keep existing instrumentation available for debugging.

### Constraints

- Do not implement SIP, carrier calling, or messenger voice-call integrations.
- Do not build native mobile applications.
- Do not introduce unnecessary infrastructure.
- Preserve existing translation architecture.
- Do not consume ElevenLabs credits or paid calling resources without authorization.
- Keep the implementation modular enough to support additional calling platforms later.

## Definition of success

I can open our website on one device, create a translated call, copy its invitation link, open it on another device, and establish a real bidirectional conversation.

One person speaks Korean.

The other speaks English.

Both hear accurate translations through our existing Runtime.

The call works on desktop and mobile browsers, and both participants can disconnect normally.

## Development instructions

You have discretion over the technical implementation.

First inspect the repository and determine what already exists.

Implement the smallest maintainable solution that meets the objectives.

Write appropriate automated tests and run the relevant test suites.

Do not claim that browser compatibility or real-call performance has been verified until actual device testing has occurred.

Commit completed and verified work.

At completion, report:

1. What was implemented.
2. How to start the application locally.
3. How to create and join a translated call.
4. What external configuration is required.
5. What has been tested automatically.
6. What requires manual testing on real devices.
7. Any limitations affecting real-world use.

Prioritize getting two real people into a working translated conversation over adding features.
