# Yes Chef

Browser-first AI cooking assistant with:
- live webcam capture from `getUserMedia()`
- OpenAI vision + tool-calling orchestration on a Node backend
- spoken guidance audio
- on-screen overlay instructions (example: `ADD 4 CUPS OF FLOUR`)

## Setup

```bash
npm install
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`.

## Run (Browser Mode)

```bash
npm run dev
```

Open [http://localhost:8787](http://localhost:8787), allow camera + microphone access, then press `Start` and say what you want to cook.

Interface flow:
- Full-screen camera with live overlay instructions.
- Compact top-left briefing panel with `COOKING` (dish name) only.
- Large centered on-screen step text for the current instruction.
- Press `Start` once to begin hands-free voice capture.
- Say your dish to begin (for example: "pancakes"), then keep speaking updates naturally.
- Say `new dish` or `start over` to reset the session.
- Completed recipes are saved to `data/recipes/recipes.yaml`.
- The assistant only saves a recipe when the dish is confirmed finished, and reuses saved recipes by default for the same dish name.
- The assistant can call a catalog lookup tool (`lookup_recipe`) to retrieve saved recipes; if nothing matches, it generates a best-practice recipe.

## Environment

- `OPENAI_MODEL` defaults to `gpt-5.2`
- `OPENAI_TTS_MODEL` defaults to `gpt-4o-mini-tts`
- `OPENAI_TTS_VOICE` defaults to `marin`
- `OPENAI_TTS_INSTRUCTIONS` controls voice personality/style
- `PORT` defaults to `8787`
- `CAPTURE_INTERVAL_MS` controls frame cadence from browser to backend (default `10000`; countdown starts after each frame response returns)
- `REQUIRE_VOICE=true` fails fast if TTS output fails
- `VERBOSE_TEXT_LOGS=false` keeps terminal transcript noise down
- `SPEECH_REPEAT_COOLDOWN_MS` throttles repeated spoken prompts (default `120000`)
- `QUESTION_REPEAT_COOLDOWN_MS` throttles repeated questions while waiting for an answer (default `180000`)
- `MIN_QUESTION_GAP_MS` enforces a minimum gap between non-urgent questions (default `600000`)
- Assistant is configured for a fixed benchtop camera viewpoint and patient turn-taking (silence-first when waiting).

## Scripts

- `npm run dev`: browser server mode
- `npm run dev:cli`: legacy terminal/webcam mode
- `npm run build`: compile TypeScript
- `npm start`: run compiled browser server
