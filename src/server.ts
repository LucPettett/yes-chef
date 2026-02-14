import "dotenv/config";
import express from "express";
import http from "node:http";
import path from "node:path";
import { WebSocketServer, WebSocket } from "ws";
import {
  AssistantOutputs,
  BrowserCookingAssistant,
  CompletedRecipePayload,
  InstructionPanel,
  OverlayUpdate,
  VisionFrame
} from "./assistant-browser";
import { RecipeStore, normalizeDishKey } from "./recipe-store";
import { CookingStateStore } from "./state";
import { TimerManager } from "./timers";
import { TextToSpeechService } from "./tts";

type ServerEvent =
  | {
      type: "ready";
      captureIntervalMs: number;
      state: unknown;
      overlay: OverlayPayload | null;
      panel: PanelPayload | null;
    }
  | { type: "speech"; text: string; audioBase64: string; mimeType: string; timestamp: string }
  | { type: "overlay_set"; overlay: OverlayPayload }
  | { type: "overlay_clear" }
  | { type: "panel_set"; panel: PanelPayload }
  | { type: "panel_clear" }
  | { type: "timer_fired"; label: string; firedAt: string }
  | { type: "state"; state: unknown }
  | { type: "error"; message: string };

interface OverlayPayload {
  text: string;
  priority: "normal" | "urgent";
  expiresAt: string | null;
}

interface PanelPayload {
  cooking: string;
  nextStep: string;
  updatedAt: string;
}

interface FrameRequestBody {
  imageBase64?: string;
  mimeType?: string;
  takenAt?: string;
}

function getRequiredEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) {
    throw new Error(`Missing required environment variable: ${name}`);
  }
  return value;
}

function getNumberEnv(name: string, fallback: number): number {
  const raw = process.env[name];
  if (!raw) {
    return fallback;
  }
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return fallback;
  }
  return parsed;
}

function getBooleanEnv(name: string, fallback: boolean): boolean {
  const raw = process.env[name];
  if (!raw) {
    return fallback;
  }

  const normalized = raw.trim().toLowerCase();
  if (["1", "true", "yes", "on"].includes(normalized)) {
    return true;
  }
  if (["0", "false", "no", "off"].includes(normalized)) {
    return false;
  }
  return fallback;
}

function normalizeBase64(input: string): string {
  const trimmed = input.trim();
  const dataUrlIndex = trimmed.indexOf("base64,");
  if (dataUrlIndex >= 0) {
    return trimmed.slice(dataUrlIndex + "base64,".length);
  }
  return trimmed;
}

function userConfirmedCompletion(
  message: string,
  waitingForAnswer: boolean,
  lastQuestionText: string
): boolean {
  const normalized = message.trim().toLowerCase();
  if (!normalized) {
    return false;
  }

  if (/\b(not|isn't|isnt|don't|dont)\s+(done|finished|complete|completed)\b/i.test(normalized)) {
    return false;
  }

  if (/\b(done|finished|all done|complete|completed|that's it|thats it|we('?| a)re done|it's done|its done)\b/i.test(normalized)) {
    return true;
  }

  if (!waitingForAnswer) {
    return false;
  }

  const askedAboutCompletion = /\b(done|finished|complete|completed|wrap up|stop now)\b/i.test(
    lastQuestionText
  );
  const affirmative = /^(yes|yep|yeah|correct|sure|ok|okay|affirmative)[.!]*$/i.test(normalized);
  return askedAboutCompletion && affirmative;
}

async function main(): Promise<void> {
  const apiKey = getRequiredEnv("OPENAI_API_KEY");
  const model = process.env.OPENAI_MODEL?.trim() || "gpt-5.2";
  const ttsModel = process.env.OPENAI_TTS_MODEL?.trim() || "gpt-4o-mini-tts";
  const ttsVoice = process.env.OPENAI_TTS_VOICE?.trim() || "marin";
  const ttsInstructions = process.env.OPENAI_TTS_INSTRUCTIONS?.trim() || undefined;
  const port = getNumberEnv("PORT", 8787);
  const captureIntervalMs = getNumberEnv("CAPTURE_INTERVAL_MS", 10000);
  const requireVoice = getBooleanEnv("REQUIRE_VOICE", true);
  const verboseTextLogs = getBooleanEnv("VERBOSE_TEXT_LOGS", false);
  const speechRepeatCooldownMs = getNumberEnv("SPEECH_REPEAT_COOLDOWN_MS", 120000);
  const questionRepeatCooldownMs = getNumberEnv("QUESTION_REPEAT_COOLDOWN_MS", 180000);
  const minQuestionGapMs = getNumberEnv("MIN_QUESTION_GAP_MS", 600000);

  const app = express();
  const server = http.createServer(app);
  const wsServer = new WebSocketServer({ server });
  const clients = new Set<WebSocket>();

  const state = new CookingStateStore();
  const timers = new TimerManager();
  const recipeStore = new RecipeStore();
  await recipeStore.ensureFile();
  const tts = new TextToSpeechService({
    apiKey,
    model: ttsModel,
    voice: ttsVoice,
    instructions: ttsInstructions
  });

  let overlay: OverlayPayload | null = null;
  let overlayTimeout: NodeJS.Timeout | null = null;
  let panel: PanelPayload | null = null;
  let lastSpokenTextKey = "";
  let lastSpokenAtMs = 0;
  let waitingForUserAnswer = false;
  let lastQuestionAtMs = 0;
  let lastQuestionText = "";
  let activeDishKey = "";
  let recipeStoredForSession = false;
  let completionConfirmedForSession = false;

  const broadcast = (event: ServerEvent): void => {
    const payload = JSON.stringify(event);
    for (const client of clients) {
      if (client.readyState === WebSocket.OPEN) {
        client.send(payload);
      }
    }
  };

  const publishState = (): void => {
    broadcast({ type: "state", state: state.getSnapshot() });
  };

  const clearOverlay = (): void => {
    if (overlayTimeout) {
      clearTimeout(overlayTimeout);
      overlayTimeout = null;
    }
    overlay = null;
    broadcast({ type: "overlay_clear" });
  };

  const setOverlay = (update: OverlayUpdate): void => {
    if (overlayTimeout) {
      clearTimeout(overlayTimeout);
      overlayTimeout = null;
    }

    const ttlMs = update.ttlSeconds ? Math.max(1000, Math.round(update.ttlSeconds * 1000)) : null;
    overlay = {
      text: update.text,
      priority: update.priority,
      expiresAt: ttlMs ? new Date(Date.now() + ttlMs).toISOString() : null
    };
    broadcast({ type: "overlay_set", overlay });

    if (ttlMs) {
      overlayTimeout = setTimeout(() => {
        clearOverlay();
      }, ttlMs);
    }
  };

  const setPanel = (next: InstructionPanel): void => {
    panel = {
      cooking: next.cooking,
      nextStep: next.nextStep,
      updatedAt: new Date().toISOString()
    };
    broadcast({ type: "panel_set", panel });
  };

  const clearPanel = (): void => {
    panel = null;
    broadcast({ type: "panel_clear" });
  };

  const completeRecipe = async (
    payload: CompletedRecipePayload
  ): Promise<Record<string, unknown>> => {
    const dish = payload.dish.trim();
    const recipe = payload.recipe.trim();
    if (!dish || !recipe) {
      return { ok: false, saved: false, error: "Both dish and recipe are required." };
    }

    const dishKey = normalizeDishKey(dish);
    if (!dishKey) {
      return { ok: false, saved: false, error: "Could not normalize dish name." };
    }

    if (!activeDishKey) {
      return { ok: false, saved: false, error: "No active cooking session." };
    }

    if (dishKey !== activeDishKey) {
      return {
        ok: false,
        saved: false,
        error: `Dish "${dish}" does not match active session dish.`
      };
    }

    if (recipeStoredForSession) {
      return { ok: true, saved: false, reason: "already_saved_for_session" };
    }

    if (!completionConfirmedForSession) {
      return {
        ok: false,
        saved: false,
        error: "Recipe can only be saved after user confirms the dish is finished."
      };
    }

    const saved = await recipeStore.saveCompletedRecipe({ dish, recipe });
    recipeStoredForSession = true;
    completionConfirmedForSession = false;
    state.addObservation(
      `Saved completed recipe "${saved.dish}" (${saved.timesCooked} total cook${saved.timesCooked === 1 ? "" : "s"}).`
    );

    return { ok: true, saved: true, recipe: saved };
  };

  const lookupRecipe = async (dishInput: string): Promise<Record<string, unknown>> => {
    const dish = dishInput.trim();
    if (!dish) {
      return {
        found: false,
        query: dishInput,
        matchType: "none",
        error: "dish is required"
      };
    }

    const lookup = await recipeStore.lookupByDish(dish);
    if (!lookup.found || !lookup.recipe) {
      return {
        found: false,
        query: dish,
        matchType: "none"
      };
    }

    return {
      found: true,
      query: dish,
      matchType: lookup.matchType,
      dish: lookup.recipe.dish,
      dishKey: lookup.recipe.dishKey,
      recipe: lookup.recipe.recipe,
      completedAt: lookup.recipe.completedAt,
      timesCooked: lookup.recipe.timesCooked
    };
  };

  const outputs: AssistantOutputs = {
    speak: async (message) => {
      const text = message.trim();
      if (!text) {
        return;
      }

      const now = Date.now();
      const textKey = text.replace(/\s+/g, " ").trim().toLowerCase();
      const isQuestion = /\?\s*$/.test(text);
      const isUrgent = /\b(stop|danger|urgent|fire|smoke|burn|hot oil|knife|raw chicken|gas)\b/i.test(
        text
      );
      const isRoutinePreferenceQuestion =
        isQuestion &&
        /\b(what|which)\b.*\b(kind|type|brand|milk|flour|pan|oil|butter|sugar|salt)\b/i.test(text);

      if (textKey === lastSpokenTextKey && now - lastSpokenAtMs < speechRepeatCooldownMs) {
        return;
      }

      if (waitingForUserAnswer && !isUrgent) {
        return;
      }

      if (isQuestion && waitingForUserAnswer && now - lastQuestionAtMs < questionRepeatCooldownMs) {
        return;
      }

      if (isRoutinePreferenceQuestion) {
        return;
      }

      if (isQuestion && !isUrgent && now - lastQuestionAtMs < minQuestionGapMs) {
        return;
      }

      try {
        const audio = await tts.synthesize(text);
        broadcast({
          type: "speech",
          text,
          audioBase64: audio.base64,
          mimeType: audio.mimeType,
          timestamp: new Date().toISOString()
        });
        lastSpokenTextKey = textKey;
        lastSpokenAtMs = now;
        if (isQuestion) {
          waitingForUserAnswer = true;
          lastQuestionAtMs = now;
          lastQuestionText = text;
        }
      } catch (error) {
        const errText = error instanceof Error ? error.message : String(error);
        if (requireVoice) {
          throw new Error(`Voice output failed: ${errText}`);
        }
        broadcast({ type: "error", message: `Voice output failed: ${errText}` });
      }
    },
    setOverlay,
    clearOverlay,
    setPanel,
    clearPanel,
    lookupRecipe,
    completeRecipe
  };

  const assistant = new BrowserCookingAssistant({
    apiKey,
    model,
    state,
    timers,
    outputs
  });

  timers.on("changed", (activeTimers) => {
    state.setActiveTimers(activeTimers);
    publishState();
  });

  timers.on("fired", (timer) => {
    broadcast({ type: "timer_fired", label: timer.label, firedAt: timer.firedAt });
    void assistant.handleTimerFired(timer).catch((error) => {
      const message = error instanceof Error ? error.message : String(error);
      broadcast({ type: "error", message: `Failed handling timer event: ${message}` });
    });
  });

  app.use(express.json({ limit: "12mb" }));

  app.get("/api/config", (_req, res) => {
    res.json({
      captureIntervalMs,
      requireVoice
    });
  });

  app.get("/api/state", (_req, res) => {
    res.json({
      state: state.getSnapshot(),
      overlay,
      panel
    });
  });

  app.post("/api/session/reset", (_req, res) => {
    try {
      timers.clearAllTimers();
      state.reset();
      assistant.resetSession();
      waitingForUserAnswer = false;
      lastQuestionAtMs = 0;
      lastQuestionText = "";
      lastSpokenAtMs = 0;
      lastSpokenTextKey = "";
      activeDishKey = "";
      recipeStoredForSession = false;
      completionConfirmedForSession = false;
      clearOverlay();
      clearPanel();
      publishState();
      res.json({ ok: true });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      res.status(500).json({ error: message });
    }
  });

  app.post("/api/session/start", async (req, res) => {
    const recipeIdea = typeof req.body?.recipeIdea === "string" ? req.body.recipeIdea.trim() : "";
    if (!recipeIdea) {
      res.status(400).json({ error: "recipeIdea is required." });
      return;
    }

    try {
      const priorRecipe = await recipeStore.findByDish(recipeIdea);
      await assistant.initializeWithRecipeIdea(recipeIdea, {
        priorRecipe:
          priorRecipe === null
            ? null
            : {
                dish: priorRecipe.dish,
                recipe: priorRecipe.recipe,
                completedAt: priorRecipe.completedAt,
                timesCooked: priorRecipe.timesCooked
              }
      });
      if (priorRecipe) {
        state.addObservation(
          `Loaded saved recipe for "${priorRecipe.dish}" (${priorRecipe.timesCooked} previous cook${priorRecipe.timesCooked === 1 ? "" : "s"}).`
        );
      }
      activeDishKey = normalizeDishKey(recipeIdea);
      recipeStoredForSession = false;
      completionConfirmedForSession = false;
      waitingForUserAnswer = false;
      lastQuestionText = "";
      publishState();
      res.json({ ok: true, restoredRecipe: Boolean(priorRecipe) });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      res.status(500).json({ error: message });
    }
  });

  app.post("/api/frame", async (req, res) => {
    const body = req.body as FrameRequestBody;
    if (!body?.imageBase64 || typeof body.imageBase64 !== "string") {
      res.status(400).json({ error: "imageBase64 is required." });
      return;
    }

    const frame: VisionFrame = {
      takenAt:
        typeof body.takenAt === "string" && body.takenAt.trim()
          ? body.takenAt
          : new Date().toISOString(),
      mimeType:
        typeof body.mimeType === "string" && body.mimeType.trim() ? body.mimeType : "image/jpeg",
      base64: normalizeBase64(body.imageBase64)
    };

    try {
      await assistant.processFrame(frame);
      publishState();
      res.json({ ok: true });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      res.status(500).json({ error: message });
    }
  });

  app.post("/api/user-message", async (req, res) => {
    const message = typeof req.body?.message === "string" ? req.body.message.trim() : "";
    if (!message) {
      res.status(400).json({ error: "message is required." });
      return;
    }

    try {
      if (userConfirmedCompletion(message, waitingForUserAnswer, lastQuestionText)) {
        completionConfirmedForSession = true;
      }
      waitingForUserAnswer = false;
      lastQuestionText = "";
      await assistant.handleUserMessage(message);
      publishState();
      res.json({ ok: true });
    } catch (error) {
      const errText = error instanceof Error ? error.message : String(error);
      res.status(500).json({ error: errText });
    }
  });

  const webRoot = path.resolve(__dirname, "../web");
  app.use(express.static(webRoot));
  app.get("*", (_req, res) => {
    res.sendFile(path.join(webRoot, "index.html"));
  });

  wsServer.on("connection", (socket) => {
    clients.add(socket);

    socket.send(
      JSON.stringify({
        type: "ready",
        captureIntervalMs,
        state: state.getSnapshot(),
        overlay,
        panel
      } satisfies ServerEvent)
    );

    socket.on("message", (data) => {
      if (!verboseTextLogs) {
        return;
      }
      const text = typeof data === "string" ? data : data.toString();
      console.log(`[ws] ${text}`);
    });

    socket.on("close", () => {
      clients.delete(socket);
    });

    socket.on("error", () => {
      clients.delete(socket);
    });
  });

  server.listen(port, () => {
    console.log(`Yes Chef server running: http://localhost:${port}`);
  });
}

main().catch((error) => {
  const message = error instanceof Error ? error.stack || error.message : String(error);
  console.error(message);
  process.exit(1);
});
