import "dotenv/config";
import { createInterface } from "node:readline";
import { CameraCapture, CameraDependencyError } from "./camera";
import { CookingAssistant } from "./assistant";
import { AudioPlayer } from "./audio";
import { CookingStateStore } from "./state";
import { TimerManager } from "./timers";

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
  if (Number.isNaN(parsed) || parsed <= 0) {
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

async function askQuestion(prompt: string): Promise<string> {
  const rl = createInterface({
    input: process.stdin,
    output: process.stdout
  });

  return new Promise<string>((resolve) => {
    rl.question(prompt, (answer) => {
      rl.close();
      resolve(answer.trim());
    });
  });
}

async function main(): Promise<void> {
  const apiKey = getRequiredEnv("OPENAI_API_KEY");
  const model = process.env.OPENAI_MODEL?.trim() || "gpt-5.2";
  const ttsModel = process.env.OPENAI_TTS_MODEL?.trim() || "gpt-4o-mini-tts";
  const ttsVoice = process.env.OPENAI_TTS_VOICE?.trim() || "marin";
  const ttsInstructions = process.env.OPENAI_TTS_INSTRUCTIONS?.trim() || undefined;

  const captureIntervalMs = getNumberEnv("CAPTURE_INTERVAL_MS", 10000);
  const cameraDeviceId = process.env.CAMERA_DEVICE_ID?.trim() || undefined;
  const requireCamera = getBooleanEnv("REQUIRE_CAMERA", true);
  const requireVoice = getBooleanEnv("REQUIRE_VOICE", true);
  const verboseTextLogs = getBooleanEnv("VERBOSE_TEXT_LOGS", false);

  const state = new CookingStateStore();
  const timers = new TimerManager();
  const audio = new AudioPlayer({
    apiKey,
    model: ttsModel,
    voice: ttsVoice,
    instructions: ttsInstructions
  });

  const assistant = new CookingAssistant({
    apiKey,
    model,
    state,
    timers,
    audio,
    verboseTextLogs
  });

  let camera: CameraCapture | null = null;
  try {
    camera = new CameraCapture({
      intervalMs: captureIntervalMs,
      deviceId: cameraDeviceId
    });
  } catch (error) {
    if (error instanceof CameraDependencyError) {
      if (requireCamera) {
        throw error;
      }
      console.error(`[camera] ${error.message}`);
    } else {
      const message = error instanceof Error ? error.message : String(error);
      if (requireCamera) {
        throw new Error(`Camera initialization failed: ${message}`);
      }
      console.error(`[camera] Initialization failed: ${message}`);
    }
  }

  if (requireVoice) {
    await audio.enqueueSpeech("Yes Chef is online.");
  }

  timers.on("changed", (activeTimers) => {
    state.setActiveTimers(activeTimers);
  });

  timers.on("fired", (timer) => {
    console.log(`[timer] "${timer.label}" finished.`);
    void assistant.handleTimerFired(timer).catch((error) => {
      const message = error instanceof Error ? error.message : String(error);
      console.error(`[assistant] Failed to handle timer event: ${message}`);
    });
  });

  const recipeIdea = await askQuestion("What do you want to cook today? ");
  if (!recipeIdea) {
    throw new Error("Recipe idea is required to start.");
  }

  await assistant.initializeWithRecipeIdea(recipeIdea);

  if (camera) {
    try {
      console.log("[startup] Capturing camera test frame...");
      const testFrame = await camera.captureFrame();
      console.log(
        `[startup] Camera frame OK (${Math.round(testFrame.buffer.length / 1024)} KB, ${testFrame.takenAt}).`
      );
      await assistant.processFrame(testFrame);
      camera.start((frame) => assistant.processFrame(frame));
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (requireCamera) {
        const permissionHint = message.includes("Camera access not granted")
          ? "\nGrant camera access to your terminal app in System Settings > Privacy & Security > Camera, then restart the app."
          : "";
        throw new Error(`Camera startup capture failed: ${message}${permissionHint}`);
      }
      console.error(`[camera] Startup capture failed: ${message}`);
      console.error("[camera] Continuing in text-only mode.");
      camera = null;
    }
  } else {
    console.log("[startup] Running in text-only mode (camera unavailable).");
  }

  console.log(
    `[startup] Monitoring started. Capture interval: ${captureIntervalMs}ms. Type updates anytime (/quit to exit).`
  );

  const input = createInterface({
    input: process.stdin,
    output: process.stdout
  });

  let shuttingDown = false;

  const shutdown = async (reason: string): Promise<void> => {
    if (shuttingDown) {
      return;
    }
    shuttingDown = true;

    console.log(`[shutdown] ${reason}`);
    camera?.stop();
    timers.clearAllTimers();
    input.close();
  };

  input.on("line", (line) => {
    const trimmed = line.trim();
    if (!trimmed) {
      return;
    }

    if (trimmed.toLowerCase() === "/quit") {
      void shutdown("user requested exit");
      return;
    }

    void assistant.handleUserMessage(trimmed).catch((error) => {
      const message = error instanceof Error ? error.message : String(error);
      console.error(`[assistant] Failed to process message: ${message}`);
    });
  });

  process.on("SIGINT", () => {
    void shutdown("SIGINT");
  });
}

main().catch((error) => {
  const message = error instanceof Error ? error.stack || error.message : String(error);
  console.error(message);
  process.exit(1);
});
