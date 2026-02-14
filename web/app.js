const videoEl = document.getElementById("cameraPreview");
const canvasEl = document.getElementById("frameCanvas");
const overlayEl = document.getElementById("overlay");
const overlayTextEl = document.getElementById("overlayText");
const cookingNameEl = document.getElementById("cookingName");
const nextStepTextEl = document.getElementById("nextStepText");
const introPromptEl = document.getElementById("introPrompt");
const startButtonEl = document.getElementById("startButton");

let captureIntervalMs = 10000;
let captureTimer = null;
let captureLoopToken = 0;
let frameInFlight = false;
let stream = null;
let sessionStarted = false;
let socket = null;

const speechQueue = [];
let drainingSpeech = false;

const SpeechRecognitionApi = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;
let listening = false;
let assistantActive = false;
let recognitionRestartTimer = null;

function showOverlay(text, priority = "normal") {
  overlayTextEl.textContent = text;
  overlayEl.classList.remove("overlay-hidden");
  overlayEl.classList.toggle("urgent", priority === "urgent");
}

function clearOverlay() {
  overlayEl.classList.add("overlay-hidden");
  overlayEl.classList.remove("urgent");
  overlayTextEl.textContent = "";
}

function setPanel(panel) {
  const cooking = String(panel.cooking || "What are we cooking?");
  const nextStep = typeof panel.nextStep === "string" ? panel.nextStep : "";
  cookingNameEl.textContent = cooking;
  nextStepTextEl.textContent = nextStep;
}

function resetPanelPrompt() {
  setPanel({
    cooking: "What are we cooking?",
    nextStep: "Press START and say the dish name."
  });
}

function setStartButtonState() {
  startButtonEl.classList.toggle("listening", assistantActive || listening);
  startButtonEl.textContent = listening ? "Listening..." : "START";
}

function showIntroPrompt() {
  introPromptEl.classList.remove("hidden");
}

function hideIntroPrompt() {
  introPromptEl.classList.add("hidden");
}

function scheduleRecognitionStart(delayMs = 180) {
  if (!assistantActive || !recognition) {
    return;
  }

  if (recognitionRestartTimer) {
    clearTimeout(recognitionRestartTimer);
    recognitionRestartTimer = null;
  }

  recognitionRestartTimer = setTimeout(() => {
    recognitionRestartTimer = null;
    if (!assistantActive || listening) {
      return;
    }
    try {
      recognition.start();
    } catch (error) {
      const text = error instanceof Error ? error.message : String(error);
      if (!text.includes("InvalidStateError")) {
        console.error(`Speech start failed: ${text}`);
      }
    }
  }, delayMs);
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error || `Request failed (${response.status})`);
  }
  return response.json();
}

async function loadConfig() {
  const response = await fetch("/api/config");
  if (!response.ok) {
    throw new Error("Unable to load config.");
  }
  const config = await response.json();
  captureIntervalMs = Number(config.captureIntervalMs) || 10000;
}

function connectSocket() {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  socket = new WebSocket(`${protocol}//${location.host}`);

  socket.addEventListener("message", (event) => {
    const payload = JSON.parse(event.data);
    handleServerEvent(payload);
  });

  socket.addEventListener("close", () => {
    setTimeout(connectSocket, 1200);
  });
}

function handleServerEvent(event) {
  switch (event.type) {
    case "ready":
      captureIntervalMs = Number(event.captureIntervalMs) || captureIntervalMs;
      if (event.panel) {
        setPanel(event.panel);
      } else {
        resetPanelPrompt();
      }
      if (event.overlay) {
        showOverlay(event.overlay.text, event.overlay.priority);
      }
      break;
    case "panel_set":
      setPanel(event.panel);
      break;
    case "panel_clear":
      resetPanelPrompt();
      break;
    case "overlay_set":
      showOverlay(event.overlay.text, event.overlay.priority);
      break;
    case "overlay_clear":
      clearOverlay();
      break;
    case "speech":
      enqueueSpeech(event);
      break;
    case "error":
      console.error(event.message);
      break;
    default:
      break;
  }
}

function enqueueSpeech(event) {
  speechQueue.push(event);
  if (!drainingSpeech) {
    drainingSpeech = true;
    void drainSpeechQueue();
  }
}

async function drainSpeechQueue() {
  try {
    while (speechQueue.length > 0) {
      const event = speechQueue.shift();
      if (!event) {
        continue;
      }
      const src = `data:${event.mimeType};base64,${event.audioBase64}`;
      const audio = new Audio(src);
      await audio.play();
      await new Promise((resolve, reject) => {
        audio.addEventListener("ended", resolve, { once: true });
        audio.addEventListener("error", () => reject(new Error("Audio playback failed.")), {
          once: true
        });
      });
    }
  } catch (error) {
    const text = error instanceof Error ? error.message : String(error);
    console.error(`Audio blocked: ${text}`);
  } finally {
    drainingSpeech = false;
  }
}

async function startCamera() {
  if (stream) {
    return;
  }

  stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "environment" },
    audio: false
  });
  videoEl.srcObject = stream;
  await videoEl.play();
}

function startCaptureLoop() {
  stopCaptureLoop();
  const loopToken = captureLoopToken;
  void captureTick(loopToken);
}

function stopCaptureLoop() {
  captureLoopToken += 1;
  if (captureTimer) {
    clearTimeout(captureTimer);
    captureTimer = null;
  }
}

function scheduleNextCapture(loopToken) {
  if (!sessionStarted || loopToken !== captureLoopToken) {
    return;
  }

  captureTimer = setTimeout(() => {
    captureTimer = null;
    void captureTick(loopToken);
  }, captureIntervalMs);
}

async function captureTick(loopToken) {
  if (!sessionStarted || loopToken !== captureLoopToken) {
    return;
  }
  if (frameInFlight) {
    scheduleNextCapture(loopToken);
    return;
  }
  if (!videoEl.videoWidth || !videoEl.videoHeight) {
    scheduleNextCapture(loopToken);
    return;
  }

  frameInFlight = true;
  try {
    const maxWidth = 960;
    const scale = Math.min(1, maxWidth / videoEl.videoWidth);
    canvasEl.width = Math.round(videoEl.videoWidth * scale);
    canvasEl.height = Math.round(videoEl.videoHeight * scale);

    const ctx = canvasEl.getContext("2d");
    if (!ctx) {
      throw new Error("Could not prepare frame.");
    }

    ctx.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height);
    const dataUrl = canvasEl.toDataURL("image/jpeg", 0.72);
    const imageBase64 = dataUrl.split(",")[1];
    await postJson("/api/frame", {
      imageBase64,
      mimeType: "image/jpeg",
      takenAt: new Date().toISOString()
    });
  } catch (error) {
    const text = error instanceof Error ? error.message : String(error);
    console.error(`Frame send failed: ${text}`);
  } finally {
    frameInFlight = false;
    scheduleNextCapture(loopToken);
  }
}

function setListeningState(active) {
  listening = active;
  setStartButtonState();
}

async function processTranscript(transcript) {
  const normalized = transcript.trim();
  if (!normalized) {
    return;
  }

  const lower = normalized.toLowerCase();
  if (lower === "new dish" || lower === "start over" || lower === "reset") {
    await resetSessionPrompt();
    return;
  }

  if (!sessionStarted) {
    await postJson("/api/session/start", { recipeIdea: normalized });
    sessionStarted = true;
    hideIntroPrompt();
    setPanel({
      cooking: normalized.toUpperCase(),
      nextStep: ""
    });
    clearOverlay();
    startCaptureLoop();
    return;
  }

  await postJson("/api/user-message", { message: normalized });
}

function setupRecognition() {
  if (!SpeechRecognitionApi) {
    return;
  }

  recognition = new SpeechRecognitionApi();
  recognition.lang = "en-US";
  recognition.interimResults = false;
  recognition.continuous = false;
  recognition.maxAlternatives = 1;

  recognition.addEventListener("start", () => {
    setListeningState(true);
  });

  recognition.addEventListener("end", () => {
    setListeningState(false);
    scheduleRecognitionStart(140);
  });

  recognition.addEventListener("error", (event) => {
    setListeningState(false);

    if (event.error === "no-speech" || event.error === "aborted") {
      scheduleRecognitionStart(150);
      return;
    }

    if (event.error === "not-allowed" || event.error === "service-not-allowed") {
      assistantActive = false;
      setStartButtonState();
      return;
    }

    scheduleRecognitionStart(220);
  });

  recognition.addEventListener("result", (event) => {
    const transcript = event.results?.[0]?.[0]?.transcript?.trim();
    if (!transcript) {
      return;
    }

    void processTranscript(transcript).catch((error) => {
      const text = error instanceof Error ? error.message : String(error);
      console.error(`Request failed: ${text}`);
    });
  });
}

async function fallbackPromptFlow() {
  const text = window.prompt(sessionStarted ? "Say update:" : "What are we cooking?");
  const transcript = text ? text.trim() : "";
  if (!transcript) {
    return;
  }

  await processTranscript(transcript);
}

async function handleStart() {
  if (!assistantActive) {
    assistantActive = true;
    setStartButtonState();
  }

  if (!recognition) {
    await fallbackPromptFlow();
    return;
  }

  scheduleRecognitionStart(40);
}

async function resetSessionPrompt() {
  await postJson("/api/session/reset", {});
  sessionStarted = false;
  stopCaptureLoop();
  showIntroPrompt();
  resetPanelPrompt();
}

startButtonEl.addEventListener("click", () => {
  void handleStart().catch((error) => {
    const text = error instanceof Error ? error.message : String(error);
    console.error(text);
  });
});

async function boot() {
  setStartButtonState();
  await loadConfig();
  await startCamera();
  setupRecognition();
  connectSocket();
  showIntroPrompt();
  resetPanelPrompt();
}

void boot().catch((error) => {
  const text = error instanceof Error ? error.message : String(error);
  console.error(`Startup failed: ${text}`);
});
