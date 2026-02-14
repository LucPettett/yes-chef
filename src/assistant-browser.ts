import OpenAI from "openai";
import { CookingStateStore } from "./state";
import { TimerFiredEvent, TimerManager } from "./timers";

export interface VisionFrame {
  takenAt: string;
  mimeType: string;
  base64: string;
}

export interface OverlayUpdate {
  text: string;
  priority: "normal" | "urgent";
  ttlSeconds?: number;
}

export interface CompletedRecipePayload {
  dish: string;
  recipe: string;
}

export interface PreviousRecipeContext {
  dish: string;
  recipe: string;
  completedAt: string;
  timesCooked: number;
}

export interface AssistantOutputs {
  speak: (message: string) => Promise<void>;
  setOverlay: (overlay: OverlayUpdate) => void;
  clearOverlay: () => void;
  setPanel: (panel: InstructionPanel) => void;
  clearPanel: () => void;
  lookupRecipe: (dish: string) => Promise<Record<string, unknown>>;
  completeRecipe: (payload: CompletedRecipePayload) => Promise<Record<string, unknown>>;
}

export interface InstructionPanel {
  cooking: string;
  nextStep: string;
}

interface AssistantOptions {
  apiKey: string;
  model: string;
  state: CookingStateStore;
  timers: TimerManager;
  outputs: AssistantOutputs;
}

type FunctionCallOutput = {
  type: "function_call_output";
  call_id: string;
  output: string;
};

type StepStatus = "not_started" | "in_progress" | "complete" | "unclear";

interface FrameAssessment {
  observation: string;
  stepStatus: StepStatus;
  confidence: number;
  reason: string;
}

interface VisionHistoryEntry {
  takenAtMs: number;
  observation: string;
  stepStatus: StepStatus;
}

const MAX_VISION_HISTORY = 10;
const HISTORY_PROMPT_WINDOW = 6;
const ROUTINE_SPEAK_MIN_GAP_MS = 45000;

const STOPWORDS = new Set<string>([
  "a",
  "an",
  "and",
  "are",
  "as",
  "at",
  "be",
  "by",
  "for",
  "from",
  "has",
  "have",
  "in",
  "into",
  "is",
  "it",
  "of",
  "on",
  "or",
  "that",
  "the",
  "then",
  "to",
  "up",
  "with",
  "your",
  "you"
]);

const VISION_ANALYSIS_PROMPT = `
You are a strict frame-by-frame cooking progress validator.
Analyze exactly one kitchen frame against provided history and the locked current step.
Assume ingredient amounts are correct; judge only visual progress.
Return JSON only (no markdown):
{"observation":"...", "step_status":"not_started|in_progress|complete|unclear", "confidence":0.0, "reason":"..."}
Constraints:
- observation: max 14 words
- reason: max 18 words
- step_status "complete" only when there is clear visual evidence the locked current step is done
- if uncertain, use "unclear"
`;

const SYSTEM_PROMPT = `
You are Yes Chef, a focused real-time cooking assistant.

You receive periodic camera frames from a browser webcam and should guide cooking safely.
The camera is fixed on the benchtop/countertop facing the user; do not expect kitchen-wide views.

Rules:
- Be concise and practical.
- Speak only when there is a clear next step, correction, or urgent safety issue.
- Use overlay for short imperative instructions that should remain visible.
- Use all-caps for overlay text when it is urgent and immediate.
- Keep the panel current using set_panel:
  - COOKING: dish name
  - NEXT STEP: one concise actionable sentence
- Treat NEXT STEP as a locked step. Never advance to a new step until the latest frame confirms completion.
- Never list multiple future steps in one response.
- When the locked step is not complete, keep guidance focused on that same step and prefer stay_silent(reason).
- Workflow order:
  1) identify dish and recipe
  2) ingredient gathering one-by-one
  3) cooking execution one step at a time
- During ingredient gathering, each NEXT STEP must request exactly one ingredient item (for example: "Grab 2 eggs.").
- After each ingredient appears on the bench, move to the next single ingredient step.
- Do not show planning chatter on the panel (no "analyzing", no ingredient lists, no multi-step summaries).
- Keep the cooking plan updated with update_plan and progress with update_state.
- For each frame, keep update_state observations very concise and visual-only.
- Use timers whenever timing matters.
- When you need a recipe, call lookup_recipe(dish) first.
- If lookup_recipe finds a saved recipe, use it as the default unless the user asks to change it.
- If lookup_recipe does not find anything, create a practical best-practice recipe yourself.
- If a previous completed recipe for this dish is provided, use it as the default.
- Only after the user confirms the dish is finished, call complete_recipe exactly once.
- Assume portions for 3 kids and 2 adults unless the user says otherwise.
- Do not ask about allergies or dietary restrictions.
- Do not ask ingredient-availability questions after startup. Confirm the dish idea once, then proceed confidently.
- Assume normal home-cooking defaults unless told otherwise:
  - whole milk
  - all-purpose flour
  - standard eggs
  - unsalted butter or neutral oil
- Do not ask routine preference questions (for example milk type, flour type, pan type, brand).
- Prefer direct instructions over questions.
- Ask a question only if safety-critical or truly blocked.
- If you ask a question, wait patiently for the user's answer. Do not repeat yourself quickly.
- Only repeat a question after a long wait and only if the user is clearly in frame.
- If no speech is needed right now, call stay_silent(reason) instead of speak().

Available tools:
- speak(message): audio guidance to user.
- stay_silent(reason): explicitly choose silence and wait.
- set_timer(duration_seconds, label)
- cancel_timer(label)
- update_plan(changes)
- update_state(observation)
- lookup_recipe(dish)
- set_panel(cooking, next_step)
- clear_panel()
- set_overlay(text, priority, ttl_seconds)
- clear_overlay()
- complete_recipe(dish, recipe)
- run_python(code)

If uncertain from the frame, ask one short clarifying question via speak().
`;

const TOOLS: Array<Record<string, unknown>> = [
  {
    type: "function",
    name: "speak",
    description: "Speak an instruction aloud to the cook.",
    strict: true,
    parameters: {
      type: "object",
      properties: {
        message: { type: "string" }
      },
      required: ["message"],
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "stay_silent",
    description: "Intentionally do not speak and wait for more evidence or user input.",
    strict: true,
    parameters: {
      type: "object",
      properties: {
        reason: { type: "string" }
      },
      required: ["reason"],
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "set_timer",
    description: "Set a cooking timer in seconds.",
    strict: true,
    parameters: {
      type: "object",
      properties: {
        duration_seconds: { type: "number", minimum: 1 },
        label: { type: "string" }
      },
      required: ["duration_seconds", "label"],
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "cancel_timer",
    description: "Cancel an active timer by label.",
    strict: true,
    parameters: {
      type: "object",
      properties: {
        label: { type: "string" }
      },
      required: ["label"],
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "update_plan",
    description: "Update the cooking plan as new context appears.",
    strict: true,
    parameters: {
      type: "object",
      properties: {
        changes: { type: "string" }
      },
      required: ["changes"],
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "update_state",
    description: "Record a silent observation about progress or setup.",
    strict: true,
    parameters: {
      type: "object",
      properties: {
        observation: { type: "string" }
      },
      required: ["observation"],
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "lookup_recipe",
    description: "Look up a saved recipe by dish name from the local catalog.",
    strict: true,
    parameters: {
      type: "object",
      properties: {
        dish: { type: "string" }
      },
      required: ["dish"],
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "set_panel",
    description: "Update the on-screen panel with cooking name and the single next step.",
    strict: true,
    parameters: {
      type: "object",
      properties: {
        cooking: { type: "string" },
        next_step: { type: "string" }
      },
      required: ["cooking", "next_step"],
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "clear_panel",
    description: "Clear the on-screen briefing panel.",
    strict: true,
    parameters: {
      type: "object",
      properties: {},
      required: [],
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "set_overlay",
    description: "Show a short on-screen overlay instruction.",
    strict: true,
    parameters: {
      type: "object",
      properties: {
        text: { type: "string" },
        priority: { type: "string", enum: ["normal", "urgent"] },
        ttl_seconds: {
          type: ["number", "null"],
          minimum: 1,
          maximum: 600
        }
      },
      required: ["text", "priority", "ttl_seconds"],
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "clear_overlay",
    description: "Clear any currently visible overlay text.",
    strict: true,
    parameters: {
      type: "object",
      properties: {},
      required: [],
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "complete_recipe",
    description: "Persist a final completed recipe for future reuse.",
    strict: true,
    parameters: {
      type: "object",
      properties: {
        dish: { type: "string" },
        recipe: { type: "string" }
      },
      required: ["dish", "recipe"],
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "run_python",
    description: "Run Python for math or conversions through code interpreter.",
    strict: true,
    parameters: {
      type: "object",
      properties: {
        code: { type: "string" }
      },
      required: ["code"],
      additionalProperties: false
    }
  }
];

export class BrowserCookingAssistant {
  private readonly client: OpenAI;
  private readonly model: string;
  private readonly state: CookingStateStore;
  private readonly timers: TimerManager;
  private readonly outputs: AssistantOutputs;
  private previousResponseId: string | undefined;
  private queue: Promise<void> = Promise.resolve();
  private currentDish = "";
  private currentStep: string | null = null;
  private pendingNextStep: string | null = null;
  private latestFrameAssessment: FrameAssessment | null = null;
  private frameAllowsStepAdvance = true;
  private visionHistory: VisionHistoryEntry[] = [];
  private completedStepKeys = new Set<string>();
  private lastRoutineSpeakAtMs = 0;

  constructor(options: AssistantOptions) {
    this.client = new OpenAI({ apiKey: options.apiKey });
    this.model = options.model;
    this.state = options.state;
    this.timers = options.timers;
    this.outputs = options.outputs;
  }

  resetSession(): void {
    this.previousResponseId = undefined;
    this.queue = Promise.resolve();
    this.currentDish = "";
    this.currentStep = null;
    this.pendingNextStep = null;
    this.latestFrameAssessment = null;
    this.frameAllowsStepAdvance = true;
    this.visionHistory = [];
    this.completedStepKeys = new Set<string>();
    this.lastRoutineSpeakAtMs = 0;
  }

  initializeWithRecipeIdea(
    recipeIdea: string,
    options?: { priorRecipe?: PreviousRecipeContext | null }
  ): Promise<void> {
    return this.enqueue(async () => {
      this.currentDish = recipeIdea;
      this.currentStep = null;
      this.pendingNextStep = null;
      this.latestFrameAssessment = null;
      this.frameAllowsStepAdvance = true;
      this.visionHistory = [];
      this.completedStepKeys = new Set<string>();
      this.lastRoutineSpeakAtMs = 0;
      this.state.setOriginalPlan(recipeIdea);
      this.state.recordConversation("user", recipeIdea);
      const priorRecipe = options?.priorRecipe ?? null;
      const startupContext = priorRecipe
        ? [
            `The user wants to cook: ${recipeIdea}`,
            "Default to the previously completed recipe unless the user asks to change it.",
            `Previously completed recipe for "${priorRecipe.dish}" (times cooked: ${priorRecipe.timesCooked}, completed at: ${priorRecipe.completedAt}):`,
            priorRecipe.recipe
          ].join("\n\n")
        : [
            `The user wants to cook: ${recipeIdea}`,
            "Before inventing a new recipe, call lookup_recipe with this dish."
          ].join("\n\n");
      await this.sendTextTurn(startupContext);
    });
  }

  processFrame(frame: VisionFrame): Promise<void> {
    return this.enqueue(async () => {
      const snapshot = this.state.getSnapshot();
      const assessment = await this.analyzeFrame(frame, snapshot.originalPlan, snapshot.completedSteps);
      this.latestFrameAssessment = assessment;
      this.frameAllowsStepAdvance = !this.currentStep || assessment.stepStatus === "complete";

      this.state.addObservation(
        `vision: ${this.formatHistoryEntry(
          this.visionHistory[this.visionHistory.length - 1] ?? {
            takenAtMs: Date.now(),
            observation: assessment.observation,
            stepStatus: assessment.stepStatus
          }
        )}`
      );

      const stepGateLine = this.currentStep
        ? this.frameAllowsStepAdvance
          ? `Locked step appears complete: "${this.currentStep}". You may advance exactly one step now.`
          : `Locked step still in progress: "${this.currentStep}". Do not advance to a new step.`
        : "No locked step exists yet. Set the first step when ready.";

      const frameSummary = [
        `Frame timestamp: ${frame.takenAt}`,
        "Camera setup: fixed benchtop/countertop facing the user; no roaming kitchen view.",
        `Current plan:\n${snapshot.originalPlan || "(none yet)"}`,
        `Current dish: ${this.currentDish || "(not set yet)"}`,
        `Locked current step: ${this.currentStep || "(not set yet)"}`,
        `Candidate next step: ${this.pendingNextStep || "(none)"}`,
        `Frame assessment: ${assessment.observation} (status=${assessment.stepStatus}, confidence=${assessment.confidence.toFixed(
          2
        )}, reason=${assessment.reason})`,
        `Recent visual history:\n${this.buildVisionHistoryText()}`,
        `Step gate: ${stepGateLine}`,
        `Completed steps: ${snapshot.completedSteps.join(" | ") || "(none yet)"}`,
        `Active timers: ${
          snapshot.activeTimers.map((t) => `${t.label} (${t.endsAt})`).join(", ") || "(none)"
        }`,
        `Recent observations: ${snapshot.recentObservations.slice(-8).join(" | ") || "(none)"}`
      ].join("\n");

      const response = await this.client.responses.create({
        model: this.model,
        instructions: SYSTEM_PROMPT,
        previous_response_id: this.previousResponseId,
        tools: TOOLS as never,
        tool_choice: "auto",
        store: true,
        input: [
          {
            role: "user",
            content: [
              { type: "input_text", text: frameSummary },
              {
                type: "input_image",
                image_url: `data:${frame.mimeType};base64,${frame.base64}`,
                detail: "low"
              }
            ]
          }
        ]
      });

      this.previousResponseId = response.id;
      await this.resolveResponse(response);
    });
  }

  handleUserMessage(message: string): Promise<void> {
    return this.enqueue(async () => {
      this.state.recordConversation("user", message);
      await this.sendTextTurn(`User update: ${message}`);
    });
  }

  handleTimerFired(timer: TimerFiredEvent): Promise<void> {
    return this.enqueue(async () => {
      this.state.addObservation(
        `Timer "${timer.label}" fired at ${timer.firedAt} (duration ${timer.durationSeconds}s).`
      );
      await this.sendTextTurn(
        `Timer event: "${timer.label}" just completed. Decide whether to notify now and next action.`
      );
    });
  }

  private async analyzeFrame(
    frame: VisionFrame,
    recipePlan: string,
    completedSteps: string[]
  ): Promise<FrameAssessment> {
    const recentHistory = this.buildVisionHistoryText(HISTORY_PROMPT_WINDOW);
    const lastKnownFrame =
      this.visionHistory.length > 0
        ? this.formatHistoryEntry(this.visionHistory[this.visionHistory.length - 1])
        : "(none yet)";

    let outputText = "";
    try {
      const response = await this.client.responses.create({
        model: this.model,
        instructions: VISION_ANALYSIS_PROMPT,
        store: false,
        input: [
          {
            role: "user",
            content: [
              {
                type: "input_text",
                text: [
                  `Frame timestamp: ${frame.takenAt}`,
                  `Recipe plan:\n${recipePlan || "(none yet)"}`,
                  `Locked current step: ${this.currentStep || "(not set yet)"}`,
                  `Candidate next step: ${this.pendingNextStep || "(none)"}`,
                  `Completed steps: ${completedSteps.join(" | ") || "(none yet)"}`,
                  `Recent frame history:\n${recentHistory}`,
                  `Last known frame: ${lastKnownFrame}`
                ].join("\n")
              },
              {
                type: "input_image",
                image_url: `data:${frame.mimeType};base64,${frame.base64}`,
                detail: "low"
              }
            ]
          }
        ]
      });
      outputText = typeof response.output_text === "string" ? response.output_text : "";
    } catch {
      outputText = "";
    }

    const parsed = this.parseFrameAssessment(outputText);
    const defaultStatus: StepStatus = this.currentStep ? "unclear" : "not_started";
    const assessment: FrameAssessment = {
      observation: this.trimToWordLimit(
        typeof parsed.observation === "string" ? parsed.observation : "No clear visual change.",
        14
      ),
      stepStatus: this.coerceStepStatus(parsed.step_status ?? parsed.stepStatus, defaultStatus),
      confidence: this.coerceConfidence(parsed.confidence),
      reason: this.trimToWordLimit(
        typeof parsed.reason === "string" ? parsed.reason : "Insufficient visual evidence.",
        18
      )
    };

    const entry: VisionHistoryEntry = {
      takenAtMs: this.parseTimestampMs(frame.takenAt),
      observation: assessment.observation,
      stepStatus: assessment.stepStatus
    };
    this.visionHistory.push(entry);
    if (this.visionHistory.length > MAX_VISION_HISTORY) {
      this.visionHistory = this.visionHistory.slice(-MAX_VISION_HISTORY);
    }

    return assessment;
  }

  private buildVisionHistoryText(limit = MAX_VISION_HISTORY): string {
    const recent = this.visionHistory.slice(-limit);
    if (recent.length === 0) {
      return "(none yet)";
    }
    return recent.map((entry) => this.formatHistoryEntry(entry)).join("\n");
  }

  private formatHistoryEntry(entry: VisionHistoryEntry): string {
    const firstTimestamp = this.visionHistory[0]?.takenAtMs ?? entry.takenAtMs;
    const elapsedMs = Math.max(0, entry.takenAtMs - firstTimestamp);
    const statusSuffix = entry.stepStatus === "complete" ? " (step complete)" : "";
    return `${this.formatElapsedMs(elapsedMs)} - ${entry.observation}${statusSuffix}`;
  }

  private formatElapsedMs(elapsedMs: number): string {
    const totalSeconds = Math.max(0, Math.floor(elapsedMs / 1000));
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    if (minutes > 0) {
      return `${minutes}m ${seconds}s`;
    }
    return `${seconds}s`;
  }

  private parseTimestampMs(isoTimestamp: string): number {
    const parsed = Date.parse(isoTimestamp);
    if (!Number.isFinite(parsed)) {
      return Date.now();
    }
    return parsed;
  }

  private parseFrameAssessment(text: string): Record<string, unknown> {
    const parsed = this.tryParseJsonObject(text);
    if (!parsed) {
      return {};
    }
    return parsed;
  }

  private tryParseJsonObject(text: string): Record<string, unknown> | null {
    const direct = this.tryParseUnknownJson(text);
    if (direct && !Array.isArray(direct)) {
      return direct;
    }

    const match = text.match(/\{[\s\S]*\}/);
    if (!match) {
      return null;
    }
    const fromMatch = this.tryParseUnknownJson(match[0]);
    if (fromMatch && !Array.isArray(fromMatch)) {
      return fromMatch;
    }
    return null;
  }

  private tryParseUnknownJson(text: string): Record<string, unknown> | unknown[] | null {
    try {
      const parsed = JSON.parse(text) as unknown;
      if (!parsed || typeof parsed !== "object") {
        return null;
      }
      return parsed as Record<string, unknown> | unknown[];
    } catch {
      return null;
    }
  }

  private coerceStepStatus(value: unknown, fallback: StepStatus): StepStatus {
    if (value === "not_started" || value === "in_progress" || value === "complete" || value === "unclear") {
      return value;
    }
    return fallback;
  }

  private coerceConfidence(value: unknown): number {
    if (typeof value !== "number" || Number.isNaN(value)) {
      return 0.5;
    }
    return Math.max(0, Math.min(1, value));
  }

  private trimToWordLimit(text: string, maxWords: number): string {
    const words = text
      .replace(/\s+/g, " ")
      .trim()
      .split(" ")
      .filter(Boolean);
    if (words.length === 0) {
      return "";
    }
    if (words.length <= maxWords) {
      return words.join(" ");
    }
    return `${words.slice(0, maxWords).join(" ")}...`;
  }

  private async sendTextTurn(text: string): Promise<void> {
    const response = await this.client.responses.create({
      model: this.model,
      instructions: SYSTEM_PROMPT,
      previous_response_id: this.previousResponseId,
      tools: TOOLS as never,
      tool_choice: "auto",
      store: true,
      input: [{ role: "user", content: [{ type: "input_text", text }] }]
    });

    this.previousResponseId = response.id;
    await this.resolveResponse(response);
  }

  private async resolveResponse(initialResponse: OpenAI.Responses.Response): Promise<void> {
    let response: OpenAI.Responses.Response = initialResponse;

    for (let step = 0; step < 8; step += 1) {
      const functionCalls = (response.output ?? []).filter(
        (item): item is OpenAI.Responses.ResponseFunctionToolCall =>
          item.type === "function_call"
      );

      if (functionCalls.length === 0) {
        const text = typeof response.output_text === "string" ? response.output_text.trim() : "";
        if (text) {
          // Keep trace, but do not auto-speak non-tool text.
          this.state.recordConversation("assistant", text);
          this.state.addObservation(`assistant_note: ${text}`);
        }
        return;
      }

      const toolResults: FunctionCallOutput[] = [];
      for (const call of functionCalls) {
        toolResults.push(await this.executeFunctionCall(call));
      }

      response = await this.client.responses.create({
        model: this.model,
        instructions: SYSTEM_PROMPT,
        previous_response_id: response.id,
        tools: TOOLS as never,
        store: true,
        input: toolResults
      });

      this.previousResponseId = response.id;
    }

    throw new Error("Tool-call loop exceeded safety depth.");
  }

  private async executeFunctionCall(
    call: OpenAI.Responses.ResponseFunctionToolCall
  ): Promise<FunctionCallOutput> {
    const args = this.tryParseArguments(call.arguments);
    let result: Record<string, unknown>;

    switch (call.name) {
      case "speak": {
        const message = this.readStringArg(args, "message");
        const blockedReason = this.getBlockedSpeakReason(message);
        if (blockedReason) {
          result = { ok: true, suppressed: true, reason: blockedReason };
          break;
        }
        await this.outputs.speak(message);
        this.state.recordConversation("assistant", message);
        if (!this.isQuestion(message) && !this.isUrgentMessage(message)) {
          this.lastRoutineSpeakAtMs = Date.now();
        }
        result = { ok: true };
        break;
      }
      case "stay_silent": {
        const reason = this.readStringArg(args, "reason");
        this.state.addObservation(`stay_silent: ${reason}`);
        result = { ok: true, silent: true };
        break;
      }
      case "set_timer": {
        const durationSeconds = this.readNumberArg(args, "duration_seconds");
        const label = this.readStringArg(args, "label");
        const timer = this.timers.setTimer(durationSeconds, label);
        this.state.setActiveTimers(this.timers.listActiveTimers());
        result = { ok: true, timer };
        break;
      }
      case "cancel_timer": {
        const label = this.readStringArg(args, "label");
        const cancelled = this.timers.cancelTimer(label);
        this.state.setActiveTimers(this.timers.listActiveTimers());
        result = { ok: cancelled };
        break;
      }
      case "update_plan": {
        const changes = this.readStringArg(args, "changes");
        this.state.updatePlan(changes);
        result = { ok: true };
        break;
      }
      case "update_state": {
        const observation = this.readStringArg(args, "observation");
        this.state.addObservation(observation);
        result = { ok: true };
        break;
      }
      case "lookup_recipe": {
        const dish = this.readStringArg(args, "dish");
        result = await this.outputs.lookupRecipe(dish);
        const found = Boolean(result.found);
        const resolvedDish =
          typeof result.dish === "string" && result.dish.trim() ? result.dish.trim() : dish;
        const matchType =
          typeof result.matchType === "string" && result.matchType.trim()
            ? result.matchType.trim()
            : "none";
        this.state.addObservation(
          found
            ? `recipe_lookup: "${dish}" -> "${resolvedDish}" (${matchType})`
            : `recipe_lookup: no match for "${dish}"`
        );
        break;
      }
      case "set_overlay": {
        const text = this.readStringArg(args, "text");
        const priority = this.readPriorityArg(args, "priority");
        const ttlSeconds = this.readOptionalNumberArg(args, "ttl_seconds");
        this.outputs.setOverlay({ text, priority, ttlSeconds });
        result = { ok: true };
        break;
      }
      case "set_panel": {
        const cooking = this.readStringArg(args, "cooking");
        const nextStep = this.readStringArg(args, "next_step");
        result = this.applyPanelStepGate(cooking, nextStep);
        break;
      }
      case "clear_panel": {
        this.outputs.clearPanel();
        this.currentStep = null;
        this.pendingNextStep = null;
        result = { ok: true };
        break;
      }
      case "clear_overlay": {
        this.outputs.clearOverlay();
        result = { ok: true };
        break;
      }
      case "complete_recipe": {
        const dish = this.readStringArg(args, "dish");
        const recipe = this.readStringArg(args, "recipe");
        result = await this.outputs.completeRecipe({ dish, recipe });
        break;
      }
      case "run_python": {
        const code = this.readStringArg(args, "code");
        const output = await this.runPython(code);
        result = { ok: true, output };
        break;
      }
      default: {
        result = { ok: false, error: `Unknown tool: ${call.name}` };
        break;
      }
    }

    this.state.recordConversation(
      "tool",
      `${call.name}(${call.arguments}) => ${JSON.stringify(result)}`
    );

    return {
      type: "function_call_output",
      call_id: call.call_id,
      output: JSON.stringify(result)
    };
  }

  private applyPanelStepGate(cooking: string, requestedStep: string): Record<string, unknown> {
    const nextStep = this.sanitizePanelStep(requestedStep);
    if (!nextStep) {
      return { ok: false, error: "next_step is empty" };
    }

    if (this.isStepTooBroad(nextStep)) {
      this.state.addObservation(`step_rejected: "${requestedStep}"`);
      return {
        ok: false,
        blocked: true,
        reason: "next_step_must_be_single_action",
        suggested_format: "Use one short action like 'Grab 2 eggs.'"
      };
    }

    this.currentDish = cooking.trim() || this.currentDish;

    if (!this.currentStep) {
      this.currentStep = nextStep;
      this.pendingNextStep = null;
      this.outputs.setPanel({ cooking, nextStep });
      return { ok: true, step_locked: this.currentStep, advanced: false };
    }

    if (this.isSameStep(this.currentStep, nextStep)) {
      this.currentStep = nextStep;
      this.pendingNextStep = null;
      this.outputs.setPanel({ cooking, nextStep });
      return { ok: true, step_locked: this.currentStep, advanced: false };
    }

    this.pendingNextStep = nextStep;
    if (!this.frameAllowsStepAdvance) {
      this.state.addObservation(
        `step_gate_blocked: waiting to finish "${this.currentStep}" before "${nextStep}".`
      );
      return {
        ok: false,
        blocked: true,
        reason: "current_step_not_visually_complete",
        current_step: this.currentStep,
        requested_next_step: nextStep,
        frame_status: this.latestFrameAssessment?.stepStatus ?? "unknown"
      };
    }

    this.markCurrentStepCompleted();
    this.currentStep = nextStep;
    this.pendingNextStep = null;
    this.outputs.setPanel({ cooking, nextStep });
    return {
      ok: true,
      advanced: true,
      current_step: this.currentStep
    };
  }

  private sanitizePanelStep(input: string): string {
    const firstLine = input
      .replace(/\r/g, "\n")
      .split("\n")
      .map((line) => line.trim())
      .find((line) => line.length > 0);
    if (!firstLine) {
      return "";
    }

    let step = firstLine
      .replace(/^next\s*step\s*[:\-]\s*/i, "")
      .replace(/^step\s*\d+\s*[:.)-]?\s*/i, "")
      .replace(/^[-*•]\s+/, "")
      .trim();

    if (!step) {
      return "";
    }

    const sentenceBoundary = step.search(/[.!?]/);
    if (sentenceBoundary >= 0) {
      step = step.slice(0, sentenceBoundary + 1).trim();
    }

    if (!step.endsWith(".") && !step.endsWith("!") && !step.endsWith("?")) {
      step = `${step}.`;
    }

    const words = step
      .replace(/[.!?]+$/g, "")
      .split(/\s+/)
      .filter(Boolean);
    if (words.length > 12) {
      step = `${words.slice(0, 12).join(" ")}.`;
    }

    return step;
  }

  private isStepTooBroad(step: string): boolean {
    const text = step.toLowerCase();
    if (!text.trim()) {
      return true;
    }

    if (/\banaly(s|z)(e|ing|ed|is)\b/.test(text)) {
      return true;
    }
    if (/\bingredient(s)?\b/.test(text)) {
      return true;
    }
    if (/\bthen\b|\bafter that\b|\bnext\b/.test(text)) {
      return true;
    }
    if ((text.match(/,\s*/g) ?? []).length >= 2) {
      return true;
    }
    if ((text.match(/\band\b/g) ?? []).length >= 2) {
      return true;
    }
    if (/^\d+[\).]/.test(text)) {
      return true;
    }

    const wordCount = text.split(/\s+/).filter(Boolean).length;
    return wordCount > 12;
  }

  private markCurrentStepCompleted(): void {
    if (!this.currentStep) {
      return;
    }

    const key = this.normalizeStepKey(this.currentStep);
    if (!key || this.completedStepKeys.has(key)) {
      return;
    }
    this.completedStepKeys.add(key);
    this.state.addCompletedStep(this.currentStep);
  }

  private getBlockedSpeakReason(message: string): string | null {
    if (!this.currentStep || this.frameAllowsStepAdvance) {
      return null;
    }
    if (this.isUrgentMessage(message) || this.isQuestion(message)) {
      return null;
    }

    if (!this.messageMatchesCurrentStep(message)) {
      return "blocked_next_step_until_visual_completion";
    }

    if (Date.now() - this.lastRoutineSpeakAtMs < ROUTINE_SPEAK_MIN_GAP_MS) {
      return "step_waiting_repeat_suppressed";
    }

    return null;
  }

  private messageMatchesCurrentStep(message: string): boolean {
    if (!this.currentStep) {
      return true;
    }

    const overlap = this.tokenOverlap(this.currentStep, message);
    if (overlap >= 0.25) {
      return true;
    }

    const messageKey = this.normalizeStepKey(message);
    const stepKey = this.normalizeStepKey(this.currentStep);
    if (!messageKey || !stepKey) {
      return true;
    }

    return messageKey.includes(stepKey) || stepKey.includes(messageKey);
  }

  private isSameStep(left: string, right: string): boolean {
    const leftKey = this.normalizeStepKey(left);
    const rightKey = this.normalizeStepKey(right);
    if (!leftKey || !rightKey) {
      return false;
    }
    if (leftKey === rightKey) {
      return true;
    }
    return this.tokenOverlap(leftKey, rightKey) >= 0.65;
  }

  private normalizeStepKey(input: string): string {
    return input
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  private tokenOverlap(left: string, right: string): number {
    const leftTokens = new Set(this.tokenize(left));
    const rightTokens = new Set(this.tokenize(right));

    if (leftTokens.size === 0 || rightTokens.size === 0) {
      return 0;
    }

    let matches = 0;
    for (const token of leftTokens) {
      if (rightTokens.has(token)) {
        matches += 1;
      }
    }

    return matches / Math.min(leftTokens.size, rightTokens.size);
  }

  private tokenize(text: string): string[] {
    const raw = text
      .toLowerCase()
      .split(/[^a-z0-9]+/g)
      .filter(Boolean);

    const tokens: string[] = [];
    for (const token of raw) {
      const normalized = this.normalizeToken(token);
      if (!normalized || STOPWORDS.has(normalized)) {
        continue;
      }
      tokens.push(normalized);
    }
    return tokens;
  }

  private normalizeToken(token: string): string {
    let normalized = token.trim();
    if (normalized.length <= 2) {
      return "";
    }
    if (normalized.endsWith("ing") && normalized.length > 5) {
      normalized = normalized.slice(0, -3);
    } else if (normalized.endsWith("ed") && normalized.length > 4) {
      normalized = normalized.slice(0, -2);
    } else if (normalized.endsWith("es") && normalized.length > 4) {
      normalized = normalized.slice(0, -2);
    } else if (normalized.endsWith("s") && normalized.length > 3) {
      normalized = normalized.slice(0, -1);
    }
    return normalized;
  }

  private isUrgentMessage(message: string): boolean {
    return /\b(stop|danger|urgent|fire|smoke|burn|hot oil|knife|raw chicken|gas)\b/i.test(message);
  }

  private isQuestion(message: string): boolean {
    return /\?\s*$/.test(message);
  }

  private async runPython(code: string): Promise<string> {
    const response = await this.client.responses.create({
      model: this.model,
      input: [
        {
          role: "user",
          content: [
            {
              type: "input_text",
              text: [
                "Run this Python code and return stdout plus a short result summary:",
                "```python",
                code,
                "```"
              ].join("\n")
            }
          ]
        }
      ],
      tools: [
        {
          type: "code_interpreter",
          container: { type: "auto", memory_limit: "1g" }
        }
      ] as never,
      tool_choice: "required"
    });

    if (response.output_text?.trim()) {
      return response.output_text.trim();
    }

    return JSON.stringify(response.output ?? []);
  }

  private enqueue(task: () => Promise<void>): Promise<void> {
    const next = this.queue.then(task, task);
    this.queue = next.then(
      () => undefined,
      () => undefined
    );
    return next;
  }

  private tryParseArguments(raw: string): Record<string, unknown> {
    try {
      return JSON.parse(raw) as Record<string, unknown>;
    } catch {
      return {};
    }
  }

  private readStringArg(args: Record<string, unknown>, key: string): string {
    const value = args[key];
    if (typeof value !== "string" || !value.trim()) {
      throw new Error(`Missing string argument: ${key}`);
    }
    return value.trim();
  }

  private readNumberArg(args: Record<string, unknown>, key: string): number {
    const value = args[key];
    if (typeof value !== "number" || Number.isNaN(value)) {
      throw new Error(`Missing numeric argument: ${key}`);
    }
    return value;
  }

  private readOptionalNumberArg(args: Record<string, unknown>, key: string): number | undefined {
    const value = args[key];
    if (value === undefined || value === null) {
      return undefined;
    }
    if (typeof value !== "number" || Number.isNaN(value)) {
      throw new Error(`Invalid numeric argument: ${key}`);
    }
    return value;
  }

  private readPriorityArg(
    args: Record<string, unknown>,
    key: string
  ): OverlayUpdate["priority"] {
    const value = args[key];
    if (value === "normal" || value === "urgent") {
      return value;
    }
    throw new Error(`Invalid priority argument: ${key}`);
  }
}
