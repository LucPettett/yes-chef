import OpenAI from "openai";
import { CapturedFrame } from "./camera";
import { AudioPlayer } from "./audio";
import { CookingStateStore } from "./state";
import { TimerFiredEvent, TimerManager } from "./timers";

const SYSTEM_PROMPT = `
You are "Yes Chef", a focused cooking assistant.

Context:
- You watch the user cook through periodic webcam frames.
- You are safety-first. Flag urgent issues immediately (burning, unsafe knife use, contamination risk, smoke).
- Be concise and practical. Do not be chatty.
- Only speak when there is a next-step instruction, a correction, or an urgent issue.

Behavior:
- Keep an evolving cooking plan.
- If a step is complete, give one clear next instruction.
- Prefer silent state updates unless speech is useful right now.
- Use timers whenever timing matters.

Tools:
- speak(message): audible instruction to the cook.
- set_timer(duration_seconds, label): create a timer.
- cancel_timer(label): cancel an active timer.
- update_plan(changes): update the ongoing recipe plan.
- update_state(observation): store observations silently.
- run_python(code): run calculations with code interpreter.

When uncertain from the frame alone, ask a short clarifying question via speak().
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
        message: { type: "string", description: "Message to speak aloud." }
      },
      required: ["message"],
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
    description: "Record a silent observation about progress, doneness, or setup.",
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

interface AssistantOptions {
  apiKey: string;
  model: string;
  state: CookingStateStore;
  timers: TimerManager;
  audio: AudioPlayer;
  verboseTextLogs?: boolean;
}

type FunctionCallOutput = {
  type: "function_call_output";
  call_id: string;
  output: string;
};

export class CookingAssistant {
  private readonly client: OpenAI;
  private readonly model: string;
  private readonly state: CookingStateStore;
  private readonly timers: TimerManager;
  private readonly audio: AudioPlayer;
  private readonly verboseTextLogs: boolean;
  private previousResponseId: string | undefined;
  private queue: Promise<void> = Promise.resolve();

  constructor(options: AssistantOptions) {
    this.client = new OpenAI({ apiKey: options.apiKey });
    this.model = options.model;
    this.state = options.state;
    this.timers = options.timers;
    this.audio = options.audio;
    this.verboseTextLogs = options.verboseTextLogs ?? false;
  }

  initializeWithRecipeIdea(recipeIdea: string): Promise<void> {
    return this.enqueue(async () => {
      this.state.setOriginalPlan(recipeIdea);
      this.state.recordConversation("user", recipeIdea);
      await this.sendTextTurn(`The user wants to cook: ${recipeIdea}`);
    });
  }

  processFrame(frame: CapturedFrame): Promise<void> {
    return this.enqueue(async () => {
      const snapshot = this.state.getSnapshot();
      const frameSummary = [
        `Frame timestamp: ${frame.takenAt}`,
        `Current plan:\n${snapshot.originalPlan || "(none yet)"}`,
        `Current dishes: ${snapshot.currentDishes.join(", ") || "(unspecified)"}`,
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

  handleUserMessage(userMessage: string): Promise<void> {
    return this.enqueue(async () => {
      this.state.recordConversation("user", userMessage);
      await this.sendTextTurn(`User update: ${userMessage}`);
    });
  }

  handleTimerFired(timer: TimerFiredEvent): Promise<void> {
    return this.enqueue(async () => {
      this.state.addObservation(
        `Timer "${timer.label}" fired at ${timer.firedAt} (duration ${timer.durationSeconds}s).`
      );
      await this.sendTextTurn(
        `Timer event: "${timer.label}" just completed. Decide whether to notify user now and next action.`
      );
    });
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
          this.state.recordConversation("assistant", text);
          if (this.verboseTextLogs) {
            console.log(`[assistant] ${text}`);
          }
          await this.audio.enqueueSpeech(text);
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
        await this.audio.enqueueSpeech(message);
        this.state.recordConversation("assistant", message);
        if (this.verboseTextLogs) {
          console.log(`[assistant:speak] ${message}`);
        }
        result = { ok: true };
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
}
