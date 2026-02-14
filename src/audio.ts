import { randomUUID } from "node:crypto";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn, spawnSync } from "node:child_process";
import { Readable } from "node:stream";

interface AudioPlayerOptions {
  apiKey: string;
  model: string;
  voice: string;
  instructions?: string;
}

const DEFAULT_TIMEOUT_MS = 25_000;

interface SpeechJob {
  text: string;
  resolve: () => void;
  reject: (error: unknown) => void;
}

export class AudioPlayer {
  private readonly apiKey: string;
  private readonly model: string;
  private readonly voice: string;
  private readonly instructions: string | undefined;
  private readonly ffplayAvailable: boolean;
  private queue: SpeechJob[] = [];
  private draining = false;

  constructor(options: AudioPlayerOptions) {
    this.apiKey = options.apiKey;
    this.model = options.model;
    this.voice = options.voice;
    this.instructions = options.instructions?.trim() || undefined;
    this.ffplayAvailable = this.commandExists("ffplay");
  }

  async enqueueSpeech(message: string): Promise<void> {
    const text = message.trim();
    if (!text) {
      return;
    }

    await new Promise<void>((resolve, reject) => {
      this.queue.push({ text, resolve, reject });
      if (!this.draining) {
        this.draining = true;
        void this.drainQueue();
      }
    });
  }

  private async drainQueue(): Promise<void> {
    try {
      while (this.queue.length > 0) {
        const job = this.queue.shift();
        if (!job) {
          continue;
        }

        try {
          await this.playSpeech(job.text);
          job.resolve();
        } catch (error) {
          job.reject(error);
        }
      }
    } finally {
      this.draining = false;
    }
  }

  private async playSpeech(input: string): Promise<void> {
    if (this.ffplayAvailable) {
      const streamed = await this.tryStreamToFfplay(input);
      if (streamed) {
        return;
      }
    }

    const wavBuffer = await this.requestSpeechBuffer(input);
    const tempPath = path.join(os.tmpdir(), `yes-chef-tts-${randomUUID()}.wav`);
    await fs.writeFile(tempPath, wavBuffer);
    try {
      await this.playFile(tempPath);
    } finally {
      await fs.unlink(tempPath).catch(() => {});
    }
  }

  private async tryStreamToFfplay(input: string): Promise<boolean> {
    const response = await this.requestSpeech(input);
    if (!response.ok || !response.body) {
      return false;
    }

    await new Promise<void>((resolve, reject) => {
      const ffplay = spawn(
        "ffplay",
        ["-nodisp", "-autoexit", "-loglevel", "error", "-i", "-"],
        { stdio: ["pipe", "ignore", "pipe"] }
      );

      ffplay.on("error", reject);
      ffplay.on("exit", (code) => {
        if (code === 0) {
          resolve();
          return;
        }
        reject(new Error(`ffplay exited with code ${code ?? "unknown"}`));
      });

      const nodeStream = Readable.fromWeb(response.body as never);
      nodeStream.on("error", reject);
      nodeStream.pipe(ffplay.stdin);
    }).catch(async () => {
      response.body?.cancel().catch(() => {});
      return Promise.reject(new Error("Streaming to ffplay failed."));
    });

    return true;
  }

  private async requestSpeechBuffer(input: string): Promise<Buffer> {
    const response = await this.requestSpeech(input);
    if (!response.ok) {
      const details = await response.text().catch(() => "");
      throw new Error(`Speech request failed (${response.status}): ${details}`);
    }

    const bytes = await response.arrayBuffer();
    return Buffer.from(bytes);
  }

  private requestSpeech(input: string): Promise<Response> {
    const body: {
      model: string;
      voice: string;
      response_format: string;
      input: string;
      instructions?: string;
    } = {
      model: this.model,
      voice: this.voice,
      response_format: "wav",
      input
    };

    // The API docs note `instructions` is not supported for tts-1 / tts-1-hd.
    if (this.instructions && this.model !== "tts-1" && this.model !== "tts-1-hd") {
      body.instructions = this.instructions;
    }

    return fetch("https://api.openai.com/v1/audio/speech", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(DEFAULT_TIMEOUT_MS)
    });
  }

  private playFile(filePath: string): Promise<void> {
    const [command, args] = this.getPlaybackCommand(filePath);
    return new Promise<void>((resolve, reject) => {
      const child = spawn(command, args, {
        stdio: ["ignore", "ignore", "pipe"]
      });

      child.on("error", reject);
      child.on("exit", (code) => {
        if (code === 0) {
          resolve();
          return;
        }

        reject(new Error(`${command} exited with code ${code ?? "unknown"}`));
      });
    });
  }

  private getPlaybackCommand(filePath: string): [string, string[]] {
    if (process.platform === "darwin") {
      return ["afplay", [filePath]];
    }

    if (process.platform === "linux") {
      return ["aplay", [filePath]];
    }

    if (process.platform === "win32") {
      return [
        "powershell",
        [
          "-NoProfile",
          "-Command",
          `(New-Object Media.SoundPlayer '${filePath.replace(/'/g, "''")}').PlaySync();`
        ]
      ];
    }

    throw new Error(`Unsupported platform for audio playback: ${process.platform}`);
  }

  private commandExists(command: string): boolean {
    const result = spawnSync("which", [command], { stdio: "ignore" });
    return result.status === 0;
  }
}
