interface TextToSpeechOptions {
  apiKey: string;
  model: string;
  voice: string;
  instructions?: string;
}

export interface SpeechAudio {
  mimeType: "audio/mpeg" | "audio/wav";
  base64: string;
}

const REQUEST_TIMEOUT_MS = 25_000;

export class TextToSpeechService {
  private readonly apiKey: string;
  private readonly model: string;
  private readonly voice: string;
  private readonly instructions: string | undefined;

  constructor(options: TextToSpeechOptions) {
    this.apiKey = options.apiKey;
    this.model = options.model;
    this.voice = options.voice;
    this.instructions = options.instructions?.trim() || undefined;
  }

  async synthesize(text: string): Promise<SpeechAudio> {
    const body: {
      model: string;
      voice: string;
      response_format: "mp3" | "wav";
      input: string;
      instructions?: string;
    } = {
      model: this.model,
      voice: this.voice,
      response_format: "mp3",
      input: text
    };

    // OpenAI docs note `instructions` is not supported for tts-1 / tts-1-hd.
    if (this.instructions && this.model !== "tts-1" && this.model !== "tts-1-hd") {
      body.instructions = this.instructions;
    }

    const response = await fetch("https://api.openai.com/v1/audio/speech", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS)
    });

    if (!response.ok) {
      const details = await response.text().catch(() => "");
      throw new Error(`TTS request failed (${response.status}): ${details}`);
    }

    const bytes = await response.arrayBuffer();
    return {
      mimeType: "audio/mpeg",
      base64: Buffer.from(bytes).toString("base64")
    };
  }
}
