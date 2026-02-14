import { randomUUID } from "node:crypto";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import NodeWebcam = require("node-webcam");

export interface CapturedFrame {
  takenAt: string;
  mimeType: "image/jpeg";
  base64: string;
  buffer: Buffer;
}

type FrameHandler = (frame: CapturedFrame) => Promise<void> | void;

interface CameraOptions {
  intervalMs: number;
  deviceId?: string;
  width?: number;
  height?: number;
}

export class CameraDependencyError extends Error {
  readonly missingCommand: string;
  readonly installHint: string;

  constructor(missingCommand: string, installHint: string) {
    super(
      `Missing required camera dependency "${missingCommand}". Install it (${installHint}) or run without camera.`
    );
    this.name = "CameraDependencyError";
    this.missingCommand = missingCommand;
    this.installHint = installHint;
  }
}

export class CameraCapture {
  private readonly webcam: ReturnType<typeof NodeWebcam.create>;
  private readonly intervalMs: number;
  private loopTimer: NodeJS.Timeout | undefined;
  private loopRunning = false;
  private captureInFlight = false;

  constructor(options: CameraOptions) {
    this.ensurePlatformDependencies();
    this.intervalMs = options.intervalMs;
    this.webcam = NodeWebcam.create({
      width: options.width ?? 1280,
      height: options.height ?? 720,
      quality: 90,
      delay: 0,
      saveShots: true,
      output: "jpeg",
      device: options.deviceId,
      callbackReturn: "location",
      verbose: false
    });
  }

  private ensurePlatformDependencies(): void {
    if (process.platform === "darwin" && !this.commandExists("imagesnap")) {
      throw new CameraDependencyError("imagesnap", "brew install imagesnap");
    }
  }

  private commandExists(command: string): boolean {
    const result = spawnSync("which", [command], { stdio: "ignore" });
    return result.status === 0;
  }

  async captureFrame(): Promise<CapturedFrame> {
    const basePath = path.join(os.tmpdir(), `yes-chef-${randomUUID()}`);
    const expectedFilePath = `${basePath}.jpg`;
    const capturePath = await new Promise<string>((resolve, reject) => {
      this.webcam.capture(basePath, (error, data) => {
        if (error) {
          reject(error);
          return;
        }

        if (typeof data === "string" && data.length > 0) {
          resolve(data.endsWith(".jpg") ? data : expectedFilePath);
          return;
        }

        resolve(expectedFilePath);
      });
    });

    const frameBuffer = await fs.readFile(capturePath);
    await fs.unlink(capturePath).catch(() => {});

    return {
      takenAt: new Date().toISOString(),
      mimeType: "image/jpeg",
      base64: frameBuffer.toString("base64"),
      buffer: frameBuffer
    };
  }

  start(handler: FrameHandler): void {
    if (this.loopRunning) {
      return;
    }

    this.loopRunning = true;

    const tick = async () => {
      if (this.captureInFlight) {
        return;
      }

      this.captureInFlight = true;
      try {
        const frame = await this.captureFrame();
        await handler(frame);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        console.error(`[camera] Capture failed: ${message}`);
      } finally {
        this.captureInFlight = false;
      }
    };

    void tick();
    this.loopTimer = setInterval(() => {
      void tick();
    }, this.intervalMs);
  }

  stop(): void {
    this.loopRunning = false;
    if (this.loopTimer) {
      clearInterval(this.loopTimer);
      this.loopTimer = undefined;
    }
  }
}
