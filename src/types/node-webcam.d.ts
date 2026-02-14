declare module "node-webcam" {
  interface Webcam {
    capture(
      location: string,
      callback: (error: Error | null, data: string | Buffer) => void
    ): void;
  }

  interface WebcamOptions {
    width?: number;
    height?: number;
    quality?: number;
    delay?: number;
    saveShots?: boolean;
    output?: "jpeg" | "png" | "bmp";
    device?: string;
    callbackReturn?: "location" | "base64" | "buffer";
    verbose?: boolean;
  }

  const NodeWebcam: {
    create(options?: WebcamOptions): Webcam;
  };

  export = NodeWebcam;
}
