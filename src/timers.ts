import { EventEmitter } from "node:events";
import { ActiveTimerSnapshot } from "./state";

interface InternalTimer {
  timeout: NodeJS.Timeout;
  snapshot: ActiveTimerSnapshot;
}

export interface TimerFiredEvent extends ActiveTimerSnapshot {
  firedAt: string;
}

type TimerEvents = {
  fired: (timer: TimerFiredEvent) => void;
  changed: (timers: ActiveTimerSnapshot[]) => void;
};

export class TimerManager extends EventEmitter {
  private readonly timers = new Map<string, InternalTimer>();

  on<E extends keyof TimerEvents>(event: E, listener: TimerEvents[E]): this {
    return super.on(event, listener);
  }

  emit<E extends keyof TimerEvents>(event: E, ...args: Parameters<TimerEvents[E]>): boolean {
    return super.emit(event, ...args);
  }

  setTimer(durationSeconds: number, label: string): ActiveTimerSnapshot {
    const safeDuration = Math.max(1, Math.round(durationSeconds));
    const safeLabel = label.trim();
    if (!safeLabel) {
      throw new Error("Timer label cannot be empty.");
    }

    this.cancelTimer(safeLabel);

    const startedAt = new Date();
    const endsAt = new Date(startedAt.getTime() + safeDuration * 1000);

    const snapshot: ActiveTimerSnapshot = {
      label: safeLabel,
      durationSeconds: safeDuration,
      startedAt: startedAt.toISOString(),
      endsAt: endsAt.toISOString()
    };

    const timeout = setTimeout(() => {
      this.timers.delete(safeLabel);
      this.emit("changed", this.listActiveTimers());
      this.emit("fired", { ...snapshot, firedAt: new Date().toISOString() });
    }, safeDuration * 1000);

    this.timers.set(safeLabel, { timeout, snapshot });
    this.emit("changed", this.listActiveTimers());

    return snapshot;
  }

  cancelTimer(label: string): boolean {
    const safeLabel = label.trim();
    const existing = this.timers.get(safeLabel);
    if (!existing) {
      return false;
    }

    clearTimeout(existing.timeout);
    this.timers.delete(safeLabel);
    this.emit("changed", this.listActiveTimers());
    return true;
  }

  listActiveTimers(): ActiveTimerSnapshot[] {
    const now = Date.now();
    return [...this.timers.values()]
      .map((entry) => entry.snapshot)
      .filter((timer) => new Date(timer.endsAt).getTime() > now)
      .sort((a, b) => new Date(a.endsAt).getTime() - new Date(b.endsAt).getTime());
  }

  clearAllTimers(): void {
    for (const timer of this.timers.values()) {
      clearTimeout(timer.timeout);
    }
    this.timers.clear();
    this.emit("changed", []);
  }
}
