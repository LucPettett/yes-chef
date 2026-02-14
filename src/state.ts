export type ConversationRole = "system" | "user" | "assistant" | "tool";

export interface ConversationEntry {
  timestamp: string;
  role: ConversationRole;
  content: string;
}

export interface ActiveTimerSnapshot {
  label: string;
  durationSeconds: number;
  startedAt: string;
  endsAt: string;
}

export interface CookingState {
  originalPlan: string;
  currentDishes: string[];
  completedSteps: string[];
  activeTimers: ActiveTimerSnapshot[];
  recentObservations: string[];
  availableIngredients: Record<string, boolean>;
  conversationHistory: ConversationEntry[];
}

const MAX_HISTORY_ITEMS = 60;
const MAX_OBSERVATIONS = 30;

export class CookingStateStore {
  private state: CookingState = this.createEmptyState();

  private createEmptyState(): CookingState {
    return {
      originalPlan: "",
      currentDishes: [],
      completedSteps: [],
      activeTimers: [],
      recentObservations: [],
      availableIngredients: {},
      conversationHistory: []
    };
  }

  reset(): void {
    this.state = this.createEmptyState();
  }

  getSnapshot(): CookingState {
    return structuredClone(this.state);
  }

  setOriginalPlan(plan: string): void {
    this.state.originalPlan = plan;
  }

  setCurrentDishes(dishes: string[]): void {
    this.state.currentDishes = [...dishes];
  }

  addCompletedStep(step: string): void {
    this.state.completedSteps.push(step);
  }

  setActiveTimers(timers: ActiveTimerSnapshot[]): void {
    this.state.activeTimers = [...timers];
  }

  addObservation(observation: string): void {
    if (!observation.trim()) {
      return;
    }

    this.state.recentObservations.push(observation.trim());
    if (this.state.recentObservations.length > MAX_OBSERVATIONS) {
      this.state.recentObservations = this.state.recentObservations.slice(-MAX_OBSERVATIONS);
    }
  }

  updatePlan(changes: string): void {
    if (!changes.trim()) {
      return;
    }

    if (!this.state.originalPlan) {
      this.state.originalPlan = changes.trim();
      return;
    }

    this.state.originalPlan = `${this.state.originalPlan}\n\nUpdate: ${changes.trim()}`;
  }

  setIngredientAvailability(ingredient: string, isAvailable: boolean): void {
    this.state.availableIngredients[ingredient.toLowerCase().trim()] = isAvailable;
  }

  recordConversation(role: ConversationRole, content: string): void {
    const value = content.trim();
    if (!value) {
      return;
    }

    this.state.conversationHistory.push({
      timestamp: new Date().toISOString(),
      role,
      content: value
    });

    if (this.state.conversationHistory.length > MAX_HISTORY_ITEMS) {
      this.state.conversationHistory = this.state.conversationHistory.slice(-MAX_HISTORY_ITEMS);
    }
  }
}
