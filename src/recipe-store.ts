import fs from "node:fs/promises";
import path from "node:path";
import YAML from "yaml";

export interface StoredRecipe {
  dish: string;
  dishKey: string;
  recipe: string;
  completedAt: string;
  timesCooked: number;
}

export interface CompleteRecipeInput {
  dish: string;
  recipe: string;
}

export interface RecipeLookupResult {
  found: boolean;
  query: string;
  matchType: "exact" | "fuzzy" | "none";
  recipe: StoredRecipe | null;
}

interface RecipeFile {
  version: number;
  recipes: StoredRecipe[];
}

const RECIPE_SCHEMA_VERSION = 1;

function createEmptyFile(): RecipeFile {
  return {
    version: RECIPE_SCHEMA_VERSION,
    recipes: []
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

export function normalizeDishKey(input: string): string {
  return input
    .toLowerCase()
    .replace(/'/g, "")
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
    .replace(/\s+/g, " ");
}

function tokenizeDishKey(input: string): string[] {
  const key = normalizeDishKey(input);
  if (!key) {
    return [];
  }
  return key.split(" ").filter(Boolean);
}

function scoreDishMatch(queryKey: string, candidateKey: string): number {
  if (!queryKey || !candidateKey) {
    return 0;
  }

  if (queryKey === candidateKey) {
    return 1;
  }

  if (candidateKey.includes(queryKey) || queryKey.includes(candidateKey)) {
    return 0.95;
  }

  const queryTokens = tokenizeDishKey(queryKey);
  const candidateTokens = tokenizeDishKey(candidateKey);
  if (queryTokens.length === 0 || candidateTokens.length === 0) {
    return 0;
  }

  const candidateTokenSet = new Set(candidateTokens);
  let shared = 0;
  for (const token of queryTokens) {
    if (candidateTokenSet.has(token)) {
      shared += 1;
    }
  }

  if (shared === 0) {
    return 0;
  }

  const overlap = shared / Math.min(queryTokens.length, candidateTokens.length);
  const coverage = shared / queryTokens.length;
  const prefixBonus =
    candidateTokens[0] && queryTokens[0] && candidateTokens[0] === queryTokens[0] ? 0.08 : 0;
  return Math.min(1, Math.max(overlap, coverage * 0.9) + prefixBonus);
}

function findBestFuzzyDishMatch(queryKey: string, recipes: StoredRecipe[]): StoredRecipe | null {
  let bestScore = 0;
  let bestRecipe: StoredRecipe | null = null;

  for (const candidate of recipes) {
    const score = scoreDishMatch(queryKey, candidate.dishKey);
    if (score > bestScore) {
      bestScore = score;
      bestRecipe = candidate;
    }
  }

  if (bestScore < 0.62 || !bestRecipe) {
    return null;
  }
  return bestRecipe;
}

function normalizeRecipeFile(raw: unknown): RecipeFile {
  if (!isRecord(raw)) {
    return createEmptyFile();
  }

  const recipesRaw = Array.isArray(raw.recipes) ? raw.recipes : [];
  const recipes: StoredRecipe[] = [];

  for (const item of recipesRaw) {
    if (!isRecord(item)) {
      continue;
    }

    const dish = typeof item.dish === "string" ? item.dish.trim() : "";
    const dishKeyRaw =
      typeof item.dishKey === "string"
        ? item.dishKey
        : typeof item.dish_key === "string"
          ? item.dish_key
          : "";
    const dishKey = normalizeDishKey(dishKeyRaw || dish);
    const recipe = typeof item.recipe === "string" ? item.recipe.trim() : "";
    const completedAtRaw =
      typeof item.completedAt === "string"
        ? item.completedAt
        : typeof item.completed_at === "string"
          ? item.completed_at
          : "";
    const completedAt = completedAtRaw && Number.isFinite(Date.parse(completedAtRaw))
      ? new Date(completedAtRaw).toISOString()
      : new Date().toISOString();
    const timesCookedRaw =
      typeof item.timesCooked === "number"
        ? item.timesCooked
        : typeof item.times_cooked === "number"
          ? item.times_cooked
          : 1;
    const timesCooked = Math.max(1, Math.round(timesCookedRaw));

    if (!dish || !dishKey || !recipe) {
      continue;
    }

    recipes.push({
      dish,
      dishKey,
      recipe,
      completedAt,
      timesCooked
    });
  }

  return {
    version:
      typeof raw.version === "number" && Number.isFinite(raw.version) ? raw.version : RECIPE_SCHEMA_VERSION,
    recipes
  };
}

export class RecipeStore {
  private readonly filePath: string;

  constructor(options?: { filePath?: string }) {
    this.filePath =
      options?.filePath ?? path.resolve(process.cwd(), "data/recipes/recipes.yaml");
  }

  async ensureFile(): Promise<void> {
    await fs.mkdir(path.dirname(this.filePath), { recursive: true });
    try {
      await fs.access(this.filePath);
    } catch {
      const initial = `${YAML.stringify(createEmptyFile())}`;
      await fs.writeFile(this.filePath, initial, "utf8");
    }
  }

  async findByDish(dish: string): Promise<StoredRecipe | null> {
    const dishKey = normalizeDishKey(dish);
    if (!dishKey) {
      return null;
    }

    const file = await this.readFile();
    const recipe = file.recipes.find((entry) => entry.dishKey === dishKey);
    return recipe ? { ...recipe } : null;
  }

  async lookupByDish(dishQuery: string): Promise<RecipeLookupResult> {
    const query = dishQuery.trim();
    const queryKey = normalizeDishKey(query);
    if (!queryKey) {
      return {
        found: false,
        query,
        matchType: "none",
        recipe: null
      };
    }

    const file = await this.readFile();
    const exact = file.recipes.find((entry) => entry.dishKey === queryKey);
    if (exact) {
      return {
        found: true,
        query,
        matchType: "exact",
        recipe: { ...exact }
      };
    }

    const fuzzy = findBestFuzzyDishMatch(queryKey, file.recipes);
    if (fuzzy) {
      return {
        found: true,
        query,
        matchType: "fuzzy",
        recipe: { ...fuzzy }
      };
    }

    return {
      found: false,
      query,
      matchType: "none",
      recipe: null
    };
  }

  async saveCompletedRecipe(input: CompleteRecipeInput): Promise<StoredRecipe> {
    const dish = input.dish.trim();
    const recipe = input.recipe.trim();
    if (!dish || !recipe) {
      throw new Error("Both dish and recipe are required to save a completed recipe.");
    }

    const dishKey = normalizeDishKey(dish);
    const now = new Date().toISOString();
    const file = await this.readFile();

    const existingIndex = file.recipes.findIndex((entry) => entry.dishKey === dishKey);
    let saved: StoredRecipe;

    if (existingIndex >= 0) {
      const existing = file.recipes[existingIndex];
      saved = {
        dish,
        dishKey,
        recipe,
        completedAt: now,
        timesCooked: existing.timesCooked + 1
      };
      file.recipes[existingIndex] = saved;
    } else {
      saved = {
        dish,
        dishKey,
        recipe,
        completedAt: now,
        timesCooked: 1
      };
      file.recipes.push(saved);
    }

    file.recipes.sort((a, b) => a.dish.localeCompare(b.dish, "en"));
    await this.writeFile(file);
    return { ...saved };
  }

  private async readFile(): Promise<RecipeFile> {
    await this.ensureFile();
    const raw = await fs.readFile(this.filePath, "utf8");
    if (!raw.trim()) {
      return createEmptyFile();
    }
    return normalizeRecipeFile(YAML.parse(raw));
  }

  private async writeFile(file: RecipeFile): Promise<void> {
    const payload = {
      version: file.version || RECIPE_SCHEMA_VERSION,
      recipes: file.recipes.map((recipe) => ({
        dish: recipe.dish,
        dish_key: recipe.dishKey,
        recipe: recipe.recipe,
        completed_at: recipe.completedAt,
        times_cooked: recipe.timesCooked
      }))
    };
    const serialized = `${YAML.stringify(payload)}`;
    await fs.writeFile(this.filePath, serialized, "utf8");
  }
}
