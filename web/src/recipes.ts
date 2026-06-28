/**
 * Shared access to the exported recipe data (lazy-fetched, cached once).
 *
 * Used by both the in-browser predictor (k-NN neighbours) and the Design page
 * (target-driven retrieval), so the dataset is loaded and typed in one place.
 */

export interface Recipe {
  name: string;
  rgb: [number, number, number];
  surface: string | null;
  transparency: string | null;
  /** UMF values aligned to `RecipesData.radar_oxides`. */
  ox: number[];
  /** Standardised feature vector (k-NN search space). */
  v: number[];
}

export interface RecipesData {
  feature_names: string[];
  scaler: { mean: number[]; std: number[] };
  radar_oxides: string[];
  recipes: Recipe[];
}

let cache: RecipesData | null = null;

/** Fetch the recipe dataset once and cache it. */
export async function loadRecipes(): Promise<RecipesData> {
  if (!cache) {
    const res = await fetch(`${import.meta.env.BASE_URL}recipes.json`);
    cache = (await res.json()) as RecipesData;
  }
  return cache;
}
