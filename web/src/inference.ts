/**
 * In-browser inference — the static site has no backend, so prediction runs
 * here from the exported compact models (see `make export`).
 *
 * It rebuilds the exact feature vector the models were trained on (same order,
 * cone→ordinal, atmosphere multi-hot), runs each MLP's forward pass, and does a
 * k-NN search over the shipped recipe vectors. Output matches the old API's
 * `PredictResponse` shape so the views are unchanged.
 */
import classifierColour from "./model/classifier_colour.json";
import classifierSurface from "./model/classifier_surface.json";
import classifierTransparency from "./model/classifier_transparency.json";
import regressorLab from "./model/regressor_lab.json";
import { coneToOrdinal } from "./cones";
import type {
  ClassPrediction,
  ClassProbability,
  Neighbour,
  PredictRequest,
  PredictResponse,
} from "./types";

interface Scaler {
  mean: number[];
  std: number[];
}
interface Layer {
  w: number[][];
  b: number[];
}
interface ClassifierModel {
  feature_names: string[];
  scaler: Scaler;
  layers: Layer[];
  classes: string[];
}
interface RegressorModel {
  feature_names: string[];
  scaler: Scaler;
  layers: Layer[];
}
interface RecipesData {
  feature_names: string[];
  scaler: Scaler;
  radar_oxides: string[];
  recipes: { name: string; rgb: [number, number, number]; ox: number[]; v: number[] }[];
}

const SURFACE = classifierSurface as unknown as ClassifierModel;
const TRANSPARENCY = classifierTransparency as unknown as ClassifierModel;
const COLOUR = classifierColour as unknown as ClassifierModel;
const LAB = regressorLab as unknown as RegressorModel;

/** Build the model feature vector from a request (mirrors the Python builder). */
function buildVector(featureNames: string[], req: PredictRequest): number[] {
  const cone = coneToOrdinal(req.cone);
  return featureNames.map((name) => {
    if (name === "cone_num") return Number.isNaN(cone) ? 0 : cone;
    if (name === "atm_known") return 1;
    if (name.startsWith("atm_")) return name === `atm_${req.atmosphere}` ? 1 : 0;
    const bare = name.endsWith("_umf") ? name.slice(0, -4) : name;
    return req.chemistry_umf[name] ?? req.chemistry_umf[bare] ?? 0;
  });
}

function standardise(vec: number[], scaler: Scaler): number[] {
  return vec.map((x, i) => (x - scaler.mean[i]) / (scaler.std[i] || 1));
}

/** Dense forward pass; ReLU on hidden layers, linear output (no final activation). */
function forward(layers: Layer[], input: number[]): number[] {
  let a = input;
  layers.forEach((layer, li) => {
    const out = layer.b.slice();
    for (let i = 0; i < a.length; i++) {
      const row = layer.w[i];
      const ai = a[i];
      for (let j = 0; j < out.length; j++) out[j] += ai * row[j];
    }
    a = li < layers.length - 1 ? out.map((z) => Math.max(0, z)) : out;
  });
  return a;
}

function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exps = logits.map((z) => Math.exp(z - max));
  const sum = exps.reduce((s, e) => s + e, 0);
  return exps.map((e) => e / sum);
}

function classify(model: ClassifierModel, req: PredictRequest): ClassPrediction {
  const x = standardise(buildVector(model.feature_names, req), model.scaler);
  const probs = softmax(forward(model.layers, x));
  const ranked: ClassProbability[] = model.classes
    .map((label, i) => ({ label, p: probs[i] }))
    .sort((a, b) => b.p - a.p);
  return { label: ranked[0].label, top2: ranked.slice(0, 2), confidence: ranked[0].p };
}

function regressLab(model: RegressorModel, req: PredictRequest): [number, number, number] {
  const x = standardise(buildVector(model.feature_names, req), model.scaler);
  const [l, a, b] = forward(model.layers, x);
  return [l, a, b];
}

let recipesCache: RecipesData | null = null;

async function loadRecipes(): Promise<RecipesData> {
  if (!recipesCache) {
    const res = await fetch(`${import.meta.env.BASE_URL}recipes.json`);
    recipesCache = (await res.json()) as RecipesData;
  }
  return recipesCache;
}

function nearestRecipes(data: RecipesData, req: PredictRequest, k: number): Neighbour[] {
  const query = standardise(buildVector(data.feature_names, req), data.scaler);
  const scored = data.recipes.map((recipe) => {
    let dist = 0;
    for (let i = 0; i < query.length; i++) {
      const d = query[i] - recipe.v[i];
      dist += d * d;
    }
    return { recipe, distance: Math.sqrt(dist) };
  });
  scored.sort((a, b) => a.distance - b.distance);
  return scored.slice(0, k).map(({ recipe, distance }) => {
    const n: Neighbour = {
      name: recipe.name,
      rgb_r: recipe.rgb[0],
      rgb_g: recipe.rgb[1],
      rgb_b: recipe.rgb[2],
      distance,
    };
    data.radar_oxides.forEach((col, j) => {
      n[col] = recipe.ox[j];
    });
    return n;
  });
}

/** Predict the render entirely in the browser. */
export async function predict(req: PredictRequest): Promise<PredictResponse> {
  const colour = { ...classify(COLOUR, req), lab: regressLab(LAB, req) };
  const recipes = await loadRecipes();
  return {
    surface: classify(SURFACE, req),
    transparency: classify(TRANSPARENCY, req),
    colour,
    neighbours: nearestRecipes(recipes, req, 6),
  };
}
