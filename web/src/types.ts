/** Shared types mirroring the Pyrochrome API contract (`/predict`). */

/** Firing atmosphere — a required input (drives redox-sensitive colours). */
export type Atmosphere = "oxidation" | "reduction" | "neutral";

/** A class label with the model's predicted probability for it. */
export interface ClassProbability {
  label: string;
  p: number;
}

/** A classification prediction with confidence and the top-2 classes. */
export interface ClassPrediction {
  label: string;
  top2: ClassProbability[];
  confidence: number;
}

/** Colour family prediction, plus a predicted CIELAB value. */
export interface ColourPrediction extends ClassPrediction {
  /** Predicted CIELAB [L*, a*, b*], if the Lab regressor is loaded. */
  lab: [number, number, number] | null;
}

/** A nearest real recipe returned by the k-NN index. */
export interface Neighbour {
  id?: number;
  name?: string;
  rgb_r?: number;
  rgb_g?: number;
  rgb_b?: number;
  lab_l?: number;
  lab_a?: number;
  lab_b?: number;
  distance: number;
  [key: string]: unknown;
}

/** The full `/predict` response. Each field is present only if its model loaded. */
export interface PredictResponse {
  surface: ClassPrediction | null;
  transparency: ClassPrediction | null;
  colour: ColourPrediction | null;
  neighbours: Neighbour[];
}

/** The `/predict` request body. */
export interface PredictRequest {
  chemistry_umf: Record<string, number>;
  cone: string;
  atmosphere: Atmosphere;
}
