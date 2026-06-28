/**
 * Glaze chemistry model for the form: the oxides we expose, a balanced default
 * recipe, a few real-world presets, and the heuristics that translate model
 * outputs (transparency, surface) and the input chemistry into renderer effects.
 *
 * Keys are bare oxide symbols (e.g. "SiO2"); the API accepts these or the
 * "<oxide>_umf" column names.
 */
import type { Atmosphere } from "./types";

export interface Oxide {
  key: string;
  label: string;
  /** Slider/step granularity. */
  step: number;
  max: number;
}

export interface OxideGroup {
  title: string;
  note: string;
  oxides: Oxide[];
}

/** Oxides grouped the way a ceramicist reads a UMF: glass, fluxes, colorants. */
export const OXIDE_GROUPS: OxideGroup[] = [
  {
    title: "Glass & stabiliser",
    note: "Silica forms the glass; alumina stiffens the melt.",
    oxides: [
      { key: "SiO2", label: "SiO₂", step: 0.05, max: 6 },
      { key: "Al2O3", label: "Al₂O₃", step: 0.01, max: 1 },
      { key: "B2O3", label: "B₂O₃", step: 0.01, max: 1 },
    ],
  },
  {
    title: "Fluxes",
    note: "Melt the glass; their balance shifts surface and colour.",
    oxides: [
      { key: "Na2O", label: "Na₂O", step: 0.01, max: 1 },
      { key: "K2O", label: "K₂O", step: 0.01, max: 1 },
      { key: "CaO", label: "CaO", step: 0.01, max: 1 },
      { key: "MgO", label: "MgO", step: 0.01, max: 1 },
      { key: "ZnO", label: "ZnO", step: 0.01, max: 1 },
    ],
  },
  {
    title: "Colorants",
    note: "Small amounts; their colour depends on atmosphere.",
    oxides: [
      { key: "Fe2O3", label: "Fe₂O₃", step: 0.005, max: 0.4 },
      { key: "CuO", label: "CuO", step: 0.005, max: 0.2 },
      { key: "CoO", label: "CoO", step: 0.002, max: 0.1 },
      { key: "MnO", label: "MnO", step: 0.005, max: 0.2 },
      { key: "TiO2", label: "TiO₂", step: 0.005, max: 0.3 },
      { key: "Cr2O3", label: "Cr₂O₃", step: 0.002, max: 0.1 },
    ],
  },
];

/** A balanced cone-6 base glaze, in UMF. */
export const DEFAULT_RECIPE: Record<string, number> = {
  SiO2: 3.0,
  Al2O3: 0.35,
  B2O3: 0.1,
  Na2O: 0.15,
  K2O: 0.15,
  CaO: 0.5,
  MgO: 0.1,
  ZnO: 0.0,
  Fe2O3: 0.0,
  CuO: 0.0,
  CoO: 0.0,
  MnO: 0.0,
  TiO2: 0.0,
  Cr2O3: 0.0,
};

export interface Preset {
  name: string;
  cone: string;
  atmosphere: Atmosphere;
  /** Oxide overrides merged onto the default recipe. */
  chemistry: Record<string, number>;
}

/** Familiar real glaze families, as starting points. */
export const PRESETS: Preset[] = [
  {
    name: "Celadon",
    cone: "10",
    atmosphere: "reduction",
    chemistry: { SiO2: 3.4, Al2O3: 0.45, CaO: 0.6, Fe2O3: 0.03 },
  },
  {
    name: "Tenmoku",
    cone: "10",
    atmosphere: "reduction",
    chemistry: { SiO2: 3.0, Al2O3: 0.32, CaO: 0.45, Fe2O3: 0.12 },
  },
  {
    name: "Copper turquoise",
    cone: "6",
    atmosphere: "oxidation",
    chemistry: { SiO2: 3.1, Al2O3: 0.25, Na2O: 0.3, CuO: 0.05 },
  },
  {
    name: "Cobalt blue",
    cone: "6",
    atmosphere: "oxidation",
    chemistry: { SiO2: 3.0, Al2O3: 0.35, CaO: 0.4, CoO: 0.04 },
  },
  {
    name: "Iron oatmeal",
    cone: "6",
    atmosphere: "oxidation",
    chemistry: { SiO2: 2.9, Al2O3: 0.4, MgO: 0.2, Fe2O3: 0.05, TiO2: 0.08 },
  },
];

export const CONES = ["06", "04", "1", "6", "8", "10"];

/**
 * The unit / standard the form values are entered in. Shown to the user so the
 * numbers are unambiguous.
 */
export const UMF_STANDARD = {
  short: "UMF · molar (Seger unity formula)",
  long:
    "Values are a Unity Molecular Formula (UMF / Seger): the RO + R₂O fluxes are " +
    "normalised to sum to 1, and every other oxide — silica, alumina, boron, " +
    "colorants — is expressed as a molar ratio to that flux unity. They are molar " +
    "proportions, not weight percentages.",
};

/** A radar axis: a label, a display max, and the oxides (bare keys) summed for it. */
export interface RadarAxis {
  label: string;
  max: number;
  oxides: string[];
}

/** Axes positioning a glaze in chemistry space (clockwise from the top). */
export const RADAR_AXES: RadarAxis[] = [
  { label: "Silica", max: 5.0, oxides: ["SiO2"] },
  { label: "Alumina", max: 0.8, oxides: ["Al2O3"] },
  { label: "Boron", max: 0.6, oxides: ["B2O3"] },
  { label: "Calcia", max: 0.9, oxides: ["CaO"] },
  { label: "Magnesia", max: 0.6, oxides: ["MgO"] },
  { label: "Alkali", max: 0.7, oxides: ["Na2O", "K2O"] },
  { label: "Colorant", max: 0.35, oxides: ["Fe2O3", "CuO", "CoO", "MnO", "Cr2O3"] },
];

/**
 * Normalised value (0–1) for an axis, given a lookup that returns the molar
 * amount for a bare oxide key. Clamped so the polygon stays inside the chart.
 */
export function axisValue(axis: RadarAxis, lookup: (oxide: string) => number): number {
  const sum = axis.oxides.reduce((acc, ox) => acc + (lookup(ox) || 0), 0);
  return Math.max(0, Math.min(1, sum / axis.max));
}

/** Map a predicted transparency class to a glaze opacity (0–1). */
export function transparencyToOpacity(label: string | undefined): number {
  switch (label) {
    case "Transparent":
      return 0.42;
    case "Translucent":
      return 0.62;
    case "Semi-opaque":
      return 0.82;
    case "Opaque":
      return 0.97;
    default:
      return 0.85;
  }
}

/** Map a predicted surface class to the renderer's surface key. */
export function surfaceToRenderer(label: string | undefined): "glossy" | "satin" | "matte" {
  const l = (label ?? "").toLowerCase();
  if (l.startsWith("gloss")) return "glossy";
  if (l.startsWith("matte")) return "matte";
  return "satin";
}

/**
 * Derive renderer effects from the input chemistry. These are honest,
 * documented heuristics (not model outputs): iron speckles and breaks, zinc/
 * titanium crystallise, boron runs, any colorant variegates.
 */
export function deriveEffects(chemistry: Record<string, number>): {
  speckle: boolean;
  variegation: boolean;
  crystalline: boolean;
  runny: boolean;
  breaking: boolean;
} {
  const fe = chemistry.Fe2O3 ?? 0;
  const colorant =
    (chemistry.Fe2O3 ?? 0) +
    (chemistry.CuO ?? 0) +
    (chemistry.CoO ?? 0) +
    (chemistry.MnO ?? 0) +
    (chemistry.Cr2O3 ?? 0);
  return {
    speckle: fe > 0.06 || (chemistry.MnO ?? 0) > 0.03,
    variegation: colorant > 0.01,
    crystalline: (chemistry.ZnO ?? 0) > 0.15 || (chemistry.TiO2 ?? 0) > 0.06,
    runny: (chemistry.B2O3 ?? 0) > 0.3,
    breaking: fe > 0.02,
  };
}

/** Representative colour for a predicted family, used when no Lab is available. */
const FAMILY_HEX: Record<string, string> = {
  Blanc: "#ECE6DA",
  Gris: "#9A958C",
  Noir: "#2B2B2B",
  Rouge: "#9A3B2E",
  Orange: "#C56A33",
  Jaune: "#C9A24B",
  Brun: "#6B4A2E",
  Vert: "#5E7F5A",
  Turquoise: "#2FB6B0",
  Bleu: "#3F5E9A",
  Violet: "#6E5A8F",
};

export function familyToHex(family: string | undefined): string {
  return (family && FAMILY_HEX[family]) || "#8A8278";
}
