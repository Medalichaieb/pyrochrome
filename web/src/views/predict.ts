/**
 * Predict page: set a glaze recipe + firing, ask the model, and show the
 * predicted render — surface, transparency, colour (family + confidence + Lab) —
 * as a procedurally rendered tile, alongside the nearest real recipes.
 */
import { renderNeighbourChart, type ChartNeighbour } from "../components/neighbourChart";
import { predict } from "../inference";
import {
  CONES,
  DEFAULT_RECIPE,
  OXIDE_GROUPS,
  PRESETS,
  UMF_STANDARD,
  deriveEffects,
  familyToHex,
  radarValues,
  surfaceToRenderer,
  transparencyToOpacity,
} from "../chemistry";
import { labToRgb, rgbToHex } from "../color";
import { el } from "../dom";
import { renderTile, type TileAttributes } from "../renderer";
import type { Atmosphere, ClassPrediction, Neighbour, PredictResponse } from "../types";

interface State {
  chemistry: Record<string, number>;
  cone: string;
  atmosphere: Atmosphere;
  seed: number;
  lastAttributes: TileAttributes | null;
}

function confidenceBar(value: number): HTMLElement {
  const pct = Math.round(value * 100);
  return el(
    "div",
    { class: "conf", title: `${pct}% confidence`, role: "img", "aria-label": `${pct}% confidence` },
    el("span", { class: "conf-fill", style: `width:${pct}%` }),
  );
}

function pct(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function classRow(title: string, p: ClassPrediction | null): HTMLElement {
  if (!p) return el("div", { class: "readout-row muted" }, title, el("b", {}, "—"));
  const second = p.top2[1];
  const altText =
    second && second.label !== p.label ? ` · then ${second.label} ${pct(second.p)}` : "";
  return el(
    "div",
    { class: "readout-row" },
    el("span", { class: "readout-key" }, title),
    el(
      "span",
      { class: "readout-val" },
      el("b", {}, p.label),
      el("span", { class: "pct mono" }, pct(p.confidence)),
      el("span", { class: "alt" }, altText),
      confidenceBar(p.confidence),
    ),
  );
}

function neighbourHex(n: Neighbour): string {
  return rgbToHex({ r: n.rgb_r ?? 200, g: n.rgb_g ?? 200, b: n.rgb_b ?? 200 });
}

/** Human-readable summary of the (illustrative) texture effects in play. */
function describeEffects(fx: ReturnType<typeof deriveEffects>): string {
  const labels: [keyof typeof fx, string][] = [
    ["breaking", "breaks on edges"],
    ["variegation", "variegation"],
    ["speckle", "iron speckle"],
    ["crystalline", "crystals"],
    ["runny", "runs & pools"],
  ];
  const active = labels.filter(([key]) => fx[key]).map(([, label]) => label);
  return active.length ? active.join(" · ") : "smooth, even";
}

export function renderPredict(host: HTMLElement): void {
  const state: State = {
    chemistry: { ...DEFAULT_RECIPE },
    cone: "6",
    atmosphere: "oxidation",
    seed: 1734,
    lastAttributes: null,
  };

  // --- result side ------------------------------------------------------
  const canvas = el("canvas", {
    class: "tile",
    width: 860,
    height: 620,
    role: "img",
    "aria-label": "Rendered glaze test tile",
  }) as HTMLCanvasElement;
  const stamp = el("span", { class: "stamp" }, "cone 6 · oxidation");
  const stage = el("figure", { class: "lightbox" }, canvas, stamp);

  const readout = el("div", { class: "readout" });
  const neighboursWrap = el("div", { class: "neighbours" });
  const status = el("p", { class: "status" });

  function paint(attrs: TileAttributes): void {
    state.lastAttributes = attrs;
    renderTile(canvas, attrs);
    stamp.textContent = `cone ${state.cone} · ${state.atmosphere}`;
  }

  function showResult(r: PredictResponse): void {
    const colour = r.colour;
    const hex = colour?.lab ? rgbToHex(labToRgb(colour.lab)) : familyToHex(colour?.label);
    const effects = deriveEffects(state.chemistry);
    paint({
      color: hex,
      surface: surfaceToRenderer(r.surface?.label),
      opacity: transparencyToOpacity(r.transparency?.label),
      reduction: state.atmosphere === "reduction",
      seed: state.seed,
      ...effects,
    });

    readout.replaceChildren(
      el(
        "p",
        { class: "readout-note" },
        "The model's best guess and its confidence (the bar / %), then its next most likely. Colour's top-2 is right ~80% of the time even when the single best guess is less sure.",
      ),
      classRow("Colour", colour),
      classRow("Surface", r.surface),
      classRow("Transparency", r.transparency),
      el(
        "div",
        { class: "readout-row" },
        el("span", { class: "readout-key" }, "Lab"),
        el(
          "span",
          { class: "readout-val mono" },
          colour?.lab ? colour.lab.map((v) => v.toFixed(0)).join(" / ") : "—",
        ),
      ),
      el(
        "div",
        { class: "readout-row" },
        el("span", { class: "readout-key" }, "Texture *"),
        el("span", { class: "readout-val" }, describeEffects(effects)),
      ),
    );

    // Texture is illustrative (rule-of-thumb from chemistry), so the redraw can
    // now re-roll its random placement.
    rerollBtn.disabled = false;

    const referenceValues = radarValues((oxide) => state.chemistry[oxide] ?? 0);
    const chartNeighbours: ChartNeighbour[] = r.neighbours.slice(0, 6).map((n) => ({
      name: n.name ?? "Untitled",
      values: radarValues((oxide) => Number(n[`${oxide}_umf`] ?? 0)),
      colour: neighbourHex(n),
    }));

    neighboursWrap.replaceChildren(
      el("p", { class: "eyebrow" }, "Nearest real recipes"),
      renderNeighbourChart(referenceValues, chartNeighbours, "Your recipe"),
      el(
        "p",
        { class: "fine" },
        "The spider plots your recipe (in unity formula) against the real fired recipes closest to it in chemistry. Where your outline sits inside their spread is where the model is on familiar ground.",
      ),
    );
  }

  async function runPredict(): Promise<void> {
    status.textContent = "Predicting…";
    status.className = "status is-busy";
    try {
      const result = await predict({
        chemistry_umf: state.chemistry,
        cone: state.cone,
        atmosphere: state.atmosphere,
      });
      status.textContent = "";
      status.className = "status";
      showResult(result);
    } catch {
      status.textContent = "Could not run the model. Please try again.";
      status.className = "status is-error";
    }
  }

  // --- form side --------------------------------------------------------
  const inputs = new Map<string, HTMLInputElement>();

  function syncInputs(): void {
    for (const [key, input] of inputs) input.value = String(state.chemistry[key] ?? 0);
    coneSelect.value = state.cone;
    for (const b of atmButtons)
      b.setAttribute("aria-pressed", String(b.dataset.atm === state.atmosphere));
  }

  const oxideGroups = OXIDE_GROUPS.map((group) =>
    el(
      "div",
      { class: "ox-group" },
      el("p", { class: "ox-title" }, group.title, el("span", { class: "ox-note" }, group.note)),
      el(
        "div",
        { class: "ox-grid" },
        ...group.oxides.map((ox) => {
          const input = el("input", {
            type: "number",
            id: `ox-${ox.key}`,
            class: "ox-input mono",
            min: 0,
            max: ox.max,
            step: ox.step,
            value: String(state.chemistry[ox.key] ?? 0),
            oninput: (e: Event) => {
              state.chemistry[ox.key] = parseFloat((e.target as HTMLInputElement).value) || 0;
            },
          }) as HTMLInputElement;
          inputs.set(ox.key, input);
          return el(
            "label",
            { class: "ox-field", for: `ox-${ox.key}` },
            el("span", { class: "ox-label" }, ox.label),
            input,
          );
        }),
      ),
    ),
  );

  const coneSelect = el(
    "select",
    {
      class: "select mono",
      id: "cone",
      onchange: (e: Event) => (state.cone = (e.target as HTMLSelectElement).value),
    },
    ...CONES.map((c) => el("option", { value: c, selected: c === state.cone }, `cone ${c}`)),
  ) as HTMLSelectElement;

  const atmButtons = (["oxidation", "reduction", "neutral"] as Atmosphere[]).map((atm) =>
    el(
      "button",
      {
        type: "button",
        class: "seg-btn",
        "data-atm": atm,
        "aria-pressed": String(atm === state.atmosphere),
        onclick: () => {
          state.atmosphere = atm;
          syncInputs();
        },
      },
      atm,
    ),
  );

  const presetRow = el(
    "div",
    { class: "presets" },
    ...PRESETS.map((p) =>
      el(
        "button",
        {
          type: "button",
          class: "preset",
          onclick: () => {
            state.chemistry = { ...DEFAULT_RECIPE, ...p.chemistry };
            state.cone = p.cone;
            state.atmosphere = p.atmosphere;
            syncInputs();
            void runPredict();
          },
        },
        p.name,
      ),
    ),
  );

  const predictBtn = el(
    "button",
    { type: "button", class: "btn-primary", onclick: () => void runPredict() },
    "Predict the render",
  );
  const rerollBtn = el("button", {
    type: "button",
    class: "btn-ghost",
    disabled: true,
    title: "Redraw the stylised texture with a new random seed — same prediction",
    onclick: () => {
      state.seed = (Math.random() * 100000) | 0;
      if (state.lastAttributes) paint({ ...state.lastAttributes, seed: state.seed });
    },
  }) as HTMLButtonElement;
  rerollBtn.append("↻ Redraw");

  const form = el(
    "section",
    { class: "form-col" },
    el("p", { class: "eyebrow" }, "Start from a known glaze"),
    presetRow,
    el(
      "div",
      { class: "firing" },
      el("label", { class: "field" }, el("span", { class: "field-label" }, "Cone"), coneSelect),
      el(
        "div",
        { class: "field" },
        el("span", { class: "field-label" }, "Atmosphere"),
        el(
          "div",
          { class: "seg", role: "group", "aria-label": "Firing atmosphere" },
          ...atmButtons,
        ),
      ),
    ),
    el(
      "div",
      { class: "composition-head" },
      el(
        "p",
        { class: "comp-title" },
        "Composition",
        el("span", { class: "unit-tag" }, UMF_STANDARD.short),
      ),
      el("p", { class: "unit-note" }, UMF_STANDARD.long),
    ),
    ...oxideGroups,
    el("div", { class: "actions" }, predictBtn, rerollBtn),
  );

  const tileNote = el(
    "p",
    { class: "tile-note" },
    "A stylised illustration of the predicted colour, surface and transparency — not a photo. ",
    el("strong", {}, "* Texture"),
    " (variegation, breaks, speckle…) is indicative: inferred from the recipe chemistry by simple rules, ",
    el("strong", {}, "not predicted by the model"),
    ". ",
    el("strong", {}, "↻ Redraw"),
    " re-rolls only the random placement of that texture — the prediction stays the same.",
  );

  const result = el(
    "section",
    { class: "result-col" },
    stage,
    status,
    readout,
    tileNote,
    neighboursWrap,
  );

  host.append(
    el(
      "div",
      { class: "predict" },
      el(
        "header",
        { class: "page-head" },
        el("p", { class: "eyebrow" }, "Predict"),
        el("h1", {}, "Predict the glaze before the kiln"),
        el(
          "p",
          { class: "lede" },
          "Set the oxide chemistry (unity formula), the Orton cone and the atmosphere. The model predicts surface, transparency and colour — then the test tile is drawn from those attributes.",
        ),
      ),
      el("div", { class: "predict-grid" }, form, result),
    ),
  );

  // Initial neutral tile so the stage is never empty.
  paint({
    color: "#8A8278",
    surface: "satin",
    opacity: 0.85,
    speckle: false,
    variegation: false,
    crystalline: false,
    runny: false,
    breaking: false,
    reduction: false,
    seed: state.seed,
  });
  status.textContent = "Set a recipe (or pick a preset) and press Predict.";
}
