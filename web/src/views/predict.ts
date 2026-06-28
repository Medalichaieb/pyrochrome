/**
 * Predict page: set a glaze recipe + firing, ask the model, and show the
 * predicted render — surface, transparency, colour (family + confidence + Lab) —
 * as a procedurally rendered tile, alongside the nearest real recipes.
 */
import { ApiError, predict } from "../api";
import {
  CONES,
  DEFAULT_RECIPE,
  OXIDE_GROUPS,
  PRESETS,
  RADAR_AXES,
  UMF_STANDARD,
  axisValue,
  deriveEffects,
  familyToHex,
  surfaceToRenderer,
  transparencyToOpacity,
} from "../chemistry";
import { hexToRgb, hslToRgb, labToRgb, rgbToHex, rgbToHsl } from "../color";
import { el } from "../dom";
import { renderRadar, type RadarSeries } from "../radar";
import { renderTile, type TileAttributes } from "../renderer";
import type { Atmosphere, ClassPrediction, Neighbour, PredictResponse } from "../types";

// Radar palette (literal colours — SVG presentation attrs don't read CSS vars).
const INK_FAINT = "rgba(28, 26, 23, 0.18)";
const CLAY = "#bd5b33";
const CLAY_FILL = "rgba(189, 91, 51, 0.12)";

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

function classRow(title: string, p: ClassPrediction | null): HTMLElement {
  if (!p) return el("div", { class: "readout-row muted" }, title, el("b", {}, "—"));
  const alt = p.top2[1] && p.top2[1] !== p.label ? ` · or ${p.top2[1]}` : "";
  return el(
    "div",
    { class: "readout-row" },
    el("span", { class: "readout-key" }, title),
    el(
      "span",
      { class: "readout-val" },
      el("b", {}, p.label),
      el("span", { class: "alt" }, alt),
      confidenceBar(p.confidence),
    ),
  );
}

function neighbourHex(n: Neighbour): string {
  return rgbToHex({ r: n.rgb_r ?? 200, g: n.rgb_g ?? 200, b: n.rgb_b ?? 200 });
}

/** Radar values for a recipe, given a per-oxide molar lookup. */
function radarValues(lookup: (oxide: string) => number): number[] {
  return RADAR_AXES.map((axis) => axisValue(axis, lookup));
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

/** Darken a too-light fired colour so its radar outline stays visible on paper. */
function readableStroke(hex: string): string {
  const { h, s, l } = rgbToHsl(hexToRgb(hex));
  return l > 0.7 ? rgbToHex(hslToRgb(h, Math.min(1, s + 0.1), 0.5)) : hex;
}

/** A clickable legend entry: fired-colour dot + name, toggles its radar outline. */
function neighbourLegendButton(
  n: Neighbour,
  index: number,
  selected: boolean,
  onToggle: (index: number) => void,
): HTMLElement {
  return el(
    "li",
    { class: "legend-item" },
    el(
      "button",
      {
        type: "button",
        class: "legend-btn",
        "aria-pressed": String(selected),
        onclick: () => onToggle(index),
      },
      el("span", { class: "legend-dot", style: `background:${neighbourHex(n)}` }),
      el("span", { class: "legend-name" }, n.name ?? "Untitled"),
    ),
  );
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

    const neighbours = r.neighbours.slice(0, 6);
    const labels = RADAR_AXES.map((a) => a.label);
    const inputSeries: RadarSeries = {
      values: radarValues((oxide) => state.chemistry[oxide] ?? 0),
      stroke: CLAY,
      fill: CLAY_FILL,
      width: 2,
      z: 2,
    };
    const neighbourValues = neighbours.map((n) =>
      radarValues((oxide) => Number(n[`${oxide}_umf`] ?? 0)),
    );

    // null = show the whole neighbourhood faintly; an index = isolate that one.
    let selected: number | null = null;
    const radarHost = el("div", { class: "radar-wrap" });
    const legend = el("ul", { class: "legend" });
    const hint = el("p", { class: "legend-hint" });

    const draw = (): void => {
      const series: RadarSeries[] = [];
      if (selected === null) {
        neighbours.forEach((_, i) =>
          series.push({ values: neighbourValues[i], stroke: INK_FAINT, width: 1, z: 0 }),
        );
      } else {
        const stroke = readableStroke(neighbourHex(neighbours[selected]));
        series.push({ values: neighbourValues[selected], stroke, width: 2.5, z: 1 });
      }
      series.push(inputSeries);
      radarHost.replaceChildren(renderRadar(labels, series));

      legend.replaceChildren(
        ...neighbours.map((n, i) =>
          neighbourLegendButton(n, i, selected === i, (index) => {
            selected = selected === index ? null : index;
            draw();
          }),
        ),
      );
      hint.textContent =
        selected === null
          ? "Select a recipe to isolate it on the chart."
          : `Showing ${neighbours[selected].name ?? "selected recipe"} — select again to show all.`;
    };
    draw();

    neighboursWrap.replaceChildren(
      el("p", { class: "eyebrow" }, "Nearest real recipes"),
      radarHost,
      el(
        "div",
        { class: "radar-legend-wrap" },
        el("p", { class: "legend-key" }, el("span", { class: "legend-swatch you" }), "Your recipe"),
        legend,
        hint,
      ),
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
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : "Something went wrong. Please try again.";
      status.textContent = msg;
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
