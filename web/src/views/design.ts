/**
 * Design page (inverse lookup): start from a target look — colour, surface,
 * transparency — and get an estimated recipe plus the real recipes closest to
 * that target. Retrieval is by perceptual colour distance (ΔE) over the real
 * recipes, optionally filtered by surface and transparency; the estimated
 * recipe is the median chemistry of the closest matches. This is honest
 * "retrieval-first" inverse design: every suggestion is a real, fireable glaze.
 */
import { renderNeighbourChart, type ChartNeighbour } from "../components/neighbourChart";
import { oxideLabel, radarValues, surfaceToRenderer, transparencyToOpacity } from "../chemistry";
import { deltaE76, hexToRgb, rgbToHex, srgbToLab } from "../color";
import { el } from "../dom";
import { loadRecipes, type Recipe } from "../recipes";
import { renderTile, type TileAttributes } from "../renderer";

const SURFACES = ["Any", "Glossy", "Satin", "Matte"];
const TRANSPARENCIES = ["Any", "Opaque", "Semi-opaque", "Translucent", "Transparent"];
const EFFECTS = ["variegation", "speckle", "crystalline", "runny", "breaking"] as const;
const MATCHES = 6;

type EffectKey = (typeof EFFECTS)[number];

interface Target {
  colour: string;
  surface: string;
  transparency: string;
  effects: Record<EffectKey, boolean>;
  seed: number;
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/** Build a bare-oxide → value lookup from an `ox` array aligned to radar_oxides. */
function oxLookup(radarOxides: string[], ox: number[]): (oxide: string) => number {
  const index = new Map(radarOxides.map((col, i) => [col.replace("_umf", ""), i]));
  return (oxide) => ox[index.get(oxide) ?? -1] ?? 0;
}

export function renderDesign(host: HTMLElement): void {
  const target: Target = {
    colour: "#3f6f63",
    surface: "Any",
    transparency: "Any",
    effects: {
      variegation: true,
      speckle: false,
      crystalline: false,
      runny: false,
      breaking: true,
    },
    seed: 1734,
  };

  // --- target preview ---------------------------------------------------
  const canvas = el("canvas", {
    class: "tile",
    width: 860,
    height: 620,
    role: "img",
    "aria-label": "Target look preview",
  }) as HTMLCanvasElement;
  const stage = el(
    "figure",
    { class: "lightbox" },
    canvas,
    el("span", { class: "stamp" }, "target"),
  );

  const paint = (): void => {
    const attrs: TileAttributes = {
      color: target.colour,
      surface: surfaceToRenderer(target.surface === "Any" ? "Satin" : target.surface),
      opacity: transparencyToOpacity(
        target.transparency === "Any" ? undefined : target.transparency,
      ),
      reduction: false,
      seed: target.seed,
      ...EFFECTS.reduce(
        (acc, fx) => ({ ...acc, [fx]: target.effects[fx] }),
        {} as Record<EffectKey, boolean>,
      ),
    };
    renderTile(canvas, attrs);
  };

  // --- results ----------------------------------------------------------
  const recipeOut = el("div", { class: "readout" });
  const chartOut = el("div", { class: "neighbours" });
  const status = el("p", { class: "status" });

  const suggest = async (): Promise<void> => {
    status.textContent = "Searching real recipes…";
    status.className = "status is-busy";
    try {
      const data = await loadRecipes();
      const targetLab = srgbToLab(hexToRgb(target.colour));
      const pool = data.recipes.filter(
        (recipe) =>
          (target.surface === "Any" || recipe.surface === target.surface) &&
          (target.transparency === "Any" || recipe.transparency === target.transparency),
      );
      if (pool.length === 0) {
        status.textContent =
          "No real recipe matches that surface and transparency. Loosen a filter.";
        status.className = "status is-error";
        recipeOut.replaceChildren();
        chartOut.replaceChildren();
        return;
      }
      const ranked = pool
        .map((recipe) => ({
          recipe,
          de: deltaE76(
            srgbToLab({ r: recipe.rgb[0], g: recipe.rgb[1], b: recipe.rgb[2] }),
            targetLab,
          ),
        }))
        .sort((a, b) => a.de - b.de)
        .slice(0, MATCHES);

      status.textContent = "";
      status.className = "status";
      showRecipe(data.radar_oxides, ranked);
      showChart(data.radar_oxides, ranked);
    } catch {
      status.textContent = "Could not load the recipe data. Please try again.";
      status.className = "status is-error";
    }
  };

  function showRecipe(radarOxides: string[], ranked: { recipe: Recipe; de: number }[]): void {
    // Estimated recipe = median oxide values across the closest matches.
    const estimate = radarOxides.map((_, j) => median(ranked.map(({ recipe }) => recipe.ox[j])));
    const cells = radarOxides.map((col, j) =>
      el(
        "div",
        { class: "est-cell" },
        el("span", { class: "est-label" }, oxideLabel(col.replace("_umf", ""))),
        el("span", { class: "est-value mono" }, estimate[j].toFixed(estimate[j] < 0.1 ? 3 : 2)),
      ),
    );
    const closest = ranked[0];
    recipeOut.replaceChildren(
      el(
        "p",
        { class: "readout-note" },
        "An estimated unity formula to aim for — the median chemistry of the closest real recipes. Use the spider below to pick one real recipe to start from.",
      ),
      el("div", { class: "est-grid" }, ...cells),
      el(
        "p",
        { class: "fine" },
        `Closest real match: ${closest.recipe.name} (ΔE ${Math.round(closest.de)}).`,
      ),
      el(
        "p",
        { class: "fine" },
        "Note: this estimate doesn't fix the firing atmosphere, and some colours depend on it. Copper, for example, fires red in reduction but green or turquoise in oxidation, so fire these recipes in the atmosphere their colour implies.",
      ),
    );
  }

  function showChart(radarOxides: string[], ranked: { recipe: Recipe; de: number }[]): void {
    const estimate = radarOxides.map((_, j) => median(ranked.map(({ recipe }) => recipe.ox[j])));
    const referenceValues = radarValues(oxLookup(radarOxides, estimate));
    const neighbours: ChartNeighbour[] = ranked.map(({ recipe, de }) => ({
      name: recipe.name,
      values: radarValues(oxLookup(radarOxides, recipe.ox)),
      colour: rgbToHex({ r: recipe.rgb[0], g: recipe.rgb[1], b: recipe.rgb[2] }),
      meta: `ΔE ${Math.round(de)}`,
    }));
    chartOut.replaceChildren(
      el("p", { class: "eyebrow" }, "Closest real recipes"),
      renderNeighbourChart(referenceValues, neighbours, "Estimated recipe"),
    );
  }

  // --- controls ---------------------------------------------------------
  const colourInput = el("input", {
    type: "color",
    class: "colour-input",
    value: target.colour,
    "aria-label": "Target colour",
    oninput: (e: Event) => {
      target.colour = (e.target as HTMLInputElement).value;
      paint();
    },
  });

  const segGroup = (
    label: string,
    options: string[],
    get: () => string,
    set: (value: string) => void,
  ): HTMLElement => {
    const buttons = options.map((opt) =>
      el(
        "button",
        {
          type: "button",
          class: "seg-btn",
          "aria-pressed": String(get() === opt),
          onclick: (e: Event) => {
            set(opt);
            const group = (e.target as HTMLElement).parentElement;
            group
              ?.querySelectorAll("button")
              .forEach((b) => b.setAttribute("aria-pressed", String(b.textContent === opt)));
            paint();
          },
        },
        opt,
      ),
    );
    return el(
      "div",
      { class: "field" },
      el("span", { class: "field-label" }, label),
      el("div", { class: "seg seg-wrap", role: "group", "aria-label": label }, ...buttons),
    );
  };

  const effectChips = el(
    "div",
    { class: "chips" },
    ...EFFECTS.map((fx) =>
      el(
        "button",
        {
          type: "button",
          class: "chip",
          "aria-pressed": String(target.effects[fx]),
          onclick: (e: Event) => {
            target.effects[fx] = !target.effects[fx];
            (e.target as HTMLElement).setAttribute("aria-pressed", String(target.effects[fx]));
            paint();
          },
        },
        fx,
      ),
    ),
  );

  const form = el(
    "section",
    { class: "form-col" },
    el("p", { class: "eyebrow" }, "Target look"),
    el(
      "div",
      { class: "field" },
      el("span", { class: "field-label" }, "Colour"),
      el(
        "div",
        { class: "colour-row" },
        colourInput,
        el("span", { class: "colour-hint" }, "Pick any colour you're aiming for"),
      ),
    ),
    segGroup(
      "Surface",
      SURFACES,
      () => target.surface,
      (v) => (target.surface = v),
    ),
    segGroup(
      "Transparency",
      TRANSPARENCIES,
      () => target.transparency,
      (v) => (target.transparency = v),
    ),
    el(
      "div",
      { class: "field" },
      el("span", { class: "field-label" }, "Effects (preview only)"),
      effectChips,
    ),
    el(
      "div",
      { class: "actions" },
      el(
        "button",
        { type: "button", class: "btn-primary", onclick: () => void suggest() },
        "Suggest a recipe",
      ),
    ),
  );

  const result = el("section", { class: "result-col" }, stage, status, recipeOut, chartOut);

  host.append(
    el(
      "div",
      { class: "predict" },
      el(
        "header",
        { class: "page-head" },
        el("p", { class: "eyebrow" }, "Design"),
        el("h1", {}, "Start from the look you want"),
        el(
          "p",
          { class: "lede" },
          "Pick a target colour, surface and transparency, and Pyrochrome finds the real fired recipes closest to it and an estimated unity formula to aim for. Inverse design is one-to-many, so it suggests real, fireable recipes rather than a single answer.",
        ),
      ),
      el("div", { class: "predict-grid" }, form, result),
    ),
  );

  paint();
  status.textContent = "Set a target and press Suggest a recipe.";
}
