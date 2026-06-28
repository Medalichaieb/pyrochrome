/**
 * Shared spider-chart + legend used by both Predict and Design.
 *
 * Plots a reference recipe (the prediction input, or a target estimate) against
 * a set of real recipes in chemistry space. Clicking a legend entry isolates
 * that recipe; clicking again shows the whole neighbourhood. Kept in one place
 * so both pages stay visually and behaviourally identical.
 */
import { RADAR_AXES } from "../chemistry";
import { hexToRgb, hslToRgb, rgbToHex, rgbToHsl } from "../color";
import { el } from "../dom";
import { renderRadar, type RadarSeries } from "../radar";

// Literal colours — SVG presentation attributes don't read CSS variables.
const INK_FAINT = "rgba(28, 26, 23, 0.18)";
const CLAY = "#bd5b33";
const CLAY_FILL = "rgba(189, 91, 51, 0.12)";

/** Darken a too-light fired colour so its isolated outline stays visible on paper. */
function readableStroke(hex: string): string {
  const { h, s, l } = rgbToHsl(hexToRgb(hex));
  return l > 0.7 ? rgbToHex(hslToRgb(h, Math.min(1, s + 0.1), 0.5)) : hex;
}

/** One real recipe to plot: normalised radar values + a legend colour/label. */
export interface ChartNeighbour {
  name: string;
  values: number[];
  /** Legend dot + isolated-outline colour (e.g. the recipe's fired colour). */
  colour: string;
  /** Optional trailing detail in the legend (e.g. "ΔE 6"). */
  meta?: string;
}

/**
 * Build the interactive neighbour chart.
 *
 * @param referenceValues Normalised radar values for the reference recipe.
 * @param neighbours Real recipes to compare against.
 * @param referenceLabel Legend label for the reference (e.g. "Your recipe").
 */
export function renderNeighbourChart(
  referenceValues: number[],
  neighbours: ChartNeighbour[],
  referenceLabel: string,
): HTMLElement {
  const labels = RADAR_AXES.map((axis) => axis.label);
  const referenceSeries: RadarSeries = {
    values: referenceValues,
    stroke: CLAY,
    fill: CLAY_FILL,
    width: 2,
    z: 2,
  };

  let selected: number | null = null;
  const radarHost = el("div", { class: "radar-wrap" });
  const legend = el("ul", { class: "legend" });
  const hint = el("p", { class: "legend-hint" });

  const draw = (): void => {
    const series: RadarSeries[] =
      selected === null
        ? neighbours.map((n) => ({ values: n.values, stroke: INK_FAINT, width: 1, z: 0 }))
        : [
            {
              values: neighbours[selected].values,
              stroke: readableStroke(neighbours[selected].colour),
              width: 2.5,
              z: 1,
            },
          ];
    series.push(referenceSeries);
    radarHost.replaceChildren(renderRadar(labels, series));

    legend.replaceChildren(
      ...neighbours.map((n, i) =>
        el(
          "li",
          { class: "legend-item" },
          el(
            "button",
            {
              type: "button",
              class: "legend-btn",
              "aria-pressed": String(selected === i),
              onclick: () => {
                selected = selected === i ? null : i;
                draw();
              },
            },
            el("span", { class: "legend-dot", style: `background:${n.colour}` }),
            el("span", { class: "legend-name" }, n.meta ? `${n.name} · ${n.meta}` : n.name),
          ),
        ),
      ),
    );
    hint.textContent =
      selected === null
        ? "Select a recipe to isolate it on the chart."
        : `Showing ${neighbours[selected].name} — select again to show all.`;
  };
  draw();

  return el(
    "div",
    {},
    radarHost,
    el(
      "div",
      { class: "radar-legend-wrap" },
      el("p", { class: "legend-key" }, el("span", { class: "legend-swatch you" }), referenceLabel),
      legend,
      hint,
    ),
  );
}
