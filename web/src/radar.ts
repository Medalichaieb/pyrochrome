/**
 * Minimal SVG radar (spider) chart — positions a recipe in chemistry space
 * against the real recipes nearest to it.
 *
 * Built with createElementNS and numeric geometry only (no innerHTML), so there
 * is no injection surface. Axis labels are caller-supplied static strings.
 */

const NS = "http://www.w3.org/2000/svg";

/** One polygon on the chart: normalised values (0–1) aligned to the axes. */
export interface RadarSeries {
  values: number[];
  stroke: string;
  fill?: string;
  width: number;
  /** Lower = drawn first (under). */
  z?: number;
}

interface RadarOptions {
  size?: number;
  rings?: number;
}

function svg<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs: Record<string, string | number>,
): SVGElementTagNameMap[K] {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  return node;
}

function point(cx: number, cy: number, r: number, angle: number, value: number): [number, number] {
  return [cx + r * value * Math.cos(angle), cy + r * value * Math.sin(angle)];
}

function polygonPoints(
  values: number[],
  cx: number,
  cy: number,
  r: number,
  angles: number[],
): string {
  return values
    .map((v, i) =>
      point(cx, cy, r, angles[i], v)
        .map((n) => n.toFixed(1))
        .join(","),
    )
    .join(" ");
}

/** Render the radar chart as an inline SVG element. */
export function renderRadar(
  labels: string[],
  series: RadarSeries[],
  options: RadarOptions = {},
): SVGSVGElement {
  const size = options.size ?? 340;
  const rings = options.rings ?? 4;
  const cx = size / 2;
  const cy = size / 2;
  const r = size / 2 - 46; // leave room for labels
  const n = labels.length;
  const angles = labels.map((_, i) => -Math.PI / 2 + (i * 2 * Math.PI) / n);

  const root = svg("svg", {
    viewBox: `0 0 ${size} ${size}`,
    class: "radar",
    role: "img",
    "aria-label": "Recipe positioned in chemistry space against nearest real recipes",
  });

  // Grid rings.
  for (let ring = 1; ring <= rings; ring++) {
    const v = ring / rings;
    root.append(
      svg("polygon", {
        points: polygonPoints(new Array(n).fill(v), cx, cy, r, angles),
        class: "radar-ring",
      }),
    );
  }

  // Axis spokes + labels.
  for (let i = 0; i < n; i++) {
    const [ex, ey] = point(cx, cy, r, angles[i], 1);
    root.append(svg("line", { x1: cx, y1: cy, x2: ex, y2: ey, class: "radar-spoke" }));
    const [lx, ly] = point(cx, cy, r + 18, angles[i], 1);
    const label = svg("text", {
      x: lx,
      y: ly,
      class: "radar-label",
      "text-anchor": lx < cx - 1 ? "end" : lx > cx + 1 ? "start" : "middle",
      "dominant-baseline": "middle",
    });
    label.textContent = labels[i];
    root.append(label);
  }

  // Series polygons (sorted so the input draws on top).
  for (const s of [...series].sort((a, b) => (a.z ?? 0) - (b.z ?? 0))) {
    root.append(
      svg("polygon", {
        points: polygonPoints(s.values, cx, cy, r, angles),
        fill: s.fill ?? "none",
        stroke: s.stroke,
        "stroke-width": s.width,
        "stroke-linejoin": "round",
      }),
    );
  }

  return root;
}
