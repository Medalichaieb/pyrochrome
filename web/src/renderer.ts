/**
 * Procedural glaze-tile renderer (ported from prototypes/glaze_renderer.html).
 *
 * This is the *visualisation* layer: it turns predicted attributes (colour,
 * surface, transparency, effects, firing) into a stylised test tile drawn on a
 * canvas — client-side, no GPU. It does not reproduce a real photo; it renders
 * faithfully what the model predicted.
 */
import { clamp, hexToRgb, hslToRgb, rgbToHsl, type Rgb } from "./color";

export type Surface = "glossy" | "satin" | "matte";

/** Everything the renderer needs to draw one tile. */
export interface TileAttributes {
  /** Base glaze colour as `#rrggbb`. */
  color: string;
  surface: Surface;
  /** Glaze opacity in 0–1 (from predicted transparency). */
  opacity: number;
  speckle: boolean;
  variegation: boolean;
  crystalline: boolean;
  runny: boolean;
  /** Lighter "breaking" colour along the dip edge. */
  breaking: boolean;
  /** "oxidation" warms/deepens slightly less than "reduction". */
  reduction: boolean;
  /** Texture seed — change to re-roll the same attributes. */
  seed: number;
}

const TILE = { x: 255, y: 70, w: 360, h: 490, r: 26 };
const DIP_FRAC = 0.34; // glaze dipped to ~mid-height

function makeNoise(seed: number): (x: number, y: number) => number {
  const rand = (x: number, y: number) => {
    const n = Math.sin(x * 127.1 + y * 311.7 + seed * 13.73) * 43758.5453;
    return n - Math.floor(n);
  };
  const sm = (t: number) => t * t * (3 - 2 * t);
  return (x, y) => {
    const xi = Math.floor(x);
    const yi = Math.floor(y);
    const xf = x - xi;
    const yf = y - yi;
    const tl = rand(xi, yi);
    const tr = rand(xi + 1, yi);
    const bl = rand(xi, yi + 1);
    const br = rand(xi + 1, yi + 1);
    const u = sm(xf);
    const v = sm(yf);
    return (tl * (1 - u) + tr * u) * (1 - v) + (bl * (1 - u) + br * u) * v;
  };
}

function fbm(noise: (x: number, y: number) => number, x: number, y: number): number {
  return noise(x, y) * 0.6 + noise(x * 2.1, y * 2.1) * 0.3 + noise(x * 4.3, y * 4.3) * 0.1;
}

/** Deterministic RNG for effect placement. */
function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function tilePath(ctx: CanvasRenderingContext2D): void {
  const { x, y, w, h, r } = TILE;
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

/** Draw the tile for the given attributes onto the canvas 2D context. */
export function renderTile(canvas: HTMLCanvasElement, attrs: TileAttributes): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const noise = makeNoise(attrs.seed);
  const rgb0 = hexToRgb(attrs.color);
  const hsl = rgbToHsl(rgb0);
  const atmWarm = attrs.reduction;

  // Drop shadow.
  ctx.save();
  ctx.shadowColor = "rgba(0,0,0,.28)";
  ctx.shadowBlur = 46;
  ctx.shadowOffsetY = 26;
  tilePath(ctx);
  ctx.fillStyle = "#000";
  ctx.fill();
  ctx.restore();

  // Clay body.
  ctx.save();
  tilePath(ctx);
  ctx.clip();
  const clay = ctx.createLinearGradient(0, TILE.y, 0, TILE.y + TILE.h);
  clay.addColorStop(0, "#C7A988");
  clay.addColorStop(1, "#A6855F");
  ctx.fillStyle = clay;
  ctx.fillRect(TILE.x, TILE.y, TILE.w, TILE.h);
  const cl = rng(attrs.seed ^ 99);
  for (let i = 0; i < 1400; i++) {
    const px = TILE.x + cl() * TILE.w;
    const py = TILE.y + cl() * TILE.h;
    ctx.fillStyle = `rgba(80,55,35,${cl() * 0.06})`;
    ctx.fillRect(px, py, 1.4, 1.4);
  }
  ctx.restore();

  // Glaze layer (dipped region).
  const gx = TILE.x;
  const gw = TILE.w;
  const gy = Math.round(TILE.y + TILE.h * DIP_FRAC);
  const gh = Math.round(TILE.y + TILE.h - gy);
  const off = document.createElement("canvas");
  off.width = gw;
  off.height = gh;
  const octx = off.getContext("2d");
  if (!octx) return;
  const img = octx.createImageData(gw, gh);
  const d = img.data;
  const matte = attrs.surface === "matte";
  const satin = attrs.surface === "satin";

  for (let yy = 0; yy < gh; yy++) {
    const t = yy / gh; // 0 top (dip edge) -> 1 bottom (pooled)
    for (let xx = 0; xx < gw; xx++) {
      let L = hsl.l;
      let S = hsl.s;
      let Hh = hsl.h;
      L += -0.1 * t;
      S += 0.14 * t;
      if (attrs.breaking) {
        const edge = Math.max(0, 1 - t * 6);
        L += 0.2 * edge * edge;
        S -= 0.1 * edge;
      }
      if (attrs.variegation) {
        const n = fbm(noise, xx / 70, yy / 70) - 0.5;
        L += n * 0.2;
        Hh += n * 0.02;
        S += n * 0.06;
      }
      if (attrs.runny) {
        const v = fbm(noise, xx / 14, yy / 120);
        L += (v - 0.5) * 0.12 * t;
      }
      if (matte) {
        const hf = fbm(noise, xx / 3.2, yy / 3.2);
        L += (hf - 0.5) * 0.06;
      } else if (satin) {
        const hf = fbm(noise, xx / 5, yy / 5);
        L += (hf - 0.5) * 0.03;
      }
      if (atmWarm) {
        S += 0.05;
        L -= 0.02;
        Hh -= 0.006;
      }
      L = clamp(L, 0.03, 0.97);
      S = clamp(S, 0, 1);
      const c: Rgb = hslToRgb((Hh + 1) % 1, S, L);
      let a = attrs.opacity * (0.62 + 0.38 * t);
      if (attrs.breaking) a *= 0.55 + 0.45 * Math.min(1, t * 5);
      a = Math.min(1, a);
      const cr = 199 - 40 * t;
      const cg = 169 - 40 * t;
      const cb = 136 - 40 * t;
      const i4 = (yy * gw + xx) * 4;
      d[i4] = c.r * a + cr * (1 - a);
      d[i4 + 1] = c.g * a + cg * (1 - a);
      d[i4 + 2] = c.b * a + cb * (1 - a);
      d[i4 + 3] = 255;
    }
  }
  octx.putImageData(img, 0, 0);

  ctx.save();
  tilePath(ctx);
  ctx.clip();
  ctx.drawImage(off, gx, gy);

  // Dip edge (breaking line).
  if (attrs.breaking) {
    const lg = ctx.createLinearGradient(0, gy - 10, 0, gy + 14);
    lg.addColorStop(0, "rgba(0,0,0,0)");
    lg.addColorStop(0.5, "rgba(255,248,235,.18)");
    lg.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = lg;
    ctx.fillRect(gx, gy - 10, gw, 24);
  }

  // Speckle (iron / manganese).
  if (attrs.speckle) {
    const r = rng(attrs.seed ^ 7);
    for (let i = 0; i < 260; i++) {
      const t = r();
      const px = gx + r() * gw;
      const py = gy + t * gh;
      const sz = 0.6 + r() * 1.8;
      ctx.fillStyle = `rgba(40,28,20,${0.25 + r() * 0.5 * (0.4 + t)})`;
      ctx.beginPath();
      ctx.ellipse(px, py, sz, sz * 0.8, 0, 0, 7);
      ctx.fill();
    }
  }

  // Crystalline (zinc / titanium).
  if (attrs.crystalline) {
    const r = rng(attrs.seed ^ 21);
    const lc = hslToRgb(hsl.h, Math.max(0, hsl.s - 0.2), Math.min(0.92, hsl.l + 0.22));
    ctx.strokeStyle = `rgba(${lc.r | 0},${lc.g | 0},${lc.b | 0},.5)`;
    for (let i = 0; i < 14; i++) {
      const px = gx + r() * gw;
      const py = gy + (0.2 + r() * 0.8) * gh;
      const spokes = 6 + ((r() * 6) | 0);
      const rad = 6 + r() * 16;
      ctx.lineWidth = 1;
      for (let k = 0; k < spokes; k++) {
        const ang = (k / spokes) * Math.PI * 2;
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(px + Math.cos(ang) * rad, py + Math.sin(ang) * rad * 0.7);
        ctx.stroke();
      }
      ctx.fillStyle = `rgba(${lc.r | 0},${lc.g | 0},${lc.b | 0},.25)`;
      ctx.beginPath();
      ctx.arc(px, py, 2.2, 0, 7);
      ctx.fill();
    }
  }

  // Surface sheen.
  if (attrs.surface === "glossy") {
    const hi = ctx.createRadialGradient(
      gx + gw * 0.34,
      gy + gh * 0.18,
      8,
      gx + gw * 0.34,
      gy + gh * 0.18,
      gw * 0.7,
    );
    hi.addColorStop(0, "rgba(255,250,240,.42)");
    hi.addColorStop(0.25, "rgba(255,250,240,.12)");
    hi.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = hi;
    ctx.fillRect(gx, gy, gw, gh);
    ctx.fillStyle = "rgba(255,252,245,.20)";
    ctx.fillRect(gx, gy, gw, 3);
  } else if (satin) {
    const sh = ctx.createLinearGradient(gx, gy, gx + gw, gy + gh);
    sh.addColorStop(0, "rgba(255,250,240,.10)");
    sh.addColorStop(0.5, "rgba(255,250,240,.02)");
    sh.addColorStop(1, "rgba(0,0,0,.05)");
    ctx.fillStyle = sh;
    ctx.fillRect(gx, gy, gw, gh);
  }
  ctx.restore();

  // Runny: drips below the bottom edge.
  if (attrs.runny) {
    ctx.save();
    const r = rng(attrs.seed ^ 55);
    const dark = hslToRgb(hsl.h, Math.min(1, hsl.s + 0.18), Math.max(0.05, hsl.l - 0.16));
    for (let i = 0; i < 5; i++) {
      const px = gx + 18 + r() * (gw - 36);
      const drop = 6 + r() * 18;
      ctx.fillStyle = `rgba(${dark.r | 0},${dark.g | 0},${dark.b | 0},.85)`;
      ctx.beginPath();
      ctx.ellipse(px, TILE.y + TILE.h - 2, 3 + r() * 2, drop, 0, 0, 7);
      ctx.fill();
    }
    ctx.restore();
  }

  // Hanging hole.
  ctx.save();
  ctx.fillStyle = "#15100D";
  ctx.beginPath();
  ctx.arc(TILE.x + TILE.w / 2, TILE.y + 34, 13, 0, 7);
  ctx.fill();
  ctx.strokeStyle = "rgba(0,0,0,.5)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(TILE.x + TILE.w / 2, TILE.y + 34, 13, 0, 7);
  ctx.stroke();
  ctx.restore();

  // Tile rim.
  ctx.save();
  tilePath(ctx);
  ctx.lineWidth = 1.4;
  ctx.strokeStyle = "rgba(120,90,60,.18)";
  ctx.stroke();
  ctx.restore();
}
