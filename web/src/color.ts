/**
 * Colour-space helpers for the renderer and swatches.
 *
 * Mirrors the server's CIELAB conversion (D65) so a predicted Lab value maps to
 * the same sRGB the model was trained against, and adds the inverse (Lab→sRGB)
 * needed to paint the tile from a predicted Lab colour.
 */

export interface Rgb {
  r: number;
  g: number;
  b: number;
}

const WHITE_D65: [number, number, number] = [0.95047, 1.0, 1.08883];

/** Clamp a number to the [lo, hi] range. */
export function clamp(value: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, value));
}

/** Parse a `#rrggbb` string into RGB (0–255). */
export function hexToRgb(hex: string): Rgb {
  const h = hex.replace("#", "");
  return {
    r: parseInt(h.slice(0, 2), 16),
    g: parseInt(h.slice(2, 4), 16),
    b: parseInt(h.slice(4, 6), 16),
  };
}

/** Format RGB (0–255) as a `#rrggbb` string. */
export function rgbToHex({ r, g, b }: Rgb): string {
  const to2 = (v: number) => clamp(Math.round(v), 0, 255).toString(16).padStart(2, "0");
  return `#${to2(r)}${to2(g)}${to2(b)}`;
}

/** Convert CIELAB (D65) to sRGB (0–255). Inverse of the server's srgb_to_lab. */
export function labToRgb([l, a, b]: [number, number, number]): Rgb {
  const fy = (l + 16) / 116;
  const fx = fy + a / 500;
  const fz = fy - b / 200;
  const eps = 216 / 24389;
  const kappa = 24389 / 27;
  const inv = (f: number) => {
    const f3 = f ** 3;
    return f3 > eps ? f3 : (116 * f - 16) / kappa;
  };
  const [xn, yn, zn] = WHITE_D65;
  const x = inv(fx) * xn;
  const y = inv(fy) * yn;
  const z = inv(fz) * zn;

  // XYZ → linear sRGB (D65).
  const rl = x * 3.2404542 + y * -1.5371385 + z * -0.4985314;
  const gl = x * -0.969266 + y * 1.8760108 + z * 0.041556;
  const bl = x * 0.0556434 + y * -0.2040259 + z * 1.0572252;

  const gamma = (c: number) => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);
  return {
    r: clamp(gamma(rl) * 255, 0, 255),
    g: clamp(gamma(gl) * 255, 0, 255),
    b: clamp(gamma(bl) * 255, 0, 255),
  };
}

/** RGB (0–255) → HSL (h,s,l in 0–1). */
export function rgbToHsl({ r, g, b }: Rgb): { h: number; s: number; l: number } {
  r /= 255;
  g /= 255;
  b /= 255;
  const mx = Math.max(r, g, b);
  const mn = Math.min(r, g, b);
  let h = 0;
  let s = 0;
  const l = (mx + mn) / 2;
  if (mx !== mn) {
    const d = mx - mn;
    s = l > 0.5 ? d / (2 - mx - mn) : d / (mx + mn);
    switch (mx) {
      case r:
        h = (g - b) / d + (g < b ? 6 : 0);
        break;
      case g:
        h = (b - r) / d + 2;
        break;
      default:
        h = (r - g) / d + 4;
    }
    h /= 6;
  }
  return { h, s, l };
}

/** HSL (0–1) → RGB (0–255). */
export function hslToRgb(h: number, s: number, l: number): Rgb {
  if (s === 0) {
    const v = l * 255;
    return { r: v, g: v, b: v };
  }
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  const hue = (t: number) => {
    if (t < 0) t += 1;
    if (t > 1) t -= 1;
    if (t < 1 / 6) return p + (q - p) * 6 * t;
    if (t < 1 / 2) return q;
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
    return p;
  };
  return { r: hue(h + 1 / 3) * 255, g: hue(h) * 255, b: hue(h - 1 / 3) * 255 };
}
