/**
 * Colour-family confusion matrix (row-normalised), for the Docs page.
 *
 * A static snapshot from the held-out evaluation (`make eval`, 2026-06-28,
 * HistGradientBoosting). Row = true family, column = predicted; each row sums to
 * ~1. Regenerate the figures with `make eval`; refresh these numbers from
 * reports/REPORT.md when the model changes.
 */
export interface ConfusionData {
  labels: string[];
  /** matrix[i][j] = P(predicted = labels[j] | true = labels[i]). */
  matrix: number[][];
}

export const COLOUR_CONFUSION: ConfusionData = {
  labels: [
    "Blanc",
    "Bleu",
    "Brun",
    "Jaune",
    "Noir",
    "Orange",
    "Rouge",
    "Turquoise",
    "Vert",
    "Violet",
  ],
  matrix: [
    [0.874, 0.02, 0.002, 0.004, 0.02, 0.014, 0.022, 0.018, 0.022, 0.002],
    [0.16, 0.606, 0.0, 0.021, 0.032, 0.011, 0.021, 0.117, 0.032, 0.0],
    [0.074, 0.037, 0.407, 0.0, 0.185, 0.074, 0.185, 0.0, 0.037, 0.0],
    [0.49, 0.02, 0.0, 0.314, 0.0, 0.039, 0.118, 0.0, 0.02, 0.0],
    [0.276, 0.041, 0.031, 0.02, 0.5, 0.02, 0.031, 0.031, 0.041, 0.01],
    [0.391, 0.043, 0.0, 0.087, 0.043, 0.283, 0.13, 0.0, 0.022, 0.0],
    [0.178, 0.01, 0.03, 0.02, 0.05, 0.02, 0.634, 0.02, 0.04, 0.0],
    [0.169, 0.13, 0.0, 0.0, 0.013, 0.013, 0.013, 0.506, 0.143, 0.013],
    [0.271, 0.035, 0.0, 0.012, 0.012, 0.024, 0.035, 0.106, 0.494, 0.012],
    [0.107, 0.286, 0.0, 0.0, 0.036, 0.0, 0.179, 0.107, 0.036, 0.25],
  ],
};
