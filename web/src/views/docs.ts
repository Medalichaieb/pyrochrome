/**
 * Docs page: a clear, editorial account of the model and the data logic behind
 * Pyrochrome — what it learns, how, how well, and where it is honestly bounded.
 */
import { COLOUR_CONFUSION, type ConfusionData } from "../data/confusion";
import { el } from "../dom";

/**
 * Render a row-normalised confusion matrix as a designed, accessible table —
 * clay intensity = probability, mono labels, hairline cells. Matches the page.
 */
function confusionMatrix(data: ConfusionData): HTMLElement {
  const abbr = (label: string): string => label.slice(0, 3);

  const headRow = el(
    "tr",
    {},
    el("td", { class: "cm-corner" }),
    ...data.labels.map((label) =>
      el("th", { scope: "col", class: "cm-col", title: label }, abbr(label)),
    ),
  );

  const bodyRows = data.matrix.map((row, i) =>
    el(
      "tr",
      {},
      el("th", { scope: "row", class: "cm-row" }, data.labels[i]),
      ...row.map((v, j) => {
        const pct = Math.round(v * 100);
        const diagonal = i === j;
        return el(
          "td",
          {
            class: diagonal ? "cm-cell cm-diag" : "cm-cell",
            // Clay intensity encodes the probability; text only where it reads.
            style: `background:rgba(189,91,51,${v.toFixed(3)});color:${v > 0.5 ? "#fff" : "var(--ink-soft)"}`,
            title: `true ${data.labels[i]} → predicted ${data.labels[j]}: ${pct}%`,
          },
          v >= 0.1 || diagonal ? String(pct) : "",
        );
      }),
    ),
  );

  return el(
    "figure",
    { class: "cm-figure" },
    el(
      "div",
      { class: "table-wrap" },
      el(
        "table",
        { class: "cm", "aria-label": "Colour family confusion matrix, row-normalised" },
        el("thead", {}, headRow),
        el("tbody", {}, ...bodyRows),
      ),
    ),
    el(
      "figcaption",
      { class: "cm-caption" },
      el("span", { class: "cm-axis" }, "rows = true · columns = predicted"),
      el("span", { class: "cm-scale" }, "0%", el("span", { class: "cm-scale-bar" }), "100%"),
    ),
  );
}

function section(eyebrow: string, title: string, ...body: (Node | string)[]): HTMLElement {
  return el(
    "section",
    { class: "doc-section" },
    el("p", { class: "eyebrow" }, eyebrow),
    el("h2", {}, title),
    ...body,
  );
}

function p(...content: (Node | string)[]): HTMLElement {
  return el("p", { class: "doc-p" }, ...content);
}

function table(headers: string[], rows: string[][]): HTMLElement {
  return el(
    "div",
    { class: "table-wrap" },
    el(
      "table",
      { class: "doc-table" },
      el("thead", {}, el("tr", {}, ...headers.map((h) => el("th", {}, h)))),
      el(
        "tbody",
        {},
        ...rows.map((r) => el("tr", {}, ...r.map((c, i) => el(i === 0 ? "th" : "td", {}, c)))),
      ),
    ),
  );
}

function stat(value: string, label: string): HTMLElement {
  return el(
    "div",
    { class: "stat" },
    el("span", { class: "stat-value" }, value),
    el("span", { class: "stat-label" }, label),
  );
}

export function renderDocs(host: HTMLElement): void {
  host.append(
    el(
      "article",
      { class: "docs" },
      el(
        "header",
        { class: "page-head" },
        el("p", { class: "eyebrow" }, "Documentation"),
        el("h1", {}, "How Pyrochrome works"),
        el(
          "p",
          { class: "lede" },
          "Pyrochrome learns the link between a glaze's chemistry and its fired appearance from thousands of real recipes. This page explains the data, the features, the models, what they achieve, and — just as important — where they are bounded.",
        ),
      ),

      el(
        "div",
        { class: "stat-grid" },
        stat("~12,900", "recipes (cleaned)"),
        stat("76.9%", "surface accuracy"),
        stat("79.9%", "colour top-2"),
        stat("ΔE ≈ 33", "colour (Lab) error"),
      ),

      section(
        "The data",
        "Where the knowledge comes from",
        p(
          "Everything is trained on the ",
          el(
            "a",
            {
              href: "https://github.com/derekphilipau/glazy-data",
              target: "_blank",
              rel: "noopener",
            },
            "Glazy",
          ),
          " public dataset — about 12,900 cleaned recipes. Each carries its composition pre-computed as a unity molecular formula (UMF), an Orton cone, and labels for surface, transparency and an RGB colour sampled from a photo.",
        ),
        p(
          "One crucial field — the firing ",
          el("strong", {}, "atmosphere"),
          " (oxidation / reduction / …) — lives only in Glazy's YAML dump, not the flat CSV. We parse it and join it back by recipe id.",
        ),
        p(
          "We keep only real, fired recipes (dropping chemical analyses, raw materials and theoretical entries). The honest caveat starts here: colour labels come from photographs taken under non-standardised lighting, so they are inherently noisy.",
        ),
      ),

      section(
        "Features",
        "How a recipe becomes numbers",
        p("Each recipe is represented by the validated feature set:"),
        el(
          "ul",
          { class: "doc-list" },
          el(
            "li",
            {},
            el("strong", {}, "UMF oxides"),
            " — silica, alumina, the fluxes and the colorant oxides, plus aggregates (R₂O, RO, SiO₂:Al₂O₃ ratio).",
          ),
          el(
            "li",
            {},
            el("strong", {}, "Cone"),
            " — Orton cones are not linear, so they are mapped to an ordinal scale (hotter = larger).",
          ),
          el(
            "li",
            {},
            el("strong", {}, "Atmosphere"),
            " — encoded multi-hot (oxidation, reduction, neutral, salt & soda, wood, raku, luster) plus a “known” flag, since a recipe is often tagged with several.",
          ),
        ),
      ),

      section(
        "The models",
        "What predicts what",
        p(
          "Surface, transparency and colour family are classification targets; colour is additionally modelled as a regression in CIELAB. We compared seven candidates by 5-fold cross-validation (RandomForest, ExtraTrees, HistGradientBoosting, LightGBM, XGBoost, an MLP and logistic regression).",
        ),
        p(
          "The tree ensembles were statistically tied. We selected ",
          el("strong", {}, "HistGradientBoosting"),
          " for the classifiers: it matches LightGBM/XGBoost without a heavy native dependency (pure scikit-learn, reproducible), and its probabilities calibrate better than a RandomForest — which matters because we show a confidence index.",
        ),
        table(
          ["Target", "n", "Accuracy", "Macro-F1", "Top-2"],
          [
            ["Surface", "8,397", "76.9%", "0.65", "91.3%"],
            ["Transparency", "6,886", "66.0%", "0.61", "85.8%"],
            ["Colour family", "5,511", "65.7%", "0.50", "79.9%"],
          ],
        ),
        el(
          "p",
          { class: "fine" },
          "Top-2 colour accuracy (≈80%) is the product-relevant metric: the right family is almost always in the model's top two guesses.",
        ),
        el("p", { class: "eyebrow cm-eyebrow" }, "Where colour goes wrong"),
        p(
          "The confusion matrix reads row by row — of every recipe truly of one family, where does the model place it? A strong diagonal is good; the bright left column is the tell-tale ",
          el("strong", {}, "bias toward “Blanc”"),
          ", an artefact of white photo backgrounds bleeding into the colour labels.",
        ),
        confusionMatrix(COLOUR_CONFUSION),
      ),

      section(
        "Lever 1 — atmosphere",
        "A measured (and surprising) result",
        p(
          "Atmosphere was expected to be the single biggest lever. We added it and measured the gain honestly: it is ",
          el("strong", {}, "negligible in aggregate"),
          " (colour accuracy +0.6%, top-2 +0.3%; surface ~0).",
        ),
        p(
          "Yet the physical effect is unmistakable where it should be. Among copper recipes, reduction yields red (copper-red / sang-de-bœuf) while oxidation yields turquoise and green:",
        ),
        table(
          ["Copper, oxidation", "Copper, reduction"],
          [["Turquoise 22% · Green 19% · Blue 9%", "Red 35% · Violet 14%"]],
        ),
        p(
          "The effect is real but diluted: copper-with-known-atmosphere is only ~13% of the colour set. The conclusion reframed the project — the binding ceiling is not the missing atmosphere, it is colour-label noise.",
        ),
      ),

      section(
        "Lever 2 — colour as Lab + real neighbours",
        "Working around the label-noise ceiling",
        p(
          "Rather than forcing a brittle 10-family label, colour is also modelled as a regression in ",
          el("strong", {}, "CIELAB"),
          ", reporting error as ΔE. The best model (ExtraTrees) reaches ΔE ≈ 33 (R² ≈ 0.40) — a 36% cut versus predicting the mean, but a large residual that directly quantifies the photo-label-noise ceiling.",
        ),
        p(
          "Because a single predicted colour is therefore unreliable, every prediction is shown beside its ",
          el("strong", {}, "nearest real recipes"),
          " — a k-NN search in chemistry space returns real, fireable glazes whose actual fired tiles bracket the likely outcome. That is the honest way to communicate uncertainty.",
        ),
      ),

      section(
        "Limits & honesty",
        "What this does not claim",
        el(
          "ul",
          { class: "doc-list" },
          el(
            "li",
            {},
            "Colour labels are photo-derived and noisy — the dominant source of error. Pyrochrome never shows a single, falsely-precise colour.",
          ),
          el(
            "li",
            {},
            "The atmosphere tag is the set a recipe is associated with, not the exact atmosphere of the photographed sample — weakly informative per sample.",
          ),
          el(
            "li",
            {},
            "Kiln-to-kiln variance is a fundamental source of difference that no composition model can remove.",
          ),
          el(
            "li",
            {},
            "Rare colour families (< 40 samples) are dropped, so they are not predicted.",
          ),
        ),
        p("This is decision-support to reduce test firings — not a replacement for them."),
      ),

      section(
        "Reproduce & attribution",
        "Everything is reproducible",
        p("From a clone, with fixed seeds and a pinned environment:"),
        el(
          "pre",
          { class: "code" },
          el(
            "code",
            {},
            "make setup\nmake data        # clone Glazy\nmake atmosphere  # parse the YAML, cache the join\nmake train       # surface / transparency / colour\nmake color       # Lab colour regression\nmake neighbors   # nearest-recipe index",
          ),
        ),
        p(
          "Data: Glazy (CC BY-NC-SA 4.0 — attribution, non-commercial, share-alike). Methodology reference: ",
          el(
            "a",
            { href: "https://arxiv.org/abs/2605.06641", target: "_blank", rel: "noopener" },
            "GlazyBench",
          ),
          ". Pyrochrome is released under the same licence.",
        ),
      ),
    ),
  );
}
