/**
 * Pyrochrome web entry point: build the chrome (nav + footer), mount the router,
 * and render the Predict / Docs views into the main region.
 */
import "./style.css";
import { el } from "./dom";
import { startRouter } from "./router";
import { renderPredict } from "./views/predict";
import { renderDocs } from "./views/docs";

function navLink(route: string, label: string): HTMLElement {
  return el("a", { href: `#${route}`, class: "nav-link", "data-route": route }, label);
}

function mount(): void {
  const root = document.querySelector<HTMLElement>("#app");
  if (!root) return;

  const main = el("main", { id: "view", class: "view", tabindex: "-1" });

  const header = el(
    "header",
    { class: "site-head" },
    el(
      "div",
      { class: "site-head-inner" },
      el(
        "a",
        { href: "#/predict", class: "wordmark" },
        el("span", { class: "wordmark-name" }, "Pyrochrome"),
        el("span", { class: "wordmark-tag" }, "Predict the glaze before the kiln"),
      ),
      el(
        "nav",
        { class: "nav", "aria-label": "Primary" },
        navLink("/predict", "Predict"),
        navLink("/docs", "Docs"),
      ),
    ),
  );

  const footer = el(
    "footer",
    { class: "site-foot" },
    el(
      "div",
      { class: "site-foot-inner" },
      el(
        "p",
        {},
        "Data: ",
        el(
          "a",
          {
            href: "https://github.com/derekphilipau/glazy-data",
            target: "_blank",
            rel: "noopener",
          },
          "Glazy",
        ),
        " (CC BY-NC-SA 4.0) · Reference: ",
        el(
          "a",
          { href: "https://arxiv.org/abs/2605.06641", target: "_blank", rel: "noopener" },
          "GlazyBench",
        ),
        " · Open source, non-commercial.",
      ),
    ),
  );

  root.replaceChildren(header, main, footer);

  startRouter(
    main,
    [
      { path: "/predict", view: renderPredict },
      { path: "/docs", view: renderDocs },
    ],
    "/predict",
  );
}

mount();
