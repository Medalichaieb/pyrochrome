/**
 * About page: the personal context behind the project and where to find the
 * code. Contact is intentionally via GitHub only. Plain, editorial, matches the
 * rest of the site.
 */
import { el } from "../dom";

const REPO_URL = "https://github.com/Medalichaieb/pyrochrome";

function section(eyebrow: string, ...body: (Node | string)[]): HTMLElement {
  return el("section", { class: "doc-section" }, el("p", { class: "eyebrow" }, eyebrow), ...body);
}

function p(...content: (Node | string)[]): HTMLElement {
  return el("p", { class: "doc-p" }, ...content);
}

export function renderAbout(host: HTMLElement): void {
  host.append(
    el(
      "article",
      { class: "docs" },
      el(
        "header",
        { class: "page-head" },
        el("p", { class: "eyebrow" }, "About"),
        el("h1", {}, "About this project"),
        el(
          "p",
          { class: "lede" },
          "Pyrochrome is a personal project, built to scratch a real itch from my own studio.",
        ),
      ),

      section(
        "The story",
        p(
          "I make ceramics as a hobby, and glazing always got to me: the same recipe can fire to completely different results depending on the temperature and the kiln atmosphere, so you end up firing test after test and hoping. I wanted to see whether a bit of data and machine learning could take some of the guesswork out of it.",
        ),
        p(
          "So I built Pyrochrome from start to finish, from the data pipeline to this site. It predicts how a glaze will look once fired from its chemistry, suggests recipes for a colour you have in mind, and tries to stay honest about its own uncertainty. The Docs tab explains how it works and where it is bounded.",
        ),
      ),

      section(
        "Code & contact",
        p(
          "Pyrochrome is open source and non-commercial. The whole thing, from the data pipeline and models to the evaluation and this site, lives on GitHub. Browse the code, open an issue, or get in touch there.",
        ),
        el(
          "p",
          { class: "doc-p" },
          el(
            "a",
            { href: REPO_URL, target: "_blank", rel: "noopener" },
            "github.com/Medalichaieb/pyrochrome",
          ),
        ),
      ),
    ),
  );
}
