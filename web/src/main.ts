/**
 * Pyrochrome web entry point.
 *
 * The app mounts into #app. For v1 it will host: the composition/cone/atmosphere
 * input form, the prediction readout (colour ± confidence, surface, transparency),
 * the procedural tile renderer (ported from prototypes/glaze_renderer.html), and
 * the nearest real recipes.
 *
 * This is the scaffold — the form, renderer and model wiring are TODO.
 */
import "./style.css";

const app = document.querySelector<HTMLElement>("#app");

if (app) {
  app.innerHTML = `
    <p style="color: var(--ash); font-family: var(--mono); font-size: 13px;">
      Squelette du frontend — à venir : formulaire de composition, prédiction +
      confiance, rendu procédural et recettes réelles proches.
    </p>
  `;
}
