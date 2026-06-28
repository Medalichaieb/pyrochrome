/**
 * Minimal hash router (no framework — keeps the SPA small and legible).
 *
 * A route is a `#/path` mapped to a mount function that renders into the host
 * element. Navigation updates `location.hash`; the active link is reflected via
 * an `is-active` class on elements with a matching `data-route`.
 */

export type View = (host: HTMLElement) => void;

export interface Route {
  path: string;
  view: View;
}

export function startRouter(host: HTMLElement, routes: Route[], fallback: string): void {
  const render = () => {
    const path = location.hash.replace(/^#/, "") || fallback;
    const route = routes.find((r) => r.path === path) ?? routes.find((r) => r.path === fallback);
    host.replaceChildren();
    route?.view(host);
    document.querySelectorAll<HTMLElement>("[data-route]").forEach((el) => {
      el.classList.toggle(
        "is-active",
        `#${el.dataset.route}` === (location.hash || `#${fallback}`),
      );
    });
    host.focus({ preventScroll: true });
    window.scrollTo({ top: 0 });
  };
  window.addEventListener("hashchange", render);
  render();
}
