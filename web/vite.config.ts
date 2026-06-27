import { defineConfig } from "vite";

// Static SPA, deployable to any static host. The procedural renderer and the
// compact model run entirely client-side.
export default defineConfig({
  root: ".",
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
