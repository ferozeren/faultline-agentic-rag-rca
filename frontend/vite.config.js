import { defineConfig } from "vitest/config";
import { loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  return {
    base: "./",
    plugins: [react()],
    server: {
      host: "0.0.0.0",
      port: 5173,
      strictPort: true,
      proxy: {
        "/api": {
          target: env.VITE_API_PROXY_TARGET || "http://127.0.0.1:8000",
          changeOrigin: true,
        },
      },
    },
    test: {
      environment: "jsdom",
      environmentOptions: { jsdom: { url: "http://localhost" } },
      setupFiles: ["./src/test/setup.js"],
      restoreMocks: true,
    },
    build: {
      rollupOptions: {
        output: {
          manualChunks(id) {
            if (id.includes("/node_modules/recharts/") || id.includes("/node_modules/d3-") || id.includes("/node_modules/victory-vendor/")) {
              return "charts";
            }
            if (/\/node_modules\/(react|react-dom|react-router|react-router-dom|scheduler)\//.test(id)) {
              return "framework";
            }
            if (id.includes("/node_modules/axios/")) return "network";
          },
        },
      },
    },
  };
});