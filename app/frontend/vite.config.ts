import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The frontend never talks to Python directly. In development it
// proxies /api to the FastAPI backend; in production VITE_API_BASE
// points at the deployed API origin.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_API_TARGET || "http://127.0.0.1:8077",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
