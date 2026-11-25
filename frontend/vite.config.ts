import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => ({
  plugins: [react()],
  server: {
    proxy: {
      "/api": mode === "demo" ? "http://127.0.0.1:8004" : "http://127.0.0.1:8000",
    },
  },
}));
