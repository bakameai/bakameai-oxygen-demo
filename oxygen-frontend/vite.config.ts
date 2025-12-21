import path from "path"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  build: {
    rollupOptions: {
      input: {
        customer: path.resolve(__dirname, 'index.html'),
        worker: path.resolve(__dirname, 'worker.html'),
      },
    },
  },
})

