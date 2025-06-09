import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import wasmPack from 'vite-plugin-wasm-pack'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    wasmPack("../noodles-wasm")
  ],
  base: "/"
})
