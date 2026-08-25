import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
// Trivial change to force a fresh Cloudflare Workers Build after the Root
// Directory setting was corrected, verifying the build actually re-runs.
export default defineConfig({
  // VITE_API_BASE_URL is read here at build time (see src/api.js) --
  // forcing a fresh build after it was added to Cloudflare's Workers
  // Build project settings, since it's a compile-time substitution.
  plugins: [react()],
})
