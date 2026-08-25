import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
// Trivial change to force a fresh Cloudflare Workers Build after the Root
// Directory setting was corrected, verifying the build actually re-runs.
export default defineConfig({
  plugins: [react()],
})
