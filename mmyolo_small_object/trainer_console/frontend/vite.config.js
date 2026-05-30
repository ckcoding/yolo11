import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// We build into trainer_console/static/dist so FastAPI can serve it
export default defineConfig({
  plugins: [vue()],
  base: '/',
  build: {
    outDir: '../static/dist',
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:18080',
        changeOrigin: true,
      },
    },
  },
})
