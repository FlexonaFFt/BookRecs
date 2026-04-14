import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const apiProxyTarget = process.env.VITE_API_PROXY_TARGET || 'http://localhost:8000';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
    proxy: {
      '/recommendations': {
        target: apiProxyTarget,
        changeOrigin: true,
      },
      '/demo': {
        target: apiProxyTarget,
        changeOrigin: true,
      },
      '/items': {
        target: apiProxyTarget,
        changeOrigin: true,
      },
      '/interactions': {
        target: apiProxyTarget,
        changeOrigin: true,
      },
      '/admin': {
        target: apiProxyTarget,
        changeOrigin: true,
      },
      '/healthz': {
        target: apiProxyTarget,
        changeOrigin: true,
      },
      '/readyz': {
        target: apiProxyTarget,
        changeOrigin: true,
      },
    },
  },
});
