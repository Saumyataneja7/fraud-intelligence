# Frontend Production Build

## Purpose

Phase 12.6 prepares the React + TypeScript frontend for production
deployment.

The frontend is built with Vite and produces a static `dist/` directory.

## Environment configuration

The frontend API endpoint is configured through:

```text
VITE_API_BASE_URL

Example:
VITE_API_BASE_URL=http://127.0.0.1:8000

.env.local is used for local development and is ignored by Git.

.env.example documents the required environment variable without exposing
deployment-specific configuration.

Vite environment variables are resolved at build time.

### Production build

From the frontend/ directory:
npm install
npm run build

The build performs:
1. TypeScript project compilation
2. Vite production bundling
3. Static asset generation

The generated output is:
frontend/dist/

### Local production preview

To test the generated production bundle:
npm run preview -- --host 127.0.0.1

The Vite preview server serves the contents of dist/.

Example validation:
curl -I http://127.0.0.1:4173/

Expected response:
HTTP/1.1 200 OK

### Current build validation

The Phase 12.6 production build was validated successfully.

Generated output:
dist/index.html
dist/favicon.svg
dist/icons.svg
dist/assets/index-*.js
dist/assets/index-*.css

The generated dist/ directory is approximately 388 KB and is not committed
to Git.

### API deployment relationship

The frontend is a static application and communicates with the Fraud
Intelligence FastAPI backend through VITE_API_BASE_URL.

For local development:
React/Vite frontend
        ↓
http://127.0.0.1:8000
        ↓
FastAPI

For production, the deployment environment must provide the appropriate
backend URL before the frontend production build is generated.

CI/CD and automated deployment are intentionally handled separately in
Phase 12.7.