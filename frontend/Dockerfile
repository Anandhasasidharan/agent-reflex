# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Builder: compile the Vite/TypeScript app.
# ---------------------------------------------------------------------------
FROM node:22-alpine AS builder

WORKDIR /build

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# ---------------------------------------------------------------------------
# Runtime: static nginx with SPA fallback, no node tooling.
# ---------------------------------------------------------------------------
FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /build/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["wget", "-qO-", "http://localhost/"]
