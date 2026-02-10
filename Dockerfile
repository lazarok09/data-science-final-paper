# Build stage – install deps and build the Next.js app
FROM oven/bun:1-alpine AS builder

WORKDIR /app

# Copy package files and install dependencies
COPY web/package.json ./
RUN bun install

# Copy the rest of the frontend and build
COPY web/ .
RUN bun run build

# Production stage – minimal image to run the app
FROM oven/bun:1-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISCRETION=1

# Copy built output and runtime files
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./
COPY --from=builder /app/public ./public

EXPOSE 3000

CMD ["bun", "run", "start"]
