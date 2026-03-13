FROM node:20-slim AS base
WORKDIR /app

FROM base AS deps
RUN corepack enable
COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

FROM base AS build
RUN corepack enable
COPY --from=deps /app/node_modules ./node_modules
COPY . .
# Build-time env vars (NEXT_PUBLIC_* is inlined at build time)
ENV NEXT_PUBLIC_LIVEKIT_URL=ws://localhost:7880
RUN pnpm run build

FROM base AS runner
ENV NODE_ENV=production
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs
COPY --from=build /app/public ./public
COPY --from=build --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=build --chown=nextjs:nodejs /app/.next/static ./.next/static
USER nextjs
EXPOSE 3000
CMD ["node", "server.js"]
