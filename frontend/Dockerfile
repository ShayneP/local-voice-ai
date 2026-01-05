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
ENV LIVEKIT_URL=http://livekit:7880 \
    LIVEKIT_API_KEY=devkey \
    LIVEKIT_API_SECRET=secret \
    NEXT_PUBLIC_LIVEKIT_URL=http://localhost:7880
RUN pnpm run build

FROM base AS runner
ENV NODE_ENV=production
ENV LIVEKIT_URL=http://livekit:7880 \
    LIVEKIT_API_KEY=devkey \
    LIVEKIT_API_SECRET=secret \
    NEXT_PUBLIC_LIVEKIT_URL=http://localhost:7880
COPY --from=build /app ./
EXPOSE 3000
CMD ["node", "node_modules/next/dist/bin/next", "start", "-H", "0.0.0.0", "-p", "3000"]
