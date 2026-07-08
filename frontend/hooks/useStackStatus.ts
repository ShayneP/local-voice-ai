'use client';

import { useEffect, useState } from 'react';

export interface StackChild {
  name: string;
  ready: boolean;
  running: boolean;
  restarts: number;
}

export interface StackStatus {
  ready: boolean;
  children: StackChild[];
}

const POLL_INTERVAL_MS = 2000;

/**
 * Polls /api/status while the backend supervisor is still starting its
 * children — the first boot downloads model weights, which can take a long
 * time, and without this the app is just a start button that doesn't work.
 *
 * Fails open: if the endpoint is missing or unreachable (e.g. `pnpm dev`
 * without the Python backend), the stack is treated as ready.
 */
export function useStackStatus(): StackStatus {
  const [status, setStatus] = useState<StackStatus>({ ready: false, children: [] });

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    async function poll() {
      let next: StackStatus;
      try {
        const res = await fetch('/api/status', { cache: 'no-store' });
        if (!res.ok) throw new Error(`status ${res.status}`);
        next = await res.json();
      } catch {
        next = { ready: true, children: [] }; // fail open
      }
      if (cancelled) return;
      setStatus(next);
      if (!next.ready) {
        timer = setTimeout(poll, POLL_INTERVAL_MS);
      }
    }

    poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, []);

  return status;
}
