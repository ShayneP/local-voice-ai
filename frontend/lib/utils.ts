import { type ClassValue, clsx } from 'clsx';
import { TokenSource } from 'livekit-client';
import { twMerge } from 'tailwind-merge';
import type { AppConfig } from '@/app-config';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Get styles for the app
 * @param appConfig - The app configuration
 * @returns A string of styles
 */
export function getStyles(appConfig: AppConfig) {
  const { accent, accentDark } = appConfig;

  return [
    accent
      ? `:root { --primary: ${accent}; --primary-hover: color-mix(in srgb, ${accent} 80%, #000); }`
      : '',
    accentDark
      ? `.dark { --primary: ${accentDark}; --primary-hover: color-mix(in srgb, ${accentDark} 80%, #000); }`
      : '',
  ]
    .filter(Boolean)
    .join('\n');
}

export function getBackendUrl(path: string) {
  const configured = process.env.NEXT_PUBLIC_BACKEND_URL;
  const base = configured ? `${configured.replace(/\/$/, '')}/` : `${window.location.origin}/`;
  return new URL(path.replace(/^\//, ''), base);
}

/**
 * Get a token source for a sandboxed LiveKit session
 * @param appConfig - The app configuration
 * @returns A token source for a sandboxed LiveKit session
 */
export function getSandboxTokenSource(appConfig: AppConfig) {
  return TokenSource.custom(async () => {
    const configuredEndpoint = process.env.NEXT_PUBLIC_CONN_DETAILS_ENDPOINT;
    const url = configuredEndpoint
      ? new URL(configuredEndpoint, window.location.origin)
      : getBackendUrl('/api/connection-details');
    const sandboxId = appConfig.sandboxId ?? '';
    const roomConfig = appConfig.agentName
      ? {
          agents: [{ agent_name: appConfig.agentName }],
        }
      : undefined;

    try {
      const res = await fetch(url.toString(), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Sandbox-Id': sandboxId,
        },
        body: JSON.stringify({
          room_config: roomConfig,
        }),
      });
      return await res.json();
    } catch (error) {
      console.error('Error fetching connection details:', error);
      throw new Error('Error fetching connection details!');
    }
  });
}
