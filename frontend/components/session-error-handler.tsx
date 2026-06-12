'use client';

import { useSession, signIn } from 'next-auth/react';
import { useEffect } from 'react';

/**
 * - トークン期限60秒前に自動リフレッシュ（先読み更新）
 * - RefreshAccessTokenError 発生時は再サインイン
 * app/(app)/layout.tsx に組み込む。
 */
export function SessionErrorHandler() {
  const { data: session, update } = useSession();

  // エラー時: 再ログイン
  useEffect(() => {
    if ((session as any)?.error === 'RefreshAccessTokenError') {
      signIn('keycloak');
    }
  }, [session]);

  // 先読みリフレッシュ: 期限60秒前に update() を呼ぶ
  useEffect(() => {
    const expiresAt = (session as any)?.expires_at as number | undefined;
    if (!expiresAt) return;

    const msUntilExpiry = expiresAt * 1000 - Date.now();
    const refreshAt = msUntilExpiry - 60_000;

    if (refreshAt <= 0) {
      update();
      return;
    }

    const timer = setTimeout(() => update(), refreshAt);
    return () => clearTimeout(timer);
  }, [(session as any)?.expires_at]);

  return null;
}
