'use client';

import { useSession, signIn } from 'next-auth/react';
import { useEffect } from 'react';

/**
 * RefreshAccessTokenError が発生した場合に自動で再サインインを促す。
 * app/(app)/layout.tsx に組み込む。
 */
export function SessionErrorHandler() {
  const { data: session } = useSession();

  useEffect(() => {
    if ((session as any)?.error === 'RefreshAccessTokenError') {
      // リフレッシュ失敗 → 再ログイン
      signIn('keycloak');
    }
  }, [session]);

  return null;
}
