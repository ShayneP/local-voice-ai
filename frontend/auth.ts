import NextAuth from 'next-auth';
import Keycloak from 'next-auth/providers/keycloak';

/** アクセストークンをKeycloakのrefresh_token grantで更新する */
async function refreshAccessToken(token: any) {
  try {
    const issuer = process.env.AUTH_KEYCLOAK_ISSUER!;
    const tokenEndpoint = `${issuer}/protocol/openid-connect/token`;

    const response = await fetch(tokenEndpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        grant_type: 'refresh_token',
        client_id: process.env.AUTH_KEYCLOAK_ID!,
        client_secret: process.env.AUTH_KEYCLOAK_SECRET!,
        refresh_token: token.refresh_token,
      }),
    });

    const refreshed = await response.json();

    if (!response.ok) {
      throw refreshed;
    }

    return {
      ...token,
      access_token: refreshed.access_token,
      refresh_token: refreshed.refresh_token ?? token.refresh_token,
      expires_at: Math.floor(Date.now() / 1000) + refreshed.expires_in,
      error: undefined,
    };
  } catch (error) {
    console.error('[auth] refreshAccessToken failed:', error);
    return { ...token, error: 'RefreshAccessTokenError' as const };
  }
}

export const { handlers, auth, signIn, signOut } = NextAuth({
  providers: [
    Keycloak({
      clientId: process.env.AUTH_KEYCLOAK_ID!,
      clientSecret: process.env.AUTH_KEYCLOAK_SECRET!,
      issuer: process.env.AUTH_KEYCLOAK_ISSUER!,
    }),
  ],
  callbacks: {
    async jwt({ token, account }) {
      // 初回ログイン時: Keycloakから渡された値を保存
      if (account) {
        return {
          ...token,
          access_token: account.access_token,
          refresh_token: account.refresh_token,
          expires_at: account.expires_at, // UNIX秒
        };
      }

      // expires_at まで余裕があればそのまま返す（30秒前にリフレッシュ）
      const expiresAt = token.expires_at as number | undefined;
      if (expiresAt && Date.now() / 1000 < expiresAt - 30) {
        return token;
      }

      // 期限切れ or 期限不明 → リフレッシュ
      return refreshAccessToken(token);
    },

    async session({ session, token }) {
      (session as any).access_token = token.access_token;
      (session as any).expires_at = token.expires_at;
      (session as any).error = token.error;

      // entity_id (sub) を取得
      if (token.access_token) {
        try {
          const payload = JSON.parse(
            Buffer.from((token.access_token as string).split('.')[1], 'base64').toString()
          );
          session.user.id = payload.sub ?? token.sub ?? 'anonymous';
        } catch {
          session.user.id = token.sub ?? 'anonymous';
        }
      } else if (token.sub) {
        session.user.id = token.sub;
      }

      return session;
    },
  },
});
