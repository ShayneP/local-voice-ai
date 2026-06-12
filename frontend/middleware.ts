import { auth } from '@/auth';
import { NextResponse } from 'next/server';

export default auth((req) => {
  // 未ログインの場合はサインインページにリダイレクト
  if (!req.auth) {
    const signInUrl = new URL('/api/auth/signin', req.url);
    signInUrl.searchParams.set('callbackUrl', req.url);
    return NextResponse.redirect(signInUrl);
  }
});

export const config = {
  // 認証が必要なパス（APIルートと静的ファイルは除外）
  matcher: [
    '/((?!api/auth|_next/static|_next/image|favicon.ico).*)',
  ],
};
