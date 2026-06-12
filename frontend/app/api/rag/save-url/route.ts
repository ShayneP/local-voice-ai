import { NextResponse } from 'next/server';
import { auth } from '@/auth';

const FASTAPI_URL = process.env.FASTAPI_URL ?? 'http://localhost:8900';

export const revalidate = 0;

export async function POST(req: Request) {
  try {
    // サーバー側でKeycloak access_tokenを取得（ブラウザに露出しない）
    const session = await auth();
    const token = (session as any)?.access_token;

    if (!token) {
      return NextResponse.json(
        { error: 'Not authenticated' },
        { status: 401 }
      );
    }

    const body = await req.json();
    const { url } = body;

    if (!url || typeof url !== 'string') {
      return NextResponse.json(
        { error: 'url is required' },
        { status: 400 }
      );
    }

    // FastAPIへ転送（サーバー→サーバー・CORS不要）
    const res = await fetch(`${FASTAPI_URL}/rag/save-url`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ url }),
    });

    const data = await res.json();
    return NextResponse.json(data, { status: res.status });

  } catch (error) {
    console.error('[/api/rag/save-url]', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
