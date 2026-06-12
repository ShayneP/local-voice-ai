'use client';

import { useEffect, useState } from 'react';
import { useSession } from 'next-auth/react';
import { useParams } from 'next/navigation';
import Link from 'next/link';

type Message = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  created_at: string;
};

type SessionMeta = {
  id: string;
  title: string;
  summary: string | null;
  started_at: string;
  ended_at: string | null;
  message_count: number;
};

export default function SessionDetailPage() {
  const { data: session } = useSession();
  const params = useParams();
  const sessionId = params.id as string;

  const [meta, setMeta] = useState<SessionMeta | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!session || !sessionId) return;
    fetchDetail();
  }, [session, sessionId]);

  async function fetchDetail() {
    setLoading(true);
    try {
      const res = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/history/sessions/${sessionId}`,
        {
          headers: {
            Authorization: `Bearer ${(session as any)?.access_token ?? ''}`,
          },
        }
      );
      if (!res.ok) throw new Error(`エラー: ${res.status}`);
      const data = await res.json();
      setMeta(data.session);
      setMessages(data.messages);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-background p-6 pt-20">
      <div className="mx-auto max-w-3xl">
        <div className="mb-6 flex items-center justify-between">
          <div>
            <Link
              href="/history"
              className="font-mono text-xs underline underline-offset-4"
            >
              ← 一覧に戻る
            </Link>
            {meta && (
              <h1 className="mt-2 font-mono text-xl font-bold">{meta.title}</h1>
            )}
          </div>
          {meta && (
            <div className="text-right">
              <p className="font-mono text-xs text-muted-foreground">
                {new Date(meta.started_at).toLocaleString('ja-JP')}
              </p>
              <p className="font-mono text-xs text-muted-foreground">
                💬 {meta.message_count}件
              </p>
            </div>
          )}
        </div>

        {error && (
          <div className="mb-4 rounded border border-red-500 bg-red-50 p-3 font-mono text-xs text-red-700">
            {error}
          </div>
        )}

        {loading ? (
          <p className="font-mono text-xs text-muted-foreground">読み込み中...</p>
        ) : messages.length === 0 ? (
          <p className="font-mono text-xs text-muted-foreground">メッセージがありません</p>
        ) : (
          <div className="space-y-3">
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`rounded-lg border p-3 ${
                  msg.role === 'user'
                    ? 'border-border bg-muted'
                    : 'border-border bg-background'
                }`}
              >
                <div className="mb-1 flex items-center justify-between">
                  <span className="font-mono text-xs font-bold">
                    {msg.role === 'user' ? '👤 ユーザー' : '🤖 AI'}
                  </span>
                  <span className="font-mono text-xs text-muted-foreground">
                    {new Date(msg.created_at).toLocaleTimeString('ja-JP', {
                      hour: '2-digit',
                      minute: '2-digit',
                      second: '2-digit',
                    })}
                  </span>
                </div>
                <p className="font-mono text-sm">{msg.content}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
