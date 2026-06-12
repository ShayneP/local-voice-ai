'use client';

import { useEffect, useState } from 'react';
import { useSession } from 'next-auth/react';
import Link from 'next/link';

type SessionItem = {
  id: string;
  title: string;
  summary: string | null;
  started_at: string;
  ended_at: string | null;
  last_message_at: string;
  message_count: number;
};

export default function HistoryPage() {
  const { data: session } = useSession();
  const [sessions, setSessions] = useState<SessionItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [confirmId, setConfirmId] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    if (!session) return;
    fetchSessions();
  }, [session]);

  async function fetchSessions() {
    setLoading(true);
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/history/sessions`, {
        headers: {
          Authorization: `Bearer ${(session as any)?.access_token ?? ''}`,
        },
      });
      if (!res.ok) throw new Error(`エラー: ${res.status}`);
      const data = await res.json();
      setSessions(data.sessions);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function deleteSession(sessionId: string) {
    setDeleting(true);
    try {
      const res = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/history/sessions/${sessionId}`,
        {
          method: 'DELETE',
          headers: {
            Authorization: `Bearer ${(session as any)?.access_token ?? ''}`,
          },
        }
      );
      if (!res.ok) throw new Error(`削除エラー: ${res.status}`);
      setSessions((prev) => prev.filter((s) => s.id !== sessionId));
    } catch (e: any) {
      setError(e.message);
    } finally {
      setDeleting(false);
      setConfirmId(null);
    }
  }

  function formatDate(iso: string) {
    const d = new Date(iso);
    const now = new Date();
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const target = new Date(d.getFullYear(), d.getMonth(), d.getDate());
    const diffDays = Math.floor((today.getTime() - target.getTime()) / 86400000);

    const time = d.toLocaleTimeString('ja-JP', { hour: '2-digit', minute: '2-digit' });
    if (diffDays === 0) return `今日 ${time}`;
    if (diffDays === 1) return `昨日 ${time}`;
    return d.toLocaleDateString('ja-JP', { month: 'short', day: 'numeric' }) + ` ${time}`;
  }

  const confirmTarget = sessions.find((s) => s.id === confirmId);

  return (
    <div className="min-h-screen bg-background p-6 pt-20">
      <div className="mx-auto max-w-3xl">
        <div className="mb-6 flex items-center justify-between">
          <h1 className="font-mono text-xl font-bold">会話履歴</h1>
          <div className="flex gap-3">
            <Link
              href="/"
              className="font-mono text-xs underline underline-offset-4"
            >
              ← 音声通話
            </Link>
            <Link
              href="/rag"
              className="font-mono text-xs underline underline-offset-4"
            >
              RAG管理 →
            </Link>
          </div>
        </div>

        {error && (
          <div className="mb-4 rounded border border-red-500 bg-red-50 p-3 font-mono text-xs text-red-700">
            {error}
          </div>
        )}

        {loading ? (
          <p className="font-mono text-xs text-muted-foreground">読み込み中...</p>
        ) : sessions.length === 0 ? (
          <p className="font-mono text-xs text-muted-foreground">会話履歴がありません</p>
        ) : (
          <div className="space-y-2">
            {sessions.map((s) => (
              <div key={s.id} className="flex items-stretch gap-2">
                <Link
                  href={`/history/${s.id}`}
                  className="block flex-1 rounded-lg border border-border p-4 transition-colors hover:bg-muted"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-sm font-bold">
                      {s.title}
                    </span>
                    <span className="font-mono text-xs text-muted-foreground">
                      {formatDate(s.last_message_at)}
                    </span>
                  </div>
                  <div className="mt-1 flex items-center gap-3">
                    <span className="font-mono text-xs text-muted-foreground">
                      💬 {s.message_count}件
                    </span>
                    {s.ended_at ? (
                      <span className="font-mono text-xs text-muted-foreground">完了</span>
                    ) : (
                      <span className="font-mono text-xs text-orange-500">進行中</span>
                    )}
                  </div>
                </Link>
                <button
                  onClick={() => setConfirmId(s.id)}
                  className="flex items-center justify-center rounded-lg border border-border px-3 text-muted-foreground transition-colors hover:border-red-400 hover:bg-red-50 hover:text-red-500"
                  aria-label="削除"
                >
                  🗑️
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* 確認ダイアログ */}
      {confirmId && confirmTarget && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="mx-4 w-full max-w-sm rounded-xl border border-border bg-background p-6 shadow-lg">
            <h2 className="font-mono text-sm font-bold">履歴を削除しますか？</h2>
            <p className="mt-2 font-mono text-xs text-muted-foreground">
              「{confirmTarget.title}」を削除します。この操作は元に戻せません。
            </p>
            <div className="mt-5 flex gap-3">
              <button
                onClick={() => setConfirmId(null)}
                disabled={deleting}
                className="flex-1 rounded-lg border border-border py-2 font-mono text-xs transition-colors hover:bg-muted disabled:opacity-50"
              >
                キャンセル
              </button>
              <button
                onClick={() => deleteSession(confirmId)}
                disabled={deleting}
                className="flex-1 rounded-lg bg-red-500 py-2 font-mono text-xs text-white transition-colors hover:bg-red-600 disabled:opacity-50"
              >
                {deleting ? '削除中...' : '削除する'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
