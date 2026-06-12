'use client';

import { useEffect, useRef, useState } from 'react';
import { useSession } from 'next-auth/react';
import Link from 'next/link';

type Document = {
  id: string;
  filename: string;
  file_size: number;
  mime_type: string;
  created_at: string;
};

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export default function RagPage() {
  const { data: session } = useSession();
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!session) return;
    fetchDocuments();
  }, [session]);

  async function fetchDocuments() {
    setLoading(true);
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/rag/documents`, {
        headers: {
          Authorization: `Bearer ${(session as any)?.access_token ?? ''}`,
        },
      });
      if (!res.ok) throw new Error(`エラー: ${res.status}`);
      const data = await res.json();
      setDocuments(data.documents);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function uploadFile(file: File) {
    setUploading(true);
    setError('');
    setMessage('');
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/rag/upload`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${(session as any)?.access_token ?? ''}`,
        },
        body: formData,
      });
      const data = await res.json();
      if (data.status === 'ok') {
        setMessage(`完了: 「${data.filename}」を登録しました（${data.chunks}チャンク）`);
        fetchDocuments();
      } else if (data.status === 'skipped') {
        setMessage(`スキップ: 「${data.existing_filename}」と重複しています`);
      } else {
        throw new Error(JSON.stringify(data));
      }
    } catch (e: any) {
      setError(e.message);
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  }

  async function deleteDocument(id: string, filename: string) {
    if (!confirm(`「${filename}」を削除しますか？`)) return;
    try {
      await fetch(`${process.env.NEXT_PUBLIC_API_URL}/rag/documents/${id}`, {
        method: 'DELETE',
        headers: {
          Authorization: `Bearer ${(session as any)?.access_token ?? ''}`,
        },
      });
      setDocuments((prev) => prev.filter((d) => d.id !== id));
      setMessage(`削除しました: 「${filename}」`);
    } catch (e: any) {
      setError(e.message);
    }
  }

  return (
    <div className="min-h-screen bg-background p-6 pt-20">
      <div className="mx-auto max-w-3xl">
        <div className="mb-6 flex items-center justify-between">
          <h1 className="font-mono text-xl font-bold">RAG管理</h1>
          <div className="flex gap-3">
            <Link href="/history" className="font-mono text-xs underline underline-offset-4">
              会話履歴
            </Link>
            <Link href="/" className="font-mono text-xs underline underline-offset-4">
              音声通話
            </Link>
          </div>
        </div>

        <div className="mb-6 rounded-lg border-2 border-dashed border-border p-6 text-center">
          <p className="mb-3 font-mono text-sm text-muted-foreground">
            PDF・Word・テキストファイルをアップロード
          </p>
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.docx,.txt,.md"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) uploadFile(file);
            }}
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            className="rounded border border-border px-4 py-2 font-mono text-xs hover:bg-muted disabled:opacity-50"
          >
            {uploading ? 'アップロード中...' : 'ファイルを選択'}
          </button>
        </div>

        {error && (
          <div className="mb-4 rounded border border-red-500 p-3 font-mono text-xs text-red-700">
            {error}
          </div>
        )}
        {message && (
          <div className="mb-4 rounded border border-green-500 p-3 font-mono text-xs text-green-700">
            {message}
          </div>
        )}

        <h2 className="mb-3 font-mono text-sm font-bold">登録済みドキュメント</h2>
        {loading ? (
          <p className="font-mono text-xs text-muted-foreground">読み込み中...</p>
        ) : documents.length === 0 ? (
          <p className="font-mono text-xs text-muted-foreground">登録済みドキュメントがありません</p>
        ) : (
          <div className="space-y-2">
            {documents.map((doc) => (
              <div key={doc.id} className="flex items-center justify-between rounded-lg border border-border p-3">
                <div>
                  <p className="font-mono text-sm font-bold">{doc.filename}</p>
                  <p className="font-mono text-xs text-muted-foreground">
                    {formatBytes(doc.file_size)} · {new Date(doc.created_at).toLocaleString('ja-JP')}
                  </p>
                </div>
                <button
                  onClick={() => deleteDocument(doc.id, doc.filename)}
                  className="font-mono text-xs text-red-500 underline underline-offset-4"
                >
                  削除
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
