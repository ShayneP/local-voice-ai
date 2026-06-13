'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import type { Room } from 'livekit-client';
import { useTranscriptions } from '@livekit/components-react';

export interface HelixMessage {
  id: string;
  role: 'user' | 'assistant';
  text: string; // raw — 一切加工しない
  ts: number;
  part?: number; // 分割送信時のパート番号
  parts?: number; // 総パート数
}

const TOPIC = 'helix.chat';
// interim(発話中テキスト)はデフォルト非表示。URLに ?debug=1 を付けた場合のみ表示。
const SHOW_INTERIM = () =>
  typeof window !== 'undefined' &&
  new URLSearchParams(window.location.search).get('debug') === '1';

/**
 * 統一メッセージストリーム(helix.chat)の購読。
 * agent側がuser/assistant両方を確定テキストとしてJSON送信するため、
 * frontendはマージ・置換なしのappend-onlyで保持する。
 *
 * interimText: lk.transcription のローカル参加者セグメント（表示専用レイヤ）。
 * helix.chat でuser確定メッセージが届いたらクリアされる。履歴には入らない。
 */
export function useHelixMessages(room?: Room): {
  messages: HelixMessage[];
  interimText: string | null;
} {
  const [messages, setMessages] = useState<HelixMessage[]>([]);
  const [interimText, setInterimText] = useState<string | null>(null);
  // user確定時に表示中だったセグメントIDを記録し、同一セグメントの再表示を抑止
  const consumedSegmentIdRef = useRef<string | null>(null);
  const currentSegmentIdRef = useRef<string | null>(null);
  const partsBufRef = useRef<Map<string, Map<number, string>>>(new Map());
  const lastConfirmedUserTextRef = useRef<string | null>(null);

  useEffect(() => {
    if (!room) return;
    const handler = async (reader: any) => {
      const dbgId = `dbg_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
      setMessages((prev) => [
        ...prev,
        {
          id: dbgId,
          role: 'assistant',
          text: `📩 受信開始 (size=${reader?.info?.size ?? '?'})`,
          ts: Date.now(),
        },
      ]);
      try {
        const raw = await reader.readAll();
        // 受信成功 → マーカーを除去
        setMessages((prev) => prev.filter((m) => m.id !== dbgId));
        const msg = JSON.parse(raw) as HelixMessage;
        if (!msg.id || !msg.role || typeof msg.text !== 'string') return;
        // 分割メッセージの再結合（agentが800バイト単位で分割送信する）
        const totalParts = msg.parts ?? 1;
        if (totalParts > 1) {
          const buf = partsBufRef.current.get(msg.id) ?? new Map<number, string>();
          buf.set(msg.part ?? 0, msg.text);
          partsBufRef.current.set(msg.id, buf);
          if (buf.size < totalParts) return; // 全パート未着
          const joined = Array.from({ length: totalParts }, (_, i) => buf.get(i) ?? '').join('');
          partsBufRef.current.delete(msg.id);
          msg.text = joined;
        }
        setMessages((prev) =>
          prev.some((m) => m.id === msg.id) ? prev : [...prev, msg]
        );
        if (msg.role === 'user') {
          // 確定でinterim破棄 + 表示中セグメントを消費済みに
          consumedSegmentIdRef.current = currentSegmentIdRef.current;
          lastConfirmedUserTextRef.current = msg.text;
          setInterimText(null);
        }
      } catch (e) {
        console.error('helix.chat parse failed:', e);
        // 受信失敗を画面で確認できるようsystemメッセージとして表示（調査用）
        setMessages((prev) => [
          ...prev,
          {
            id: `err_${Date.now()}`,
            role: 'assistant',
            text: `⚠️ メッセージ受信エラー: ${String(e)}`,
            ts: Date.now(),
          },
        ]);
      }
    };
    room.registerTextStreamHandler(TOPIC, handler);
    return () => {
      try {
        room.unregisterTextStreamHandler(TOPIC);
      } catch {
        /* already unregistered */
      }
    };
  }, [room]);

  // ── interim表示レイヤ（lk.transcription / ローカル参加者のみ・表示専用）──
  const transcriptionOptions = useMemo(() => ({ room }), [room]);
  const transcriptions = useTranscriptions(transcriptionOptions as any);

  useEffect(() => {
    if (!room) return;
    const latestLocal = [...transcriptions]
      .reverse()
      .find((t) => t.participantInfo?.identity === room.localParticipant?.identity);
    if (!latestLocal) return;
    const attrs = latestLocal.streamInfo?.attributes ?? {};
    const segKey = attrs['lk.segment_id'] ?? latestLocal.streamInfo?.id ?? null;
    currentSegmentIdRef.current = segKey;
    // user確定時に表示していたセグメント(発話)は再表示しない（滞留・二重表示防止）
    if (segKey && segKey === consumedSegmentIdRef.current) return;
    const isFinal = attrs['lk.transcription_final'] === 'true';
    // 属性が無い環境向けフォールバック: 確定済みuserメッセージと同一テキストは破棄
    const matchesConfirmed =
      lastConfirmedUserTextRef.current !== null &&
      (latestLocal.text || '').trim() === lastConfirmedUserTextRef.current.trim();
    if (isFinal || matchesConfirmed) {
      setInterimText(null);
    } else {
      setInterimText(latestLocal.text || null);
    }
  }, [transcriptions, room]);

  return { messages, interimText: SHOW_INTERIM() ? interimText : null };
}
