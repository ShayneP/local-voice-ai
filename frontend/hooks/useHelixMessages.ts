'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import type { Room } from 'livekit-client';
import { useTranscriptions } from '@livekit/components-react';

export interface HelixMessage {
  id: string;
  role: 'user' | 'assistant';
  text: string; // raw — 一切加工しない
  ts: number;
}

const TOPIC = 'helix.chat';

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

  useEffect(() => {
    if (!room) return;
    const handler = async (reader: any) => {
      try {
        const raw = await reader.readAll();
        const msg = JSON.parse(raw) as HelixMessage;
        if (!msg.id || !msg.role || typeof msg.text !== 'string') return;
        setMessages((prev) =>
          prev.some((m) => m.id === msg.id) ? prev : [...prev, msg]
        );
        if (msg.role === 'user') {
          // 確定でinterim破棄 + 表示中セグメントを消費済みに
          consumedSegmentIdRef.current = currentSegmentIdRef.current;
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
    const segId = latestLocal.streamInfo?.id ?? null;
    currentSegmentIdRef.current = segId;
    // user確定時に表示していたセグメントは再表示しない（滞留・二重表示防止）
    if (segId && segId === consumedSegmentIdRef.current) return;
    const isFinal =
      latestLocal.streamInfo?.attributes?.['lk.transcription_final'] === 'true';
    if (isFinal) {
      setInterimText(null);
    } else {
      setInterimText(latestLocal.text || null);
    }
  }, [transcriptions, room]);

  return { messages, interimText };
}
