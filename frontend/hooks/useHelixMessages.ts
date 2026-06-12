'use client';

import { useEffect, useMemo, useState } from 'react';
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
        if (msg.role === 'user') setInterimText(null); // 確定でinterim破棄
      } catch (e) {
        console.warn('helix.chat parse failed:', e);
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
    const isFinal =
      latestLocal.streamInfo?.attributes?.['lk.transcription_final'] === 'true';
    if (isFinal) {
      // finalセグメントはhelix.chat側で確定表示されるため、interimは即破棄
      setInterimText(null);
    } else {
      setInterimText(latestLocal.text || null);
    }
  }, [transcriptions, room]);

  return { messages, interimText };
}
