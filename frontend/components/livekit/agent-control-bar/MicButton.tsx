'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import { Track } from 'livekit-client';
import {
  useLocalParticipant,
  useRoomContext,
  useTrackToggle,
} from '@livekit/components-react';
import { MicrophoneIcon, MicrophoneSlashIcon, LockSimpleIcon } from '@phosphor-icons/react/dist/ssr';
import { cn } from '@/lib/utils';

// ── 定数 ────────────────────────────────────────────────────
const VAD_TIMER_MS = 3000;
const SPEAKING_WAIT_MS = 2000;
const SPEAKING_CHECK_INTERVAL = 50;
const AUDIO_LEVEL_THRESHOLD = 0.02;
const AUDIO_LEVEL_OFF_DELAY = 300;
const LONG_PRESS_MS = 600;

type MicMode = 'off' | 'normal' | 'locked';

interface MicButtonProps {
  className?: string;
}

export function MicButton({ className }: MicButtonProps) {
  const room = useRoomContext();
  const { localParticipant } = useLocalParticipant();
  const micToggle = useTrackToggle({ source: Track.Source.Microphone });

  // ── 状態 ─────────────────────────────────────────────────
  const [micMode, setMicMode] = useState<MicMode>('off');
  const micModeRef = useRef<MicMode>('off');  // staleクロージャ回避用ref

  // setMicModeとmicModeRefを同時に更新するヘルパー
  const setMicModeWithRef = useCallback((mode: MicMode) => {
    micModeRef.current = mode;
    setMicMode(mode);
  }, []);

  const vadTimerRef = useRef<NodeJS.Timeout | null>(null);
  const waitTimerRef = useRef<NodeJS.Timeout | null>(null);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const isSpeakingRef = useRef(false);
  const speakingOffTimerRef = useRef<NodeJS.Timeout | null>(null);
  const longPressTimerRef = useRef<NodeJS.Timeout | null>(null);
  const isLongPressRef = useRef(false);

  const updateIsSpeaking = useCallback((val: boolean) => {
    isSpeakingRef.current = val;
    setIsSpeaking(val);
  }, []);

  // ── マイクOFF処理 ────────────────────────────────────────
  const turnMicOff = useCallback(async () => {
    if (vadTimerRef.current) clearTimeout(vadTimerRef.current);
    if (waitTimerRef.current) clearInterval(waitTimerRef.current);
    vadTimerRef.current = null;
    waitTimerRef.current = null;
    updateIsSpeaking(false);
    setMicModeWithRef('off');
    try {
      await micToggle.toggle(false);
    } catch (e) {
      console.warn('micToggle off error:', e);
    }
  }, [micToggle, updateIsSpeaking, setMicModeWithRef]);

  // ── タイマー満了時のOFF判定 ──────────────────────────────
  const handleVadTimerExpired = useCallback(() => {
    vadTimerRef.current = null;
    if (micModeRef.current === 'locked') return;
    if (!isSpeakingRef.current) {
      turnMicOff();
      return;
    }
    const deadline = Date.now() + SPEAKING_WAIT_MS;
    waitTimerRef.current = setInterval(() => {
      if (!isSpeakingRef.current || Date.now() >= deadline) {
        if (waitTimerRef.current) clearInterval(waitTimerRef.current);
        waitTimerRef.current = null;
        if (micModeRef.current !== 'locked') {
          turnMicOff();
        }
      }
    }, SPEAKING_CHECK_INTERVAL);
  }, [turnMicOff]);

  // ── VADイベント受信 ──────────────────────────────────────
  useEffect(() => {
    if (!room) return;
    const handleData = (payload: Uint8Array) => {
      try {
        const msg = JSON.parse(new TextDecoder().decode(payload));
        if (msg.type === 'vad_end') {
          if (micModeRef.current === 'off') return;
          if (vadTimerRef.current) clearTimeout(vadTimerRef.current);
          vadTimerRef.current = setTimeout(handleVadTimerExpired, VAD_TIMER_MS);
        } else if (msg.type === 'vad_start') {
          if (vadTimerRef.current) {
            clearTimeout(vadTimerRef.current);
            vadTimerRef.current = null;
          }
        }
      } catch (e) {
        // 無視
      }
    };
    room.on('dataReceived', handleData);
    return () => { room.off('dataReceived', handleData); };
  }, [room, handleVadTimerExpired]);

  // ── audioLevel監視 ───────────────────────────────────────
  useEffect(() => {
    if (micMode === 'off') {
      updateIsSpeaking(false);
      return;
    }
    const interval = setInterval(() => {
      const level = localParticipant?.audioLevel ?? 0;
      if (level > AUDIO_LEVEL_THRESHOLD) {
        if (speakingOffTimerRef.current) {
          clearTimeout(speakingOffTimerRef.current);
          speakingOffTimerRef.current = null;
        }
        updateIsSpeaking(true);
      } else {
        if (isSpeakingRef.current && !speakingOffTimerRef.current) {
          speakingOffTimerRef.current = setTimeout(() => {
            updateIsSpeaking(false);
            speakingOffTimerRef.current = null;
          }, AUDIO_LEVEL_OFF_DELAY);
        }
      }
    }, 50);
    return () => {
      clearInterval(interval);
      if (speakingOffTimerRef.current) {
        clearTimeout(speakingOffTimerRef.current);
        speakingOffTimerRef.current = null;
      }
    };
  }, [micMode, localParticipant, updateIsSpeaking]);

  // ── マイクON処理 ────────────────────────────────────────
  const turnMicOn = useCallback(async (mode: MicMode) => {
    setMicModeWithRef(mode);
    try {
      await micToggle.toggle(true);
    } catch (e) {
      console.warn('micToggle on error:', e);
    }
  }, [micToggle, setMicModeWithRef]);

  // ── 長押し判定 ───────────────────────────────────────────
  const handlePointerDown = useCallback(() => {
    isLongPressRef.current = false;
    longPressTimerRef.current = setTimeout(() => {
      isLongPressRef.current = true;
      if (micModeRef.current === 'off') {
        turnMicOn('locked');
      } else {
        setMicModeWithRef('locked');
        if (vadTimerRef.current) clearTimeout(vadTimerRef.current);
        vadTimerRef.current = null;
      }
    }, LONG_PRESS_MS);
  }, [turnMicOn, setMicModeWithRef]);

  const handlePointerUp = useCallback(() => {
    if (longPressTimerRef.current) {
      clearTimeout(longPressTimerRef.current);
      longPressTimerRef.current = null;
    }
  }, []);

  // ── タップ処理 ───────────────────────────────────────────
  const handleClick = useCallback(async () => {
    if (isLongPressRef.current) {
      isLongPressRef.current = false;
      return;
    }
    const currentMode = micModeRef.current;
    if (currentMode === 'off') {
      await turnMicOn('normal');
    } else if (currentMode === 'normal') {
      await turnMicOff();
    } else if (currentMode === 'locked') {
      await turnMicOff();
    }
  }, [turnMicOn, turnMicOff]);

  // ── クリーンアップ ───────────────────────────────────────
  useEffect(() => {
    return () => {
      if (vadTimerRef.current) clearTimeout(vadTimerRef.current);
      if (waitTimerRef.current) clearInterval(waitTimerRef.current);
      if (longPressTimerRef.current) clearTimeout(longPressTimerRef.current);
      if (speakingOffTimerRef.current) clearTimeout(speakingOffTimerRef.current);
    };
  }, []);

  // ── UI ──────────────────────────────────────────────────
  const isOn = micMode !== 'off';
  const isLocked = micMode === 'locked';

  return (
    <button
      aria-label={
        isLocked ? 'マイクロック中（タップで解除）'
        : isOn    ? 'マイクON（タップでOFF）'
                  : 'マイクOFF（タップでON・長押しでロック）'
      }
      onPointerDown={handlePointerDown}
      onPointerUp={handlePointerUp}
      onPointerLeave={handlePointerUp}
      onClick={handleClick}
      className={cn(
        'relative flex h-9 w-9 items-center justify-center rounded-full transition-all duration-200',
        !isOn && 'bg-muted text-foreground hover:bg-muted/80',
        isOn && !isLocked && 'bg-primary text-primary-foreground hover:bg-primary/90 ring-2 ring-primary/30',
        isLocked && 'bg-amber-500 text-white hover:bg-amber-400 ring-2 ring-amber-400/50',
        className
      )}
    >
      {!isOn
        ? <MicrophoneSlashIcon weight="bold" className="h-4 w-4" />
        : isLocked
          ? <LockSimpleIcon weight="bold" className="h-4 w-4" />
          : <MicrophoneIcon weight="bold" className="h-4 w-4" />
      }
      {isOn && !isLocked && isSpeaking && (
        <span className="absolute -top-1 -right-1 h-3 w-3 rounded-full bg-green-400 animate-pulse" />
      )}
      {isLocked && (
        <span className="absolute -top-1 -right-1 h-3 w-3 rounded-full bg-amber-300" />
      )}
    </button>
  );
}
