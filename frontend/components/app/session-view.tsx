'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { motion } from 'motion/react';
import { useSessionContext } from '@livekit/components-react';
import type { AppConfig } from '@/app-config';
import { ChatTranscript, type StatusIndicator } from '@/components/app/chat-transcript';
import { PreConnectMessage } from '@/components/app/preconnect-message';
import { TileLayout } from '@/components/app/tile-layout';
import {
  AgentControlBar,
  type ControlBarControls,
} from '@/components/livekit/agent-control-bar/agent-control-bar';
import { cn } from '@/lib/utils';
import { ScrollArea } from '../livekit/scroll-area/scroll-area';
import { useHelixMessages } from '@/hooks/useHelixMessages';

const MotionBottom = motion.create('div');

const BOTTOM_VIEW_MOTION_PROPS = {
  variants: {
    visible: {
      opacity: 1,
      translateY: '0%',
    },
    hidden: {
      opacity: 0,
      translateY: '100%',
    },
  },
  initial: 'hidden',
  animate: 'visible',
  exit: 'hidden',
  transition: {
    duration: 0.3,
    delay: 0.5,
    ease: 'easeOut',
  },
};

interface FadeProps {
  top?: boolean;
  bottom?: boolean;
  className?: string;
}

export function Fade({ top = false, bottom = false, className }: FadeProps) {
  return (
    <div
      className={cn(
        'from-background pointer-events-none h-4 bg-linear-to-b to-transparent',
        top && 'bg-linear-to-b',
        bottom && 'bg-linear-to-t',
        className
      )}
    />
  );
}

// ── 状態インジケータ定義 ──
type AgentStatusState = 'thinking' | 'speaking' | null;
const STATUS_THINKING: StatusIndicator = { emoji: '💭', text: '回答を考えています…' };
const STATUS_LISTENING: StatusIndicator = { emoji: '👂', text: '聞いています…' };
const STATUS_SEARCHING: StatusIndicator = { emoji: '🔍', text: '検索中です…' };

interface SessionViewProps {
  appConfig: AppConfig;
}

export const SessionView = ({
  appConfig,
  ...props
}: React.ComponentProps<'section'> & SessionViewProps) => {
  const session = useSessionContext();
  const { messages, streaming } = useHelixMessages(session?.room);
  const [chatOpen, setChatOpen] = useState(true);
  const [agentState, setAgentState] = useState<AgentStatusState>(null);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [deepSearching, setDeepSearching] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  const controls: ControlBarControls = {
    leave: true,
    microphone: true,
    chat: appConfig.supportsChatInput,
    camera: appConfig.supportsVideoInput,
    screenShare: appConfig.supportsVideoInput,
  };

  // Deep Search status 購読（DataChannel経由）
  const myAgentIdentityRef = useRef<string | null>(null);
  useEffect(() => {
    const room = session?.room;
    if (!room) return;
    const handler = (payload: Uint8Array, participant: any) => {
      try {
        const data = JSON.parse(new TextDecoder().decode(payload));
        if (data.type !== 'deep_search_status') return;
        const identity = participant?.identity;
        if (!myAgentIdentityRef.current) {
          if (!identity || identity.startsWith('user_')) return;
          myAgentIdentityRef.current = identity;
        }
        if (identity !== myAgentIdentityRef.current) return;
        setDeepSearching(data.status === 'start');
      } catch { /* ignore */ }
    };
    room.on('dataReceived', handler);
    return () => { room.off('dataReceived', handler); };
  }, [session?.room]);

  // 状態インジケータの優先度決定
  // isSpeaking > deepSearch > thinking（delta受信中はthinking非表示）
  const statusIndicator: StatusIndicator | null = (() => {
    if (isSpeaking) return STATUS_LISTENING;
    if (deepSearching) return STATUS_SEARCHING;
    if (agentState === 'thinking' && !streaming) return STATUS_THINKING;
    return null;
  })();

  // 新しいメッセージが来たら最下部へスクロール
  useEffect(() => {
    const el = scrollAreaRef.current;
    if (!el) return;
    const isNearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 100;
    const lastMessage = messages.at(-1);
    const lastMessageIsLocal = lastMessage?.role === 'user';
    if (lastMessageIsLocal || isNearBottom) {
      el.scrollTop = el.scrollHeight;
    }
  }, [messages]);

  // インジケータ変化時も自動スクロール（末尾付近のときのみ）
  useEffect(() => {
    const el = scrollAreaRef.current;
    if (!el || !statusIndicator) return;
    const isNearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 100;
    if (isNearBottom) {
      el.scrollTop = el.scrollHeight;
    }
  }, [statusIndicator]);

  return (
    <section className="bg-background relative z-10 h-full w-full overflow-hidden" {...props}>
      {/* Chat Transcript */}
      <div
        className={cn(
          'fixed inset-0 grid grid-cols-1 grid-rows-1',
          !chatOpen && 'pointer-events-none'
        )}
      >
        <Fade top className="absolute inset-x-4 top-0 h-40" />
        <ScrollArea ref={scrollAreaRef} className="px-4 pt-16 pb-[220px] md:px-6 md:pb-[260px]">
          <ChatTranscript
            hidden={!chatOpen}
            messages={messages}
            status={statusIndicator}
            className="mx-auto max-w-2xl space-y-3 transition-opacity duration-300 ease-out"
          />
          {appConfig.isPreConnectBufferEnabled && (
            <PreConnectMessage messages={messages} className="pb-4" />
          )}
        </ScrollArea>
      </div>

      {/* Tile Layout */}
      <TileLayout chatOpen={chatOpen} />

      {/* Bottom */}
      <MotionBottom
        {...BOTTOM_VIEW_MOTION_PROPS}
        className="fixed inset-x-3 bottom-0 z-50 md:inset-x-12"
      >
        <div className="bg-background relative mx-auto max-w-2xl pb-3 md:pb-12">
          <Fade bottom className="absolute inset-x-0 top-0 h-4 -translate-y-full" />
          <AgentControlBar
            controls={controls}
            isConnected={session.isConnected}
            onDisconnect={session.end}
            onChatOpenChange={setChatOpen}
          />
        </div>
      </MotionBottom>
    </section>
  );
};
