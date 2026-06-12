'use client';

import { AnimatePresence, type HTMLMotionProps, motion } from 'motion/react';
import { ChatEntry } from '@/components/livekit/chat-entry';
import type { HelixMessage } from '@/hooks/useHelixMessages';

const MotionContainer = motion.create('div');
const MotionChatEntry = motion.create(ChatEntry);

const CONTAINER_MOTION_PROPS = {
  variants: {
    hidden: {
      opacity: 0,
      transition: {
        ease: 'easeOut',
        duration: 0.3,
        staggerChildren: 0.1,
        staggerDirection: -1,
      },
    },
    visible: {
      opacity: 1,
      transition: {
        delay: 0.2,
        ease: 'easeOut',
        duration: 0.3,
        stagerDelay: 0.2,
        staggerChildren: 0.1,
        staggerDirection: 1,
      },
    },
  },
  initial: 'hidden',
  animate: 'visible',
  exit: 'hidden',
};

const MESSAGE_MOTION_PROPS = {
  variants: {
    hidden: {
      opacity: 0,
      translateY: 10,
    },
    visible: {
      opacity: 1,
      translateY: 0,
    },
  },
};

interface ChatTranscriptProps {
  hidden?: boolean;
  messages?: HelixMessage[];
  interimText?: string | null;
}

export function ChatTranscript({
  hidden = false,
  messages = [],
  interimText = null,
  ...props
}: ChatTranscriptProps & Omit<HTMLMotionProps<'div'>, 'ref'>) {
  return (
    <AnimatePresence>
      {!hidden && (
        <MotionContainer {...CONTAINER_MOTION_PROPS} {...props}>
          {messages.map((msg) => {
            const locale = navigator?.language ?? 'en-US';
            return (
              <MotionChatEntry
                key={msg.id}
                locale={locale}
                timestamp={msg.ts}
                message={msg.text}
                messageOrigin={msg.role === 'user' ? 'local' : 'remote'}
                {...MESSAGE_MOTION_PROPS}
              />
            );
          })}
          {/* interim表示レイヤ: 発話中バブル（履歴には入らない・表示専用） */}
          {interimText && (
            <li className="flex w-full flex-col gap-0.5">
              <span className="bg-muted/50 text-muted-foreground ml-auto max-w-4/5 rounded-[20px] p-2 italic whitespace-pre-wrap break-words">
                {interimText}
              </span>
            </li>
          )}
        </MotionContainer>
      )}
    </AnimatePresence>
  );
}
