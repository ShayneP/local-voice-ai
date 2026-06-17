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
        ease: 'easeOut' as const,
        duration: 0.3,
        staggerChildren: 0.1,
        staggerDirection: -1,
      },
    },
    visible: {
      opacity: 1,
      transition: {
        delay: 0.2,
        ease: 'easeOut' as const,
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

export interface StatusIndicator {
  emoji: string;
  text: string;
}

interface ChatTranscriptProps {
  hidden?: boolean;
  messages?: HelixMessage[];
  status?: StatusIndicator | null;
}

export function ChatTranscript({
  hidden = false,
  messages = [],
  status = null,
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
          {/* 状態インジケータ（エフェメラル・履歴に残さない） */}
          <AnimatePresence>
            {status && (
              <motion.div
                key="status-indicator"
                initial={{ opacity: 0, translateY: 6 }}
                animate={{ opacity: 1, translateY: 0 }}
                exit={{ opacity: 0, translateY: 6 }}
                transition={{ duration: 0.2 }}
                className="flex items-center gap-1.5 px-1 py-1.5"
              >
                <span className="text-sm">{status.emoji}</span>
                <span className="text-muted-foreground text-xs">{status.text}</span>
              </motion.div>
            )}
          </AnimatePresence>
        </MotionContainer>
      )}
    </AnimatePresence>
  );
}
