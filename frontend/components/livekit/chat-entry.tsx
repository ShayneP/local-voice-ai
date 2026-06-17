import * as React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';
import { cn } from '@/lib/utils';

export interface ChatEntryProps extends React.HTMLAttributes<HTMLLIElement> {
  locale: string;
  timestamp: number;
  message: string;
  messageOrigin: 'local' | 'remote';
  name?: string;
  hasBeenEdited?: boolean;
}

// 感情タグを除去する（例: <e:happy>テキスト → テキスト）
function removeEmotionTags(text: string): string {
  return text.replace(/<e:\w+>/g, '').trim();
}

export const ChatEntry = ({
  name,
  locale,
  timestamp,
  message,
  messageOrigin,
  hasBeenEdited = false,
  className,
  ...props
}: ChatEntryProps) => {
  const time = new Date(timestamp);
  const title = time.toLocaleTimeString(locale, { timeStyle: 'full' });
  // assistant(remote)のみ感情タグ除去。user(local)は完全raw。
  const displayMessage = messageOrigin === 'remote' ? removeEmotionTags(message) : message;

  return (
    <li
      title={title}
      data-lk-message-origin={messageOrigin}
      className={cn('group flex w-full flex-col gap-0.5', className)}
      {...props}
    >
      <header
        className={cn(
          'text-muted-foreground flex items-center gap-2 text-sm',
          messageOrigin === 'local' ? 'flex-row-reverse' : 'text-left'
        )}
      >
        {name && <strong>{name}</strong>}
        <span className="font-mono text-xs opacity-0 transition-opacity ease-linear group-hover:opacity-100">
          {hasBeenEdited && '*'}
          {time.toLocaleTimeString(locale, { timeStyle: 'short' })}
        </span>
      </header>
      <span
        className={cn(
          'rounded-[20px] break-words',
          messageOrigin === 'local' ? 'max-w-4/5 bg-muted ml-auto p-2 whitespace-pre-wrap' : 'w-full text-left'
        )}
      >
        {messageOrigin === 'remote' ? (
          <div className="helix-prose">
            <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>{displayMessage}</ReactMarkdown>
          </div>
        ) : (
          displayMessage
        )}
      </span>
    </li>
  );
};
