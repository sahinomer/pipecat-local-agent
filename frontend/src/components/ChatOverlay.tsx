import { useEffect, useRef } from 'react';

import type { ChatMessage } from '../types';
import './ChatOverlay.css';

interface ChatOverlayProps {
  history: ChatMessage[];
}

export const ChatOverlay = ({ history }: ChatOverlayProps) => {
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const frame = requestAnimationFrame(() => {
      const scrollArea = scrollAreaRef.current;

      if (scrollArea) {
        scrollArea.scrollTo({ top: scrollArea.scrollHeight, behavior: 'smooth' });
      }
    });

    return () => cancelAnimationFrame(frame);
  }, [history]);

  return (
    <section
      className={`chat-overlay${history.length === 0 ? ' chat-overlay--empty' : ''}`}
      aria-label="Conversation"
      aria-live="polite"
    >
      <div ref={scrollAreaRef} className="chat-overlay__scroll-area">
        {history.map((message, index) => (
          <div
            key={`${message.timestamp}-${message.role}-${index}`}
            className={`chat-overlay__message-row chat-overlay__message-row--${message.role}`}
          >
            <div className={`chat-overlay__bubble chat-overlay__bubble--${message.role}`}>
              <div>{message.text}</div>
              <div className="chat-overlay__timestamp">{message.timestamp}</div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
};
