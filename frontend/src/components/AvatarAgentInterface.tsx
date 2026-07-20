import type { ReactNode } from 'react';

import type { ConnectionStatus } from '../hooks/useAvatarAgent';
import type { ChatMessage } from '../types';
import { ChatOverlay } from './ChatOverlay';
import './AvatarAgentInterface.css';

interface AvatarAgentInterfaceProps {
  children: ReactNode;
  chatHistory: ChatMessage[];
  connectionStatus: ConnectionStatus;
  isMuted: boolean;
  onConnect: () => void | Promise<void>;
  onDisconnect: () => void | Promise<void>;
  onToggleMute: () => void;
}

const MicIcon = () => (
  <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
    <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
    <line x1="12" y1="19" x2="12" y2="23" />
    <line x1="8" y1="23" x2="16" y2="23" />
  </svg>
);

const MicOffIcon = () => (
  <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="1" y1="1" x2="23" y2="23" />
    <path d="M9 9v3a3 3 0 0 0 5.12 2.12M15 9.34V4a3 3 0 0 0-5.94-.6" />
    <path d="M17 16.95A7 7 0 0 1 5 12v-2" />
    <line x1="12" y1="19" x2="12" y2="23" />
    <line x1="8" y1="23" x2="16" y2="23" />
  </svg>
);

const statusLabel: Record<ConnectionStatus, string> = {
  disconnected: 'Ready to connect',
  connecting: 'Connecting',
  connected: 'Connected',
};

export function AvatarAgentInterface({
  children,
  chatHistory,
  connectionStatus,
  isMuted,
  onConnect,
  onDisconnect,
  onToggleMute,
}: AvatarAgentInterfaceProps) {
  const isConnecting = connectionStatus === 'connecting';
  const isConnected = connectionStatus === 'connected';

  return (
    <main className="avatar-agent">
      <div className="avatar-agent__stage" aria-label="3D avatar">
        {children}
      </div>

      <div className={`avatar-agent__status avatar-agent__status--${connectionStatus}`} role="status" aria-live="polite">
        <span className="avatar-agent__status-dot" />
        {statusLabel[connectionStatus]}
      </div>

      <ChatOverlay history={chatHistory} />

      <div className="avatar-agent__controls" aria-label="Voice agent controls">
        {!isConnected && (
          <button
            className="avatar-agent__connect-button"
            type="button"
            onClick={() => void onConnect()}
            disabled={isConnecting}
          >
            {isConnecting && <span className="avatar-agent__spinner" aria-hidden="true" />}
            {isConnecting ? 'Connecting...' : 'Connect'}
          </button>
        )}

        {isConnected && (
          <>
            <button
              className={`avatar-agent__mute-button${isMuted ? ' avatar-agent__mute-button--muted' : ''}`}
              type="button"
              onClick={onToggleMute}
              title={isMuted ? 'Unmute' : 'Mute'}
              aria-label={isMuted ? 'Unmute microphone' : 'Mute microphone'}
              aria-pressed={isMuted}
            >
              {isMuted ? <MicOffIcon /> : <MicIcon />}
            </button>

            <button
              className="avatar-agent__disconnect-button"
              type="button"
              onClick={() => void onDisconnect()}
            >
              Disconnect
            </button>
          </>
        )}
      </div>
    </main>
  );
}
