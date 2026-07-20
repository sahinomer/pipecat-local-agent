import { useCallback, useEffect, useRef, useState } from 'react';
import { AggregationType, LogLevel, PipecatClient } from '@pipecat-ai/client-js';
import { ProtobufFrameSerializer, WebSocketTransport } from '@pipecat-ai/websocket-transport';
import type { ChatMessage } from '../types';

export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected';
type ConnectionQueryValue = string | number | boolean;
export type ConnectionQueryParams = Record<string, ConnectionQueryValue | null | undefined>;

type BlendshapeMessage = {
  type: 'blendshapes';
  data: number[][];
};

const SENTENCE_END_PATTERN = /[.!?\u2026]["'\u2019\u201d\])}]*$/u;

const appendTranscriptText = (currentText: string, incomingText: string) => {
  const alreadySeparated = /\s$/u.test(currentText) || /^\s/u.test(incomingText);
  const separator = !alreadySeparated && SENTENCE_END_PATTERN.test(currentText) ? ' ' : '';

  return `${currentText}${separator}${incomingText}`;
};

const buildConnectionUrl = (backendUrl: string, queryParams: ConnectionQueryParams) => {
  const url = new URL(backendUrl);

  Object.entries(queryParams).forEach(([key, value]) => {
    if (value !== undefined && value !== null) {
      url.searchParams.set(key, String(value));
    }
  });

  return url.toString();
};

const isBlendshapeMessage = (message: unknown): message is BlendshapeMessage => {
  if (typeof message !== 'object' || message === null) return false;

  const { type, data } = message as { type?: unknown; data?: unknown };
  return type === 'blendshapes'
    && Array.isArray(data)
    && data.every(frame => Array.isArray(frame) && frame.every(Number.isFinite));
};

export const useAvatarAgent = (backendUrl: string) => {
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('disconnected');
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [isBotSpeaking, setIsBotSpeaking] = useState(false);
  const [isMuted, setIsMuted] = useState(false);

  const animationQueue = useRef<number[][]>([]);
  const clientRef = useRef<PipecatClient | null>(null);

  useEffect(() => {
    const transport = new WebSocketTransport({
      playerSampleRate: 24000,
      serializer: new ProtobufFrameSerializer(),
    });

    const handleNewTranscript = (role: 'user' | 'bot', text: string) => {
      setChatHistory((previousHistory) => {
        const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        if (previousHistory.length === 0) {
          return [{ role, text, timestamp }];
        }

        const lastMessage = previousHistory[previousHistory.length - 1];

        if (lastMessage.role === role) {
          const updatedMessage = {
            ...lastMessage,
            text: appendTranscriptText(lastMessage.text, text),
            timestamp,
          };
          return [...previousHistory.slice(0, -1), updatedMessage];
        }

        return [...previousHistory, { role, text, timestamp }];
      });
    };

    const client = new PipecatClient({
      transport,
      callbacks: {
        onConnected: () => {
          setConnectionStatus('connected');
        },
        onDisconnected: () => {
          setConnectionStatus('disconnected');
          setChatHistory([]);
          setIsBotSpeaking(false);
          setIsMuted(false);
          animationQueue.current = [];
        },

        onUserTranscript: (data) => {
          if (data && data.final) {
            handleNewTranscript('user', data.text);
          }
        },

        onBotOutput: (data) => {
          if (!data.text || data.aggregated_by === AggregationType.WORD) return;

          const isInitialOutput = data.spoken_status === 'new'
            || (data.spoken_status === undefined && data.spoken !== true);

          if (isInitialOutput) handleNewTranscript('bot', data.text);
        },

        onBotStartedSpeaking: () => {
          setIsBotSpeaking(true);
        },

        onBotStoppedSpeaking: () => {
          setIsBotSpeaking(false);
          animationQueue.current = [];
        },

        onServerMessage: (msg: unknown) => {
          if (isBlendshapeMessage(msg)) {
            animationQueue.current.push(...msg.data);
          }
        },

        onError: (error) => {
          console.error('Pipecat error:', error);
          setConnectionStatus('disconnected');
          setChatHistory([]);
          setIsBotSpeaking(false);
          setIsMuted(false);
          animationQueue.current = [];
        },
      },
    });

    client.setLogLevel(LogLevel.WARN);
    clientRef.current = client;

    return () => {
      client.disconnect().catch(() => {});
    };
  }, []);

  const connect = useCallback(async (queryParams: ConnectionQueryParams = {}) => {
    const client = clientRef.current;
    if (!client) return;

    try {
      setConnectionStatus('connecting');
      await client.connect({ wsUrl: buildConnectionUrl(backendUrl, queryParams) });
    } catch (err) {
      console.error('Connection failed:', err);
      setConnectionStatus('disconnected');
    }
  }, [backendUrl]);

  const disconnect = useCallback(async () => {
    if (!clientRef.current) return;

    try {
      await clientRef.current.disconnect();
    } catch (err) {
      console.error('Disconnect failed:', err);
      setConnectionStatus('disconnected');
      setChatHistory([]);
      setIsBotSpeaking(false);
      setIsMuted(false);
      animationQueue.current = [];
    }
  }, []);

  const toggleMute = useCallback(() => {
    const client = clientRef.current;
    if (!client) return;

    const newMutedState = !isMuted;
    client.enableMic(!newMutedState);
    setIsMuted(newMutedState);
  }, [isMuted]);

  return {
    connect,
    disconnect,
    connectionStatus,
    chatHistory,
    animationQueue,
    isBotSpeaking,
    isMuted,
    toggleMute,
  };
};
