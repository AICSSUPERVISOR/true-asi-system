/**
 * WebSocketProvider - Real-time updates using Socket.io
 * Provides live metric updates, analysis notifications, and connection status
 */

import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';
import { useToast } from '@/hooks/use-toast';

interface WebSocketContextType {
  socket: Socket | null;
  isConnected: boolean;
  subscribe: (event: string, callback: (data: any) => void) => void;
  unsubscribe: (event: string, callback: (data: any) => void) => void;
  emit: (event: string, data: any) => void;
}

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined);

export function WebSocketProvider({ children }: { children: React.ReactNode }) {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const { toast } = useToast();

  useEffect(() => {
    // Connect to WebSocket server
    // In production, this would be your backend URL
    const socketInstance = io(window.location.origin, {
      path: '/api/socket.io',
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      reconnectionAttempts: 5,
    });

    socketInstance.on('connect', () => {
      console.log('[WebSocket] Connected');
      setIsConnected(true);
      toast({
        title: 'Real-time updates enabled',
        description: 'You will receive live updates automatically',
      });
    });

    socketInstance.on('disconnect', () => {
      console.log('[WebSocket] Disconnected');
      setIsConnected(false);
    });

    socketInstance.on('connect_error', (error) => {
      console.error('[WebSocket] Connection error:', error);
      setIsConnected(false);
    });

    socketInstance.on('reconnect', (attemptNumber) => {
      console.log('[WebSocket] Reconnected after', attemptNumber, 'attempts');
      setIsConnected(true);
      toast({
        title: 'Reconnected',
        description: 'Real-time updates restored',
      });
    });

    // Listen for real-time events
    socketInstance.on('metric:update', (data) => {
      console.log('[WebSocket] Metric update:', data);
    });

    socketInstance.on('analysis:complete', (data) => {
      console.log('[WebSocket] Analysis complete:', data);
      toast({
        title: 'Analysis Complete',
        description: `Analysis for ${data.companyName} is ready`,
      });
    });

    socketInstance.on('execution:progress', (data) => {
      console.log('[WebSocket] Execution progress:', data);
    });

    setSocket(socketInstance);

    // Cleanup on unmount
    return () => {
      socketInstance.disconnect();
    };
  }, []);

  const subscribe = useCallback((event: string, callback: (data: any) => void) => {
    if (socket) {
      socket.on(event, callback);
    }
  }, [socket]);

  const unsubscribe = useCallback((event: string, callback: (data: any) => void) => {
    if (socket) {
      socket.off(event, callback);
    }
  }, [socket]);

  const emit = useCallback((event: string, data: any) => {
    if (socket && isConnected) {
      socket.emit(event, data);
    }
  }, [socket, isConnected]);

  return (
    <WebSocketContext.Provider value={{ socket, isConnected, subscribe, unsubscribe, emit }}>
      {children}
    </WebSocketContext.Provider>
  );
}

export function useWebSocket() {
  const context = useContext(WebSocketContext);
  if (context === undefined) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
}

/**
 * Hook for subscribing to specific WebSocket events
 */
export function useWebSocketEvent(event: string, callback: (data: any) => void) {
  const { subscribe, unsubscribe } = useWebSocket();

  useEffect(() => {
    subscribe(event, callback);
    return () => {
      unsubscribe(event, callback);
    };
  }, [event, callback, subscribe, unsubscribe]);
}

/**
 * Hook for real-time metric updates
 */
export function useRealtimeMetrics(analysisId: string) {
  const [metrics, setMetrics] = useState<any>(null);
  const { subscribe, unsubscribe } = useWebSocket();

  useEffect(() => {
    const handleMetricUpdate = (data: any) => {
      if (data.analysisId === analysisId) {
        setMetrics(data.metrics);
      }
    };

    subscribe('metric:update', handleMetricUpdate);
    return () => {
      unsubscribe('metric:update', handleMetricUpdate);
    };
  }, [analysisId, subscribe, unsubscribe]);

  return metrics;
}

/**
 * Hook for real-time execution progress
 */
export function useRealtimeExecution(workflowId: string) {
  const [progress, setProgress] = useState<any>(null);
  const { subscribe, unsubscribe } = useWebSocket();

  useEffect(() => {
    const handleProgressUpdate = (data: any) => {
      if (data.workflowId === workflowId) {
        setProgress(data);
      }
    };

    subscribe('execution:progress', handleProgressUpdate);
    return () => {
      unsubscribe('execution:progress', handleProgressUpdate);
    };
  }, [workflowId, subscribe, unsubscribe]);

  return progress;
}
