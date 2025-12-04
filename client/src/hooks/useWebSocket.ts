import { useEffect, useState } from "react";
import { io, Socket } from "socket.io-client";

let socket: Socket | null = null;

export function useWebSocket() {
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    // Initialize socket connection if not already connected
    if (!socket) {
      socket = io({
        path: "/api/socket.io",
        transports: ["websocket", "polling"],
      });

      socket.on("connect", () => {
        console.log("[WebSocket] Connected:", socket?.id);
        setIsConnected(true);
      });

      socket.on("disconnect", () => {
        console.log("[WebSocket] Disconnected");
        setIsConnected(false);
      });
    }

    return () => {
      // Don't disconnect on unmount, keep connection alive
      // socket?.disconnect();
    };
  }, []);

  return { socket, isConnected };
}

// Hook for joining specific rooms
export function useWebSocketRoom(roomName: string) {
  const { socket, isConnected } = useWebSocket();

  useEffect(() => {
    if (socket && isConnected) {
      socket.emit(`join-${roomName}`);
      console.log(`[WebSocket] Joined room: ${roomName}`);
    }
  }, [socket, isConnected, roomName]);

  return { socket, isConnected };
}

// Hook for S-7 session collaboration
export function useS7Session(sessionId: string) {
  const { socket, isConnected } = useWebSocket();
  const [sessionData, setSessionData] = useState<any>(null);

  useEffect(() => {
    if (socket && isConnected && sessionId) {
      socket.emit("join-s7-workspace", sessionId);

      socket.on("s7-session-update", (update: any) => {
        setSessionData(update);
      });

      return () => {
        socket.off("s7-session-update");
      };
    }
  }, [socket, isConnected, sessionId]);

  const updateSession = (update: any) => {
    if (socket && isConnected) {
      socket.emit("s7-session-update", { sessionId, update });
    }
  };

  return { sessionData, updateSession, isConnected };
}

// Hook for real-time leaderboard
export function useLeaderboard() {
  const { socket, isConnected } = useWebSocket();
  const [leaderboardData, setLeaderboardData] = useState<any>(null);

  useEffect(() => {
    if (socket && isConnected) {
      socket.on("leaderboard-update", (update: any) => {
        setLeaderboardData(update);
      });

      return () => {
        socket.off("leaderboard-update");
      };
    }
  }, [socket, isConnected]);

  return { leaderboardData, isConnected };
}

// Hook for user presence
export function useUserPresence() {
  const { socket, isConnected } = useWebSocket();
  const [onlineUsers, setOnlineUsers] = useState<number[]>([]);

  useEffect(() => {
    if (socket && isConnected) {
      socket.on("user-presence", (data: { userId: number; status: string }) => {
        setOnlineUsers((prev) => {
          if (data.status === "online") {
            const uniqueSet = new Set([...prev, data.userId]);
            return Array.from(uniqueSet);
          } else {
            return prev.filter((id) => id !== data.userId);
          }
        });
      });

      return () => {
        socket.off("user-presence");
      };
    }
  }, [socket, isConnected]);

  return { onlineUsers, isConnected };
}

// Hook for agent pool collaboration
export function useAgentPool(poolId: string) {
  const { socket, isConnected } = useWebSocket();
  const [poolData, setPoolData] = useState<any>(null);

  useEffect(() => {
    if (socket && isConnected && poolId) {
      socket.emit("join-agent-pool", poolId);

      socket.on("agent-pool-update", (update: any) => {
        setPoolData(update);
      });

      return () => {
        socket.off("agent-pool-update");
      };
    }
  }, [socket, isConnected, poolId]);

  const updatePool = (update: any) => {
    if (socket && isConnected) {
      socket.emit("agent-pool-update", { poolId, update });
    }
  };

  return { poolData, updatePool, isConnected };
}

// Hook for real-time notifications
export function useNotifications(userId: number) {
  const { socket, isConnected } = useWebSocket();
  const [notifications, setNotifications] = useState<any[]>([]);

  useEffect(() => {
    if (socket && isConnected && userId) {
      socket.on(`notification-${userId}`, (notification: any) => {
        setNotifications((prev) => [notification, ...prev]);
      });

      return () => {
        socket.off(`notification-${userId}`);
      };
    }
  }, [socket, isConnected, userId]);

  return { notifications, isConnected };
}
