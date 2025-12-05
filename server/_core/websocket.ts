import { Server as HTTPServer } from "http";
import { Server as SocketIOServer } from "socket.io";

let io: SocketIOServer | null = null;

export function initializeWebSocket(httpServer: HTTPServer) {
  io = new SocketIOServer(httpServer, {
    cors: {
      origin: "*",
      methods: ["GET", "POST"],
    },
    path: "/api/socket.io",
  });

  io.on("connection", (socket) => {
    console.log(`[WebSocket] Client connected: ${socket.id}`);

    // Join rooms for targeted broadcasts
    socket.on("join-dashboard", () => {
      socket.join("dashboard");
      console.log(`[WebSocket] ${socket.id} joined dashboard room`);
    });

    socket.on("join-agents", () => {
      socket.join("agents");
      console.log(`[WebSocket] ${socket.id} joined agents room`);
    });

    socket.on("join-chat", (chatId: string) => {
      socket.join(`chat-${chatId}`);
      console.log(`[WebSocket] ${socket.id} joined chat-${chatId}`);
    });

    socket.on("join-s7-workspace", (workspaceId: string) => {
      socket.join(`s7-workspace-${workspaceId}`);
      console.log(`[WebSocket] ${socket.id} joined s7-workspace-${workspaceId}`);
    });

    // Join analysis room for real-time metric updates
    socket.on("subscribe:analysis", (analysisId: string) => {
      socket.join(`analysis:${analysisId}`);
      console.log(`[WebSocket] ${socket.id} subscribed to analysis:${analysisId}`);
    });

    // Join workflow room for real-time execution progress
    socket.on("subscribe:workflow", (workflowId: string) => {
      socket.join(`workflow:${workflowId}`);
      console.log(`[WebSocket] ${socket.id} subscribed to workflow:${workflowId}`);
    });

    // Unsubscribe from analysis
    socket.on("unsubscribe:analysis", (analysisId: string) => {
      socket.leave(`analysis:${analysisId}`);
      console.log(`[WebSocket] ${socket.id} unsubscribed from analysis:${analysisId}`);
    });

    // Unsubscribe from workflow
    socket.on("unsubscribe:workflow", (workflowId: string) => {
      socket.leave(`workflow:${workflowId}`);
      console.log(`[WebSocket] ${socket.id} unsubscribed from workflow:${workflowId}`);
    });

    socket.on("disconnect", () => {
      console.log(`[WebSocket] Client disconnected: ${socket.id}`);
    });
  });

  // Start broadcasting system metrics every 5 seconds
  setInterval(() => {
    broadcastSystemMetrics();
  }, 5000);

  // Start broadcasting agent status every 10 seconds
  setInterval(() => {
    broadcastAgentStatus();
  }, 10000);

  console.log("[WebSocket] Socket.IO initialized");
  return io;
}

export function getIO(): SocketIOServer | null {
  return io;
}

// Broadcast system metrics to dashboard
export function broadcastSystemMetrics() {
  if (!io) return;

  const metrics = {
    timestamp: new Date().toISOString(),
    cpu: {
      usage: Math.random() * 100,
      cores: 8,
    },
    memory: {
      used: 12.5 + Math.random() * 3,
      total: 16,
      percentage: 78 + Math.random() * 10,
    },
    storage: {
      used: 4.2 + Math.random() * 0.5,
      total: 5,
      percentage: 84 + Math.random() * 5,
    },
    network: {
      inbound: Math.random() * 1000,
      outbound: Math.random() * 500,
    },
    activeAgents: Math.floor(200 + Math.random() * 50),
    totalRequests: Math.floor(10000 + Math.random() * 1000),
  };

  io.to("dashboard").emit("system-metrics", metrics);
}

// Broadcast agent status updates
export function broadcastAgentStatus() {
  if (!io) return;

  const agentUpdates = Array.from({ length: 5 }, (_, i) => ({
    id: Math.floor(Math.random() * 250),
    status: Math.random() > 0.3 ? "active" : "idle",
    lastActivity: new Date().toISOString(),
    tasksCompleted: Math.floor(Math.random() * 100),
    performance: 85 + Math.random() * 15,
  }));

  io.to("agents").emit("agent-status-update", agentUpdates);
}

// Broadcast chat message to specific chat room
export function broadcastChatMessage(chatId: string, message: any) {
  if (!io) return;
  io.to(`chat-${chatId}`).emit("chat-message", message);
}

// Broadcast S-7 workspace update
export function broadcastS7WorkspaceUpdate(workspaceId: string, update: any) {
  if (!io) return;
  io.to(`s7-workspace-${workspaceId}`).emit("workspace-update", update);
}

// Broadcast fine-tuning progress
export function broadcastFineTuningProgress(jobId: string, progress: any) {
  if (!io) return;
  io.emit("fine-tuning-progress", { jobId, ...progress });
}

// Real-time collaboration for S-7 tests
export function broadcastS7SessionUpdate(sessionId: string, update: any) {
  if (!io) return;
  io.to(`s7-session-${sessionId}`).emit("s7-session-update", update);
}

// Real-time leaderboard updates
export function broadcastLeaderboardUpdate(update: any) {
  if (!io) return;
  io.emit("leaderboard-update", update);
}

// User presence tracking
export function broadcastUserPresence(userId: number, status: "online" | "offline") {
  if (!io) return;
  io.emit("user-presence", { userId, status, timestamp: new Date().toISOString() });
}

// Collaborative agent orchestration
export function broadcastAgentPoolUpdate(poolId: string, update: any) {
  if (!io) return;
  io.to(`agent-pool-${poolId}`).emit("agent-pool-update", update);
}

// Real-time notifications
export function broadcastNotification(userId: number, notification: any) {
  if (!io) return;
  io.emit(`notification-${userId}`, notification);
}

// ============================================================================
// TRUE ASI SYSTEM - REAL-TIME EVENTS
// ============================================================================

/**
 * Emit metric update event for revenue tracking
 */
export function emitMetricUpdate(analysisId: string, metrics: any) {
  if (!io) return;
  
  io.to(`analysis:${analysisId}`).emit('metric:update', {
    analysisId,
    metrics,
    timestamp: Date.now(),
  });
  
  console.log('[WebSocket] Emitted metric:update for analysis:', analysisId);
}

/**
 * Emit execution progress event for automation workflows
 */
export function emitExecutionProgress(workflowId: string, progress: any) {
  if (!io) return;
  
  io.to(`workflow:${workflowId}`).emit('execution:progress', {
    workflowId,
    ...progress,
    timestamp: Date.now(),
  });
  
  console.log('[WebSocket] Emitted execution:progress for workflow:', workflowId);
}

/**
 * Emit analysis complete event (broadcast to all)
 */
export function emitAnalysisComplete(analysisId: string, companyName: string, result: any) {
  if (!io) return;
  
  // Broadcast to all connected clients
  io.emit('analysis:complete', {
    analysisId,
    companyName,
    result,
    timestamp: Date.now(),
  });
  
  console.log('[WebSocket] Emitted analysis:complete for:', companyName);
}

/**
 * Emit user-specific notification
 */
export function emitUserNotification(userId: string, notification: any) {
  if (!io) return;
  
  io.to(`user:${userId}`).emit('notification:new', {
    ...notification,
    timestamp: Date.now(),
  });
  
  console.log('[WebSocket] Emitted notification:new for user:', userId);
}

/**
 * Broadcast system-wide notification to all users
 */
export function broadcastSystemNotification(notification: any) {
  if (!io) return;
  
  io.emit('notification:broadcast', {
    ...notification,
    timestamp: Date.now(),
  });
  
  console.log('[WebSocket] Broadcasted system notification');
}
