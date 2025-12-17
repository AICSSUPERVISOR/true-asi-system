import { Response } from "express";
import { SkattFlowAgentV2, getSkattFlowAgentV2, AgentMessage } from "../agents/skattFlowAgentV2";

// ============================================================================
// STREAMING CHAT IMPLEMENTATION
// Server-Sent Events (SSE) for real-time AI responses
// ============================================================================

export interface StreamMessage {
  type: "start" | "content" | "tool_call" | "tool_result" | "end" | "error";
  content?: string;
  toolName?: string;
  toolArgs?: Record<string, unknown>;
  toolResult?: unknown;
  error?: string;
  metadata?: {
    tokensUsed?: number;
    processingTime?: number;
    actions?: unknown[];
  };
}

export interface ChatStreamOptions {
  companyId: number;
  userId: number;
  message: string;
  conversationHistory?: AgentMessage[];
  enableTools?: boolean;
}

/**
 * Stream chat response using Server-Sent Events
 */
export async function streamChatResponse(
  res: Response,
  options: ChatStreamOptions
): Promise<void> {
  const startTime = Date.now();
  
  // Set SSE headers
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Accel-Buffering", "no");

  // Helper to send SSE message
  const sendEvent = (message: StreamMessage) => {
    res.write(`data: ${JSON.stringify(message)}\n\n`);
  };

  try {
    // Send start event
    sendEvent({ type: "start" });

    // Get agent instance
    const agent = getSkattFlowAgentV2();

    // Call agent chat method
    const response = await agent.chat(
      options.message,
      { companyId: options.companyId },
      options.conversationHistory,
      options.userId
    );

    // Stream tool results if any
    if (response.toolResults && response.toolResults.length > 0) {
      for (const toolResult of response.toolResults) {
        sendEvent({
          type: "tool_call",
          toolName: toolResult.name,
        });
        sendEvent({
          type: "tool_result",
          toolName: toolResult.name,
          toolResult: toolResult.result,
        });
      }
    }

    // Stream the response content
    const content = response.message;
    await streamContent(content, sendEvent);

    // Send end event with metadata
    sendEvent({
      type: "end",
      metadata: {
        processingTime: Date.now() - startTime,
        actions: response.actions,
      },
    });

  } catch (error) {
    console.error("[ChatStream] Error:", error);
    sendEvent({
      type: "error",
      error: error instanceof Error ? error.message : "Unknown error",
    });
  } finally {
    res.end();
  }
}

/**
 * Stream content word by word with delays for natural feel
 */
async function streamContent(
  content: string,
  sendEvent: (msg: StreamMessage) => void
): Promise<void> {
  // Split into chunks (sentences for more natural streaming)
  const chunks = content.match(/[^.!?]+[.!?]+|[^.!?]+$/g) || [content];
  
  for (const chunk of chunks) {
    // Split chunk into words
    const words = chunk.split(/(\s+)/);
    
    for (const word of words) {
      if (word.trim()) {
        sendEvent({ type: "content", content: word });
        // Small delay for natural feel
        await new Promise((resolve) => setTimeout(resolve, 15));
      } else if (word) {
        sendEvent({ type: "content", content: word });
      }
    }
  }
}

/**
 * Create SSE endpoint handler
 */
export function createChatStreamHandler() {
  return async (req: { body: ChatStreamOptions }, res: Response) => {
    await streamChatResponse(res, req.body);
  };
}

// ============================================================================
// CHAT ROOM MANAGEMENT (for multi-user sessions)
// ============================================================================

export interface ChatRoom {
  companyId: number;
  users: Set<string>;
  history: AgentMessage[];
}

// In-memory chat rooms (in production, use Redis)
const chatRooms = new Map<number, ChatRoom>();

export function getChatRoom(companyId: number): ChatRoom {
  if (!chatRooms.has(companyId)) {
    chatRooms.set(companyId, {
      companyId,
      users: new Set(),
      history: [],
    });
  }
  return chatRooms.get(companyId)!;
}

export function addMessageToHistory(
  companyId: number,
  role: "user" | "assistant",
  content: string
): void {
  const room = getChatRoom(companyId);
  room.history.push({
    role,
    content,
  });

  // Keep only last 100 messages
  if (room.history.length > 100) {
    room.history = room.history.slice(-100);
  }
}

export function getConversationHistory(
  companyId: number,
  limit: number = 20
): AgentMessage[] {
  const room = getChatRoom(companyId);
  return room.history.slice(-limit);
}
