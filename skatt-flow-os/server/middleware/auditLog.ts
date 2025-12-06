import { TRPCError } from "@trpc/server";
import { nanoid } from "nanoid";
import * as db from "../db";

// ============================================================================
// AUDIT LOGGING MIDDLEWARE
// Logs all write operations for compliance with Bokføringsloven
// ============================================================================

export interface AuditLogEntry {
  userId: number;
  companyId?: number;
  action: string;
  entityType: string;
  entityId?: number;
  oldValue?: Record<string, unknown>;
  newValue?: Record<string, unknown>;
  ipAddress?: string;
  userAgent?: string;
  correlationId: string;
  timestamp: Date;
}

// In-memory buffer for batch logging (production should use a queue)
const logBuffer: AuditLogEntry[] = [];
const FLUSH_INTERVAL = 5000; // 5 seconds
const MAX_BUFFER_SIZE = 100;

// Flush buffer periodically
setInterval(async () => {
  if (logBuffer.length > 0) {
    await flushLogs();
  }
}, FLUSH_INTERVAL);

async function flushLogs(): Promise<void> {
  const logsToFlush = logBuffer.splice(0, logBuffer.length);
  
  for (const log of logsToFlush) {
    try {
      await db.createApiLog({
        userId: log.userId,
        companyId: log.companyId,
        endpoint: `${log.action}:${log.entityType}`,
        method: "AUDIT",
        statusCode: 200,
        correlationId: log.correlationId,
        durationMs: 0,
      });
    } catch (error) {
      console.error("[AuditLog] Failed to persist log:", error);
    }
  }
}

/**
 * Log an audit event
 */
export function logAudit(entry: Omit<AuditLogEntry, "correlationId" | "timestamp">): string {
  const correlationId = nanoid(16);
  
  const fullEntry: AuditLogEntry = {
    ...entry,
    correlationId,
    timestamp: new Date(),
  };

  logBuffer.push(fullEntry);

  // Flush if buffer is full
  if (logBuffer.length >= MAX_BUFFER_SIZE) {
    flushLogs().catch(console.error);
  }

  console.log(`[AuditLog] ${entry.action} ${entry.entityType} by user ${entry.userId}`, {
    entityId: entry.entityId,
    companyId: entry.companyId,
    correlationId,
  });

  return correlationId;
}

/**
 * Audit log decorator for tRPC mutations
 */
export function withAuditLog<T extends Record<string, unknown>>(
  action: string,
  entityType: string,
  getEntityId?: (input: T) => number | undefined,
  getCompanyId?: (input: T) => number | undefined
) {
  return function auditMiddleware<TContext extends { user?: { id: number } }>(
    opts: { ctx: TContext; input: T; next: () => Promise<unknown> }
  ) {
    const { ctx, input, next } = opts;

    if (!ctx.user) {
      throw new TRPCError({ code: "UNAUTHORIZED" });
    }

    const entityId = getEntityId ? getEntityId(input) : undefined;
    const companyId = getCompanyId ? getCompanyId(input) : undefined;

    // Log before execution
    const correlationId = logAudit({
      userId: ctx.user.id,
      companyId,
      action,
      entityType,
      entityId,
      newValue: input as Record<string, unknown>,
    });

    // Execute the mutation
    return next().then((result) => {
      console.log(`[AuditLog] ${action} ${entityType} completed`, { correlationId });
      return result;
    }).catch((error) => {
      console.error(`[AuditLog] ${action} ${entityType} failed`, { correlationId, error });
      throw error;
    });
  };
}

/**
 * Check if user has access to a company
 */
export async function checkCompanyAccess(
  userId: number,
  companyId: number,
  requiredRoles: Array<"OWNER" | "ADMIN" | "ACCOUNTANT" | "VIEWER"> = ["OWNER", "ADMIN", "ACCOUNTANT", "VIEWER"]
): Promise<boolean> {
  const access = await db.getUserCompanyAccess(userId, companyId);
  
  if (!access) {
    return false;
  }

  return requiredRoles.includes(access.accessRole as "OWNER" | "ADMIN" | "ACCOUNTANT" | "VIEWER");
}

/**
 * Middleware to verify company access
 */
export async function requireCompanyAccess(
  userId: number,
  companyId: number,
  requiredRoles: Array<"OWNER" | "ADMIN" | "ACCOUNTANT" | "VIEWER"> = ["OWNER", "ADMIN", "ACCOUNTANT", "VIEWER"]
): Promise<void> {
  const hasAccess = await checkCompanyAccess(userId, companyId, requiredRoles);
  
  if (!hasAccess) {
    throw new TRPCError({
      code: "FORBIDDEN",
      message: "Du har ikke tilgang til dette selskapet",
    });
  }
}

/**
 * Log a security event (failed access, suspicious activity)
 */
export function logSecurityEvent(
  event: "ACCESS_DENIED" | "INVALID_TOKEN" | "RATE_LIMIT" | "SUSPICIOUS_ACTIVITY",
  details: {
    userId?: number;
    ipAddress?: string;
    endpoint?: string;
    reason?: string;
  }
): void {
  console.warn(`[SecurityEvent] ${event}`, details);
  
  // In production, this should alert security team
  logAudit({
    userId: details.userId || 0,
    action: "SECURITY_EVENT",
    entityType: event,
    newValue: details as Record<string, unknown>,
  });
}
