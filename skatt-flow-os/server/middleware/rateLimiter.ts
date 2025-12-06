// ============================================================================
// RATE LIMITING MIDDLEWARE
// Protects endpoints from abuse and ensures fair usage
// ============================================================================

interface RateLimitEntry {
  count: number;
  resetAt: number;
}

interface RateLimitConfig {
  windowMs: number;      // Time window in milliseconds
  maxRequests: number;   // Max requests per window
  message?: string;      // Custom error message
}

class RateLimiter {
  private limits: Map<string, RateLimitEntry> = new Map();
  private cleanupInterval: NodeJS.Timeout;

  constructor() {
    // Cleanup expired entries every minute
    this.cleanupInterval = setInterval(() => this.cleanup(), 60000);
  }

  /**
   * Check if a request should be rate limited
   * @returns true if request is allowed, false if rate limited
   */
  check(key: string, config: RateLimitConfig): { allowed: boolean; remaining: number; resetAt: number } {
    const now = Date.now();
    const entry = this.limits.get(key);

    if (!entry || now >= entry.resetAt) {
      // New window
      const newEntry: RateLimitEntry = {
        count: 1,
        resetAt: now + config.windowMs,
      };
      this.limits.set(key, newEntry);
      return {
        allowed: true,
        remaining: config.maxRequests - 1,
        resetAt: newEntry.resetAt,
      };
    }

    if (entry.count >= config.maxRequests) {
      // Rate limited
      return {
        allowed: false,
        remaining: 0,
        resetAt: entry.resetAt,
      };
    }

    // Increment count
    entry.count++;
    return {
      allowed: true,
      remaining: config.maxRequests - entry.count,
      resetAt: entry.resetAt,
    };
  }

  /**
   * Reset rate limit for a specific key
   */
  reset(key: string): void {
    this.limits.delete(key);
  }

  /**
   * Cleanup expired entries
   */
  private cleanup(): void {
    const now = Date.now();
    for (const [key, entry] of Array.from(this.limits.entries())) {
      if (now >= entry.resetAt) {
        this.limits.delete(key);
      }
    }
  }

  /**
   * Destroy the rate limiter
   */
  destroy(): void {
    clearInterval(this.cleanupInterval);
    this.limits.clear();
  }
}

// Singleton instance
export const rateLimiter = new RateLimiter();

// ============================================================================
// PREDEFINED RATE LIMIT CONFIGURATIONS
// ============================================================================

export const rateLimitConfigs = {
  // Standard API endpoints
  standard: {
    windowMs: 60 * 1000,    // 1 minute
    maxRequests: 100,
    message: "For mange forespørsler. Prøv igjen om et minutt.",
  },

  // AI/LLM endpoints (more expensive)
  ai: {
    windowMs: 60 * 1000,    // 1 minute
    maxRequests: 20,
    message: "AI-grense nådd. Prøv igjen om et minutt.",
  },

  // File upload endpoints
  upload: {
    windowMs: 60 * 1000,    // 1 minute
    maxRequests: 10,
    message: "Opplastingsgrense nådd. Prøv igjen om et minutt.",
  },

  // Authentication endpoints (strict)
  auth: {
    windowMs: 15 * 60 * 1000,  // 15 minutes
    maxRequests: 10,
    message: "For mange påloggingsforsøk. Prøv igjen om 15 minutter.",
  },

  // Altinn submission (very strict)
  altinn: {
    windowMs: 60 * 60 * 1000,  // 1 hour
    maxRequests: 5,
    message: "Altinn-grense nådd. Prøv igjen om en time.",
  },
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Generate rate limit key from user ID and endpoint
 */
export function getRateLimitKey(userId: number | string, endpoint: string): string {
  return `${userId}:${endpoint}`;
}

/**
 * Generate rate limit key from IP address
 */
export function getIpRateLimitKey(ip: string, endpoint: string): string {
  return `ip:${ip}:${endpoint}`;
}

/**
 * Check rate limit and throw error if exceeded
 */
export function checkRateLimit(
  key: string,
  config: RateLimitConfig
): { remaining: number; resetAt: number } {
  const result = rateLimiter.check(key, config);

  if (!result.allowed) {
    const error = new Error(config.message || "Rate limit exceeded");
    (error as Error & { code: string }).code = "TOO_MANY_REQUESTS";
    throw error;
  }

  return {
    remaining: result.remaining,
    resetAt: result.resetAt,
  };
}

// ============================================================================
// TRPC MIDDLEWARE INTEGRATION
// ============================================================================

import { TRPCError } from "@trpc/server";

/**
 * Create a tRPC middleware for rate limiting
 */
export function createRateLimitMiddleware(config: RateLimitConfig) {
  return async function rateLimitMiddleware({
    ctx,
    next,
    path,
  }: {
    ctx: { user?: { id: number } | null };
    next: () => Promise<unknown>;
    path: string;
  }) {
    const userId = ctx.user?.id || "anonymous";
    const key = getRateLimitKey(userId, path);

    const result = rateLimiter.check(key, config);

    if (!result.allowed) {
      throw new TRPCError({
        code: "TOO_MANY_REQUESTS",
        message: config.message || "Rate limit exceeded",
      });
    }

    return next();
  };
}

// ============================================================================
// EXPRESS MIDDLEWARE (for non-tRPC routes)
// ============================================================================

import type { Request, Response, NextFunction } from "express";

/**
 * Create Express middleware for rate limiting
 */
export function createExpressRateLimiter(config: RateLimitConfig) {
  return (req: Request, res: Response, next: NextFunction) => {
    const ip = req.ip || req.socket.remoteAddress || "unknown";
    const key = getIpRateLimitKey(ip, req.path);

    const result = rateLimiter.check(key, config);

    // Set rate limit headers
    res.setHeader("X-RateLimit-Limit", config.maxRequests);
    res.setHeader("X-RateLimit-Remaining", result.remaining);
    res.setHeader("X-RateLimit-Reset", Math.ceil(result.resetAt / 1000));

    if (!result.allowed) {
      res.status(429).json({
        error: "Too Many Requests",
        message: config.message || "Rate limit exceeded",
        retryAfter: Math.ceil((result.resetAt - Date.now()) / 1000),
      });
      return;
    }

    next();
  };
}
