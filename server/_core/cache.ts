/**
 * Redis Caching Layer for S-7 Answers
 * Provides <100ms response times with automatic cache invalidation
 */

import Redis from "ioredis";

let redis: Redis | null = null;

// Initialize Redis connection
export function getRedisClient(): Redis {
  if (!redis) {
    redis = new Redis({
      host: process.env.REDIS_HOST || "localhost",
      port: parseInt(process.env.REDIS_PORT || "6379"),
      password: process.env.REDIS_PASSWORD,
      retryStrategy: (times) => {
        const delay = Math.min(times * 50, 2000);
        return delay;
      },
      maxRetriesPerRequest: 3,
    });

    redis.on("error", (err) => {
      console.error("[Redis] Connection error:", err);
    });

    redis.on("connect", () => {
      console.log("[Redis] Connected successfully");
    });
  }

  return redis;
}

// Cache key prefixes
const CACHE_PREFIXES = {
  S7_ANSWER: "s7:answer:",
  S7_EVALUATION: "s7:evaluation:",
  S7_LEADERBOARD: "s7:leaderboard",
  SYSTEM_METRICS: "system:metrics",
};

// Cache TTLs (in seconds)
const CACHE_TTLS = {
  S7_ANSWER: 3600 * 24 * 7, // 7 days (enhanced answers rarely change)
  S7_EVALUATION: 3600 * 24, // 24 hours
  S7_LEADERBOARD: 300, // 5 minutes (frequently updated)
  SYSTEM_METRICS: 60, // 1 minute
};

/**
 * Get cached S-7 answer
 */
export async function getCachedS7Answer(questionNumber: number): Promise<any | null> {
  try {
    const client = getRedisClient();
    const key = `${CACHE_PREFIXES.S7_ANSWER}${questionNumber}`;
    const cached = await client.get(key);
    
    if (cached) {
      return JSON.parse(cached);
    }
    
    return null;
  } catch (error) {
    console.error("[Cache] Error getting S-7 answer:", error);
    return null;
  }
}

/**
 * Set cached S-7 answer
 */
export async function setCachedS7Answer(questionNumber: number, answer: any): Promise<void> {
  try {
    const client = getRedisClient();
    const key = `${CACHE_PREFIXES.S7_ANSWER}${questionNumber}`;
    await client.setex(key, CACHE_TTLS.S7_ANSWER, JSON.stringify(answer));
  } catch (error) {
    console.error("[Cache] Error setting S-7 answer:", error);
  }
}

/**
 * Invalidate S-7 answer cache
 */
export async function invalidateS7Answer(questionNumber: number): Promise<void> {
  try {
    const client = getRedisClient();
    const key = `${CACHE_PREFIXES.S7_ANSWER}${questionNumber}`;
    await client.del(key);
    console.log(`[Cache] Invalidated S-7 answer Q${questionNumber}`);
  } catch (error) {
    console.error("[Cache] Error invalidating S-7 answer:", error);
  }
}

/**
 * Invalidate all S-7 answers
 */
export async function invalidateAllS7Answers(): Promise<void> {
  try {
    const client = getRedisClient();
    const keys = await client.keys(`${CACHE_PREFIXES.S7_ANSWER}*`);
    
    if (keys.length > 0) {
      await client.del(...keys);
      console.log(`[Cache] Invalidated ${keys.length} S-7 answers`);
    }
  } catch (error) {
    console.error("[Cache] Error invalidating all S-7 answers:", error);
  }
}

/**
 * Get cached evaluation
 */
export async function getCachedEvaluation(userId: string, questionNumber: number): Promise<any | null> {
  try {
    const client = getRedisClient();
    const key = `${CACHE_PREFIXES.S7_EVALUATION}${userId}:${questionNumber}`;
    const cached = await client.get(key);
    
    if (cached) {
      return JSON.parse(cached);
    }
    
    return null;
  } catch (error) {
    console.error("[Cache] Error getting evaluation:", error);
    return null;
  }
}

/**
 * Set cached evaluation
 */
export async function setCachedEvaluation(userId: string, questionNumber: number, evaluation: any): Promise<void> {
  try {
    const client = getRedisClient();
    const key = `${CACHE_PREFIXES.S7_EVALUATION}${userId}:${questionNumber}`;
    await client.setex(key, CACHE_TTLS.S7_EVALUATION, JSON.stringify(evaluation));
  } catch (error) {
    console.error("[Cache] Error setting evaluation:", error);
  }
}

/**
 * Get cached leaderboard
 */
export async function getCachedLeaderboard(): Promise<any[] | null> {
  try {
    const client = getRedisClient();
    const cached = await client.get(CACHE_PREFIXES.S7_LEADERBOARD);
    
    if (cached) {
      return JSON.parse(cached);
    }
    
    return null;
  } catch (error) {
    console.error("[Cache] Error getting leaderboard:", error);
    return null;
  }
}

/**
 * Set cached leaderboard
 */
export async function setCachedLeaderboard(leaderboard: any[]): Promise<void> {
  try {
    const client = getRedisClient();
    await client.setex(CACHE_PREFIXES.S7_LEADERBOARD, CACHE_TTLS.S7_LEADERBOARD, JSON.stringify(leaderboard));
  } catch (error) {
    console.error("[Cache] Error setting leaderboard:", error);
  }
}

/**
 * Invalidate leaderboard cache
 */
export async function invalidateLeaderboard(): Promise<void> {
  try {
    const client = getRedisClient();
    await client.del(CACHE_PREFIXES.S7_LEADERBOARD);
    console.log("[Cache] Invalidated leaderboard");
  } catch (error) {
    console.error("[Cache] Error invalidating leaderboard:", error);
  }
}

/**
 * Get cache statistics
 */
export async function getCacheStats(): Promise<{
  connected: boolean;
  keys: number;
  memory: string;
  hits: number;
  misses: number;
}> {
  try {
    const client = getRedisClient();
    const info = await client.info("stats");
    const dbsize = await client.dbsize();
    const memory = await client.info("memory");
    
    // Parse stats
    const hitsMatch = info.match(/keyspace_hits:(\d+)/);
    const missesMatch = info.match(/keyspace_misses:(\d+)/);
    const memoryMatch = memory.match(/used_memory_human:([^\r\n]+)/);
    
    return {
      connected: true,
      keys: dbsize,
      memory: memoryMatch ? memoryMatch[1] : "unknown",
      hits: hitsMatch ? parseInt(hitsMatch[1]) : 0,
      misses: missesMatch ? parseInt(missesMatch[1]) : 0,
    };
  } catch (error) {
    console.error("[Cache] Error getting cache stats:", error);
    return {
      connected: false,
      keys: 0,
      memory: "0B",
      hits: 0,
      misses: 0,
    };
  }
}

/**
 * Warm up cache with all enhanced S-7 answers
 */
export async function warmUpCache(enhancedAnswers: Record<string, any>): Promise<void> {
  try {
    console.log("[Cache] Warming up cache with enhanced S-7 answers...");
    
    for (const [key, answer] of Object.entries(enhancedAnswers)) {
      const questionNumber = parseInt(key.substring(1));
      await setCachedS7Answer(questionNumber, answer);
    }
    
    console.log(`[Cache] Warmed up ${Object.keys(enhancedAnswers).length} S-7 answers`);
  } catch (error) {
    console.error("[Cache] Error warming up cache:", error);
  }
}

// Close Redis connection on process exit
process.on("SIGTERM", async () => {
  if (redis) {
    await redis.quit();
    console.log("[Redis] Connection closed");
  }
});
