/**
 * Redis Cache Helper
 * 
 * Provides caching layer for Forvalt.no results to improve performance
 * and reduce load on Forvalt.no servers
 */

import Redis from 'ioredis';

// Redis client singleton
let redisClient: Redis | null = null;

/**
 * Get or create Redis client
 * 
 * Note: Redis is optional. If connection fails, caching is disabled
 * and the system continues to work without caching.
 */
function getRedisClient(): Redis | null {
  if (redisClient) {
    return redisClient;
  }

  try {
    // Try to connect to Redis (optional service)
    redisClient = new Redis({
      host: process.env.REDIS_HOST || 'localhost',
      port: parseInt(process.env.REDIS_PORT || '6379'),
      password: process.env.REDIS_PASSWORD,
      retryStrategy: (times) => {
        // Stop retrying after 3 attempts
        if (times > 3) {
          console.log('[Redis] Max retry attempts reached, disabling cache');
          return null;
        }
        return Math.min(times * 100, 3000);
      },
      lazyConnect: true, // Don't connect immediately
    });

    // Handle connection errors gracefully
    redisClient.on('error', (err) => {
      console.log('[Redis] Connection error (cache disabled):', err.message);
      redisClient = null;
    });

    redisClient.on('connect', () => {
      console.log('[Redis] Connected successfully');
    });

    // Try to connect
    redisClient.connect().catch(() => {
      console.log('[Redis] Failed to connect, disabling cache');
      redisClient = null;
    });

    return redisClient;
  } catch (error) {
    console.log('[Redis] Initialization failed, disabling cache');
    return null;
  }
}

/**
 * Get cached value
 * 
 * @param key Cache key
 * @returns Cached value or null if not found/error
 */
export async function getCached<T>(key: string): Promise<T | null> {
  const client = getRedisClient();
  if (!client) return null;

  try {
    const value = await client.get(key);
    if (!value) return null;

    return JSON.parse(value) as T;
  } catch (error) {
    console.log('[Redis] Get error:', error);
    return null;
  }
}

/**
 * Set cached value with TTL
 * 
 * @param key Cache key
 * @param value Value to cache
 * @param ttlSeconds Time to live in seconds (default: 24 hours)
 */
export async function setCached<T>(
  key: string,
  value: T,
  ttlSeconds: number = 86400 // 24 hours default
): Promise<void> {
  const client = getRedisClient();
  if (!client) return;

  try {
    await client.setex(key, ttlSeconds, JSON.stringify(value));
  } catch (error) {
    console.log('[Redis] Set error:', error);
  }
}

/**
 * Delete cached value
 * 
 * @param key Cache key
 */
export async function deleteCached(key: string): Promise<void> {
  const client = getRedisClient();
  if (!client) return;

  try {
    await client.del(key);
  } catch (error) {
    console.log('[Redis] Delete error:', error);
  }
}

/**
 * Delete all cached values matching pattern
 * 
 * @param pattern Key pattern (e.g., "forvalt:*")
 */
export async function deleteCachedPattern(pattern: string): Promise<void> {
  const client = getRedisClient();
  if (!client) return;

  try {
    const keys = await client.keys(pattern);
    if (keys.length > 0) {
      await client.del(...keys);
    }
  } catch (error) {
    console.log('[Redis] Delete pattern error:', error);
  }
}

/**
 * Generate cache key for Forvalt.no results
 * 
 * @param orgNumber Organization number
 * @returns Cache key
 */
export function getForvaltCacheKey(orgNumber: string): string {
  return `forvalt:${orgNumber}`;
}

/**
 * Get cached Forvalt.no result
 * 
 * @param orgNumber Organization number
 * @returns Cached result or null
 */
export async function getCachedForvaltData(orgNumber: string) {
  return getCached(getForvaltCacheKey(orgNumber));
}

/**
 * Cache Forvalt.no result
 * 
 * @param orgNumber Organization number
 * @param data Forvalt data to cache
 * @param ttlSeconds Time to live in seconds (default: 24 hours)
 */
export async function setCachedForvaltData(
  orgNumber: string,
  data: any,
  ttlSeconds: number = 86400 // 24 hours
): Promise<void> {
  return setCached(getForvaltCacheKey(orgNumber), data, ttlSeconds);
}

/**
 * Invalidate cached Forvalt.no result
 * 
 * @param orgNumber Organization number
 */
export async function invalidateForvaltCache(orgNumber: string): Promise<void> {
  return deleteCached(getForvaltCacheKey(orgNumber));
}

/**
 * Invalidate all Forvalt.no cached results
 */
export async function invalidateAllForvaltCache(): Promise<void> {
  return deleteCachedPattern('forvalt:*');
}
