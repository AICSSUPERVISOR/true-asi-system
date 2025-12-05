/**
 * Simple in-memory cache for API responses
 * In production, this should use Redis or similar
 */

interface CacheEntry<T> {
  data: T;
  expiresAt: number;
}

const cache = new Map<string, CacheEntry<any>>();

/**
 * Get cached value
 */
export async function cacheGet<T>(key: string): Promise<T | null> {
  const entry = cache.get(key);
  
  if (!entry) {
    return null;
  }
  
  // Check if expired
  if (Date.now() > entry.expiresAt) {
    cache.delete(key);
    return null;
  }
  
  return entry.data as T;
}

/**
 * Set cached value
 */
export async function cacheSet<T>(key: string, data: T, ttlSeconds: number): Promise<void> {
  const expiresAt = Date.now() + (ttlSeconds * 1000);
  
  cache.set(key, {
    data,
    expiresAt
  });
}

/**
 * Delete cached value
 */
export async function cacheDelete(key: string): Promise<void> {
  cache.delete(key);
}

/**
 * Clear all cache
 */
export async function cacheClear(): Promise<void> {
  cache.clear();
}

/**
 * Get cache stats
 */
export function cacheStats() {
  let totalSize = 0;
  let expiredCount = 0;
  const now = Date.now();
  
  cache.forEach((entry, key) => {
    totalSize++;
    if (now > entry.expiresAt) {
      expiredCount++;
    }
  });
  
  return {
    totalEntries: totalSize,
    expiredEntries: expiredCount,
    activeEntries: totalSize - expiredCount
  };
}

// Clean up expired entries every 5 minutes
setInterval(() => {
  const now = Date.now();
  let cleanedCount = 0;
  
  cache.forEach((entry, key) => {
    if (now > entry.expiresAt) {
      cache.delete(key);
      cleanedCount++;
    }
  });
  
  if (cleanedCount > 0) {
    console.log(`[Cache] Cleaned up ${cleanedCount} expired entries`);
  }
}, 5 * 60 * 1000);
