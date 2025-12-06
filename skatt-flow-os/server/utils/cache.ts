// ============================================================================
// IN-MEMORY CACHE WITH TTL
// Simple caching for frequently accessed data
// ============================================================================

interface CacheEntry<T> {
  value: T;
  expiresAt: number;
}

class MemoryCache {
  private cache: Map<string, CacheEntry<unknown>> = new Map();
  private cleanupInterval: NodeJS.Timeout | null = null;

  constructor(cleanupIntervalMs: number = 60000) {
    // Periodic cleanup of expired entries
    this.cleanupInterval = setInterval(() => this.cleanup(), cleanupIntervalMs);
  }

  /**
   * Get a value from cache
   */
  get<T>(key: string): T | undefined {
    const entry = this.cache.get(key);
    if (!entry) return undefined;

    if (Date.now() > entry.expiresAt) {
      this.cache.delete(key);
      return undefined;
    }

    return entry.value as T;
  }

  /**
   * Set a value in cache with TTL (in seconds)
   */
  set<T>(key: string, value: T, ttlSeconds: number): void {
    this.cache.set(key, {
      value,
      expiresAt: Date.now() + ttlSeconds * 1000,
    });
  }

  /**
   * Delete a specific key
   */
  delete(key: string): boolean {
    return this.cache.delete(key);
  }

  /**
   * Delete all keys matching a pattern
   */
  deletePattern(pattern: string): number {
    let deleted = 0;
    const regex = new RegExp(pattern);
    
    for (const key of Array.from(this.cache.keys())) {
      if (regex.test(key)) {
        this.cache.delete(key);
        deleted++;
      }
    }
    
    return deleted;
  }

  /**
   * Clear all cache entries
   */
  clear(): void {
    this.cache.clear();
  }

  /**
   * Get cache statistics
   */
  stats(): { size: number; keys: string[] } {
    return {
      size: this.cache.size,
      keys: Array.from(this.cache.keys()),
    };
  }

  /**
   * Cleanup expired entries
   */
  private cleanup(): void {
    const now = Date.now();
    for (const [key, entry] of Array.from(this.cache.entries())) {
      if (now > entry.expiresAt) {
        this.cache.delete(key);
      }
    }
  }

  /**
   * Stop the cleanup interval
   */
  destroy(): void {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
      this.cleanupInterval = null;
    }
    this.clear();
  }
}

// Singleton cache instance
export const cache = new MemoryCache();

// Cache key generators
export const cacheKeys = {
  company: (id: number) => `company:${id}`,
  companyList: () => `companies:list`,
  forvaltProfile: (orgNumber: string) => `forvalt:${orgNumber}`,
  ledgerEntries: (companyId: number, page: number) => `ledger:${companyId}:${page}`,
  filings: (companyId: number) => `filings:${companyId}`,
  documents: (companyId: number, page: number) => `docs:${companyId}:${page}`,
  dashboardStats: () => `dashboard:stats`,
};

// Cache TTL constants (in seconds)
export const cacheTTL = {
  short: 60,           // 1 minute
  medium: 300,         // 5 minutes
  long: 3600,          // 1 hour
  forvalt: 86400,      // 24 hours (Forvalt data changes rarely)
};

/**
 * Decorator-style cache wrapper for async functions
 */
export async function withCache<T>(
  key: string,
  ttlSeconds: number,
  fn: () => Promise<T>
): Promise<T> {
  const cached = cache.get<T>(key);
  if (cached !== undefined) {
    return cached;
  }

  const result = await fn();
  cache.set(key, result, ttlSeconds);
  return result;
}

/**
 * Invalidate cache entries related to a company
 */
export function invalidateCompanyCache(companyId: number): void {
  cache.deletePattern(`^company:${companyId}`);
  cache.deletePattern(`^ledger:${companyId}`);
  cache.deletePattern(`^filings:${companyId}`);
  cache.deletePattern(`^docs:${companyId}`);
  cache.delete(cacheKeys.companyList());
  cache.delete(cacheKeys.dashboardStats());
}
