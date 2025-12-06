/**
 * Redis Connection Test
 * Validates Redis credentials by testing connection
 */

import { describe, it, expect } from 'vitest';
import Redis from 'ioredis';

describe('Redis Connection', () => {
  it('should connect to Redis with provided credentials', async () => {
    const redis = new Redis({
      host: process.env.REDIS_HOST || 'chief-grouper-44799.upstash.io',
      port: parseInt(process.env.REDIS_PORT || '6379'),
      password: process.env.REDIS_PASSWORD,
      tls: {}, // Upstash requires TLS
      maxRetriesPerRequest: 3,
      retryStrategy: (times) => {
        if (times > 3) return null;
        return Math.min(times * 50, 2000);
      },
    });

    try {
      // Test connection with PING
      const result = await redis.ping();
      expect(result).toBe('PONG');

      // Test SET operation
      await redis.set('test_key', 'test_value', 'EX', 10);
      
      // Test GET operation
      const value = await redis.get('test_key');
      expect(value).toBe('test_value');

      // Cleanup
      await redis.del('test_key');
      
      console.log('✅ Redis connection successful');
    } catch (error) {
      console.error('❌ Redis connection failed:', error);
      throw error;
    } finally {
      await redis.quit();
    }
  }, 10000); // 10 second timeout
});
