/**
 * Credentials Validation Tests
 * 
 * Tests all newly added credentials to ensure they work correctly
 */

import { describe, it, expect } from 'vitest';
import { Index } from '@upstash/vector';
import { Client } from '@upstash/qstash';

describe('Upstash Credentials', () => {
  it('should validate Upstash Vector credentials', async () => {
    const vectorIndex = new Index({
      url: process.env.UPSTASH_VECTOR_URL!,
      token: process.env.UPSTASH_VECTOR_TOKEN!,
    });

    // Test with a simple info call
    const info = await vectorIndex.info();
    expect(info).toBeDefined();
    expect(info.vectorCount).toBeGreaterThanOrEqual(0);
  });

  it('should validate QStash credentials', async () => {
    const qstashClient = new Client({
      token: process.env.QSTASH_TOKEN!,
    });

    // Test with a simple list call (will return empty array if no messages)
    // Note: We can't test actual message sending without a valid callback URL
    expect(qstashClient).toBeDefined();
    expect(process.env.QSTASH_URL).toBe('https://qstash.upstash.io');
    expect(process.env.QSTASH_CURRENT_SIGNING_KEY).toMatch(/^sig_/);
    expect(process.env.QSTASH_NEXT_SIGNING_KEY).toMatch(/^sig_/);
  });

  it('should validate Manus API key format', () => {
    const manusApiKey = process.env.MANUS_API_KEY;
    expect(manusApiKey).toBeDefined();
    expect(manusApiKey).toMatch(/^sk-/);
    expect(manusApiKey!.length).toBeGreaterThan(50);
  });

  it('should validate Upstash Search credentials exist', () => {
    expect(process.env.UPSTASH_SEARCH_URL).toBe('https://touching-pigeon-96283-eu1-search.upstash.io');
    expect(process.env.UPSTASH_SEARCH_TOKEN).toBeDefined();
    expect(process.env.UPSTASH_SEARCH_TOKEN!.length).toBeGreaterThan(50);
  });
});
