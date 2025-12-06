/**
 * Firecrawl API Test
 * Validates Firecrawl API key by testing web scraping
 */

import { describe, it, expect } from 'vitest';
import axios from 'axios';

describe('Firecrawl API', () => {
  it('should validate Firecrawl API key with scrape request', async () => {
    const apiKey = process.env.FIRECRAWL_API_KEY;
    
    if (!apiKey) {
      throw new Error('FIRECRAWL_API_KEY not set');
    }

    try {
      // Test with a simple scrape request
      const response = await axios.post(
        'https://api.firecrawl.dev/v1/scrape',
        {
          url: 'https://example.com',
          formats: ['markdown'],
        },
        {
          headers: {
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type': 'application/json',
          },
          timeout: 10000,
        }
      );

      expect(response.status).toBe(200);
      expect(response.data).toBeDefined();
      expect(response.data.success).toBe(true);
      
      console.log('✅ Firecrawl API key valid');
      console.log('   Scraped content length:', response.data.data?.markdown?.length || 0);
    } catch (error: any) {
      if (error.response?.status === 401) {
        throw new Error('Invalid Firecrawl API key');
      }
      if (error.response?.status === 402) {
        console.log('⚠️  Firecrawl API key valid but quota exceeded');
        return; // Key is valid, just out of credits
      }
      console.error('❌ Firecrawl API test failed:', error.message);
      throw error;
    }
  }, 15000); // 15 second timeout
});
