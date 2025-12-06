/**
 * Upstash Vector Search Integration
 * 
 * Provides semantic search capabilities for the 6.54TB AWS S3 knowledge base
 * using Upstash Vector database for similarity search and competitor analysis.
 */

import { Index } from '@upstash/vector';

// Initialize Upstash Vector client
const vectorIndex = new Index({
  url: 'https://polished-monster-32312-us1-vector.upstash.io',
  token: 'ABoFMHBvbGlzaGVkLW1vbnN0ZXItMzIzMTItdXMxYWRtaW5NR1ZtTnpRMlltRXRNVGhoTVMwME1HTmpMV0ptWVdVdFptTTRNRFExTW1Zek9XUmw=',
});

export interface VectorSearchResult {
  id: string;
  score: number;
  metadata: {
    title: string;
    content: string;
    source: string;
    category: string;
    timestamp: string;
  };
}

export interface CompanyEmbedding {
  companyId: string;
  orgNumber: string;
  name: string;
  industry: string;
  embedding: number[];
  metadata: {
    revenue?: number;
    employees?: number;
    creditRating?: string;
    description?: string;
  };
}

/**
 * Generate embedding for text using simple hash-based approach
 * In production, use OpenAI embeddings or similar
 */
function generateEmbedding(text: string): number[] {
  // Simple 1536-dimensional embedding (matching OpenAI ada-002)
  const embedding = new Array(1536).fill(0);
  
  // Hash-based simple embedding generation
  for (let i = 0; i < text.length; i++) {
    const charCode = text.charCodeAt(i);
    const index = (charCode * i) % 1536;
    embedding[index] += charCode / 1000;
  }
  
  // Normalize
  const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
  return embedding.map(val => val / (magnitude || 1));
}

/**
 * Index company data for semantic search
 */
export async function indexCompanyData(company: CompanyEmbedding): Promise<void> {
  try {
    const text = `${company.name} ${company.industry} ${company.metadata.description || ''}`;
    const embedding = generateEmbedding(text);
    
    await vectorIndex.upsert({
      id: company.companyId,
      vector: embedding,
      metadata: {
        orgNumber: company.orgNumber,
        name: company.name,
        industry: company.industry,
        ...company.metadata,
      },
    });
    
    console.log(`[Vector] Indexed company: ${company.name}`);
  } catch (error) {
    console.error('[Vector] Error indexing company:', error);
    throw error;
  }
}

/**
 * Search for similar companies based on description
 */
export async function searchSimilarCompanies(
  query: string,
  topK: number = 10
): Promise<VectorSearchResult[]> {
  try {
    const queryEmbedding = generateEmbedding(query);
    
    const results = await vectorIndex.query({
      vector: queryEmbedding,
      topK,
      includeMetadata: true,
    });
    
    return results.map((result: any) => ({
      id: result.id,
      score: result.score,
      metadata: {
        title: result.metadata.name || 'Unknown',
        content: result.metadata.description || '',
        source: `Company ${result.metadata.orgNumber}`,
        category: result.metadata.industry || 'Unknown',
        timestamp: new Date().toISOString(),
      },
    }));
  } catch (error) {
    console.error('[Vector] Error searching similar companies:', error);
    return [];
  }
}

/**
 * Find competitor companies in the same industry
 */
export async function findCompetitors(
  companyId: string,
  topK: number = 5
): Promise<VectorSearchResult[]> {
  try {
    // Fetch the company's embedding
    const company = await vectorIndex.fetch([companyId]);
    
    if (!company || company.length === 0 || !company[0]?.vector) {
      console.warn(`[Vector] Company ${companyId} not found in index`);
      return [];
    }
    
    // Search for similar companies (excluding the company itself)
    const results = await vectorIndex.query({
      vector: company[0].vector as number[],
      topK: topK + 1, // +1 to account for the company itself
      includeMetadata: true,
    });
    
    // Filter out the company itself
    return results
      .filter((result: any) => result.id !== companyId)
      .slice(0, topK)
      .map((result: any) => ({
        id: result.id,
        score: result.score,
        metadata: {
          title: result.metadata.name || 'Unknown',
          content: result.metadata.description || '',
          source: `Company ${result.metadata.orgNumber}`,
          category: result.metadata.industry || 'Unknown',
          timestamp: new Date().toISOString(),
        },
      }));
  } catch (error) {
    console.error('[Vector] Error finding competitors:', error);
    return [];
  }
}

/**
 * Search knowledge base documents
 */
export async function searchKnowledgeBase(
  query: string,
  category?: string,
  topK: number = 20
): Promise<VectorSearchResult[]> {
  try {
    const queryEmbedding = generateEmbedding(query);
    
    const results = await vectorIndex.query({
      vector: queryEmbedding,
      topK,
      includeMetadata: true,
      filter: category ? `category = '${category}'` : undefined,
    });
    
    return results.map((result: any) => ({
      id: result.id,
      score: result.score,
      metadata: result.metadata || {
        title: 'Unknown',
        content: '',
        source: 'Knowledge Base',
        category: 'General',
        timestamp: new Date().toISOString(),
      },
    }));
  } catch (error) {
    console.error('[Vector] Error searching knowledge base:', error);
    return [];
  }
}

/**
 * Get vector search statistics
 */
export async function getVectorStats(): Promise<{
  totalVectors: number;
  dimensions: number;
  status: string;
}> {
  try {
    const info = await vectorIndex.info();
    
    return {
      totalVectors: info.vectorCount || 0,
      dimensions: info.dimension || 1536,
      status: 'operational',
    };
  } catch (error) {
    console.error('[Vector] Error getting stats:', error);
    return {
      totalVectors: 0,
      dimensions: 1536,
      status: 'error',
    };
  }
}

/**
 * Delete company from vector index
 */
export async function deleteCompanyVector(companyId: string): Promise<void> {
  try {
    await vectorIndex.delete([companyId]);
    console.log(`[Vector] Deleted company: ${companyId}`);
  } catch (error) {
    console.error('[Vector] Error deleting company:', error);
    throw error;
  }
}

/**
 * Batch index multiple companies
 */
export async function batchIndexCompanies(companies: CompanyEmbedding[]): Promise<void> {
  try {
    const vectors = companies.map(company => {
      const text = `${company.name} ${company.industry} ${company.metadata.description || ''}`;
      const embedding = generateEmbedding(text);
      
      return {
        id: company.companyId,
        vector: embedding,
        metadata: {
          orgNumber: company.orgNumber,
          name: company.name,
          industry: company.industry,
          ...company.metadata,
        },
      };
    });
    
    await vectorIndex.upsert(vectors);
    console.log(`[Vector] Batch indexed ${companies.length} companies`);
  } catch (error) {
    console.error('[Vector] Error batch indexing:', error);
    throw error;
  }
}
