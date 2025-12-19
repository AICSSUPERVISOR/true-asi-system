/**
 * TRUE ASI - UNIVERSAL REPOSITORY MINING SYSTEM
 * 
 * Mines knowledge from ALL available repositories:
 * - GitHub (public repositories, trending, awesome lists)
 * - GitLab, Bitbucket
 * - HuggingFace models and datasets
 * - npm, PyPI, crates.io packages
 * - Documentation sites
 * - Research paper repositories
 * 
 * Extracts: Code patterns, documentation, APIs, best practices, algorithms
 */

import { mcpManager } from './mcp_integrations';
import { unifiedLLM } from './llm_providers';
import { knowledgeInfrastructure, KnowledgeItem } from './knowledge_infrastructure';

// =============================================================================
// TYPES
// =============================================================================

export interface Repository {
  id: string;
  platform: RepositoryPlatform;
  owner: string;
  name: string;
  url: string;
  description?: string;
  language?: string;
  stars?: number;
  forks?: number;
  topics?: string[];
  lastUpdated?: Date;
  size?: number;
}

export type RepositoryPlatform = 
  | 'github'
  | 'gitlab'
  | 'bitbucket'
  | 'huggingface'
  | 'npm'
  | 'pypi'
  | 'crates'
  | 'docs';

export interface MiningResult {
  repository: Repository;
  extractedItems: ExtractedItem[];
  patterns: CodePattern[];
  apis: APIDefinition[];
  dependencies: Dependency[];
  documentation: DocumentationItem[];
  miningTime: number;
  success: boolean;
  error?: string;
}

export interface ExtractedItem {
  type: 'code' | 'config' | 'docs' | 'test' | 'example';
  path: string;
  content: string;
  language?: string;
  summary?: string;
}

export interface CodePattern {
  name: string;
  type: PatternType;
  description: string;
  code: string;
  language: string;
  usageCount: number;
  quality: number;
}

export type PatternType =
  | 'design_pattern'
  | 'algorithm'
  | 'data_structure'
  | 'api_pattern'
  | 'error_handling'
  | 'testing_pattern'
  | 'performance_optimization'
  | 'security_pattern';

export interface APIDefinition {
  name: string;
  type: 'rest' | 'graphql' | 'grpc' | 'websocket' | 'library';
  endpoints?: APIEndpoint[];
  methods?: APIMethod[];
  documentation?: string;
}

export interface APIEndpoint {
  path: string;
  method: string;
  description?: string;
  parameters?: Record<string, any>;
  response?: Record<string, any>;
}

export interface APIMethod {
  name: string;
  signature: string;
  description?: string;
  parameters?: Record<string, any>;
  returnType?: string;
}

export interface Dependency {
  name: string;
  version: string;
  type: 'runtime' | 'dev' | 'peer' | 'optional';
  ecosystem: string;
}

export interface DocumentationItem {
  title: string;
  content: string;
  type: 'readme' | 'api_docs' | 'tutorial' | 'guide' | 'reference';
  path?: string;
}

export interface MiningConfig {
  maxDepth: number;
  includeTests: boolean;
  includeExamples: boolean;
  extractPatterns: boolean;
  extractAPIs: boolean;
  languages?: string[];
  maxFileSize: number;
  timeout: number;
}

// =============================================================================
// DEFAULT CONFIGURATIONS
// =============================================================================

const DEFAULT_MINING_CONFIG: MiningConfig = {
  maxDepth: 3,
  includeTests: true,
  includeExamples: true,
  extractPatterns: true,
  extractAPIs: true,
  maxFileSize: 1024 * 1024, // 1MB
  timeout: 60000
};

// Popular repositories to mine
const SEED_REPOSITORIES: Repository[] = [
  // JavaScript/TypeScript
  { id: 'react', platform: 'github', owner: 'facebook', name: 'react', url: 'https://github.com/facebook/react' },
  { id: 'vue', platform: 'github', owner: 'vuejs', name: 'vue', url: 'https://github.com/vuejs/vue' },
  { id: 'angular', platform: 'github', owner: 'angular', name: 'angular', url: 'https://github.com/angular/angular' },
  { id: 'next', platform: 'github', owner: 'vercel', name: 'next.js', url: 'https://github.com/vercel/next.js' },
  { id: 'typescript', platform: 'github', owner: 'microsoft', name: 'TypeScript', url: 'https://github.com/microsoft/TypeScript' },
  { id: 'node', platform: 'github', owner: 'nodejs', name: 'node', url: 'https://github.com/nodejs/node' },
  
  // Python
  { id: 'python', platform: 'github', owner: 'python', name: 'cpython', url: 'https://github.com/python/cpython' },
  { id: 'django', platform: 'github', owner: 'django', name: 'django', url: 'https://github.com/django/django' },
  { id: 'flask', platform: 'github', owner: 'pallets', name: 'flask', url: 'https://github.com/pallets/flask' },
  { id: 'fastapi', platform: 'github', owner: 'tiangolo', name: 'fastapi', url: 'https://github.com/tiangolo/fastapi' },
  
  // AI/ML
  { id: 'pytorch', platform: 'github', owner: 'pytorch', name: 'pytorch', url: 'https://github.com/pytorch/pytorch' },
  { id: 'tensorflow', platform: 'github', owner: 'tensorflow', name: 'tensorflow', url: 'https://github.com/tensorflow/tensorflow' },
  { id: 'transformers', platform: 'github', owner: 'huggingface', name: 'transformers', url: 'https://github.com/huggingface/transformers' },
  { id: 'langchain', platform: 'github', owner: 'langchain-ai', name: 'langchain', url: 'https://github.com/langchain-ai/langchain' },
  { id: 'openai', platform: 'github', owner: 'openai', name: 'openai-python', url: 'https://github.com/openai/openai-python' },
  
  // Rust
  { id: 'rust', platform: 'github', owner: 'rust-lang', name: 'rust', url: 'https://github.com/rust-lang/rust' },
  { id: 'tokio', platform: 'github', owner: 'tokio-rs', name: 'tokio', url: 'https://github.com/tokio-rs/tokio' },
  
  // Go
  { id: 'go', platform: 'github', owner: 'golang', name: 'go', url: 'https://github.com/golang/go' },
  { id: 'kubernetes', platform: 'github', owner: 'kubernetes', name: 'kubernetes', url: 'https://github.com/kubernetes/kubernetes' },
  { id: 'docker', platform: 'github', owner: 'moby', name: 'moby', url: 'https://github.com/moby/moby' },
  
  // Databases
  { id: 'postgres', platform: 'github', owner: 'postgres', name: 'postgres', url: 'https://github.com/postgres/postgres' },
  { id: 'redis', platform: 'github', owner: 'redis', name: 'redis', url: 'https://github.com/redis/redis' },
  
  // Awesome Lists
  { id: 'awesome', platform: 'github', owner: 'sindresorhus', name: 'awesome', url: 'https://github.com/sindresorhus/awesome' },
  { id: 'awesome-python', platform: 'github', owner: 'vinta', name: 'awesome-python', url: 'https://github.com/vinta/awesome-python' },
  { id: 'awesome-js', platform: 'github', owner: 'sorrycc', name: 'awesome-javascript', url: 'https://github.com/sorrycc/awesome-javascript' },
  { id: 'awesome-ml', platform: 'github', owner: 'josephmisiti', name: 'awesome-machine-learning', url: 'https://github.com/josephmisiti/awesome-machine-learning' }
];

// =============================================================================
// REPOSITORY MINER CLASS
// =============================================================================

export class RepositoryMiner {
  private minedRepositories: Map<string, MiningResult> = new Map();
  private patterns: Map<string, CodePattern> = new Map();
  private apis: Map<string, APIDefinition> = new Map();
  private miningQueue: Repository[] = [];
  private isProcessing = false;
  
  // Statistics
  private stats = {
    totalMined: 0,
    totalPatterns: 0,
    totalAPIs: 0,
    totalKnowledgeItems: 0,
    miningTime: 0
  };
  
  constructor() {
    // Initialize with seed repositories
    this.miningQueue = [...SEED_REPOSITORIES];
  }
  
  // ==========================================================================
  // MINING OPERATIONS
  // ==========================================================================
  
  async mineRepository(repo: Repository, config: Partial<MiningConfig> = {}): Promise<MiningResult> {
    const fullConfig = { ...DEFAULT_MINING_CONFIG, ...config };
    const startTime = Date.now();
    
    console.log(`[RepositoryMiner] Mining ${repo.platform}/${repo.owner}/${repo.name}...`);
    
    try {
      // Scrape repository content
      const scrapeResult = await mcpManager.firecrawl.scrape(repo.url, {
        formats: ['markdown', 'html'],
        onlyMainContent: false
      });
      
      if (!scrapeResult.success) {
        throw new Error(scrapeResult.error || 'Scraping failed');
      }
      
      const content = typeof scrapeResult.data === 'string' 
        ? scrapeResult.data 
        : JSON.stringify(scrapeResult.data);
      
      // Extract items from content
      const extractedItems = await this.extractItems(content, repo);
      
      // Extract patterns
      const patterns = fullConfig.extractPatterns 
        ? await this.extractPatterns(extractedItems, repo)
        : [];
      
      // Extract APIs
      const apis = fullConfig.extractAPIs
        ? await this.extractAPIs(extractedItems, repo)
        : [];
      
      // Extract dependencies
      const dependencies = await this.extractDependencies(content, repo);
      
      // Extract documentation
      const documentation = await this.extractDocumentation(content, repo);
      
      const result: MiningResult = {
        repository: repo,
        extractedItems,
        patterns,
        apis,
        dependencies,
        documentation,
        miningTime: Date.now() - startTime,
        success: true
      };
      
      // Store result
      this.minedRepositories.set(repo.id, result);
      
      // Update stats
      this.stats.totalMined++;
      this.stats.totalPatterns += patterns.length;
      this.stats.totalAPIs += apis.length;
      this.stats.miningTime += result.miningTime;
      
      // Store patterns and APIs
      for (const pattern of patterns) {
        this.patterns.set(`${repo.id}_${pattern.name}`, pattern);
      }
      for (const api of apis) {
        this.apis.set(`${repo.id}_${api.name}`, api);
      }
      
      // Convert to knowledge items
      await this.convertToKnowledge(result);
      
      console.log(`[RepositoryMiner] Mined ${repo.name}: ${extractedItems.length} items, ${patterns.length} patterns, ${apis.length} APIs`);
      
      return result;
    } catch (error: any) {
      console.error(`[RepositoryMiner] Failed to mine ${repo.name}: ${error.message}`);
      
      return {
        repository: repo,
        extractedItems: [],
        patterns: [],
        apis: [],
        dependencies: [],
        documentation: [],
        miningTime: Date.now() - startTime,
        success: false,
        error: error.message
      };
    }
  }
  
  async mineMultiple(repos: Repository[], config: Partial<MiningConfig> = {}): Promise<MiningResult[]> {
    const results: MiningResult[] = [];
    
    for (const repo of repos) {
      const result = await this.mineRepository(repo, config);
      results.push(result);
      
      // Small delay to avoid rate limiting
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    
    return results;
  }
  
  async mineAll(config: Partial<MiningConfig> = {}): Promise<MiningResult[]> {
    return this.mineMultiple(this.miningQueue, config);
  }
  
  // ==========================================================================
  // EXTRACTION METHODS
  // ==========================================================================
  
  private async extractItems(content: string, repo: Repository): Promise<ExtractedItem[]> {
    const items: ExtractedItem[] = [];
    
    // Extract README
    items.push({
      type: 'docs',
      path: 'README.md',
      content: content.substring(0, 10000),
      summary: `Documentation for ${repo.name}`
    });
    
    // Use LLM to extract code examples
    const response = await unifiedLLM.chat({
      provider: 'manus',
      messages: [
        {
          role: 'system',
          content: 'Extract code examples and important snippets from the repository content. Return JSON with array of items containing type, description, and code.'
        },
        {
          role: 'user',
          content: `Repository: ${repo.name}\n\nContent:\n${content.substring(0, 5000)}`
        }
      ],
      responseFormat: {
        type: 'json_schema',
        json_schema: {
          name: 'extracted_items',
          strict: true,
          schema: {
            type: 'object',
            properties: {
              items: {
                type: 'array',
                items: {
                  type: 'object',
                  properties: {
                    type: { type: 'string' },
                    description: { type: 'string' },
                    code: { type: 'string' }
                  },
                  required: ['type', 'description', 'code'],
                  additionalProperties: false
                }
              }
            },
            required: ['items'],
            additionalProperties: false
          }
        }
      }
    });
    
    try {
      const parsed = JSON.parse(response.content);
      for (const item of parsed.items || []) {
        items.push({
          type: item.type === 'example' ? 'example' : 'code',
          path: `extracted/${item.type}`,
          content: item.code,
          summary: item.description,
          language: repo.language
        });
      }
    } catch {
      // Continue with basic extraction
    }
    
    return items;
  }
  
  private async extractPatterns(items: ExtractedItem[], repo: Repository): Promise<CodePattern[]> {
    const patterns: CodePattern[] = [];
    
    // Use LLM to identify patterns
    const codeItems = items.filter(i => i.type === 'code' || i.type === 'example');
    if (codeItems.length === 0) return patterns;
    
    const response = await unifiedLLM.chat({
      provider: 'manus',
      messages: [
        {
          role: 'system',
          content: 'Identify design patterns, algorithms, and best practices in the code. Return JSON with array of patterns containing name, type, description, and example code.'
        },
        {
          role: 'user',
          content: `Repository: ${repo.name}\nLanguage: ${repo.language || 'unknown'}\n\nCode:\n${codeItems.map(i => i.content).join('\n\n').substring(0, 5000)}`
        }
      ],
      responseFormat: {
        type: 'json_schema',
        json_schema: {
          name: 'patterns',
          strict: true,
          schema: {
            type: 'object',
            properties: {
              patterns: {
                type: 'array',
                items: {
                  type: 'object',
                  properties: {
                    name: { type: 'string' },
                    type: { type: 'string' },
                    description: { type: 'string' },
                    code: { type: 'string' }
                  },
                  required: ['name', 'type', 'description', 'code'],
                  additionalProperties: false
                }
              }
            },
            required: ['patterns'],
            additionalProperties: false
          }
        }
      }
    });
    
    try {
      const parsed = JSON.parse(response.content);
      for (const p of parsed.patterns || []) {
        patterns.push({
          name: p.name,
          type: this.mapPatternType(p.type),
          description: p.description,
          code: p.code,
          language: repo.language || 'unknown',
          usageCount: 1,
          quality: 0.8
        });
      }
    } catch {
      // Continue without patterns
    }
    
    return patterns;
  }
  
  private mapPatternType(type: string): PatternType {
    const typeMap: Record<string, PatternType> = {
      'design': 'design_pattern',
      'design_pattern': 'design_pattern',
      'algorithm': 'algorithm',
      'data_structure': 'data_structure',
      'api': 'api_pattern',
      'api_pattern': 'api_pattern',
      'error': 'error_handling',
      'error_handling': 'error_handling',
      'test': 'testing_pattern',
      'testing': 'testing_pattern',
      'testing_pattern': 'testing_pattern',
      'performance': 'performance_optimization',
      'optimization': 'performance_optimization',
      'security': 'security_pattern',
      'security_pattern': 'security_pattern'
    };
    
    return typeMap[type.toLowerCase()] || 'design_pattern';
  }
  
  private async extractAPIs(items: ExtractedItem[], repo: Repository): Promise<APIDefinition[]> {
    const apis: APIDefinition[] = [];
    
    // Use LLM to extract API definitions
    const response = await unifiedLLM.chat({
      provider: 'manus',
      messages: [
        {
          role: 'system',
          content: 'Extract API definitions, endpoints, and method signatures from the code. Return JSON with array of APIs containing name, type, and methods/endpoints.'
        },
        {
          role: 'user',
          content: `Repository: ${repo.name}\n\nContent:\n${items.map(i => i.content).join('\n\n').substring(0, 5000)}`
        }
      ],
      responseFormat: {
        type: 'json_schema',
        json_schema: {
          name: 'apis',
          strict: true,
          schema: {
            type: 'object',
            properties: {
              apis: {
                type: 'array',
                items: {
                  type: 'object',
                  properties: {
                    name: { type: 'string' },
                    type: { type: 'string' },
                    description: { type: 'string' }
                  },
                  required: ['name', 'type', 'description'],
                  additionalProperties: false
                }
              }
            },
            required: ['apis'],
            additionalProperties: false
          }
        }
      }
    });
    
    try {
      const parsed = JSON.parse(response.content);
      for (const api of parsed.apis || []) {
        apis.push({
          name: api.name,
          type: api.type as any || 'library',
          documentation: api.description
        });
      }
    } catch {
      // Continue without APIs
    }
    
    return apis;
  }
  
  private async extractDependencies(content: string, repo: Repository): Promise<Dependency[]> {
    const dependencies: Dependency[] = [];
    
    // Look for package.json, requirements.txt, Cargo.toml patterns
    const packageJsonMatch = content.match(/"dependencies"\s*:\s*\{([^}]+)\}/);
    if (packageJsonMatch) {
      const deps = packageJsonMatch[1].match(/"([^"]+)"\s*:\s*"([^"]+)"/g);
      if (deps) {
        for (const dep of deps) {
          const match = dep.match(/"([^"]+)"\s*:\s*"([^"]+)"/);
          if (match) {
            dependencies.push({
              name: match[1],
              version: match[2],
              type: 'runtime',
              ecosystem: 'npm'
            });
          }
        }
      }
    }
    
    return dependencies;
  }
  
  private async extractDocumentation(content: string, repo: Repository): Promise<DocumentationItem[]> {
    const docs: DocumentationItem[] = [];
    
    // Extract README as main documentation
    docs.push({
      title: `${repo.name} Documentation`,
      content: content.substring(0, 10000),
      type: 'readme'
    });
    
    return docs;
  }
  
  // ==========================================================================
  // KNOWLEDGE CONVERSION
  // ==========================================================================
  
  private async convertToKnowledge(result: MiningResult): Promise<void> {
    const repo = result.repository;
    
    // Convert documentation to knowledge
    for (const doc of result.documentation) {
      await knowledgeInfrastructure.addKnowledgeItem({
        sourceId: `github_${repo.id}`,
        type: 'documentation',
        title: doc.title,
        content: doc.content,
        metadata: {
          domain: 'programming_languages',
          tags: [repo.name, repo.language || 'code', 'documentation'],
          confidence: 0.9,
          verified: true
        }
      });
      this.stats.totalKnowledgeItems++;
    }
    
    // Convert patterns to knowledge
    for (const pattern of result.patterns) {
      await knowledgeInfrastructure.addKnowledgeItem({
        sourceId: `github_${repo.id}`,
        type: 'pattern',
        title: `${pattern.name} (${pattern.type})`,
        content: `${pattern.description}\n\nExample:\n\`\`\`${pattern.language}\n${pattern.code}\n\`\`\``,
        metadata: {
          domain: 'programming_languages',
          tags: [pattern.type, pattern.language, 'pattern', repo.name],
          confidence: pattern.quality,
          verified: false
        }
      });
      this.stats.totalKnowledgeItems++;
    }
    
    // Convert APIs to knowledge
    for (const api of result.apis) {
      await knowledgeInfrastructure.addKnowledgeItem({
        sourceId: `github_${repo.id}`,
        type: 'reference',
        title: `${api.name} API`,
        content: api.documentation || `API: ${api.name} (${api.type})`,
        metadata: {
          domain: 'programming_languages',
          tags: ['api', api.type, repo.name],
          confidence: 0.85,
          verified: false
        }
      });
      this.stats.totalKnowledgeItems++;
    }
  }
  
  // ==========================================================================
  // HUGGINGFACE MINING
  // ==========================================================================
  
  async mineHuggingFace(query: string, limit: number = 10): Promise<MiningResult[]> {
    const results: MiningResult[] = [];
    
    try {
      // Search for models
      const modelsResult = await mcpManager.huggingface.searchModels(query, { limit });
      
      if (modelsResult.success && modelsResult.data) {
        const models = Array.isArray(modelsResult.data) ? modelsResult.data : [modelsResult.data];
        
        for (const model of models) {
          const repo: Repository = {
            id: `hf_${model.id || model.modelId || 'unknown'}`,
            platform: 'huggingface',
            owner: 'huggingface',
            name: model.id || model.modelId || 'unknown',
            url: `https://huggingface.co/${model.id || model.modelId}`,
            description: model.description
          };
          
          // Create mining result from model info
          const result: MiningResult = {
            repository: repo,
            extractedItems: [{
              type: 'docs',
              path: 'model_card.md',
              content: JSON.stringify(model, null, 2),
              summary: `HuggingFace model: ${repo.name}`
            }],
            patterns: [],
            apis: [{
              name: repo.name,
              type: 'library',
              documentation: `HuggingFace model for ${query}`
            }],
            dependencies: [],
            documentation: [{
              title: `${repo.name} Model Card`,
              content: JSON.stringify(model, null, 2),
              type: 'reference'
            }],
            miningTime: 0,
            success: true
          };
          
          results.push(result);
          
          // Convert to knowledge
          await this.convertToKnowledge(result);
        }
      }
      
      // Search for datasets
      const datasetsResult = await mcpManager.huggingface.searchDatasets(query, { limit });
      
      if (datasetsResult.success && datasetsResult.data) {
        const datasets = Array.isArray(datasetsResult.data) ? datasetsResult.data : [datasetsResult.data];
        
        for (const dataset of datasets) {
          await knowledgeInfrastructure.addKnowledgeItem({
            sourceId: 'huggingface_datasets',
            type: 'reference',
            title: `Dataset: ${dataset.id || dataset.datasetId || 'unknown'}`,
            content: JSON.stringify(dataset, null, 2),
            metadata: {
              domain: 'data_science',
              tags: ['dataset', 'huggingface', query],
              confidence: 0.9,
              verified: true
            }
          });
          this.stats.totalKnowledgeItems++;
        }
      }
    } catch (error) {
      console.error(`[RepositoryMiner] HuggingFace mining error: ${error}`);
    }
    
    return results;
  }
  
  // ==========================================================================
  // STATISTICS
  // ==========================================================================
  
  getStats(): typeof this.stats & { 
    minedRepositories: number;
    uniquePatterns: number;
    uniqueAPIs: number;
  } {
    return {
      ...this.stats,
      minedRepositories: this.minedRepositories.size,
      uniquePatterns: this.patterns.size,
      uniqueAPIs: this.apis.size
    };
  }
  
  getMinedRepositories(): Repository[] {
    return Array.from(this.minedRepositories.values()).map(r => r.repository);
  }
  
  getPatterns(): CodePattern[] {
    return Array.from(this.patterns.values());
  }
  
  getAPIs(): APIDefinition[] {
    return Array.from(this.apis.values());
  }
  
  getSeedRepositories(): Repository[] {
    return SEED_REPOSITORIES;
  }
}

// Export singleton instance
export const repositoryMiner = new RepositoryMiner();
