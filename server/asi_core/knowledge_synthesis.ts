/**
 * TRUE ASI - KNOWLEDGE SYNTHESIS & INFINITE EXPANSION ENGINE
 * 
 * Synthesizes new knowledge from existing knowledge:
 * - Cross-domain knowledge fusion
 * - Recursive knowledge expansion
 * - Automated insight generation
 * - Knowledge gap identification
 * - Self-improving knowledge base
 * 
 * TARGET: Infinite knowledge expansion through recursive synthesis
 */

import { unifiedLLM } from './llm_providers';
import { knowledgeInfrastructure, KnowledgeItem, KnowledgeItemType } from './knowledge_infrastructure';
import { memorySystem } from './memory_system';

// =============================================================================
// TYPES
// =============================================================================

export interface SynthesisRequest {
  topics: string[];
  depth: 'shallow' | 'medium' | 'deep' | 'exhaustive';
  crossDomain: boolean;
  generateInsights: boolean;
  expandRelated: boolean;
  maxIterations?: number;
}

export interface SynthesisResult {
  synthesizedItems: KnowledgeItem[];
  insights: Insight[];
  connections: KnowledgeConnection[];
  gaps: KnowledgeGap[];
  expansionPaths: ExpansionPath[];
  metrics: SynthesisMetrics;
}

export interface Insight {
  id: string;
  title: string;
  description: string;
  sourceTopics: string[];
  confidence: number;
  novelty: number;
  applicability: string[];
  created: Date;
}

export interface KnowledgeConnection {
  sourceId: string;
  targetId: string;
  connectionType: ConnectionType;
  strength: number;
  description: string;
}

export type ConnectionType =
  | 'causal'
  | 'correlational'
  | 'hierarchical'
  | 'analogical'
  | 'contradictory'
  | 'complementary'
  | 'prerequisite'
  | 'application';

export interface KnowledgeGap {
  id: string;
  topic: string;
  description: string;
  importance: number;
  suggestedSources: string[];
  relatedKnowledge: string[];
}

export interface ExpansionPath {
  id: string;
  startTopic: string;
  endTopic: string;
  intermediateTopics: string[];
  estimatedNewKnowledge: number;
  priority: number;
}

export interface SynthesisMetrics {
  totalSynthesized: number;
  totalInsights: number;
  totalConnections: number;
  totalGaps: number;
  processingTime: number;
  knowledgeGrowthRate: number;
}

// =============================================================================
// SYNTHESIS STRATEGIES
// =============================================================================

type SynthesisStrategy = 
  | 'fusion'           // Combine knowledge from multiple domains
  | 'abstraction'      // Extract higher-level concepts
  | 'specialization'   // Generate domain-specific knowledge
  | 'analogy'          // Find analogies across domains
  | 'contradiction'    // Identify and resolve contradictions
  | 'gap_filling'      // Fill knowledge gaps
  | 'prediction'       // Predict future developments
  | 'application';     // Generate practical applications

// =============================================================================
// KNOWLEDGE SYNTHESIS ENGINE
// =============================================================================

export class KnowledgeSynthesisEngine {
  private synthesisHistory: SynthesisResult[] = [];
  private insights: Map<string, Insight> = new Map();
  private connections: Map<string, KnowledgeConnection> = new Map();
  private gaps: Map<string, KnowledgeGap> = new Map();
  
  // Metrics
  private totalSynthesized = 0;
  private totalInsights = 0;
  private totalConnections = 0;
  
  // ==========================================================================
  // MAIN SYNTHESIS
  // ==========================================================================
  
  async synthesize(request: SynthesisRequest): Promise<SynthesisResult> {
    const startTime = Date.now();
    const maxIterations = request.maxIterations || this.getIterationsForDepth(request.depth);
    
    console.log(`[KnowledgeSynthesis] Starting synthesis for topics: ${request.topics.join(', ')}`);
    
    const result: SynthesisResult = {
      synthesizedItems: [],
      insights: [],
      connections: [],
      gaps: [],
      expansionPaths: [],
      metrics: {
        totalSynthesized: 0,
        totalInsights: 0,
        totalConnections: 0,
        totalGaps: 0,
        processingTime: 0,
        knowledgeGrowthRate: 0
      }
    };
    
    // Phase 1: Gather existing knowledge
    const existingKnowledge = await this.gatherKnowledge(request.topics);
    
    // Phase 2: Apply synthesis strategies
    const strategies = this.selectStrategies(request);
    
    for (let iteration = 0; iteration < maxIterations; iteration++) {
      console.log(`[KnowledgeSynthesis] Iteration ${iteration + 1}/${maxIterations}`);
      
      for (const strategy of strategies) {
        const synthesized = await this.applySynthesisStrategy(
          strategy,
          existingKnowledge,
          request.topics
        );
        
        result.synthesizedItems.push(...synthesized.items);
        result.insights.push(...synthesized.insights);
        result.connections.push(...synthesized.connections);
      }
      
      // Generate insights if requested
      if (request.generateInsights) {
        const newInsights = await this.generateInsights(
          result.synthesizedItems,
          request.topics
        );
        result.insights.push(...newInsights);
      }
      
      // Expand to related topics if requested
      if (request.expandRelated) {
        const expansionPaths = await this.identifyExpansionPaths(
          request.topics,
          result.synthesizedItems
        );
        result.expansionPaths.push(...expansionPaths);
      }
    }
    
    // Phase 3: Identify knowledge gaps
    result.gaps = await this.identifyKnowledgeGaps(request.topics, result.synthesizedItems);
    
    // Phase 4: Cross-domain synthesis
    if (request.crossDomain) {
      const crossDomainItems = await this.crossDomainSynthesis(request.topics);
      result.synthesizedItems.push(...crossDomainItems);
    }
    
    // Update metrics
    result.metrics = {
      totalSynthesized: result.synthesizedItems.length,
      totalInsights: result.insights.length,
      totalConnections: result.connections.length,
      totalGaps: result.gaps.length,
      processingTime: Date.now() - startTime,
      knowledgeGrowthRate: result.synthesizedItems.length / ((Date.now() - startTime) / 1000)
    };
    
    // Store results
    this.synthesisHistory.push(result);
    this.totalSynthesized += result.synthesizedItems.length;
    this.totalInsights += result.insights.length;
    this.totalConnections += result.connections.length;
    
    // Store insights and connections
    for (const insight of result.insights) {
      this.insights.set(insight.id, insight);
    }
    for (const connection of result.connections) {
      this.connections.set(`${connection.sourceId}_${connection.targetId}`, connection);
    }
    for (const gap of result.gaps) {
      this.gaps.set(gap.id, gap);
    }
    
    console.log(`[KnowledgeSynthesis] Completed: ${result.metrics.totalSynthesized} items, ${result.metrics.totalInsights} insights`);
    
    return result;
  }
  
  // ==========================================================================
  // KNOWLEDGE GATHERING
  // ==========================================================================
  
  private async gatherKnowledge(topics: string[]): Promise<KnowledgeItem[]> {
    const items: KnowledgeItem[] = [];
    
    for (const topic of topics) {
      const queryResult = await knowledgeInfrastructure.query({
        query: topic,
        limit: 20
      });
      items.push(...queryResult);
    }
    
    return items;
  }
  
  // ==========================================================================
  // SYNTHESIS STRATEGIES
  // ==========================================================================
  
  private selectStrategies(request: SynthesisRequest): SynthesisStrategy[] {
    const strategies: SynthesisStrategy[] = ['fusion', 'abstraction'];
    
    if (request.crossDomain) {
      strategies.push('analogy');
    }
    
    if (request.depth === 'deep' || request.depth === 'exhaustive') {
      strategies.push('specialization', 'prediction', 'application');
    }
    
    if (request.depth === 'exhaustive') {
      strategies.push('contradiction', 'gap_filling');
    }
    
    return strategies;
  }
  
  private async applySynthesisStrategy(
    strategy: SynthesisStrategy,
    existingKnowledge: KnowledgeItem[],
    topics: string[]
  ): Promise<{
    items: KnowledgeItem[];
    insights: Insight[];
    connections: KnowledgeConnection[];
  }> {
    const result = {
      items: [] as KnowledgeItem[],
      insights: [] as Insight[],
      connections: [] as KnowledgeConnection[]
    };
    
    switch (strategy) {
      case 'fusion':
        result.items = await this.fusionSynthesis(existingKnowledge, topics);
        break;
      case 'abstraction':
        result.items = await this.abstractionSynthesis(existingKnowledge, topics);
        break;
      case 'specialization':
        result.items = await this.specializationSynthesis(existingKnowledge, topics);
        break;
      case 'analogy':
        const analogyResult = await this.analogySynthesis(existingKnowledge, topics);
        result.items = analogyResult.items;
        result.connections = analogyResult.connections;
        break;
      case 'prediction':
        result.items = await this.predictionSynthesis(existingKnowledge, topics);
        break;
      case 'application':
        result.items = await this.applicationSynthesis(existingKnowledge, topics);
        break;
    }
    
    return result;
  }
  
  private async fusionSynthesis(knowledge: KnowledgeItem[], topics: string[]): Promise<KnowledgeItem[]> {
    const items: KnowledgeItem[] = [];
    
    const response = await unifiedLLM.chat({
      provider: 'manus',
      messages: [
        {
          role: 'system',
          content: 'You are a knowledge fusion engine. Combine the given knowledge items to create new, synthesized knowledge that integrates insights from multiple sources. Return JSON with array of synthesized items.'
        },
        {
          role: 'user',
          content: `Topics: ${topics.join(', ')}\n\nExisting Knowledge:\n${knowledge.map(k => `- ${k.title}: ${k.content.substring(0, 200)}`).join('\n')}\n\nSynthesize new knowledge by combining these sources.`
        }
      ],
      responseFormat: {
        type: 'json_schema',
        json_schema: {
          name: 'fusion_result',
          strict: true,
          schema: {
            type: 'object',
            properties: {
              items: {
                type: 'array',
                items: {
                  type: 'object',
                  properties: {
                    title: { type: 'string' },
                    content: { type: 'string' },
                    type: { type: 'string' }
                  },
                  required: ['title', 'content', 'type'],
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
        const newItem = await knowledgeInfrastructure.addKnowledgeItem({
          sourceId: 'llm_synthesis',
          type: (item.type as KnowledgeItemType) || 'insight',
          title: item.title,
          content: item.content,
          metadata: {
            domain: 'general_knowledge',
            tags: [...topics, 'synthesized', 'fusion'],
            confidence: 0.8,
            verified: false
          }
        });
        items.push(newItem);
      }
    } catch {
      // Continue without fusion items
    }
    
    return items;
  }
  
  private async abstractionSynthesis(knowledge: KnowledgeItem[], topics: string[]): Promise<KnowledgeItem[]> {
    const items: KnowledgeItem[] = [];
    
    const response = await unifiedLLM.chat({
      provider: 'manus',
      messages: [
        {
          role: 'system',
          content: 'Extract higher-level concepts and principles from the given knowledge. Identify patterns, generalizations, and abstract frameworks. Return JSON with array of abstracted concepts.'
        },
        {
          role: 'user',
          content: `Topics: ${topics.join(', ')}\n\nKnowledge:\n${knowledge.map(k => `- ${k.title}: ${k.content.substring(0, 200)}`).join('\n')}\n\nExtract abstract concepts and principles.`
        }
      ],
      responseFormat: {
        type: 'json_schema',
        json_schema: {
          name: 'abstraction_result',
          strict: true,
          schema: {
            type: 'object',
            properties: {
              concepts: {
                type: 'array',
                items: {
                  type: 'object',
                  properties: {
                    title: { type: 'string' },
                    description: { type: 'string' },
                    applications: { type: 'array', items: { type: 'string' } }
                  },
                  required: ['title', 'description', 'applications'],
                  additionalProperties: false
                }
              }
            },
            required: ['concepts'],
            additionalProperties: false
          }
        }
      }
    });
    
    try {
      const parsed = JSON.parse(response.content);
      for (const concept of parsed.concepts || []) {
        const newItem = await knowledgeInfrastructure.addKnowledgeItem({
          sourceId: 'llm_synthesis',
          type: 'concept',
          title: concept.title,
          content: `${concept.description}\n\nApplications:\n${concept.applications.map((a: string) => `- ${a}`).join('\n')}`,
          metadata: {
            domain: 'general_knowledge',
            tags: [...topics, 'abstracted', 'concept'],
            confidence: 0.75,
            verified: false
          }
        });
        items.push(newItem);
      }
    } catch {
      // Continue without abstraction items
    }
    
    return items;
  }
  
  private async specializationSynthesis(knowledge: KnowledgeItem[], topics: string[]): Promise<KnowledgeItem[]> {
    const items: KnowledgeItem[] = [];
    
    const response = await unifiedLLM.chat({
      provider: 'manus',
      messages: [
        {
          role: 'system',
          content: 'Generate specialized, domain-specific knowledge from the given general knowledge. Create detailed, practical content for specific use cases. Return JSON with array of specialized items.'
        },
        {
          role: 'user',
          content: `Topics: ${topics.join(', ')}\n\nGeneral Knowledge:\n${knowledge.map(k => `- ${k.title}: ${k.content.substring(0, 200)}`).join('\n')}\n\nGenerate specialized knowledge for practical applications.`
        }
      ],
      responseFormat: {
        type: 'json_schema',
        json_schema: {
          name: 'specialization_result',
          strict: true,
          schema: {
            type: 'object',
            properties: {
              items: {
                type: 'array',
                items: {
                  type: 'object',
                  properties: {
                    title: { type: 'string' },
                    content: { type: 'string' },
                    domain: { type: 'string' }
                  },
                  required: ['title', 'content', 'domain'],
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
        const newItem = await knowledgeInfrastructure.addKnowledgeItem({
          sourceId: 'llm_synthesis',
          type: 'procedure',
          title: item.title,
          content: item.content,
          metadata: {
            domain: item.domain || 'general_knowledge',
            tags: [...topics, 'specialized', item.domain],
            confidence: 0.8,
            verified: false
          }
        });
        items.push(newItem);
      }
    } catch {
      // Continue without specialization items
    }
    
    return items;
  }
  
  private async analogySynthesis(knowledge: KnowledgeItem[], topics: string[]): Promise<{
    items: KnowledgeItem[];
    connections: KnowledgeConnection[];
  }> {
    const items: KnowledgeItem[] = [];
    const connections: KnowledgeConnection[] = [];
    
    const response = await unifiedLLM.chat({
      provider: 'manus',
      messages: [
        {
          role: 'system',
          content: 'Find analogies and parallels between different domains in the given knowledge. Identify structural similarities and transferable patterns. Return JSON with analogies.'
        },
        {
          role: 'user',
          content: `Topics: ${topics.join(', ')}\n\nKnowledge:\n${knowledge.map(k => `- ${k.title}: ${k.content.substring(0, 200)}`).join('\n')}\n\nFind cross-domain analogies.`
        }
      ],
      responseFormat: {
        type: 'json_schema',
        json_schema: {
          name: 'analogy_result',
          strict: true,
          schema: {
            type: 'object',
            properties: {
              analogies: {
                type: 'array',
                items: {
                  type: 'object',
                  properties: {
                    source: { type: 'string' },
                    target: { type: 'string' },
                    description: { type: 'string' },
                    transferableInsights: { type: 'array', items: { type: 'string' } }
                  },
                  required: ['source', 'target', 'description', 'transferableInsights'],
                  additionalProperties: false
                }
              }
            },
            required: ['analogies'],
            additionalProperties: false
          }
        }
      }
    });
    
    try {
      const parsed = JSON.parse(response.content);
      for (const analogy of parsed.analogies || []) {
        // Create knowledge item for the analogy
        const newItem = await knowledgeInfrastructure.addKnowledgeItem({
          sourceId: 'llm_synthesis',
          type: 'insight',
          title: `Analogy: ${analogy.source} ↔ ${analogy.target}`,
          content: `${analogy.description}\n\nTransferable Insights:\n${analogy.transferableInsights.map((i: string) => `- ${i}`).join('\n')}`,
          metadata: {
            domain: 'general_knowledge',
            tags: ['analogy', analogy.source, analogy.target],
            confidence: 0.7,
            verified: false
          }
        });
        items.push(newItem);
        
        // Create connection
        connections.push({
          sourceId: analogy.source,
          targetId: analogy.target,
          connectionType: 'analogical',
          strength: 0.7,
          description: analogy.description
        });
      }
    } catch {
      // Continue without analogy items
    }
    
    return { items, connections };
  }
  
  private async predictionSynthesis(knowledge: KnowledgeItem[], topics: string[]): Promise<KnowledgeItem[]> {
    const items: KnowledgeItem[] = [];
    
    const response = await unifiedLLM.chat({
      provider: 'manus',
      messages: [
        {
          role: 'system',
          content: 'Based on the given knowledge, predict future developments, trends, and potential breakthroughs. Return JSON with predictions.'
        },
        {
          role: 'user',
          content: `Topics: ${topics.join(', ')}\n\nCurrent Knowledge:\n${knowledge.map(k => `- ${k.title}: ${k.content.substring(0, 200)}`).join('\n')}\n\nPredict future developments.`
        }
      ],
      responseFormat: {
        type: 'json_schema',
        json_schema: {
          name: 'prediction_result',
          strict: true,
          schema: {
            type: 'object',
            properties: {
              predictions: {
                type: 'array',
                items: {
                  type: 'object',
                  properties: {
                    title: { type: 'string' },
                    description: { type: 'string' },
                    timeframe: { type: 'string' },
                    confidence: { type: 'number' }
                  },
                  required: ['title', 'description', 'timeframe', 'confidence'],
                  additionalProperties: false
                }
              }
            },
            required: ['predictions'],
            additionalProperties: false
          }
        }
      }
    });
    
    try {
      const parsed = JSON.parse(response.content);
      for (const prediction of parsed.predictions || []) {
        const newItem = await knowledgeInfrastructure.addKnowledgeItem({
          sourceId: 'llm_synthesis',
          type: 'insight',
          title: `Prediction: ${prediction.title}`,
          content: `${prediction.description}\n\nTimeframe: ${prediction.timeframe}`,
          metadata: {
            domain: 'general_knowledge',
            tags: [...topics, 'prediction', 'future'],
            confidence: prediction.confidence || 0.6,
            verified: false
          }
        });
        items.push(newItem);
      }
    } catch {
      // Continue without prediction items
    }
    
    return items;
  }
  
  private async applicationSynthesis(knowledge: KnowledgeItem[], topics: string[]): Promise<KnowledgeItem[]> {
    const items: KnowledgeItem[] = [];
    
    const response = await unifiedLLM.chat({
      provider: 'manus',
      messages: [
        {
          role: 'system',
          content: 'Generate practical applications and use cases from the given knowledge. Create actionable guides and implementation strategies. Return JSON with applications.'
        },
        {
          role: 'user',
          content: `Topics: ${topics.join(', ')}\n\nKnowledge:\n${knowledge.map(k => `- ${k.title}: ${k.content.substring(0, 200)}`).join('\n')}\n\nGenerate practical applications.`
        }
      ],
      responseFormat: {
        type: 'json_schema',
        json_schema: {
          name: 'application_result',
          strict: true,
          schema: {
            type: 'object',
            properties: {
              applications: {
                type: 'array',
                items: {
                  type: 'object',
                  properties: {
                    title: { type: 'string' },
                    description: { type: 'string' },
                    steps: { type: 'array', items: { type: 'string' } },
                    benefits: { type: 'array', items: { type: 'string' } }
                  },
                  required: ['title', 'description', 'steps', 'benefits'],
                  additionalProperties: false
                }
              }
            },
            required: ['applications'],
            additionalProperties: false
          }
        }
      }
    });
    
    try {
      const parsed = JSON.parse(response.content);
      for (const app of parsed.applications || []) {
        const newItem = await knowledgeInfrastructure.addKnowledgeItem({
          sourceId: 'llm_synthesis',
          type: 'procedure',
          title: app.title,
          content: `${app.description}\n\nSteps:\n${app.steps.map((s: string, i: number) => `${i + 1}. ${s}`).join('\n')}\n\nBenefits:\n${app.benefits.map((b: string) => `- ${b}`).join('\n')}`,
          metadata: {
            domain: 'general_knowledge',
            tags: [...topics, 'application', 'practical'],
            confidence: 0.85,
            verified: false
          }
        });
        items.push(newItem);
      }
    } catch {
      // Continue without application items
    }
    
    return items;
  }
  
  // ==========================================================================
  // INSIGHT GENERATION
  // ==========================================================================
  
  private async generateInsights(items: KnowledgeItem[], topics: string[]): Promise<Insight[]> {
    const insights: Insight[] = [];
    
    const response = await unifiedLLM.chat({
      provider: 'manus',
      messages: [
        {
          role: 'system',
          content: 'Generate novel insights by analyzing the given knowledge items. Identify non-obvious patterns, implications, and opportunities. Return JSON with insights.'
        },
        {
          role: 'user',
          content: `Topics: ${topics.join(', ')}\n\nKnowledge Items:\n${items.map(k => `- ${k.title}: ${k.content.substring(0, 200)}`).join('\n')}\n\nGenerate novel insights.`
        }
      ],
      responseFormat: {
        type: 'json_schema',
        json_schema: {
          name: 'insights_result',
          strict: true,
          schema: {
            type: 'object',
            properties: {
              insights: {
                type: 'array',
                items: {
                  type: 'object',
                  properties: {
                    title: { type: 'string' },
                    description: { type: 'string' },
                    novelty: { type: 'number' },
                    applicability: { type: 'array', items: { type: 'string' } }
                  },
                  required: ['title', 'description', 'novelty', 'applicability'],
                  additionalProperties: false
                }
              }
            },
            required: ['insights'],
            additionalProperties: false
          }
        }
      }
    });
    
    try {
      const parsed = JSON.parse(response.content);
      for (const insight of parsed.insights || []) {
        insights.push({
          id: `insight_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
          title: insight.title,
          description: insight.description,
          sourceTopics: topics,
          confidence: 0.75,
          novelty: insight.novelty || 0.7,
          applicability: insight.applicability || [],
          created: new Date()
        });
      }
    } catch {
      // Continue without insights
    }
    
    return insights;
  }
  
  // ==========================================================================
  // KNOWLEDGE GAPS
  // ==========================================================================
  
  private async identifyKnowledgeGaps(topics: string[], synthesized: KnowledgeItem[]): Promise<KnowledgeGap[]> {
    const gaps: KnowledgeGap[] = [];
    
    const response = await unifiedLLM.chat({
      provider: 'manus',
      messages: [
        {
          role: 'system',
          content: 'Identify knowledge gaps and missing information in the given topics. What important aspects are not covered? Return JSON with gaps.'
        },
        {
          role: 'user',
          content: `Topics: ${topics.join(', ')}\n\nExisting Knowledge:\n${synthesized.map(k => `- ${k.title}`).join('\n')}\n\nIdentify knowledge gaps.`
        }
      ],
      responseFormat: {
        type: 'json_schema',
        json_schema: {
          name: 'gaps_result',
          strict: true,
          schema: {
            type: 'object',
            properties: {
              gaps: {
                type: 'array',
                items: {
                  type: 'object',
                  properties: {
                    topic: { type: 'string' },
                    description: { type: 'string' },
                    importance: { type: 'number' },
                    suggestedSources: { type: 'array', items: { type: 'string' } }
                  },
                  required: ['topic', 'description', 'importance', 'suggestedSources'],
                  additionalProperties: false
                }
              }
            },
            required: ['gaps'],
            additionalProperties: false
          }
        }
      }
    });
    
    try {
      const parsed = JSON.parse(response.content);
      for (const gap of parsed.gaps || []) {
        gaps.push({
          id: `gap_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
          topic: gap.topic,
          description: gap.description,
          importance: gap.importance || 0.5,
          suggestedSources: gap.suggestedSources || [],
          relatedKnowledge: topics
        });
      }
    } catch {
      // Continue without gaps
    }
    
    return gaps;
  }
  
  // ==========================================================================
  // EXPANSION PATHS
  // ==========================================================================
  
  private async identifyExpansionPaths(topics: string[], synthesized: KnowledgeItem[]): Promise<ExpansionPath[]> {
    const paths: ExpansionPath[] = [];
    
    const response = await unifiedLLM.chat({
      provider: 'manus',
      messages: [
        {
          role: 'system',
          content: 'Identify paths for expanding knowledge from the given topics. What related areas should be explored? Return JSON with expansion paths.'
        },
        {
          role: 'user',
          content: `Current Topics: ${topics.join(', ')}\n\nSynthesized Knowledge:\n${synthesized.map(k => `- ${k.title}`).join('\n')}\n\nIdentify expansion paths.`
        }
      ],
      responseFormat: {
        type: 'json_schema',
        json_schema: {
          name: 'paths_result',
          strict: true,
          schema: {
            type: 'object',
            properties: {
              paths: {
                type: 'array',
                items: {
                  type: 'object',
                  properties: {
                    startTopic: { type: 'string' },
                    endTopic: { type: 'string' },
                    intermediateTopics: { type: 'array', items: { type: 'string' } },
                    priority: { type: 'number' }
                  },
                  required: ['startTopic', 'endTopic', 'intermediateTopics', 'priority'],
                  additionalProperties: false
                }
              }
            },
            required: ['paths'],
            additionalProperties: false
          }
        }
      }
    });
    
    try {
      const parsed = JSON.parse(response.content);
      for (const path of parsed.paths || []) {
        paths.push({
          id: `path_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
          startTopic: path.startTopic,
          endTopic: path.endTopic,
          intermediateTopics: path.intermediateTopics || [],
          estimatedNewKnowledge: (path.intermediateTopics?.length || 0) * 5 + 10,
          priority: path.priority || 0.5
        });
      }
    } catch {
      // Continue without paths
    }
    
    return paths;
  }
  
  // ==========================================================================
  // CROSS-DOMAIN SYNTHESIS
  // ==========================================================================
  
  private async crossDomainSynthesis(topics: string[]): Promise<KnowledgeItem[]> {
    const items: KnowledgeItem[] = [];
    
    const response = await unifiedLLM.chat({
      provider: 'manus',
      messages: [
        {
          role: 'system',
          content: 'Create cross-domain knowledge by combining insights from different fields. Find unexpected connections and novel applications. Return JSON with cross-domain items.'
        },
        {
          role: 'user',
          content: `Topics to combine: ${topics.join(', ')}\n\nGenerate cross-domain knowledge that bridges these topics.`
        }
      ],
      responseFormat: {
        type: 'json_schema',
        json_schema: {
          name: 'cross_domain_result',
          strict: true,
          schema: {
            type: 'object',
            properties: {
              items: {
                type: 'array',
                items: {
                  type: 'object',
                  properties: {
                    title: { type: 'string' },
                    content: { type: 'string' },
                    domains: { type: 'array', items: { type: 'string' } }
                  },
                  required: ['title', 'content', 'domains'],
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
        const newItem = await knowledgeInfrastructure.addKnowledgeItem({
          sourceId: 'llm_synthesis',
          type: 'insight',
          title: item.title,
          content: item.content,
          metadata: {
            domain: 'general_knowledge',
            tags: [...(item.domains || []), 'cross_domain', 'synthesized'],
            confidence: 0.75,
            verified: false
          }
        });
        items.push(newItem);
      }
    } catch {
      // Continue without cross-domain items
    }
    
    return items;
  }
  
  // ==========================================================================
  // INFINITE EXPANSION
  // ==========================================================================
  
  async infiniteExpansion(seedTopics: string[], maxIterations: number = 10): Promise<SynthesisResult[]> {
    const results: SynthesisResult[] = [];
    let currentTopics = [...seedTopics];
    
    for (let i = 0; i < maxIterations; i++) {
      console.log(`[KnowledgeSynthesis] Infinite expansion iteration ${i + 1}/${maxIterations}`);
      
      // Synthesize knowledge for current topics
      const result = await this.synthesize({
        topics: currentTopics,
        depth: 'medium',
        crossDomain: true,
        generateInsights: true,
        expandRelated: true
      });
      
      results.push(result);
      
      // Get new topics from expansion paths
      const newTopics = result.expansionPaths
        .sort((a, b) => b.priority - a.priority)
        .slice(0, 3)
        .map(p => p.endTopic);
      
      // Add gap topics
      const gapTopics = result.gaps
        .sort((a, b) => b.importance - a.importance)
        .slice(0, 2)
        .map(g => g.topic);
      
      currentTopics = [...new Set([...newTopics, ...gapTopics])];
      
      if (currentTopics.length === 0) {
        console.log('[KnowledgeSynthesis] No more topics to expand');
        break;
      }
    }
    
    return results;
  }
  
  // ==========================================================================
  // UTILITIES
  // ==========================================================================
  
  private getIterationsForDepth(depth: SynthesisRequest['depth']): number {
    switch (depth) {
      case 'shallow': return 1;
      case 'medium': return 3;
      case 'deep': return 5;
      case 'exhaustive': return 10;
      default: return 3;
    }
  }
  
  // ==========================================================================
  // STATISTICS
  // ==========================================================================
  
  getStats(): {
    totalSynthesized: number;
    totalInsights: number;
    totalConnections: number;
    totalGaps: number;
    synthesisHistory: number;
  } {
    return {
      totalSynthesized: this.totalSynthesized,
      totalInsights: this.totalInsights,
      totalConnections: this.totalConnections,
      totalGaps: this.gaps.size,
      synthesisHistory: this.synthesisHistory.length
    };
  }
  
  getInsights(): Insight[] {
    return Array.from(this.insights.values());
  }
  
  getConnections(): KnowledgeConnection[] {
    return Array.from(this.connections.values());
  }
  
  getGaps(): KnowledgeGap[] {
    return Array.from(this.gaps.values());
  }
}

// Export singleton instance
export const knowledgeSynthesis = new KnowledgeSynthesisEngine();
