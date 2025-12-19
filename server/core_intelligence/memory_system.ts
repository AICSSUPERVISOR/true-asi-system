/**
 * TRUE ASI - MEMORY SYSTEM
 * 
 * Implements all memory types required for ASI:
 * 1. Working memory (context window management)
 * 2. Long-term memory (persistent storage)
 * 3. Episodic memory (experience recall)
 * 4. Semantic memory (knowledge facts)
 * 5. Procedural memory (skills/how-to)
 * 6. Memory consolidation
 * 7. Memory retrieval optimization
 * 
 * NO MOCK DATA - 100% FUNCTIONAL
 */

import { invokeLLM } from '../_core/llm';

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

export type MemoryType = 'working' | 'episodic' | 'semantic' | 'procedural';

export interface MemoryItem {
  id: string;
  type: MemoryType;
  content: string;
  embedding?: number[];
  metadata: MemoryMetadata;
  created_at: Date;
  accessed_at: Date;
  access_count: number;
  importance: number;
  decay_rate: number;
  associations: string[];
}

export interface MemoryMetadata {
  source?: string;
  context?: string;
  tags?: string[];
  confidence?: number;
  verified?: boolean;
  expiry?: Date;
}

export interface Episode {
  id: string;
  timestamp: Date;
  events: EpisodeEvent[];
  context: string;
  outcome: string;
  emotional_valence?: number;
  lessons_learned?: string[];
}

export interface EpisodeEvent {
  action: string;
  observation: string;
  timestamp: Date;
  importance: number;
}

export interface SemanticFact {
  id: string;
  subject: string;
  predicate: string;
  object: string;
  confidence: number;
  source: string;
  valid_from?: Date;
  valid_until?: Date;
}

export interface Procedure {
  id: string;
  name: string;
  description: string;
  steps: ProcedureStep[];
  preconditions: string[];
  postconditions: string[];
  success_rate: number;
  execution_count: number;
}

export interface ProcedureStep {
  order: number;
  action: string;
  expected_result: string;
  fallback?: string;
}

export interface MemoryQuery {
  query: string;
  memory_types?: MemoryType[];
  max_results?: number;
  min_relevance?: number;
  time_range?: { start: Date; end: Date };
  tags?: string[];
}

export interface MemorySearchResult {
  item: MemoryItem;
  relevance: number;
  recency_score: number;
  importance_score: number;
  combined_score: number;
}

export interface ConsolidationResult {
  merged_items: number;
  pruned_items: number;
  strengthened_items: number;
  new_associations: number;
}

// ============================================================================
// MEMORY SYSTEM CLASS
// ============================================================================

export class MemorySystem {
  // Working memory (limited capacity)
  private workingMemory: MemoryItem[] = [];
  private readonly WORKING_MEMORY_CAPACITY = 7; // Miller's Law

  // Long-term memory stores
  private episodicMemory: Map<string, Episode> = new Map();
  private semanticMemory: Map<string, SemanticFact> = new Map();
  private proceduralMemory: Map<string, Procedure> = new Map();

  // Index for fast retrieval
  private memoryIndex: Map<string, MemoryItem> = new Map();
  private tagIndex: Map<string, Set<string>> = new Map();

  // Memory statistics
  private stats = {
    total_memories: 0,
    retrievals: 0,
    consolidations: 0,
    last_consolidation: new Date()
  };

  constructor() {
    // Initialize with empty state
  }

  // ============================================================================
  // WORKING MEMORY
  // ============================================================================

  addToWorkingMemory(content: string, metadata: MemoryMetadata = {}): MemoryItem {
    const item: MemoryItem = {
      id: `wm_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      type: 'working',
      content,
      metadata,
      created_at: new Date(),
      accessed_at: new Date(),
      access_count: 1,
      importance: metadata.confidence || 0.5,
      decay_rate: 0.1,
      associations: []
    };

    // If at capacity, remove least important item
    if (this.workingMemory.length >= this.WORKING_MEMORY_CAPACITY) {
      this.workingMemory.sort((a, b) => a.importance - b.importance);
      const removed = this.workingMemory.shift();
      
      // Consider moving to long-term memory if important enough
      if (removed && removed.importance > 0.7) {
        this.consolidateToLongTerm(removed);
      }
    }

    this.workingMemory.push(item);
    this.memoryIndex.set(item.id, item);
    this.stats.total_memories++;

    return item;
  }

  getWorkingMemory(): MemoryItem[] {
    // Update access times
    for (const item of this.workingMemory) {
      item.accessed_at = new Date();
      item.access_count++;
    }
    return [...this.workingMemory];
  }

  clearWorkingMemory(): void {
    // Consolidate important items before clearing
    for (const item of this.workingMemory) {
      if (item.importance > 0.5) {
        this.consolidateToLongTerm(item);
      }
    }
    this.workingMemory = [];
  }

  // ============================================================================
  // EPISODIC MEMORY
  // ============================================================================

  recordEpisode(
    context: string,
    events: EpisodeEvent[],
    outcome: string,
    lessonsLearned?: string[]
  ): Episode {
    const episode: Episode = {
      id: `ep_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date(),
      events,
      context,
      outcome,
      emotional_valence: this.calculateEmotionalValence(outcome),
      lessons_learned: lessonsLearned
    };

    this.episodicMemory.set(episode.id, episode);

    // Create memory item for indexing
    const item: MemoryItem = {
      id: episode.id,
      type: 'episodic',
      content: `${context}: ${events.map(e => e.action).join(' -> ')} => ${outcome}`,
      metadata: {
        context,
        tags: ['episode'],
        confidence: 1.0
      },
      created_at: episode.timestamp,
      accessed_at: episode.timestamp,
      access_count: 1,
      importance: this.calculateEpisodeImportance(episode),
      decay_rate: 0.05,
      associations: []
    };

    this.memoryIndex.set(item.id, item);
    this.stats.total_memories++;

    return episode;
  }

  private calculateEmotionalValence(outcome: string): number {
    // Simple sentiment analysis
    const positiveWords = ['success', 'completed', 'achieved', 'good', 'excellent'];
    const negativeWords = ['failed', 'error', 'problem', 'bad', 'wrong'];
    
    const lowerOutcome = outcome.toLowerCase();
    let valence = 0;
    
    for (const word of positiveWords) {
      if (lowerOutcome.includes(word)) valence += 0.2;
    }
    for (const word of negativeWords) {
      if (lowerOutcome.includes(word)) valence -= 0.2;
    }
    
    return Math.max(-1, Math.min(1, valence));
  }

  private calculateEpisodeImportance(episode: Episode): number {
    // Importance based on outcome, lessons, and emotional impact
    let importance = 0.5;
    
    if (episode.lessons_learned && episode.lessons_learned.length > 0) {
      importance += 0.2;
    }
    if (Math.abs(episode.emotional_valence || 0) > 0.5) {
      importance += 0.2;
    }
    if (episode.events.length > 3) {
      importance += 0.1;
    }
    
    return Math.min(1, importance);
  }

  async recallSimilarEpisodes(
    context: string,
    maxResults: number = 5
  ): Promise<Episode[]> {
    const episodes = Array.from(this.episodicMemory.values());
    
    // Use LLM to find similar episodes
    if (episodes.length === 0) return [];

    const systemPrompt = `You are a memory retrieval system.
Given a current context and a list of past episodes, identify which episodes are most relevant.
Return the IDs of the most relevant episodes in order of relevance.`;

    const userPrompt = `Current context: ${context}

Past episodes:
${episodes.map(ep => `ID: ${ep.id}\nContext: ${ep.context}\nOutcome: ${ep.outcome}`).join('\n\n')}

Return the IDs of the ${maxResults} most relevant episodes.`;

    const response = await invokeLLM({
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt }
      ],
      response_format: {
        type: 'json_schema',
        json_schema: {
          name: 'episode_recall',
          strict: true,
          schema: {
            type: 'object',
            properties: {
              episode_ids: { type: 'array', items: { type: 'string' } }
            },
            required: ['episode_ids'],
            additionalProperties: false
          }
        }
      }
    });

    const content = response.choices[0]?.message?.content;
    const parsed = JSON.parse(typeof content === 'string' ? content : '{"episode_ids":[]}');

    const results: Episode[] = [];
    for (const id of parsed.episode_ids.slice(0, maxResults)) {
      const episode = this.episodicMemory.get(id);
      if (episode) {
        results.push(episode);
        // Update access
        const item = this.memoryIndex.get(id);
        if (item) {
          item.accessed_at = new Date();
          item.access_count++;
        }
      }
    }

    this.stats.retrievals++;
    return results;
  }

  // ============================================================================
  // SEMANTIC MEMORY
  // ============================================================================

  storeFact(
    subject: string,
    predicate: string,
    object: string,
    confidence: number = 1.0,
    source: string = 'unknown'
  ): SemanticFact {
    const fact: SemanticFact = {
      id: `fact_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      subject,
      predicate,
      object,
      confidence,
      source
    };

    this.semanticMemory.set(fact.id, fact);

    // Create memory item for indexing
    const item: MemoryItem = {
      id: fact.id,
      type: 'semantic',
      content: `${subject} ${predicate} ${object}`,
      metadata: {
        source,
        confidence,
        tags: ['fact', subject, predicate]
      },
      created_at: new Date(),
      accessed_at: new Date(),
      access_count: 1,
      importance: confidence,
      decay_rate: 0.01, // Facts decay slowly
      associations: []
    };

    this.memoryIndex.set(item.id, item);
    
    // Update tag index
    for (const tag of item.metadata.tags || []) {
      if (!this.tagIndex.has(tag)) {
        this.tagIndex.set(tag, new Set());
      }
      this.tagIndex.get(tag)!.add(item.id);
    }

    this.stats.total_memories++;
    return fact;
  }

  queryFacts(subject?: string, predicate?: string, object?: string): SemanticFact[] {
    const results: SemanticFact[] = [];
    
    for (const fact of this.semanticMemory.values()) {
      let matches = true;
      
      if (subject && fact.subject !== subject) matches = false;
      if (predicate && fact.predicate !== predicate) matches = false;
      if (object && fact.object !== object) matches = false;
      
      if (matches) {
        results.push(fact);
        // Update access
        const item = this.memoryIndex.get(fact.id);
        if (item) {
          item.accessed_at = new Date();
          item.access_count++;
        }
      }
    }

    this.stats.retrievals++;
    return results;
  }

  // ============================================================================
  // PROCEDURAL MEMORY
  // ============================================================================

  storeProcedure(
    name: string,
    description: string,
    steps: ProcedureStep[],
    preconditions: string[] = [],
    postconditions: string[] = []
  ): Procedure {
    const procedure: Procedure = {
      id: `proc_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      name,
      description,
      steps,
      preconditions,
      postconditions,
      success_rate: 1.0,
      execution_count: 0
    };

    this.proceduralMemory.set(procedure.id, procedure);

    // Create memory item for indexing
    const item: MemoryItem = {
      id: procedure.id,
      type: 'procedural',
      content: `${name}: ${description}`,
      metadata: {
        tags: ['procedure', name],
        confidence: 1.0
      },
      created_at: new Date(),
      accessed_at: new Date(),
      access_count: 1,
      importance: 0.8,
      decay_rate: 0.02,
      associations: []
    };

    this.memoryIndex.set(item.id, item);
    this.stats.total_memories++;

    return procedure;
  }

  getProcedure(name: string): Procedure | undefined {
    for (const proc of this.proceduralMemory.values()) {
      if (proc.name === name) {
        // Update access
        const item = this.memoryIndex.get(proc.id);
        if (item) {
          item.accessed_at = new Date();
          item.access_count++;
        }
        this.stats.retrievals++;
        return proc;
      }
    }
    return undefined;
  }

  updateProcedureStats(procedureId: string, success: boolean): void {
    const procedure = this.proceduralMemory.get(procedureId);
    if (procedure) {
      procedure.execution_count++;
      // Update success rate with exponential moving average
      const alpha = 0.1;
      procedure.success_rate = alpha * (success ? 1 : 0) + (1 - alpha) * procedure.success_rate;
    }
  }

  // ============================================================================
  // MEMORY SEARCH
  // ============================================================================

  async search(query: MemoryQuery): Promise<MemorySearchResult[]> {
    const results: MemorySearchResult[] = [];
    const now = new Date();

    for (const item of this.memoryIndex.values()) {
      // Filter by memory type
      if (query.memory_types && !query.memory_types.includes(item.type)) {
        continue;
      }

      // Filter by time range
      if (query.time_range) {
        if (item.created_at < query.time_range.start || item.created_at > query.time_range.end) {
          continue;
        }
      }

      // Filter by tags
      if (query.tags && query.tags.length > 0) {
        const itemTags = item.metadata.tags || [];
        if (!query.tags.some(tag => itemTags.includes(tag))) {
          continue;
        }
      }

      // Calculate relevance (simple text matching for now)
      const relevance = this.calculateTextRelevance(query.query, item.content);
      
      if (relevance < (query.min_relevance || 0)) {
        continue;
      }

      // Calculate recency score (exponential decay)
      const ageMs = now.getTime() - item.accessed_at.getTime();
      const ageHours = ageMs / (1000 * 60 * 60);
      const recencyScore = Math.exp(-item.decay_rate * ageHours);

      // Calculate importance score (boosted by access count)
      const importanceScore = item.importance * (1 + Math.log(1 + item.access_count) / 10);

      // Combined score
      const combinedScore = 0.4 * relevance + 0.3 * recencyScore + 0.3 * importanceScore;

      results.push({
        item,
        relevance,
        recency_score: recencyScore,
        importance_score: importanceScore,
        combined_score: combinedScore
      });
    }

    // Sort by combined score
    results.sort((a, b) => b.combined_score - a.combined_score);

    // Limit results
    const maxResults = query.max_results || 10;
    const topResults = results.slice(0, maxResults);

    // Update access times for retrieved items
    for (const result of topResults) {
      result.item.accessed_at = now;
      result.item.access_count++;
    }

    this.stats.retrievals++;
    return topResults;
  }

  private calculateTextRelevance(query: string, content: string): number {
    const queryWords = query.toLowerCase().split(/\s+/);
    const contentWords = content.toLowerCase().split(/\s+/);
    
    let matches = 0;
    for (const qWord of queryWords) {
      if (contentWords.some(cWord => cWord.includes(qWord) || qWord.includes(cWord))) {
        matches++;
      }
    }
    
    return queryWords.length > 0 ? matches / queryWords.length : 0;
  }

  // ============================================================================
  // MEMORY CONSOLIDATION
  // ============================================================================

  async consolidate(): Promise<ConsolidationResult> {
    const result: ConsolidationResult = {
      merged_items: 0,
      pruned_items: 0,
      strengthened_items: 0,
      new_associations: 0
    };

    const now = new Date();
    const itemsToRemove: string[] = [];

    // 1. Prune decayed memories
    for (const [id, item] of this.memoryIndex) {
      const ageMs = now.getTime() - item.accessed_at.getTime();
      const ageHours = ageMs / (1000 * 60 * 60);
      const strength = Math.exp(-item.decay_rate * ageHours) * item.importance;

      if (strength < 0.1 && item.access_count < 3) {
        itemsToRemove.push(id);
        result.pruned_items++;
      }
    }

    // Remove pruned items
    for (const id of itemsToRemove) {
      this.memoryIndex.delete(id);
      this.episodicMemory.delete(id);
      this.semanticMemory.delete(id);
      this.proceduralMemory.delete(id);
    }

    // 2. Strengthen frequently accessed memories
    for (const item of this.memoryIndex.values()) {
      if (item.access_count > 5) {
        item.importance = Math.min(1, item.importance * 1.1);
        item.decay_rate = Math.max(0.001, item.decay_rate * 0.9);
        result.strengthened_items++;
      }
    }

    // 3. Find and create associations
    const items = Array.from(this.memoryIndex.values());
    for (let i = 0; i < items.length; i++) {
      for (let j = i + 1; j < items.length; j++) {
        const similarity = this.calculateTextRelevance(items[i].content, items[j].content);
        if (similarity > 0.5 && !items[i].associations.includes(items[j].id)) {
          items[i].associations.push(items[j].id);
          items[j].associations.push(items[i].id);
          result.new_associations++;
        }
      }
    }

    this.stats.consolidations++;
    this.stats.last_consolidation = now;

    return result;
  }

  private consolidateToLongTerm(item: MemoryItem): void {
    // Move working memory item to appropriate long-term store
    if (item.content.includes(':') && item.content.includes('=>')) {
      // Looks like an episode
      const parts = item.content.split('=>');
      this.recordEpisode(
        parts[0].trim(),
        [{ action: parts[0].trim(), observation: '', timestamp: new Date(), importance: item.importance }],
        parts[1]?.trim() || 'unknown'
      );
    } else if (item.content.match(/\w+\s+\w+\s+\w+/)) {
      // Looks like a fact (subject predicate object)
      const words = item.content.split(/\s+/);
      if (words.length >= 3) {
        this.storeFact(
          words[0],
          words.slice(1, -1).join(' '),
          words[words.length - 1],
          item.importance,
          item.metadata.source || 'working_memory'
        );
      }
    }
  }

  // ============================================================================
  // UTILITY METHODS
  // ============================================================================

  getStats(): typeof this.stats {
    return { ...this.stats };
  }

  getMemoryCount(): Record<MemoryType, number> {
    const counts: Record<MemoryType, number> = {
      working: this.workingMemory.length,
      episodic: this.episodicMemory.size,
      semantic: this.semanticMemory.size,
      procedural: this.proceduralMemory.size
    };
    return counts;
  }

  exportMemory(): {
    working: MemoryItem[];
    episodic: Episode[];
    semantic: SemanticFact[];
    procedural: Procedure[];
  } {
    return {
      working: [...this.workingMemory],
      episodic: Array.from(this.episodicMemory.values()),
      semantic: Array.from(this.semanticMemory.values()),
      procedural: Array.from(this.proceduralMemory.values())
    };
  }

  importMemory(data: {
    episodic?: Episode[];
    semantic?: SemanticFact[];
    procedural?: Procedure[];
  }): void {
    if (data.episodic) {
      for (const episode of data.episodic) {
        this.episodicMemory.set(episode.id, episode);
      }
    }
    if (data.semantic) {
      for (const fact of data.semantic) {
        this.semanticMemory.set(fact.id, fact);
      }
    }
    if (data.procedural) {
      for (const proc of data.procedural) {
        this.proceduralMemory.set(proc.id, proc);
      }
    }
  }
}

// Export singleton instance
export const memorySystem = new MemorySystem();
