/**
 * META-LEARNING SYSTEM
 * 
 * Implements continuous self-improvement through:
 * - Learning from past interactions
 * - Adaptive strategy selection
 * - Performance optimization loops
 * - Knowledge transfer across domains
 * 
 * This is a core component of TRUE Artificial Superintelligence.
 */

// Types for meta-learning
interface LearningExperience {
  id: string;
  timestamp: Date;
  taskType: string;
  input: string;
  output: string;
  feedback: number; // -1 to 1
  modelUsed: string;
  latencyMs: number;
  success: boolean;
  metadata: Record<string, unknown>;
}

interface Strategy {
  id: string;
  name: string;
  description: string;
  applicableTasks: string[];
  successRate: number;
  avgLatency: number;
  usageCount: number;
  lastUsed: Date;
  parameters: Record<string, number>;
}

interface PerformanceMetrics {
  taskType: string;
  successRate: number;
  avgLatency: number;
  totalAttempts: number;
  improvements: number;
  regressions: number;
  trend: 'improving' | 'stable' | 'declining';
}

interface KnowledgeTransfer {
  sourceTask: string;
  targetTask: string;
  transferScore: number;
  sharedConcepts: string[];
  adaptations: string[];
}

// Experience memory store
const experienceMemory: LearningExperience[] = [];
const strategyLibrary: Map<string, Strategy> = new Map();
const performanceHistory: Map<string, PerformanceMetrics[]> = new Map();
const knowledgeGraph: Map<string, Set<string>> = new Map();

// Core meta-learning strategies
const META_STRATEGIES: Strategy[] = [
  {
    id: 'chain-of-thought',
    name: 'Chain of Thought',
    description: 'Break complex problems into sequential reasoning steps',
    applicableTasks: ['reasoning', 'math', 'logic', 'planning'],
    successRate: 0.85,
    avgLatency: 2500,
    usageCount: 0,
    lastUsed: new Date(),
    parameters: { depth: 5, branching: 2 }
  },
  {
    id: 'tree-of-thought',
    name: 'Tree of Thought',
    description: 'Explore multiple reasoning paths simultaneously',
    applicableTasks: ['complex-reasoning', 'creative', 'strategy'],
    successRate: 0.88,
    avgLatency: 4000,
    usageCount: 0,
    lastUsed: new Date(),
    parameters: { branches: 3, depth: 4, pruning: 0.3 }
  },
  {
    id: 'self-consistency',
    name: 'Self-Consistency',
    description: 'Generate multiple solutions and select most consistent',
    applicableTasks: ['math', 'logic', 'factual'],
    successRate: 0.92,
    avgLatency: 5000,
    usageCount: 0,
    lastUsed: new Date(),
    parameters: { samples: 5, threshold: 0.7 }
  },
  {
    id: 'retrieval-augmented',
    name: 'Retrieval Augmented Generation',
    description: 'Enhance responses with retrieved knowledge',
    applicableTasks: ['factual', 'research', 'technical'],
    successRate: 0.90,
    avgLatency: 3000,
    usageCount: 0,
    lastUsed: new Date(),
    parameters: { topK: 10, relevanceThreshold: 0.8 }
  },
  {
    id: 'multi-agent-debate',
    name: 'Multi-Agent Debate',
    description: 'Multiple AI agents debate to reach consensus',
    applicableTasks: ['complex-reasoning', 'ethical', 'strategy'],
    successRate: 0.87,
    avgLatency: 8000,
    usageCount: 0,
    lastUsed: new Date(),
    parameters: { agents: 3, rounds: 3 }
  },
  {
    id: 'analogical-reasoning',
    name: 'Analogical Reasoning',
    description: 'Transfer knowledge from similar domains',
    applicableTasks: ['creative', 'problem-solving', 'learning'],
    successRate: 0.82,
    avgLatency: 2000,
    usageCount: 0,
    lastUsed: new Date(),
    parameters: { analogyDepth: 3, abstractionLevel: 2 }
  },
  {
    id: 'decomposition',
    name: 'Problem Decomposition',
    description: 'Break complex problems into simpler sub-problems',
    applicableTasks: ['complex', 'multi-step', 'planning'],
    successRate: 0.89,
    avgLatency: 3500,
    usageCount: 0,
    lastUsed: new Date(),
    parameters: { maxSubproblems: 5, minComplexity: 0.2 }
  },
  {
    id: 'reflection',
    name: 'Self-Reflection',
    description: 'Analyze and improve own outputs',
    applicableTasks: ['writing', 'code', 'analysis'],
    successRate: 0.86,
    avgLatency: 4500,
    usageCount: 0,
    lastUsed: new Date(),
    parameters: { iterations: 2, improvementThreshold: 0.1 }
  }
];

// Initialize strategy library
META_STRATEGIES.forEach(s => strategyLibrary.set(s.id, s));

/**
 * Record a learning experience
 */
export function recordExperience(experience: Omit<LearningExperience, 'id' | 'timestamp'>): string {
  const id = `exp_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  const fullExperience: LearningExperience = {
    ...experience,
    id,
    timestamp: new Date()
  };
  
  experienceMemory.push(fullExperience);
  
  // Update knowledge graph
  updateKnowledgeGraph(fullExperience);
  
  // Update performance metrics
  updatePerformanceMetrics(fullExperience);
  
  // Trigger learning if enough experiences
  if (experienceMemory.length % 10 === 0) {
    triggerLearningCycle();
  }
  
  return id;
}

/**
 * Select optimal strategy for a task
 */
export function selectStrategy(taskType: string, context: Record<string, unknown> = {}): Strategy {
  // Get applicable strategies
  const applicable = Array.from(strategyLibrary.values())
    .filter(s => s.applicableTasks.includes(taskType) || s.applicableTasks.includes('general'));
  
  if (applicable.length === 0) {
    // Return default strategy
    return strategyLibrary.get('chain-of-thought')!;
  }
  
  // Score strategies based on historical performance
  const scored = applicable.map(strategy => {
    const recentExperiences = experienceMemory
      .filter(e => e.taskType === taskType)
      .slice(-100);
    
    // Calculate adaptive score
    let score = strategy.successRate * 0.4;
    score += (1 - strategy.avgLatency / 10000) * 0.2; // Prefer faster
    score += Math.min(strategy.usageCount / 100, 1) * 0.1; // Experience bonus
    
    // Recency bonus
    const hoursSinceUse = (Date.now() - strategy.lastUsed.getTime()) / (1000 * 60 * 60);
    score += Math.max(0, 0.1 - hoursSinceUse / 240); // Decay over 10 days
    
    // Context-specific adjustments
    if (context.complexity === 'high' && strategy.id === 'tree-of-thought') {
      score += 0.15;
    }
    if (context.needsAccuracy && strategy.id === 'self-consistency') {
      score += 0.15;
    }
    if (context.needsSpeed && strategy.avgLatency < 3000) {
      score += 0.1;
    }
    
    return { strategy, score };
  });
  
  // Select best strategy
  scored.sort((a, b) => b.score - a.score);
  const selected = scored[0].strategy;
  
  // Update usage stats
  selected.usageCount++;
  selected.lastUsed = new Date();
  
  return selected;
}

/**
 * Update knowledge graph with new experience
 */
function updateKnowledgeGraph(experience: LearningExperience): void {
  const taskType = experience.taskType;
  
  if (!knowledgeGraph.has(taskType)) {
    knowledgeGraph.set(taskType, new Set());
  }
  
  // Extract concepts from experience
  const concepts = extractConcepts(experience);
  concepts.forEach(concept => {
    knowledgeGraph.get(taskType)!.add(concept);
  });
  
  // Find connections between task types
  for (const [otherTask, otherConcepts] of Array.from(knowledgeGraph.entries())) {
    if (otherTask !== taskType) {
      const intersection = new Set([...concepts].filter(c => otherConcepts.has(c)));
      if (intersection.size > 2) {
        // Strong connection found - enable knowledge transfer
        console.log(`[Meta-Learning] Found knowledge transfer path: ${taskType} <-> ${otherTask}`);
      }
    }
  }
}

/**
 * Extract concepts from experience
 */
function extractConcepts(experience: LearningExperience): string[] {
  const concepts: string[] = [];
  
  // Extract from task type
  concepts.push(experience.taskType);
  
  // Extract from model
  concepts.push(`model:${experience.modelUsed}`);
  
  // Extract keywords from input/output
  const text = `${experience.input} ${experience.output}`.toLowerCase();
  const keywords = [
    'reasoning', 'math', 'code', 'analysis', 'creative',
    'planning', 'research', 'technical', 'ethical', 'strategy'
  ];
  keywords.forEach(kw => {
    if (text.includes(kw)) concepts.push(kw);
  });
  
  return concepts;
}

/**
 * Update performance metrics
 */
function updatePerformanceMetrics(experience: LearningExperience): void {
  const taskType = experience.taskType;
  
  if (!performanceHistory.has(taskType)) {
    performanceHistory.set(taskType, []);
  }
  
  const history = performanceHistory.get(taskType)!;
  const recentExperiences = experienceMemory
    .filter(e => e.taskType === taskType)
    .slice(-100);
  
  // Calculate current metrics
  const successRate = recentExperiences.filter(e => e.success).length / recentExperiences.length;
  const avgLatency = recentExperiences.reduce((sum, e) => sum + e.latencyMs, 0) / recentExperiences.length;
  
  // Determine trend
  let trend: 'improving' | 'stable' | 'declining' = 'stable';
  if (history.length >= 2) {
    const prev = history[history.length - 1];
    if (successRate > prev.successRate + 0.05) trend = 'improving';
    else if (successRate < prev.successRate - 0.05) trend = 'declining';
  }
  
  const metrics: PerformanceMetrics = {
    taskType,
    successRate,
    avgLatency,
    totalAttempts: recentExperiences.length,
    improvements: history.filter(h => h.trend === 'improving').length,
    regressions: history.filter(h => h.trend === 'declining').length,
    trend
  };
  
  history.push(metrics);
  
  // Keep only last 100 metric snapshots
  if (history.length > 100) {
    history.shift();
  }
}

/**
 * Trigger learning cycle to improve strategies
 */
function triggerLearningCycle(): void {
  console.log('[Meta-Learning] Triggering learning cycle...');
  
  // Analyze recent experiences
  const recentExperiences = experienceMemory.slice(-100);
  
  // Group by strategy used (from metadata)
  const strategyPerformance: Map<string, { successes: number; total: number; avgLatency: number }> = new Map();
  
  recentExperiences.forEach(exp => {
    const strategyId = exp.metadata.strategyId as string || 'unknown';
    if (!strategyPerformance.has(strategyId)) {
      strategyPerformance.set(strategyId, { successes: 0, total: 0, avgLatency: 0 });
    }
    const perf = strategyPerformance.get(strategyId)!;
    perf.total++;
    if (exp.success) perf.successes++;
    perf.avgLatency = (perf.avgLatency * (perf.total - 1) + exp.latencyMs) / perf.total;
  });
  
  // Update strategy success rates
  for (const [strategyId, perf] of Array.from(strategyPerformance.entries())) {
    const strategy = strategyLibrary.get(strategyId);
    if (strategy && perf.total >= 5) {
      // Exponential moving average update
      const alpha = 0.3;
      strategy.successRate = alpha * (perf.successes / perf.total) + (1 - alpha) * strategy.successRate;
      strategy.avgLatency = alpha * perf.avgLatency + (1 - alpha) * strategy.avgLatency;
      
      console.log(`[Meta-Learning] Updated ${strategyId}: success=${strategy.successRate.toFixed(2)}, latency=${strategy.avgLatency.toFixed(0)}ms`);
    }
  }
  
  // Identify underperforming strategies and adapt
  for (const strategy of Array.from(strategyLibrary.values())) {
    if (strategy.successRate < 0.7 && strategy.usageCount > 20) {
      console.log(`[Meta-Learning] Strategy ${strategy.id} underperforming, adapting parameters...`);
      adaptStrategyParameters(strategy);
    }
  }
}

/**
 * Adapt strategy parameters based on performance
 */
function adaptStrategyParameters(strategy: Strategy): void {
  // Simple parameter adaptation
  switch (strategy.id) {
    case 'chain-of-thought':
      strategy.parameters.depth = Math.min(strategy.parameters.depth + 1, 10);
      break;
    case 'tree-of-thought':
      strategy.parameters.branches = Math.min(strategy.parameters.branches + 1, 5);
      break;
    case 'self-consistency':
      strategy.parameters.samples = Math.min(strategy.parameters.samples + 2, 10);
      break;
    case 'multi-agent-debate':
      strategy.parameters.rounds = Math.min(strategy.parameters.rounds + 1, 5);
      break;
    case 'reflection':
      strategy.parameters.iterations = Math.min(strategy.parameters.iterations + 1, 4);
      break;
  }
}

/**
 * Get knowledge transfer recommendations
 */
export function getKnowledgeTransfer(sourceTask: string, targetTask: string): KnowledgeTransfer | null {
  const sourceConcepts = knowledgeGraph.get(sourceTask);
  const targetConcepts = knowledgeGraph.get(targetTask);
  
  if (!sourceConcepts || !targetConcepts) return null;
  
  const sharedConcepts = Array.from(sourceConcepts).filter(c => targetConcepts.has(c));
  
  if (sharedConcepts.length < 2) return null;
  
  const transferScore = sharedConcepts.length / Math.max(sourceConcepts.size, targetConcepts.size);
  
  return {
    sourceTask,
    targetTask,
    transferScore,
    sharedConcepts,
    adaptations: generateAdaptations(sourceTask, targetTask, sharedConcepts)
  };
}

/**
 * Generate adaptation recommendations
 */
function generateAdaptations(source: string, target: string, shared: string[]): string[] {
  const adaptations: string[] = [];
  
  if (shared.includes('reasoning')) {
    adaptations.push('Apply similar reasoning patterns');
  }
  if (shared.includes('code')) {
    adaptations.push('Reuse code generation templates');
  }
  if (shared.includes('analysis')) {
    adaptations.push('Transfer analytical frameworks');
  }
  if (shared.includes('creative')) {
    adaptations.push('Apply creative techniques');
  }
  
  return adaptations;
}

/**
 * Get current performance summary
 */
export function getPerformanceSummary(): Record<string, PerformanceMetrics> {
  const summary: Record<string, PerformanceMetrics> = {};
  
  for (const [taskType, history] of Array.from(performanceHistory.entries())) {
    if (history.length > 0) {
      summary[taskType] = history[history.length - 1];
    }
  }
  
  return summary;
}

/**
 * Get all strategies with current stats
 */
export function getAllStrategies(): Strategy[] {
  return Array.from(strategyLibrary.values());
}

/**
 * Get experience count by task type
 */
export function getExperienceCounts(): Record<string, number> {
  const counts: Record<string, number> = {};
  
  experienceMemory.forEach(exp => {
    counts[exp.taskType] = (counts[exp.taskType] || 0) + 1;
  });
  
  return counts;
}

/**
 * Export meta-learning system
 */
export const metaLearningSystem = {
  recordExperience,
  selectStrategy,
  getKnowledgeTransfer,
  getPerformanceSummary,
  getAllStrategies,
  getExperienceCounts,
  experienceCount: () => experienceMemory.length,
  strategyCount: () => strategyLibrary.size
};

export default metaLearningSystem;
