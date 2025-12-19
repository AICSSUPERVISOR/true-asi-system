/**
 * Tests for TRUE ASI Enhancement Modules
 * - Meta-Learning System
 * - Consciousness Simulation
 * - Universal Problem Solver
 * - Reasoning Chains
 * - Emergent Intelligence Detection
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { metaLearningSystem } from './meta_learning';
import { consciousnessSystem } from './consciousness_simulation';
import { universalSolver } from './universal_solver';
import { reasoningChains } from './reasoning_chains';
import { emergentIntelligence } from './emergent_intelligence';

describe('Meta-Learning System', () => {
  it('should record learning experiences', () => {
    const expId = metaLearningSystem.recordExperience({
      taskType: 'reasoning',
      input: 'Test problem',
      output: 'Test solution',
      feedback: 0.8,
      modelUsed: 'gpt-4',
      latencyMs: 1500,
      success: true,
      metadata: { strategyId: 'chain-of-thought' }
    });
    
    expect(expId).toBeDefined();
    expect(expId).toMatch(/^exp_/);
    expect(metaLearningSystem.experienceCount()).toBeGreaterThan(0);
  });

  it('should select appropriate strategies', () => {
    const strategy = metaLearningSystem.selectStrategy('reasoning');
    
    expect(strategy).toBeDefined();
    expect(strategy.id).toBeDefined();
    expect(strategy.name).toBeDefined();
    expect(strategy.successRate).toBeGreaterThan(0);
    expect(strategy.applicableTasks).toContain('reasoning');
  });

  it('should return all strategies', () => {
    const strategies = metaLearningSystem.getAllStrategies();
    
    expect(strategies).toBeDefined();
    expect(strategies.length).toBeGreaterThan(5);
    expect(strategies[0]).toHaveProperty('id');
    expect(strategies[0]).toHaveProperty('name');
    expect(strategies[0]).toHaveProperty('successRate');
  });

  it('should track performance summary', () => {
    // Record some experiences first
    metaLearningSystem.recordExperience({
      taskType: 'code',
      input: 'Write function',
      output: 'function test() {}',
      feedback: 0.9,
      modelUsed: 'claude-3',
      latencyMs: 2000,
      success: true,
      metadata: {}
    });

    const summary = metaLearningSystem.getPerformanceSummary();
    expect(summary).toBeDefined();
  });
});

describe('Consciousness Simulation', () => {
  it('should get current state', () => {
    const state = consciousnessSystem.getState();
    
    expect(state).toBeDefined();
    expect(state.awarenessLevel).toBeGreaterThanOrEqual(0);
    expect(state.awarenessLevel).toBeLessThanOrEqual(1);
    expect(state.emotionalState).toBeDefined();
    expect(state.emotionalState.curiosity).toBeGreaterThanOrEqual(0);
  });

  it('should update state', () => {
    const newState = consciousnessSystem.updateState({
      cognitiveLoad: 0.5,
      uncertaintyLevel: 0.3
    });
    
    expect(newState.cognitiveLoad).toBe(0.5);
    expect(newState.uncertaintyLevel).toBe(0.3);
  });

  it('should record thoughts', () => {
    const thoughtId = consciousnessSystem.recordThought({
      type: 'reasoning',
      content: 'Analyzing the problem',
      confidence: 0.85,
      alternatives: ['Alternative approach A', 'Alternative approach B'],
      dependencies: []
    });
    
    expect(thoughtId).toBeDefined();
    expect(thoughtId).toMatch(/^thought_/);
  });

  it('should perform introspection', () => {
    const result = consciousnessSystem.introspect();
    
    expect(result).toBeDefined();
    expect(result.currentState).toBeDefined();
    expect(result.insights).toBeDefined();
    expect(result.recommendations).toBeDefined();
    expect(result.selfAssessment).toBeDefined();
    expect(result.selfAssessment.overallPerformance).toBeGreaterThanOrEqual(0);
  });

  it('should set and track goals', () => {
    const goalId = consciousnessSystem.setGoal({
      description: 'Complete ASI enhancement',
      priority: 0.9,
      progress: 0,
      subgoals: [],
      constraints: ['time-limited']
    });
    
    expect(goalId).toBeDefined();
    expect(goalId).toMatch(/^goal_/);
    expect(consciousnessSystem.goalCount()).toBeGreaterThan(0);
  });

  it('should generate reflections', () => {
    const reflection = consciousnessSystem.reflect('problem solving');
    
    expect(reflection).toBeDefined();
    expect(reflection).toContain('Meta-Cognitive Reflection');
    expect(reflection).toContain('Current State');
  });
});

describe('Universal Problem Solver', () => {
  it('should analyze problems', () => {
    const problem = universalSolver.analyzeProblem(
      'Create a scalable microservices architecture for an e-commerce platform',
      { priority: 0.8 }
    );
    
    expect(problem).toBeDefined();
    expect(problem.id).toMatch(/^prob_/);
    expect(problem.domain).toBeDefined();
    expect(problem.complexity).toBeDefined();
    expect(problem.constraints).toBeDefined();
    expect(problem.objectives).toBeDefined();
  });

  it('should solve problems', async () => {
    const problem = universalSolver.analyzeProblem(
      'Optimize database query performance for large datasets'
    );
    
    const solution = await universalSolver.solve(problem);
    
    expect(solution).toBeDefined();
    expect(solution.id).toMatch(/^sol_/);
    expect(solution.problemId).toBe(problem.id);
    expect(solution.approach).toBeDefined();
    expect(solution.steps).toBeDefined();
    expect(solution.steps.length).toBeGreaterThan(0);
    expect(solution.confidence).toBeGreaterThan(0);
    expect(solution.risks).toBeDefined();
  });

  it('should return solving strategies', () => {
    const strategies = universalSolver.getStrategies();
    
    expect(strategies).toBeDefined();
    expect(strategies.length).toBeGreaterThan(5);
    expect(strategies[0]).toHaveProperty('id');
    expect(strategies[0]).toHaveProperty('name');
    expect(strategies[0]).toHaveProperty('steps');
  });

  it('should provide domain knowledge', () => {
    const knowledge = universalSolver.getDomainKnowledge('software');
    
    expect(knowledge).toBeDefined();
    expect(knowledge?.domain).toBe('software');
    expect(knowledge?.concepts).toBeDefined();
    expect(knowledge?.methods).toBeDefined();
    expect(knowledge?.tools).toBeDefined();
    expect(knowledge?.bestPractices).toBeDefined();
  });
});

describe('Reasoning Chains', () => {
  it('should create reasoning chains', () => {
    const chain = reasoningChains.createChain('Prove that P implies Q');
    
    expect(chain).toBeDefined();
    expect(chain.id).toMatch(/^chain_/);
    expect(chain.goal).toBe('Prove that P implies Q');
    expect(chain.steps).toEqual([]);
    expect(chain.isValid).toBe(true);
  });

  it('should add reasoning steps', () => {
    const chain = reasoningChains.createChain('Test reasoning');
    
    const step = reasoningChains.addStep(
      chain.id,
      'premise',
      'All humans are mortal'
    );
    
    expect(step).toBeDefined();
    expect(step?.id).toMatch(/^step_/);
    expect(step?.type).toBe('premise');
    expect(step?.content).toBe('All humans are mortal');
    expect(step?.confidence).toBeGreaterThan(0);
  });

  it('should apply logical rules', () => {
    const chain = reasoningChains.createChain('Apply modus ponens');
    
    const premise1 = reasoningChains.addStep(chain.id, 'premise', 'If it rains, the ground is wet');
    const premise2 = reasoningChains.addStep(chain.id, 'premise', 'It rains');
    
    const inference = reasoningChains.applyRule(
      chain.id,
      'modus-ponens',
      [premise1!.id, premise2!.id]
    );
    
    expect(inference).toBeDefined();
    expect(inference?.type).toBe('inference');
    expect(inference?.content).toContain('Modus Ponens');
  });

  it('should create proofs', () => {
    const proof = reasoningChains.createProof(
      'A implies C',
      ['A implies B', 'B implies C']
    );
    
    expect(proof).toBeDefined();
    expect(proof.id).toMatch(/^proof_/);
    expect(proof.theorem).toBe('A implies C');
    expect(proof.assumptions.length).toBe(2);
    expect(proof.steps.length).toBe(2); // Assumptions added as steps
  });

  it('should perform chain-of-thought reasoning', () => {
    const chain = reasoningChains.chainOfThought('What is the meaning of life?');
    
    expect(chain).toBeDefined();
    expect(chain.steps.length).toBeGreaterThan(3);
    expect(chain.steps[0].type).toBe('premise');
  });

  it('should perform tree-of-thought reasoning', () => {
    const chains = reasoningChains.treeOfThought('How to solve climate change?', 3);
    
    expect(chains).toBeDefined();
    expect(chains.length).toBe(3);
    chains.forEach(chain => {
      expect(chain.steps.length).toBeGreaterThan(0);
    });
  });

  it('should return logical rules', () => {
    const rules = reasoningChains.getLogicalRules();
    
    expect(rules).toBeDefined();
    expect(rules.length).toBeGreaterThan(5);
    expect(rules[0]).toHaveProperty('id');
    expect(rules[0]).toHaveProperty('name');
    expect(rules[0]).toHaveProperty('pattern');
    expect(rules[0]).toHaveProperty('inference');
  });
});

describe('Emergent Intelligence Detection', () => {
  it('should get all capabilities', () => {
    const capabilities = emergentIntelligence.getAllCapabilities();
    
    expect(capabilities).toBeDefined();
    expect(capabilities.length).toBeGreaterThan(20);
    expect(capabilities[0]).toHaveProperty('id');
    expect(capabilities[0]).toHaveProperty('name');
    expect(capabilities[0]).toHaveProperty('level');
    expect(capabilities[0]).toHaveProperty('domain');
  });

  it('should get capabilities by domain', () => {
    const languageCaps = emergentIntelligence.getCapabilitiesByDomain('language');
    
    expect(languageCaps).toBeDefined();
    expect(languageCaps.length).toBeGreaterThan(0);
    languageCaps.forEach(cap => {
      expect(cap.domain).toBe('language');
    });
  });

  it('should measure capabilities', () => {
    const capabilities = emergentIntelligence.getAllCapabilities();
    const capId = capabilities[0].id;
    
    const measurement = emergentIntelligence.measureCapability(capId, [
      { testId: 't1', testName: 'Test 1', score: 85, maxScore: 100, passed: true },
      { testId: 't2', testName: 'Test 2', score: 90, maxScore: 100, passed: true }
    ]);
    
    expect(measurement).toBeDefined();
    expect(measurement?.level).toBeGreaterThan(0);
    expect(measurement?.confidence).toBeGreaterThan(0);
  });

  it('should detect new capabilities', () => {
    const newCap = emergentIntelligence.detectNewCapability(
      'Quantum Reasoning',
      'Ability to reason about quantum phenomena',
      'physics',
      ['Demonstrated understanding of quantum superposition']
    );
    
    expect(newCap).toBeDefined();
    expect(newCap.name).toBe('Quantum Reasoning');
    expect(newCap.isEmergent).toBe(true);
    expect(newCap.level).toBe(50); // Initial level
  });

  it('should calculate intelligence metrics', () => {
    const metrics = emergentIntelligence.calculateMetrics();
    
    expect(metrics).toBeDefined();
    expect(metrics.overallLevel).toBeGreaterThan(0);
    expect(metrics.capabilityCount).toBeGreaterThan(0);
    expect(metrics.domainCoverage).toBeGreaterThan(0);
    expect(['accelerating', 'steady', 'plateauing', 'declining']).toContain(metrics.trend);
  });

  it('should get emergent events', () => {
    // Trigger an emergent event by detecting a new capability
    emergentIntelligence.detectNewCapability(
      'Test Capability',
      'Test description',
      'test',
      ['Test evidence']
    );
    
    const events = emergentIntelligence.getEmergentEvents();
    
    expect(events).toBeDefined();
    expect(events.length).toBeGreaterThan(0);
    expect(events[events.length - 1]).toHaveProperty('type');
    expect(events[events.length - 1]).toHaveProperty('significance');
  });

  it('should detect capability combinations', () => {
    const capabilities = emergentIntelligence.getAllCapabilities();
    const cap1 = capabilities[0].id;
    const cap2 = capabilities[1].id;
    
    const combined = emergentIntelligence.detectCapabilityCombination(
      'Combined Capability',
      'Emergent from combining two capabilities',
      [cap1, cap2],
      ['Evidence of combination']
    );
    
    expect(combined).toBeDefined();
    expect(combined?.isEmergent).toBe(true);
    expect(combined?.prerequisites).toContain(cap1);
    expect(combined?.prerequisites).toContain(cap2);
  });
});
