/**
 * TRUE ASI - tRPC Router
 * 
 * Exposes all ASI functionality through tRPC endpoints
 */

import { z } from 'zod';
import { router, publicProcedure, protectedProcedure } from '../_core/trpc';
import { trueASI, initializeASI, askASI, executeASITask, chatWithASI } from './true_asi';
import { benchmarkSystem } from './benchmark_system';
import { selfImprovementEngine } from './self_improvement';
import { multiAgentCoordinator } from './multi_agent_coordinator';
import { reasoningEngine } from './reasoning_engine';
import { memorySystem } from './memory_system';
import { learningSystem } from './learning_system';
import { knowledgeGraph } from './knowledge_graph';
import { agentFramework } from './agent_framework';
import { toolExecutor } from './tool_executor';

// Initialize ASI on module load
let asiInitialized = false;

async function ensureInitialized() {
  if (!asiInitialized) {
    await initializeASI();
    asiInitialized = true;
  }
}

export const asiRouter = router({
  // ==========================================================================
  // MAIN ASI ENDPOINTS
  // ==========================================================================
  
  // Initialize ASI
  initialize: publicProcedure
    .mutation(async () => {
      await ensureInitialized();
      return { success: true, message: 'TRUE ASI initialized' };
    }),
  
  // Get ASI status
  status: publicProcedure
    .query(async () => {
      await ensureInitialized();
      return trueASI.getState();
    }),
  
  // Get ASI statistics
  stats: publicProcedure
    .query(async () => {
      await ensureInitialized();
      return trueASI.getStats();
    }),
  
  // Get ASI capabilities
  capabilities: publicProcedure
    .query(async () => {
      await ensureInitialized();
      return trueASI.getCapabilities();
    }),
  
  // Process a request
  process: publicProcedure
    .input(z.object({
      type: z.enum(['query', 'task', 'analysis', 'generation', 'research', 'coding', 'planning', 'conversation']),
      content: z.string(),
      context: z.record(z.string(), z.unknown()).optional(),
      priority: z.enum(['critical', 'high', 'medium', 'low']).optional(),
      requiresReasoning: z.boolean().optional(),
      requiresLearning: z.boolean().optional()
    }))
    .mutation(async ({ input }) => {
      await ensureInitialized();
      return await trueASI.process({
        id: `req_${Date.now()}`,
        ...input
      });
    }),
  
  // Quick query
  ask: publicProcedure
    .input(z.object({ question: z.string() }))
    .mutation(async ({ input }) => {
      await ensureInitialized();
      return await askASI(input.question);
    }),
  
  // Execute task
  execute: publicProcedure
    .input(z.object({ task: z.string() }))
    .mutation(async ({ input }) => {
      await ensureInitialized();
      return await executeASITask(input.task);
    }),
  
  // Chat conversation
  chat: publicProcedure
    .input(z.object({ message: z.string() }))
    .mutation(async ({ input }) => {
      await ensureInitialized();
      return await chatWithASI(input.message);
    }),
  
  // Provide feedback
  feedback: publicProcedure
    .input(z.object({
      requestId: z.string(),
      type: z.enum(['explicit', 'implicit', 'outcome', 'comparative']).default('explicit'),
      score: z.number().min(-1).max(1),
      explanation: z.string().optional(),
      corrections: z.string().optional()
    }))
    .mutation(async ({ input }) => {
      await ensureInitialized();
      await trueASI.provideFeedback(input.requestId, {
        type: input.type,
        score: input.score,
        explanation: input.explanation,
        corrections: input.corrections
      });
      return { success: true };
    }),
  
  // ==========================================================================
  // REASONING ENDPOINTS
  // ==========================================================================
  
  reasoning: router({
    // Reason about a problem
    reason: publicProcedure
      .input(z.object({
        problem: z.string(),
        context: z.string().optional(),
        strategy: z.enum([
          'chain_of_thought', 'tree_of_thoughts', 'self_consistency',
          'react', 'reflection', 'debate', 'socratic', 'analogical',
          'causal', 'formal_logic'
        ]).optional()
      }))
      .mutation(async ({ input }) => {
        return await reasoningEngine.reason({
          id: `reason_${Date.now()}`,
          problem: input.problem,
          context: input.context
        }, input.strategy);
      }),
    
    // Get reasoning stats
    stats: publicProcedure
      .query(() => reasoningEngine.getStats())
  }),
  
  // ==========================================================================
  // MEMORY ENDPOINTS
  // ==========================================================================
  
  memory: router({
    // Store memory
    store: publicProcedure
      .input(z.object({
        content: z.string(),
        type: z.enum(['working', 'episodic', 'semantic', 'procedural', 'emotional']),
        metadata: z.object({
          source: z.string().optional(),
          tags: z.array(z.string()).optional(),
          confidence: z.number().optional()
        }).optional()
      }))
      .mutation(async ({ input }) => {
        return await memorySystem.store(input.content, input.type, input.metadata);
      }),
    
    // Recall memories
    recall: publicProcedure
      .input(z.object({
        query: z.string(),
        type: z.enum(['working', 'episodic', 'semantic', 'procedural', 'emotional']).optional(),
        limit: z.number().optional()
      }))
      .query(async ({ input }) => {
        return await memorySystem.recall(input);
      }),
    
    // Get memory stats
    stats: publicProcedure
      .query(() => memorySystem.getStats())
  }),
  
  // ==========================================================================
  // LEARNING ENDPOINTS
  // ==========================================================================
  
  learning: router({
    // Learn from example
    learn: publicProcedure
      .input(z.object({
        input: z.string(),
        output: z.string(),
        domain: z.string(),
        metadata: z.record(z.string(), z.unknown()).optional()
      }))
      .mutation(async ({ input }) => {
        return await learningSystem.learnFromExample(
          input.input,
          input.output,
          input.domain,
          input.metadata || {}
        );
      }),
    
    // Apply learning
    apply: publicProcedure
      .input(z.object({
        input: z.string(),
        domain: z.string()
      }))
      .mutation(async ({ input }) => {
        return await learningSystem.applyLearning(input.input, input.domain);
      }),
    
    // Get skills
    skills: publicProcedure
      .query(() => learningSystem.getAllSkills()),
    
    // Get learning stats
    stats: publicProcedure
      .query(() => learningSystem.getStats())
  }),
  
  // ==========================================================================
  // KNOWLEDGE GRAPH ENDPOINTS
  // ==========================================================================
  
  knowledge: router({
    // Add entity
    addEntity: publicProcedure
      .input(z.object({
        name: z.string(),
        type: z.enum(['concept', 'person', 'organization', 'location', 'event', 'object', 'process', 'abstract']),
        properties: z.record(z.string(), z.unknown()).optional()
      }))
      .mutation(async ({ input }) => {
        return await knowledgeGraph.addEntity(input.name, input.type, input.properties || {}, []);
      }),
    
    // Add relationship
    addRelationship: publicProcedure
      .input(z.object({
        from: z.string(),
        to: z.string(),
        type: z.enum(['is_a', 'part_of', 'has', 'causes', 'enables', 'requires', 'related_to', 'opposite_of', 'similar_to', 'instance_of', 'created_by', 'located_in', 'occurs_at', 'custom']),
        weight: z.number().optional()
      }))
      .mutation(async ({ input }) => {
        return await knowledgeGraph.addRelationship(
          input.from, input.to, input.type, input.weight
        );
      }),
    
    // Query knowledge
    query: publicProcedure
      .input(z.object({ query: z.string() }))
      .query(async ({ input }) => {
        return knowledgeGraph.query({ entityType: input.query as any });
      }),
    
    // Infer answer
    infer: publicProcedure
      .input(z.object({ question: z.string() }))
      .mutation(async ({ input }) => {
        return await knowledgeGraph.infer(input.question);
      }),
    
    // Extract knowledge from text
    extract: publicProcedure
      .input(z.object({ text: z.string() }))
      .mutation(async ({ input }) => {
        return await knowledgeGraph.extractKnowledge(input.text);
      }),
    
    // Get knowledge stats
    stats: publicProcedure
      .query(() => knowledgeGraph.getStats())
  }),
  
  // ==========================================================================
  // AGENT ENDPOINTS
  // ==========================================================================
  
  agents: router({
    // Create agent
    create: publicProcedure
      .input(z.object({
        name: z.string(),
        type: z.enum(['executor', 'planner', 'researcher', 'coder', 'analyst', 'creative', 'coordinator', 'specialist']),
        capabilities: z.array(z.object({
          name: z.string(),
          proficiency: z.number()
        })).optional()
      }))
      .mutation(async ({ input }) => {
        const caps = input.capabilities?.map(c => ({
          name: c.name,
          description: c.name,
          tools: [],
          proficiency: c.proficiency
        })) || [];
        return agentFramework.createAgent(input.name, input.type, caps);
      }),
    
    // Create task
    createTask: publicProcedure
      .input(z.object({
        description: z.string(),
        type: z.enum(['query', 'action', 'analysis', 'generation']),
        context: z.record(z.string(), z.unknown()).optional(),
        priority: z.enum(['critical', 'high', 'medium', 'low']).optional()
      }))
      .mutation(async ({ input }) => {
        return agentFramework.createTask(
          input.description, 
          input.type, 
          input.context || {}, 
          input.priority || 'medium'
        );
      }),
    
    // Execute task
    executeTask: publicProcedure
      .input(z.object({
        taskId: z.string(),
        agentId: z.string().optional()
      }))
      .mutation(async ({ input }) => {
        return await agentFramework.executeTask(input.taskId, input.agentId);
      }),
    
    // Get all agents
    list: publicProcedure
      .query(() => agentFramework.getAllAgents()),
    
    // Get agent stats
    stats: publicProcedure
      .query(() => agentFramework.getStats())
  }),
  
  // ==========================================================================
  // MULTI-AGENT COORDINATION ENDPOINTS
  // ==========================================================================
  
  swarm: router({
    // Execute swarm task
    execute: publicProcedure
      .input(z.object({ task: z.string() }))
      .mutation(async ({ input }) => {
        return await multiAgentCoordinator.executeSwarmTask(input.task);
      }),
    
    // Collective decision
    decide: publicProcedure
      .input(z.object({ question: z.string() }))
      .mutation(async ({ input }) => {
        return await multiAgentCoordinator.collectiveDecision(input.question);
      }),
    
    // Request consensus
    consensus: publicProcedure
      .input(z.object({
        topic: z.string(),
        options: z.array(z.string())
      }))
      .mutation(async ({ input }) => {
        return await multiAgentCoordinator.requestConsensus(input.topic, input.options);
      }),
    
    // Get swarm stats
    stats: publicProcedure
      .query(() => multiAgentCoordinator.getStats())
  }),
  
  // ==========================================================================
  // TOOL EXECUTION ENDPOINTS
  // ==========================================================================
  
  tools: router({
    // Execute code
    executeCode: publicProcedure
      .input(z.object({
        code: z.string(),
        language: z.enum(['javascript', 'python', 'typescript']).default('javascript')
      }))
      .mutation(async ({ input }) => {
        return await toolExecutor.executeCode(input.code, input.language as 'javascript' | 'python' | 'typescript');
      }),
    
    // Execute tool
    executeTool: publicProcedure
      .input(z.object({
        toolName: z.string(),
        parameters: z.record(z.string(), z.unknown())
      }))
      .mutation(async ({ input }) => {
        return await toolExecutor.execute(input.toolName, input.parameters);
      }),
    
    // Get available tools
    list: publicProcedure
      .query(() => toolExecutor.getAllTools()),
    
    // Get tool stats
    stats: publicProcedure
      .query(() => toolExecutor.getStats())
  }),
  
  // ==========================================================================
  // SELF-IMPROVEMENT ENDPOINTS
  // ==========================================================================
  
  improvement: router({
    // Reflect
    reflect: publicProcedure
      .mutation(async () => {
        return await selfImprovementEngine.reflect();
      }),
    
    // Evolve
    evolve: publicProcedure
      .mutation(async () => {
        return await selfImprovementEngine.evolve();
      }),
    
    // Get latest reflection
    latestReflection: publicProcedure
      .query(() => selfImprovementEngine.getLatestReflection()),
    
    // Get latest generation
    latestGeneration: publicProcedure
      .query(() => selfImprovementEngine.getLatestGeneration()),
    
    // Get improvement stats
    stats: publicProcedure
      .query(() => selfImprovementEngine.getStats())
  }),
  
  // ==========================================================================
  // BENCHMARK ENDPOINTS
  // ==========================================================================
  
  benchmarks: router({
    // Run all benchmarks
    runAll: publicProcedure
      .mutation(async () => {
        return await benchmarkSystem.runAllBenchmarks();
      }),
    
    // Run specific benchmark
    run: publicProcedure
      .input(z.object({ benchmarkId: z.string() }))
      .mutation(async ({ input }) => {
        return await benchmarkSystem.runBenchmark(input.benchmarkId);
      }),
    
    // Test specific capability
    testCapability: publicProcedure
      .input(z.object({
        capability: z.enum([
          'reasoning', 'mathematics', 'coding', 'knowledge',
          'learning', 'creativity', 'planning', 'self_improvement'
        ])
      }))
      .mutation(async ({ input }) => {
        switch (input.capability) {
          case 'reasoning': return await benchmarkSystem.testReasoning();
          case 'mathematics': return await benchmarkSystem.testMathematics();
          case 'coding': return await benchmarkSystem.testCoding();
          case 'knowledge': return await benchmarkSystem.testKnowledge();
          case 'learning': return await benchmarkSystem.testLearning();
          case 'creativity': return await benchmarkSystem.testCreativity();
          case 'self_improvement': return await benchmarkSystem.testSelfImprovement();
          default: return 0;
        }
      }),
    
    // Get available benchmarks
    list: publicProcedure
      .query(() => benchmarkSystem.getBenchmarks()),
    
    // Get benchmark results
    results: publicProcedure
      .query(() => benchmarkSystem.getResults()),
    
    // Get scorecards
    scorecards: publicProcedure
      .query(() => benchmarkSystem.getScorecards()),
    
    // Get benchmark stats
    stats: publicProcedure
      .query(() => benchmarkSystem.getStats())
  })
});
