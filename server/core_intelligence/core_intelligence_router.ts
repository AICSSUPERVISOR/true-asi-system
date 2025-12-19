/**
 * TRUE ASI - CORE INTELLIGENCE ROUTER
 * 
 * Unified API for all core intelligence systems:
 * - Reasoning Engine (8 reasoning types)
 * - Planning System (hierarchical task planning)
 * - Memory System (4 memory types)
 * - Learning System (8 learning modes)
 * 
 * NO MOCK DATA - 100% FUNCTIONAL
 */

import { z } from 'zod';
import { publicProcedure, router } from '../_core/trpc';
import { reasoningEngine } from './reasoning_engine';
import { planningSystem } from './planning_system';
import { memorySystem } from './memory_system';
import { learningSystem } from './learning_system';
import type { ReasoningContext } from './reasoning_engine';
import type { Goal, PlanningContext, Resource } from './planning_system';
import type { MemoryQuery } from './memory_system';
import type { LearningExample, TransferConfig } from './learning_system';

export const coreIntelligenceRouter = router({
  // ============================================================================
  // REASONING ENDPOINTS
  // ============================================================================

  reason: publicProcedure
    .input(z.object({
      type: z.enum(['deductive', 'inductive', 'abductive', 'causal', 'mathematical', 'spatial', 'temporal', 'counterfactual']),
      premises: z.array(z.string()),
      observations: z.array(z.string()).optional(),
      constraints: z.array(z.string()).optional(),
      domain: z.string().optional()
    }))
    .mutation(async ({ input }) => {
      const context: ReasoningContext = {
        premises: input.premises,
        observations: input.observations,
        constraints: input.constraints,
        domain: input.domain
      };

      switch (input.type) {
        case 'deductive':
          return await reasoningEngine.deductiveReasoning(context);
        case 'inductive':
          return await reasoningEngine.inductiveReasoning(context);
        case 'abductive':
          return await reasoningEngine.abductiveReasoning(context);
        case 'causal':
          return await reasoningEngine.causalReasoning(context);
        case 'mathematical':
          return await reasoningEngine.mathematicalReasoning(context);
        case 'spatial':
          return await reasoningEngine.spatialReasoning(context);
        case 'temporal':
          return await reasoningEngine.temporalReasoning(context);
        case 'counterfactual':
          return await reasoningEngine.counterfactualReasoning(context);
        default:
          throw new Error(`Unknown reasoning type: ${input.type}`);
      }
    }),

  analogicalReason: publicProcedure
    .input(z.object({
      sourceDomain: z.string(),
      targetDomain: z.string(),
      premises: z.array(z.string())
    }))
    .mutation(async ({ input }) => {
      return await reasoningEngine.analogicalReasoning(
        input.sourceDomain,
        input.targetDomain,
        { premises: input.premises }
      );
    }),

  abstractReason: publicProcedure
    .input(z.object({
      examples: z.array(z.object({
        input: z.unknown(),
        output: z.unknown()
      })),
      testInput: z.unknown()
    }))
    .mutation(async ({ input }) => {
      return await reasoningEngine.abstractReasoning(
        input.examples.map(e => ({ input: e.input, output: e.output })),
        input.testInput
      );
    }),

  consensusReason: publicProcedure
    .input(z.object({
      premises: z.array(z.string()),
      constraints: z.array(z.string()).optional()
    }))
    .mutation(async ({ input }) => {
      return await reasoningEngine.consensusReasoning({
        premises: input.premises,
        constraints: input.constraints
      });
    }),

  getReasoningHistory: publicProcedure.query(async () => {
    return reasoningEngine.getReasoningHistory();
  }),

  // ============================================================================
  // PLANNING ENDPOINTS
  // ============================================================================

  createPlan: publicProcedure
    .input(z.object({
      goal: z.object({
        id: z.string(),
        description: z.string(),
        priority: z.number(),
        success_criteria: z.array(z.string()),
        deadline: z.string().optional()
      }),
      available_resources: z.array(z.object({
        type: z.enum(['compute', 'memory', 'api_calls', 'time', 'tokens', 'custom']),
        name: z.string(),
        amount: z.number(),
        unit: z.string(),
        available: z.number().optional()
      })),
      constraints: z.array(z.string()),
      preferences: z.array(z.string()).optional()
    }))
    .mutation(async ({ input }) => {
      const goal: Goal = {
        ...input.goal,
        deadline: input.goal.deadline ? new Date(input.goal.deadline) : undefined
      };

      const context: PlanningContext = {
        available_resources: input.available_resources as Resource[],
        constraints: input.constraints,
        preferences: input.preferences || []
      };

      return await planningSystem.createPlan(goal, context);
    }),

  executePlan: publicProcedure
    .input(z.object({
      planId: z.string()
    }))
    .mutation(async ({ input }) => {
      return await planningSystem.executePlan(input.planId);
    }),

  adaptPlan: publicProcedure
    .input(z.object({
      planId: z.string(),
      newConstraints: z.array(z.string()),
      reason: z.string()
    }))
    .mutation(async ({ input }) => {
      return await planningSystem.adaptPlan(input.planId, input.newConstraints, input.reason);
    }),

  getPlan: publicProcedure
    .input(z.object({
      planId: z.string()
    }))
    .query(async ({ input }) => {
      return planningSystem.getPlan(input.planId);
    }),

  getAllPlans: publicProcedure.query(async () => {
    return planningSystem.getAllPlans();
  }),

  // ============================================================================
  // MEMORY ENDPOINTS
  // ============================================================================

  addToWorkingMemory: publicProcedure
    .input(z.object({
      content: z.string(),
      metadata: z.object({
        source: z.string().optional(),
        context: z.string().optional(),
        tags: z.array(z.string()).optional(),
        confidence: z.number().optional()
      }).optional()
    }))
    .mutation(async ({ input }) => {
      return memorySystem.addToWorkingMemory(input.content, input.metadata || {});
    }),

  getWorkingMemory: publicProcedure.query(async () => {
    return memorySystem.getWorkingMemory();
  }),

  recordEpisode: publicProcedure
    .input(z.object({
      context: z.string(),
      events: z.array(z.object({
        action: z.string(),
        observation: z.string(),
        importance: z.number()
      })),
      outcome: z.string(),
      lessonsLearned: z.array(z.string()).optional()
    }))
    .mutation(async ({ input }) => {
      return memorySystem.recordEpisode(
        input.context,
        input.events.map(e => ({ ...e, timestamp: new Date() })),
        input.outcome,
        input.lessonsLearned
      );
    }),

  recallEpisodes: publicProcedure
    .input(z.object({
      context: z.string(),
      maxResults: z.number().optional()
    }))
    .mutation(async ({ input }) => {
      return await memorySystem.recallSimilarEpisodes(input.context, input.maxResults);
    }),

  storeFact: publicProcedure
    .input(z.object({
      subject: z.string(),
      predicate: z.string(),
      object: z.string(),
      confidence: z.number().optional(),
      source: z.string().optional()
    }))
    .mutation(async ({ input }) => {
      return memorySystem.storeFact(
        input.subject,
        input.predicate,
        input.object,
        input.confidence,
        input.source
      );
    }),

  queryFacts: publicProcedure
    .input(z.object({
      subject: z.string().optional(),
      predicate: z.string().optional(),
      object: z.string().optional()
    }))
    .query(async ({ input }) => {
      return memorySystem.queryFacts(input.subject, input.predicate, input.object);
    }),

  storeProcedure: publicProcedure
    .input(z.object({
      name: z.string(),
      description: z.string(),
      steps: z.array(z.object({
        order: z.number(),
        action: z.string(),
        expected_result: z.string(),
        fallback: z.string().optional()
      })),
      preconditions: z.array(z.string()).optional(),
      postconditions: z.array(z.string()).optional()
    }))
    .mutation(async ({ input }) => {
      return memorySystem.storeProcedure(
        input.name,
        input.description,
        input.steps,
        input.preconditions,
        input.postconditions
      );
    }),

  getProcedure: publicProcedure
    .input(z.object({
      name: z.string()
    }))
    .query(async ({ input }) => {
      return memorySystem.getProcedure(input.name);
    }),

  searchMemory: publicProcedure
    .input(z.object({
      query: z.string(),
      memory_types: z.array(z.enum(['working', 'episodic', 'semantic', 'procedural'])).optional(),
      max_results: z.number().optional(),
      min_relevance: z.number().optional(),
      tags: z.array(z.string()).optional()
    }))
    .mutation(async ({ input }) => {
      return await memorySystem.search(input as MemoryQuery);
    }),

  consolidateMemory: publicProcedure.mutation(async () => {
    return await memorySystem.consolidate();
  }),

  getMemoryStats: publicProcedure.query(async () => {
    return {
      stats: memorySystem.getStats(),
      counts: memorySystem.getMemoryCount()
    };
  }),

  // ============================================================================
  // LEARNING ENDPOINTS
  // ============================================================================

  supervisedLearn: publicProcedure
    .input(z.object({
      taskName: z.string(),
      examples: z.array(z.object({
        input: z.unknown(),
        output: z.unknown()
      }))
    }))
    .mutation(async ({ input }) => {
      return await learningSystem.supervisedLearn(
        input.taskName,
        input.examples as LearningExample[]
      );
    }),

  reinforcementLearn: publicProcedure
    .input(z.object({
      taskName: z.string(),
      action: z.string(),
      reward: z.number(),
      state_before: z.unknown(),
      state_after: z.unknown(),
      done: z.boolean()
    }))
    .mutation(async ({ input }) => {
      return await learningSystem.reinforcementLearn(input.taskName, {
        action: input.action,
        reward: input.reward,
        state_before: input.state_before,
        state_after: input.state_after,
        done: input.done
      });
    }),

  fewShotLearn: publicProcedure
    .input(z.object({
      taskName: z.string(),
      examples: z.array(z.object({
        input: z.unknown(),
        output: z.unknown()
      })),
      query: z.unknown()
    }))
    .mutation(async ({ input }) => {
      return await learningSystem.fewShotLearn(
        input.taskName,
        input.examples as LearningExample[],
        input.query
      );
    }),

  zeroShotLearn: publicProcedure
    .input(z.object({
      taskDescription: z.string(),
      query: z.unknown()
    }))
    .mutation(async ({ input }) => {
      return await learningSystem.zeroShotLearn(input.taskDescription, input.query);
    }),

  transferLearn: publicProcedure
    .input(z.object({
      source_task: z.string(),
      target_task: z.string(),
      transfer_type: z.enum(['full', 'partial', 'feature_extraction']),
      adaptation_examples: z.array(z.object({
        input: z.unknown(),
        output: z.unknown()
      }))
    }))
    .mutation(async ({ input }) => {
      return await learningSystem.transferLearn(input as TransferConfig);
    }),

  metaLearn: publicProcedure
    .input(z.object({
      task_ids: z.array(z.string()),
      inner_learning_rate: z.number(),
      outer_learning_rate: z.number(),
      inner_steps: z.number(),
      task_batch_size: z.number()
    }))
    .mutation(async ({ input }) => {
      const tasks = input.task_ids
        .map(id => learningSystem.getTask(id))
        .filter((t): t is NonNullable<typeof t> => t !== undefined);
      
      return await learningSystem.metaLearn(tasks, {
        inner_learning_rate: input.inner_learning_rate,
        outer_learning_rate: input.outer_learning_rate,
        inner_steps: input.inner_steps,
        task_batch_size: input.task_batch_size
      });
    }),

  continualLearn: publicProcedure
    .input(z.object({
      taskName: z.string(),
      newExamples: z.array(z.object({
        input: z.unknown(),
        output: z.unknown()
      }))
    }))
    .mutation(async ({ input }) => {
      return await learningSystem.continualLearn(
        input.taskName,
        input.newExamples as LearningExample[]
      );
    }),

  curriculumLearn: publicProcedure
    .input(z.object({
      taskName: z.string(),
      examples: z.array(z.object({
        input: z.unknown(),
        output: z.unknown(),
        metadata: z.object({
          difficulty: z.number().optional()
        }).optional()
      })),
      difficulty_levels: z.number(),
      current_level: z.number(),
      promotion_threshold: z.number(),
      demotion_threshold: z.number()
    }))
    .mutation(async ({ input }) => {
      return await learningSystem.curriculumLearn(
        input.taskName,
        input.examples as LearningExample[],
        {
          difficulty_levels: input.difficulty_levels,
          current_level: input.current_level,
          promotion_threshold: input.promotion_threshold,
          demotion_threshold: input.demotion_threshold
        }
      );
    }),

  getLearningTasks: publicProcedure.query(async () => {
    return learningSystem.getAllTasks();
  }),

  getLearningTask: publicProcedure
    .input(z.object({
      taskId: z.string()
    }))
    .query(async ({ input }) => {
      return learningSystem.getTask(input.taskId);
    }),

  // ============================================================================
  // UNIFIED STATUS
  // ============================================================================

  getStatus: publicProcedure.query(async () => {
    return {
      reasoning: {
        history_count: reasoningEngine.getReasoningHistory().length,
        learned_patterns: reasoningEngine.getLearnedPatterns().length
      },
      planning: {
        active_plans: planningSystem.getAllPlans().filter(p => p.status === 'active').length,
        total_plans: planningSystem.getAllPlans().length
      },
      memory: {
        stats: memorySystem.getStats(),
        counts: memorySystem.getMemoryCount()
      },
      learning: {
        total_tasks: learningSystem.getAllTasks().length,
        meta_knowledge_items: learningSystem.getMetaKnowledge().size
      }
    };
  })
});

export type CoreIntelligenceRouter = typeof coreIntelligenceRouter;
