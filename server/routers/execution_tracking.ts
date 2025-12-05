import { z } from "zod";
import { TRPCError } from "@trpc/server";
import { protectedProcedure, router } from "../_core/trpc";

/**
 * Execution Tracking Router (Simplified)
 * Track recommendation executions, status updates, and actual ROI
 * 
 * Note: Using in-memory storage for now. Will migrate to database in Phase 2.
 */

// In-memory storage (temporary)
const executionStore: Map<string, any> = new Map();

export const executionTrackingRouter = router({
  /**
   * Track a new recommendation execution
   */
  trackExecution: protectedProcedure
    .input(
      z.object({
        companyId: z.string(),
        recommendationId: z.string(),
        recommendationAction: z.string(),
        recommendationCategory: z.enum(["revenue", "marketing", "leadership", "operations", "technology"]),
        deeplinkPlatform: z.string(),
        deeplinkUrl: z.string(),
      })
    )
    .mutation(async ({ ctx, input }) => {
      try {
        const executionId = `exec_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        
        const execution = {
          id: executionId,
          userId: ctx.user.id,
          companyId: input.companyId,
          recommendationId: input.recommendationId,
          recommendationAction: input.recommendationAction,
          recommendationCategory: input.recommendationCategory,
          deeplinkPlatform: input.deeplinkPlatform,
          deeplinkUrl: input.deeplinkUrl,
          status: "pending",
          executedAt: new Date().toISOString(),
          completedAt: null,
          actualROI: null,
          notes: null,
        };

        executionStore.set(executionId, execution);

        return {
          success: true,
          executionId,
        };
      } catch (error) {
        console.error("[ExecutionTracking] Error tracking execution:", error);
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: "Failed to track execution",
        });
      }
    }),

  /**
   * Get execution history for a company
   */
  getExecutionHistory: protectedProcedure
    .input(
      z.object({
        companyId: z.string(),
      })
    )
    .query(async ({ ctx, input }) => {
      try {
        const executions = Array.from(executionStore.values()).filter(
          (exec) => exec.userId === ctx.user.id && exec.companyId === input.companyId
        );

        return executions.sort((a, b) => 
          new Date(b.executedAt).getTime() - new Date(a.executedAt).getTime()
        );
      } catch (error) {
        console.error("[ExecutionTracking] Error fetching execution history:", error);
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: "Failed to fetch execution history",
        });
      }
    }),

  /**
   * Get all executions for current user
   */
  getAllExecutions: protectedProcedure.query(async ({ ctx }) => {
    try {
      const executions = Array.from(executionStore.values()).filter(
        (exec) => exec.userId === ctx.user.id
      );

      return executions
        .sort((a, b) => new Date(b.executedAt).getTime() - new Date(a.executedAt).getTime())
        .slice(0, 100);
    } catch (error) {
      console.error("[ExecutionTracking] Error fetching all executions:", error);
      throw new TRPCError({
        code: "INTERNAL_SERVER_ERROR",
        message: "Failed to fetch executions",
      });
    }
  }),

  /**
   * Update execution status
   */
  updateExecutionStatus: protectedProcedure
    .input(
      z.object({
        executionId: z.string(),
        status: z.enum(["pending", "in-progress", "completed", "failed"]),
        notes: z.string().optional(),
      })
    )
    .mutation(async ({ ctx, input }) => {
      try {
        const execution = executionStore.get(input.executionId);
        
        if (!execution || execution.userId !== ctx.user.id) {
          throw new TRPCError({
            code: "NOT_FOUND",
            message: "Execution not found",
          });
        }

        execution.status = input.status;
        execution.notes = input.notes || execution.notes;
        
        if (input.status === "completed") {
          execution.completedAt = new Date().toISOString();
        }

        executionStore.set(input.executionId, execution);

        return {
          success: true,
        };
      } catch (error) {
        console.error("[ExecutionTracking] Error updating execution status:", error);
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: "Failed to update execution status",
        });
      }
    }),

  /**
   * Record actual ROI achieved
   */
  recordActualROI: protectedProcedure
    .input(
      z.object({
        executionId: z.string(),
        actualROI: z.string(), // e.g., "+35%"
        notes: z.string().optional(),
      })
    )
    .mutation(async ({ ctx, input }) => {
      try {
        const execution = executionStore.get(input.executionId);
        
        if (!execution || execution.userId !== ctx.user.id) {
          throw new TRPCError({
            code: "NOT_FOUND",
            message: "Execution not found",
          });
        }

        execution.actualROI = input.actualROI;
        execution.notes = input.notes || execution.notes;
        execution.status = "completed";
        execution.completedAt = new Date().toISOString();

        executionStore.set(input.executionId, execution);

        return {
          success: true,
        };
      } catch (error) {
        console.error("[ExecutionTracking] Error recording actual ROI:", error);
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: "Failed to record actual ROI",
        });
      }
    }),

  /**
   * Get execution statistics
   */
  getExecutionStats: protectedProcedure
    .input(
      z.object({
        companyId: z.string().optional(),
      })
    )
    .query(async ({ ctx, input }) => {
      try {
        let executions = Array.from(executionStore.values()).filter(
          (exec) => exec.userId === ctx.user.id
        );

        if (input.companyId) {
          executions = executions.filter((exec) => exec.companyId === input.companyId);
        }

        const total = executions.length;
        const completed = executions.filter((e) => e.status === "completed").length;
        const inProgress = executions.filter((e) => e.status === "in-progress").length;
        const pending = executions.filter((e) => e.status === "pending").length;
        const completionRate = total > 0 ? Math.round((completed / total) * 100) : 0;

        const byCategory = executions.reduce((acc: any[], exec) => {
          const existing = acc.find((item) => item.recommendation_category === exec.recommendationCategory);
          if (existing) {
            existing.count++;
          } else {
            acc.push({
              recommendation_category: exec.recommendationCategory,
              count: 1,
            });
          }
          return acc;
        }, []);

        return {
          total,
          completed,
          inProgress,
          pending,
          completionRate,
          byCategory,
        };
      } catch (error) {
        console.error("[ExecutionTracking] Error fetching execution stats:", error);
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: "Failed to fetch execution stats",
        });
      }
    }),
});
