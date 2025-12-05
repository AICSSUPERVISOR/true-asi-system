import { z } from "zod";
import { publicProcedure, protectedProcedure, router } from "../_core/trpc";
import { getDb } from "../db";
import { analyses, recommendations, executions, revenueTracking } from "../../drizzle/schema";
import { eq, desc } from "drizzle-orm";
import { TRPCError } from "@trpc/server";

export const analysisHistoryRouter = router({
  /**
   * Save a new business analysis
   */
  saveAnalysis: protectedProcedure
    .input(z.object({
      organizationNumber: z.string(),
      companyName: z.string(),
      digitalMaturityScore: z.number().min(0).max(100),
      dataCompleteness: z.number().min(0).max(100),
      competitivePosition: z.enum(["leader", "challenger", "follower", "niche"]),
      industryCode: z.string().optional(),
      industryName: z.string().optional(),
      industryCategory: z.string().optional(),
      analysisData: z.any(), // Full JSON of analysis results
    }))
    .mutation(async ({ ctx, input }) => {
      const analysisId = `analysis_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      const db = await getDb();
      if (!db) throw new TRPCError({ code: "INTERNAL_SERVER_ERROR", message: "Database not available" });
      
      await db.insert(analyses).values({
        id: analysisId,
        userId: ctx.user.id,
        organizationNumber: input.organizationNumber,
        companyName: input.companyName,
        digitalMaturityScore: input.digitalMaturityScore,
        dataCompleteness: input.dataCompleteness,
        competitivePosition: input.competitivePosition,
        industryCode: input.industryCode,
        industryName: input.industryName,
        industryCategory: input.industryCategory,
        analysisData: JSON.stringify(input.analysisData),
      });

      return { analysisId, success: true };
    }),

  /**
   * Get all analyses for the current user
   */
  getMyAnalyses: protectedProcedure
    .input(z.object({
      limit: z.number().min(1).max(100).default(20),
      offset: z.number().min(0).default(0),
    }).optional())
    .query(async ({ ctx, input }) => {
      const limit = input?.limit || 20;
      const offset = input?.offset || 0;
      const db = await getDb();
      if (!db) throw new TRPCError({ code: "INTERNAL_SERVER_ERROR", message: "Database not available" });

      const userAnalyses = await db
        .select()
        .from(analyses)
        .where(eq(analyses.userId, ctx.user.id))
        .orderBy(desc(analyses.createdAt))
        .limit(limit)
        .offset(offset);

      return userAnalyses.map((analysis: typeof analyses.$inferSelect) => ({
        ...analysis,
        analysisData: JSON.parse(analysis.analysisData),
      }));
    }),

  /**
   * Get a specific analysis by ID
   */
  getAnalysisById: protectedProcedure
    .input(z.object({
      analysisId: z.string(),
    }))
    .query(async ({ ctx, input }) => {
      const db = await getDb();
      if (!db) throw new TRPCError({ code: "INTERNAL_SERVER_ERROR", message: "Database not available" });
      const [analysis] = await db
        .select()
        .from(analyses)
        .where(eq(analyses.id, input.analysisId));

      if (!analysis) {
        throw new TRPCError({
          code: "NOT_FOUND",
          message: "Analysis not found",
        });
      }

      // Verify ownership
      if (analysis.userId !== ctx.user.id) {
        throw new TRPCError({
          code: "FORBIDDEN",
          message: "You don't have permission to view this analysis",
        });
      }

      return {
        ...analysis,
        analysisData: JSON.parse(analysis.analysisData),
      };
    }),

  /**
   * Delete an analysis
   */
  deleteAnalysis: protectedProcedure
    .input(z.object({
      analysisId: z.string(),
    }))
    .mutation(async ({ ctx, input }) => {
      const db = await getDb();
      if (!db) throw new TRPCError({ code: "INTERNAL_SERVER_ERROR", message: "Database not available" });
      const [analysis] = await db
        .select()
        .from(analyses)
        .where(eq(analyses.id, input.analysisId));

      if (!analysis) {
        throw new TRPCError({
          code: "NOT_FOUND",
          message: "Analysis not found",
        });
      }

      // Verify ownership
      if (analysis.userId !== ctx.user.id) {
        throw new TRPCError({
          code: "FORBIDDEN",
          message: "You don't have permission to delete this analysis",
        });
      }

      // Delete the analysis
      await db.delete(analyses).where(eq(analyses.id, input.analysisId));

      return { success: true };
    }),

  /**
   * Get analysis statistics for the current user
   */
  getMyStats: protectedProcedure
    .query(async ({ ctx }) => {
      const db = await getDb();
      if (!db) throw new TRPCError({ code: "INTERNAL_SERVER_ERROR", message: "Database not available" });
      const userAnalyses = await db
        .select()
        .from(analyses)
        .where(eq(analyses.userId, ctx.user.id));

      const totalAnalyses = userAnalyses.length;
      const avgDigitalMaturity = totalAnalyses > 0
        ? Math.round(userAnalyses.reduce((sum: number, a: typeof analyses.$inferSelect) => sum + a.digitalMaturityScore, 0) / totalAnalyses)
        : 0;
      
      const avgDataCompleteness = totalAnalyses > 0
        ? Math.round(userAnalyses.reduce((sum: number, a: typeof analyses.$inferSelect) => sum + a.dataCompleteness, 0) / totalAnalyses)
        : 0;

      // Count by competitive position
      const positionCounts = {
        leader: userAnalyses.filter((a: typeof analyses.$inferSelect) => a.competitivePosition === "leader").length,
        challenger: userAnalyses.filter((a: typeof analyses.$inferSelect) => a.competitivePosition === "challenger").length,
        follower: userAnalyses.filter((a: typeof analyses.$inferSelect) => a.competitivePosition === "follower").length,
        niche: userAnalyses.filter((a: typeof analyses.$inferSelect) => a.competitivePosition === "niche").length,
      };

      // Count by industry
      const industryMap = new Map<string, number>();
      userAnalyses.forEach((a: typeof analyses.$inferSelect) => {
        if (a.industryCategory) {
          industryMap.set(a.industryCategory, (industryMap.get(a.industryCategory) || 0) + 1);
        }
      });
      const topIndustries = Array.from(industryMap.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5)
        .map(([industry, count]) => ({ industry, count }));

      return {
        totalAnalyses,
        avgDigitalMaturity,
        avgDataCompleteness,
        positionCounts,
        topIndustries,
      };
    }),
});
