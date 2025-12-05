import { z } from "zod";
import { TRPCError } from "@trpc/server";
import { protectedProcedure, router } from "../_core/trpc";
import { getDb } from "../db";
import { companies, companyFinancials, companyLinkedIn } from "../../drizzle/schema";
import { eq } from "drizzle-orm";
import { invokeLLM } from "../_core/llm";
import axios from "axios";
import { emitAnalysisProgress, emitAnalysisComplete } from "../_core/websocket";
import { selectModelsForTask, ensembleVote, trackModelPerformance, type TaskType } from "../helpers/ai_model_router";
import { scrapeForvaltData } from "../helpers/forvalt_scraper";
import { generateExecutionPlan, type Recommendation, type RecommendationCategory, type ImpactLevel, type DifficultyLevel } from "../helpers/recommendation_automation";

/**
 * Business Orchestrator
 * Complete end-to-end business automation system
 * Integrates: Brreg → Proff → Website → LinkedIn → Multi-Model AI → Recommendations → Deeplinks
 */

const API_KEYS = {
  ASI1_AI: process.env.ASI1_AI_API_KEY || "sk_26ec4938b6274ae089bfa915d02bf10036bde0326b5845c5b87c50b5dbc2c9ad",
  AIMLAPI: process.env.AIMLAPI_KEY || "147620aa16e04b96bb2f12b79527593f",
};

// Dynamic AI Model Selection using AI Router
// Models are selected based on task type for optimal performance

interface EnrichmentData {
  company: any;
  financial: any;
  website: any;
  linkedin: any;
  industry: any;
  competitors: any[];
}

interface AIAnalysisResult {
  model: string;
  analysis: string;
  recommendations: any[];
  confidence: number;
}

export const businessOrchestratorRouter = router({
  /**
   * Complete Business Analysis Orchestration
   * Fetches all data sources and runs multi-model AI analysis
   */
  runCompleteAnalysis: protectedProcedure
    .input(
      z.object({
        companyId: z.string(),
        orgnr: z.string().length(9),
      })
    )
    .mutation(async ({ ctx, input }) => {
      const db = await getDb();
      if (!db) {
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: "Database connection failed",
        });
      }

      try {
        // Step 1: Fetch company data from database
        const companyData = await db
          .select()
          .from(companies)
          .where(eq(companies.id, input.companyId))
          .limit(1);

        if (companyData.length === 0) {
          throw new TRPCError({
            code: "NOT_FOUND",
            message: "Company not found",
          });
        }

        const company = companyData[0];

        // Emit progress: Step 1 - Brreg data fetched
        emitAnalysisProgress(input.companyId, 1, "Fetched company data from Brreg.no");

        // Step 2: Enrich with all data sources
        const financial = await fetchFinancialData(input.orgnr);
        emitAnalysisProgress(input.companyId, 2, "Fetched financial data from Proff.no");

        const website = await analyzeWebsite(company);
        emitAnalysisProgress(input.companyId, 3, "Analyzed company website");

        const linkedin = await fetchLinkedInData(input.orgnr);
        emitAnalysisProgress(input.companyId, 4, "Fetched LinkedIn company data");

        const industry = await fetchIndustryBenchmarks(company.industryCode || "");
        const competitors = await identifyCompetitors(company);

        const enrichmentData: EnrichmentData = {
          company,
          financial,
          website,
          linkedin,
          industry,
          competitors,
        };

        // Step 3: Run multi-model AI analysis
        const aiAnalysisResults = await runMultiModelAnalysis(enrichmentData);

        // Emit progress: Step 5 - AI analysis complete
        emitAnalysisProgress(input.companyId, 5, "Completed multi-model AI analysis");

        // Step 4: Generate consensus recommendations
        const consensusRecommendations = generateConsensusRecommendations(aiAnalysisResults);

        // Step 5: Map recommendations to deeplinks and generate execution plans
        const executableRecommendations = await mapRecommendationsToDeeplinksWithAutomation(
          consensusRecommendations,
          company.industryCode || ""
        );

        // Calculate automation coverage
        const automationStats = calculateAutomationStats(executableRecommendations);

        // Step 6: Save results to database
        const analysisId = `analysis_${input.companyId}_${Date.now()}`;
        // TODO: Save to business_analyses table

        return {
          success: true,
          analysisId,
          company,
          aiAnalysisResults,
          consensusRecommendations,
          executableRecommendations,
          automationCoverage: automationStats.coveragePercentage,
          automationStats,
        };
      } catch (error) {
        console.error("[BusinessOrchestrator] Error:", error);
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: "Failed to run complete analysis",
        });
      }
    }),

  /**
   * Get analysis results by ID
   */
  getAnalysisResults: protectedProcedure
    .input(z.object({ analysisId: z.string() }))
    .query(async ({ input }) => {
      // TODO: Fetch from database
      return {
        analysisId: input.analysisId,
        status: "completed",
        recommendations: [],
      };
    }),
});

/**
 * Fetch financial data from Forvalt.no Premium + Proff.no
 */
async function fetchFinancialData(orgnr: string): Promise<any> {
  try {
    // Fetch real data from Forvalt.no premium platform
    const forvaltData = await scrapeForvaltData(orgnr);
    
    // Return comprehensive financial data
    return {
      // Credit Rating
      creditRating: forvaltData.creditRating,
      creditScore: forvaltData.creditScore,
      bankruptcyProbability: forvaltData.bankruptcyProbability,
      creditLimit: forvaltData.creditLimit,
      riskLevel: forvaltData.riskLevel,
      riskDescription: forvaltData.riskDescription,
      
      // Rating Components
      leadershipScore: forvaltData.leadershipScore,
      economyScore: forvaltData.economyScore,
      paymentHistoryScore: forvaltData.paymentHistoryScore,
      generalScore: forvaltData.generalScore,
      
      // Financial Metrics
      revenue: forvaltData.revenue,
      ebitda: forvaltData.ebitda,
      operatingResult: forvaltData.operatingResult,
      totalAssets: forvaltData.totalAssets,
      profitability: forvaltData.profitability,
      liquidity: forvaltData.liquidity,
      solidity: forvaltData.solidity,
      ebitdaMargin: forvaltData.ebitdaMargin,
      currency: forvaltData.currency,
      
      // Payment Remarks
      voluntaryLiens: forvaltData.voluntaryLiens,
      factoringAgreements: forvaltData.factoringAgreements,
      forcedLiens: forvaltData.forcedLiens,
      hasPaymentRemarks: forvaltData.hasPaymentRemarks,
      
      // Company Info
      companyName: forvaltData.companyName,
      employees: forvaltData.employees,
      website: forvaltData.website,
      phone: forvaltData.phone,
      
      // Leadership
      ceo: forvaltData.ceo,
      boardChairman: forvaltData.boardChairman,
      auditor: forvaltData.auditor,
      
      // Metadata
      lastUpdated: forvaltData.lastUpdated,
      forvaltUrl: forvaltData.forvaltUrl,
    };
  } catch (error) {
    console.error("[Forvalt] Error fetching financial data:", error);
    // Return null on error, orchestrator will handle gracefully
    return null;
  }
}

/**
 * Analyze company website with AI
 */
async function analyzeWebsite(company: any): Promise<any> {
  try {
    // Extract website URL from company data
    const rawData = company.rawData ? JSON.parse(company.rawData) : {};
    const website = rawData.hjemmeside || null;

    if (!website) {
      return { hasWebsite: false };
    }

    // Fetch website content
    const response = await axios.get(website, {
      timeout: 10000,
      headers: {
        "User-Agent": "Mozilla/5.0 (compatible; TrueASI/1.0; +https://trueasI.com)",
      },
    });

    const html = response.data;

    // AI-powered website analysis
    const analysis = await invokeLLM({
      messages: [
        {
          role: "system",
          content: "You are an expert web analyst. Analyze the website and provide insights on SEO, UX, content quality, and conversion optimization.",
        },
        {
          role: "user",
          content: `Analyze this website HTML and provide recommendations:\n\n${html.substring(0, 5000)}`,
        },
      ],
    });

    return {
      hasWebsite: true,
      url: website,
      analysis: analysis.choices[0].message.content,
      seoScore: 75, // TODO: Calculate real SEO score
      uxScore: 80,
      contentScore: 70,
    };
  } catch (error) {
    console.error("[Website] Error analyzing website:", error);
    return { hasWebsite: false, error: "Failed to analyze website" };
  }
}

/**
 * Fetch LinkedIn company data
 */
async function fetchLinkedInData(orgnr: string): Promise<any> {
  try {
    // TODO: Integrate real LinkedIn API
    // For now, return mock data
    return {
      followerCount: 1250,
      employeeCount: 45,
      description: "Leading Norwegian technology company",
      specialties: "Software Development, AI, Cloud Services",
      industry: "Information Technology",
      companySize: "11-50",
    };
  } catch (error) {
    console.error("[LinkedIn] Error fetching data:", error);
    return null;
  }
}

/**
 * Fetch industry benchmarks
 */
async function fetchIndustryBenchmarks(industryCode: string): Promise<any> {
  try {
    // TODO: Fetch real industry benchmarks from SSB or other sources
    return {
      avgRevenue: 12000000,
      avgProfit: 2000000,
      avgEmployees: 35,
      avgGrowthRate: 8.5,
    };
  } catch (error) {
    console.error("[Industry] Error fetching benchmarks:", error);
    return null;
  }
}

/**
 * Identify top competitors
 */
async function identifyCompetitors(company: any): Promise<any[]> {
  try {
    // TODO: Use AI to identify competitors based on industry and location
    return [
      { name: "Competitor A", revenue: 20000000 },
      { name: "Competitor B", revenue: 18000000 },
      { name: "Competitor C", revenue: 15000000 },
    ];
  } catch (error) {
    console.error("[Competitors] Error identifying competitors:", error);
    return [];
  }
}

/**
 * Run multi-model AI analysis with consensus algorithm
 */
async function runMultiModelAnalysis(data: EnrichmentData): Promise<AIAnalysisResult[]> {
  // Determine task type based on analysis needs
  const taskType: TaskType = "strategy"; // Business strategy analysis
  
  // Select optimal models for this task
  const selectedModels = selectModelsForTask(taskType, 5);
  const prompt = `Analyze this Norwegian company and provide comprehensive business recommendations:

**Company Data:**
- Name: ${data.company.name}
- Industry: ${data.company.industryDescription}
- Employees: ${data.company.employees}
- Location: ${data.company.municipality}

**Financial Data:**
- Revenue: ${data.financial?.revenue ? `${(data.financial.revenue / 1000000).toFixed(1)}M NOK` : "N/A"}
- Profit: ${data.financial?.profit ? `${(data.financial.profit / 1000000).toFixed(1)}M NOK` : "N/A"}
- Credit Rating: ${data.financial?.creditRating || "N/A"}

**Website Analysis:**
- Has Website: ${data.website?.hasWebsite ? "Yes" : "No"}
- SEO Score: ${data.website?.seoScore || "N/A"}
- UX Score: ${data.website?.uxScore || "N/A"}

**LinkedIn Data:**
- Followers: ${data.linkedin?.followerCount || "N/A"}
- Employee Count: ${data.linkedin?.employeeCount || "N/A"}

**Industry Benchmarks:**
- Avg Revenue: ${data.industry?.avgRevenue ? `${(data.industry.avgRevenue / 1000000).toFixed(1)}M NOK` : "N/A"}
- Avg Growth Rate: ${data.industry?.avgGrowthRate || "N/A"}%

Provide:
1. Revenue optimization strategies (5 specific actions)
2. Marketing recommendations (5 specific actions)
3. Leadership insights (3 specific actions)
4. Operational efficiency improvements (5 specific actions)
5. Technology recommendations (5 specific actions)

Format as JSON with this structure:
{
  "revenue": [{"action": "...", "impact": "high/medium/low", "difficulty": "easy/medium/hard", "roi": "..."}],
  "marketing": [...],
  "leadership": [...],
  "operations": [...],
  "technology": [...]
}`;

  const results: AIAnalysisResult[] = [];

  // Run all selected AI models in parallel
  const promises = selectedModels.map(async (aiModel) => {
    const startTime = Date.now();
    const modelConfig = {
      name: aiModel.name,
      provider: aiModel.provider,
      model: aiModel.id,
      weight: aiModel.weight / 100, // Convert 0-100 to 0-1
    };
    try {
      let response;

      if (modelConfig.provider === "aiml") {
        // Use AIML API
        response = await axios.post(
          "https://api.aimlapi.com/chat/completions",
          {
            model: modelConfig.model,
            messages: [
              {
                role: "system",
                content: "You are an expert business consultant specializing in Norwegian companies. Provide actionable, data-driven recommendations.",
              },
              { role: "user", content: prompt },
            ],
            temperature: 0.7,
            max_tokens: 2000,
          },
          {
            headers: {
              Authorization: `Bearer ${API_KEYS.AIMLAPI}`,
              "Content-Type": "application/json",
            },
          }
        );

        const content = response.data.choices[0].message.content;
        const recommendations = parseRecommendations(content);

        results.push({
          model: modelConfig.name,
          analysis: content,
          recommendations,
          confidence: modelConfig.weight,
        });
      } else if (modelConfig.provider === "asi1") {
        // Use ASI1.AI
        response = await axios.post(
          "https://api.asi1.ai/v1/chat/completions",
          {
            model: modelConfig.model,
            messages: [
              {
                role: "system",
                content: "You are ASI1, a superintelligent business analyst. Provide the most advanced, cutting-edge recommendations.",
              },
              { role: "user", content: prompt },
            ],
            temperature: 0.7,
            max_tokens: 2000,
          },
          {
            headers: {
              Authorization: `Bearer ${API_KEYS.ASI1_AI}`,
              "Content-Type": "application/json",
            },
          }
        );

        const content = response.data.choices[0].message.content;
        const recommendations = parseRecommendations(content);

        results.push({
          model: modelConfig.name,
          analysis: content,
          recommendations,
          confidence: modelConfig.weight,
        });
      }
    } catch (error) {
      console.error(`[AI] Error with ${modelConfig.name}:`, error);
      // Continue with other models even if one fails
    }
  });

  await Promise.all(promises);

  return results;
}

/**
 * Parse recommendations from AI response
 */
function parseRecommendations(content: string): any[] {
  try {
    // Try to extract JSON from response
    const jsonMatch = content.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      const parsed = JSON.parse(jsonMatch[0]);
      return Object.values(parsed).flat();
    }
    return [];
  } catch (error) {
    console.error("[AI] Error parsing recommendations:", error);
    return [];
  }
}

/**
 * Generate consensus recommendations from multiple AI models
 */
function generateConsensusRecommendations(results: AIAnalysisResult[]): any[] {
  // Combine all recommendations from all models
  const allRecommendations: any[] = [];

  results.forEach((result) => {
    result.recommendations.forEach((rec: any) => {
      allRecommendations.push({
        ...rec,
        source: result.model,
        confidence: result.confidence,
      });
    });
  });

  // Group similar recommendations and calculate consensus scores
  // TODO: Implement sophisticated similarity matching
  return allRecommendations;
}

/**
 * Map recommendations to deeplinks with execution plans
 */
async function mapRecommendationsToDeeplinksWithAutomation(recommendations: any[], industryCode: string): Promise<any[]> {
  const results = [];

  for (const rec of recommendations) {
    // Convert to Recommendation format
    const recommendation: Recommendation = {
      id: rec.id || `rec_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      title: rec.title || rec.recommendation || "Untitled Recommendation",
      description: rec.description || rec.details || rec.recommendation || "",
      category: (rec.category || "operations") as RecommendationCategory,
      impact: (rec.impact || "medium") as ImpactLevel,
      difficulty: (rec.difficulty || "medium") as DifficultyLevel,
      priority: rec.priority || 5,
      expectedROI: rec.expectedROI || "10-30%",
      cost: rec.cost || "$500-$5,000",
      timeframe: rec.timeframe || "1-3 months",
      isAutomated: false,
    };

    // Generate execution plan
    const executionPlan = generateExecutionPlan(recommendation);

    results.push({
      ...rec,
      executionPlan,
      automationLevel: executionPlan.automationLevel,
      platforms: executionPlan.platforms,
      estimatedTime: executionPlan.estimatedTime,
      totalCost: executionPlan.totalCost,
      expectedROI: executionPlan.expectedROI,
    });
  }

  return results;
}

/**
 * Calculate automation coverage statistics
 */
function calculateAutomationStats(recommendations: any[]) {
  const total = recommendations.length;
  const fullyAutomated = recommendations.filter(r => r.automationLevel === 'full').length;
  const partiallyAutomated = recommendations.filter(r => r.automationLevel === 'partial').length;
  const manual = recommendations.filter(r => r.automationLevel === 'manual').length;
  const totalPlatforms = recommendations.reduce((sum, r) => sum + (r.platforms?.length || 0), 0);

  return {
    total,
    fullyAutomated,
    partiallyAutomated,
    manual,
    coveragePercentage: total > 0 ? ((fullyAutomated + partiallyAutomated) / total) * 100 : 0,
    totalPlatforms,
  };
}
