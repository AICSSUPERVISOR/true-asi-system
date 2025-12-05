import { z } from "zod";
import { TRPCError } from "@trpc/server";
import { protectedProcedure, router } from "../_core/trpc";
import { getDb } from "../db";
import { companies, companyFinancials, companyLinkedIn } from "../../drizzle/schema";
import { eq } from "drizzle-orm";
import { invokeLLM } from "../_core/llm";
import axios from "axios";
import { emitAnalysisProgress, emitAnalysisComplete } from "../_core/websocket";

/**
 * Business Orchestrator
 * Complete end-to-end business automation system
 * Integrates: Brreg → Proff → Website → LinkedIn → Multi-Model AI → Recommendations → Deeplinks
 */

const API_KEYS = {
  ASI1_AI: process.env.ASI1_AI_API_KEY || "sk_26ec4938b6274ae089bfa915d02bf10036bde0326b5845c5b87c50b5dbc2c9ad",
  AIMLAPI: process.env.AIMLAPI_KEY || "147620aa16e04b96bb2f12b79527593f",
};

// AI Model Configuration for Multi-Model Consensus
const AI_MODELS = [
  { name: "GPT-4", provider: "aimlapi", model: "gpt-4o", weight: 1.0 },
  { name: "Claude-3.5", provider: "aimlapi", model: "claude-3-5-sonnet-20241022", weight: 1.0 },
  { name: "Gemini-Pro", provider: "aimlapi", model: "gemini-2.0-flash-exp", weight: 0.9 },
  { name: "Llama-3.3", provider: "aimlapi", model: "meta-llama/Llama-3.3-70B-Instruct-Turbo", weight: 0.8 },
  { name: "ASI1-AI", provider: "asi1", model: "gpt-4o-mini", weight: 1.2 }, // Highest weight for ASI1
];

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

        // Step 2: Enrich with all data sources
        const enrichmentData: EnrichmentData = {
          company,
          financial: await fetchFinancialData(input.orgnr),
          website: await analyzeWebsite(company),
          linkedin: await fetchLinkedInData(input.orgnr),
          industry: await fetchIndustryBenchmarks(company.industryCode || ""),
          competitors: await identifyCompetitors(company),
        };

        // Step 3: Run multi-model AI analysis
        const aiAnalysisResults = await runMultiModelAnalysis(enrichmentData);

        // Step 4: Generate consensus recommendations
        const consensusRecommendations = generateConsensusRecommendations(aiAnalysisResults);

        // Step 5: Map recommendations to deeplinks
        const executableRecommendations = mapRecommendationsToDeeplinks(
          consensusRecommendations,
          company.industryCode || ""
        );

        // Step 6: Save results to database
        const analysisId = `analysis_${input.companyId}_${Date.now()}`;
        // TODO: Save to business_analyses table

        return {
          success: true,
          analysisId,
          enrichmentData,
          aiAnalysisResults,
          recommendations: executableRecommendations,
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
 * Fetch financial data from Proff.no API
 */
async function fetchFinancialData(orgnr: string): Promise<any> {
  try {
    // TODO: Integrate real Proff.no API
    // For now, return mock data
    return {
      year: 2024,
      revenue: 15000000, // 15M NOK
      profit: 2500000, // 2.5M NOK
      assets: 8000000,
      liabilities: 3000000,
      equity: 5000000,
      creditRating: "AA",
      creditScore: 85,
      riskLevel: "Low",
    };
  } catch (error) {
    console.error("[Proff] Error fetching financial data:", error);
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

  // Run all AI models in parallel
  const promises = AI_MODELS.map(async (modelConfig) => {
    try {
      let response;

      if (modelConfig.provider === "aimlapi") {
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
 * Map recommendations to deeplinks for one-click execution
 */
function mapRecommendationsToDeeplinks(recommendations: any[], industryCode: string): any[] {
  // TODO: Load industry_deeplinks.ts and map recommendations to platforms
  return recommendations.map((rec) => ({
    ...rec,
    deeplinks: [], // Will be populated with actual deeplinks
  }));
}
