/**
 * AI-powered industry categorization and needs assessment engine
 */

import { invokeLLM } from "./llm";
import type { BusinessProfile } from "../business_db";
import type { LinkedInCompanyProfile } from "./linkedin";

export interface IndustryCategory {
  primary: string;
  secondary: string[];
  naceCode: string;
  description: string;
}

export interface BusinessNeeds {
  critical: string[];
  high: string[];
  medium: string[];
  low: string[];
  opportunities: string[];
}

export interface CompetitiveAnalysis {
  strengths: string[];
  weaknesses: string[];
  opportunities: string[];
  threats: string[];
  marketPosition: "leader" | "challenger" | "follower" | "niche";
  competitiveAdvantages: string[];
}

export interface RevenueOptimizationStrategy {
  currentRevenue: string;
  projectedGrowth: string;
  strategies: Array<{
    category: string;
    actions: string[];
    expectedImpact: string;
    timeframe: string;
    priority: "critical" | "high" | "medium" | "low";
  }>;
  quickWins: string[];
  longTermInitiatives: string[];
}

/**
 * Categorize business into detailed industry taxonomy
 */
export async function categorizeIndustry(
  businessProfile: BusinessProfile,
  linkedInProfile?: LinkedInCompanyProfile
): Promise<IndustryCategory> {
  const prompt = `Analyze this Norwegian business and provide detailed industry categorization:

Company: ${businessProfile.name}
NACE Code: ${businessProfile.industry_code} - ${businessProfile.industry_description}
Employees: ${businessProfile.employees}
Website: ${businessProfile.website || "N/A"}
${linkedInProfile ? `LinkedIn Industries: ${linkedInProfile.industries.join(", ")}` : ""}
${linkedInProfile ? `Specialties: ${linkedInProfile.specialities.join(", ")}` : ""}
${linkedInProfile ? `Description: ${linkedInProfile.description}` : ""}

Provide industry categorization in JSON format:
{
  "primary": "Main industry category",
  "secondary": ["Related industries"],
  "naceCode": "${businessProfile.industry_code}",
  "description": "Detailed industry description"
}`;

  const response = await invokeLLM({
    messages: [
      {
        role: "system",
        content:
          "You are an expert business analyst specializing in industry classification. Provide accurate, detailed categorization.",
      },
      { role: "user", content: prompt },
    ],
    response_format: {
      type: "json_schema",
      json_schema: {
        name: "industry_category",
        strict: true,
        schema: {
          type: "object",
          properties: {
            primary: { type: "string" },
            secondary: { type: "array", items: { type: "string" } },
            naceCode: { type: "string" },
            description: { type: "string" },
          },
          required: ["primary", "secondary", "naceCode", "description"],
          additionalProperties: false,
        },
      },
    },
  });

  const content = response.choices[0].message.content as string;
  return JSON.parse(content) as IndustryCategory;
}

/**
 * Assess business needs based on industry, size, and current state
 */
export async function assessBusinessNeeds(
  businessProfile: BusinessProfile,
  industryCategory: IndustryCategory,
  linkedInProfile?: LinkedInCompanyProfile
): Promise<BusinessNeeds> {
  const prompt = `Analyze this business and identify specific needs for improvement:

Company: ${businessProfile.name}
Industry: ${industryCategory.primary}
Employees: ${businessProfile.employees}
Website: ${businessProfile.website || "None"}
LinkedIn Followers: ${linkedInProfile?.followerCount || "Unknown"}
Staff Count: ${linkedInProfile?.staffCount || businessProfile.employees}

Assess needs in these categories:
- Digital presence & marketing
- Operations & efficiency
- Technology & automation
- Customer experience
- Employee engagement
- Revenue growth
- Compliance & risk

Provide assessment in JSON format with prioritized needs.`;

  const response = await invokeLLM({
    messages: [
      {
        role: "system",
        content:
          "You are a business consultant expert. Identify specific, actionable needs based on company data.",
      },
      { role: "user", content: prompt },
    ],
    response_format: {
      type: "json_schema",
      json_schema: {
        name: "business_needs",
        strict: true,
        schema: {
          type: "object",
          properties: {
            critical: { type: "array", items: { type: "string" } },
            high: { type: "array", items: { type: "string" } },
            medium: { type: "array", items: { type: "string" } },
            low: { type: "array", items: { type: "string" } },
            opportunities: { type: "array", items: { type: "string" } },
          },
          required: ["critical", "high", "medium", "low", "opportunities"],
          additionalProperties: false,
        },
      },
    },
  });

  const content = response.choices[0].message.content as string;
  return JSON.parse(content) as BusinessNeeds;
}

/**
 * Perform SWOT analysis
 */
export async function performSWOTAnalysis(
  businessProfile: BusinessProfile,
  industryCategory: IndustryCategory,
  linkedInProfile?: LinkedInCompanyProfile
): Promise<CompetitiveAnalysis> {
  const prompt = `Perform comprehensive SWOT analysis for this business:

Company: ${businessProfile.name}
Industry: ${industryCategory.primary}
Employees: ${businessProfile.employees}
Website: ${businessProfile.website || "None"}
${linkedInProfile ? `LinkedIn Presence: ${linkedInProfile.followerCount} followers` : ""}
${linkedInProfile ? `Specialties: ${linkedInProfile.specialities.join(", ")}` : ""}

Analyze:
1. Strengths (internal advantages)
2. Weaknesses (internal limitations)
3. Opportunities (external possibilities)
4. Threats (external challenges)
5. Market position (leader/challenger/follower/niche)
6. Competitive advantages

Provide detailed SWOT analysis in JSON format.`;

  const response = await invokeLLM({
    messages: [
      {
        role: "system",
        content:
          "You are a strategic business analyst. Provide thorough, realistic SWOT analysis.",
      },
      { role: "user", content: prompt },
    ],
    response_format: {
      type: "json_schema",
      json_schema: {
        name: "swot_analysis",
        strict: true,
        schema: {
          type: "object",
          properties: {
            strengths: { type: "array", items: { type: "string" } },
            weaknesses: { type: "array", items: { type: "string" } },
            opportunities: { type: "array", items: { type: "string" } },
            threats: { type: "array", items: { type: "string" } },
            marketPosition: {
              type: "string",
              enum: ["leader", "challenger", "follower", "niche"],
            },
            competitiveAdvantages: { type: "array", items: { type: "string" } },
          },
          required: [
            "strengths",
            "weaknesses",
            "opportunities",
            "threats",
            "marketPosition",
            "competitiveAdvantages",
          ],
          additionalProperties: false,
        },
      },
    },
  });

  const content = response.choices[0].message.content as string;
  return JSON.parse(content) as CompetitiveAnalysis;
}

/**
 * Generate revenue optimization strategy
 */
export async function generateRevenueStrategy(
  businessProfile: BusinessProfile,
  industryCategory: IndustryCategory,
  businessNeeds: BusinessNeeds,
  swotAnalysis: CompetitiveAnalysis
): Promise<RevenueOptimizationStrategy> {
  const prompt = `Create comprehensive revenue optimization strategy:

Company: ${businessProfile.name}
Industry: ${industryCategory.primary}
Employees: ${businessProfile.employees}
Market Position: ${swotAnalysis.marketPosition}

Critical Needs: ${businessNeeds.critical.join(", ")}
Opportunities: ${businessNeeds.opportunities.join(", ")}
Strengths: ${swotAnalysis.strengths.join(", ")}

Generate:
1. Current revenue estimate
2. Projected growth potential
3. Specific strategies with actions, impact, timeframe, priority
4. Quick wins (0-3 months)
5. Long-term initiatives (6-24 months)

Focus on:
- Digital transformation
- Marketing & sales optimization
- Operational efficiency
- Customer retention
- New revenue streams
- Cost reduction

Provide detailed strategy in JSON format.`;

  const response = await invokeLLM({
    messages: [
      {
        role: "system",
        content:
          "You are a revenue optimization expert. Create actionable, measurable strategies.",
      },
      { role: "user", content: prompt },
    ],
    response_format: {
      type: "json_schema",
      json_schema: {
        name: "revenue_strategy",
        strict: true,
        schema: {
          type: "object",
          properties: {
            currentRevenue: { type: "string" },
            projectedGrowth: { type: "string" },
            strategies: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  category: { type: "string" },
                  actions: { type: "array", items: { type: "string" } },
                  expectedImpact: { type: "string" },
                  timeframe: { type: "string" },
                  priority: {
                    type: "string",
                    enum: ["critical", "high", "medium", "low"],
                  },
                },
                required: ["category", "actions", "expectedImpact", "timeframe", "priority"],
                additionalProperties: false,
              },
            },
            quickWins: { type: "array", items: { type: "string" } },
            longTermInitiatives: { type: "array", items: { type: "string" } },
          },
          required: [
            "currentRevenue",
            "projectedGrowth",
            "strategies",
            "quickWins",
            "longTermInitiatives",
          ],
          additionalProperties: false,
        },
      },
    },
  });

  const content = response.choices[0].message.content as string;
  return JSON.parse(content) as RevenueOptimizationStrategy;
}

/**
 * Comprehensive business analysis
 */
export async function analyzeBusinessComprehensively(
  businessProfile: BusinessProfile,
  linkedInProfile?: LinkedInCompanyProfile
) {
  console.log(`[Industry Analysis] Starting comprehensive analysis for ${businessProfile.name}`);

  // Step 1: Categorize industry
  const industryCategory = await categorizeIndustry(businessProfile, linkedInProfile);
  console.log(`[Industry Analysis] Categorized as: ${industryCategory.primary}`);

  // Step 2: Assess needs
  const businessNeeds = await assessBusinessNeeds(
    businessProfile,
    industryCategory,
    linkedInProfile
  );
  console.log(
    `[Industry Analysis] Identified ${businessNeeds.critical.length} critical needs`
  );

  // Step 3: SWOT analysis
  const swotAnalysis = await performSWOTAnalysis(
    businessProfile,
    industryCategory,
    linkedInProfile
  );
  console.log(`[Industry Analysis] Market position: ${swotAnalysis.marketPosition}`);

  // Step 4: Revenue strategy
  const revenueStrategy = await generateRevenueStrategy(
    businessProfile,
    industryCategory,
    businessNeeds,
    swotAnalysis
  );
  console.log(
    `[Industry Analysis] Generated ${revenueStrategy.strategies.length} revenue strategies`
  );

  return {
    industryCategory,
    businessNeeds,
    swotAnalysis,
    revenueStrategy,
    generatedAt: new Date().toISOString(),
  };
}
