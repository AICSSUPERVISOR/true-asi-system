/**
 * LinkedIn Profile Optimization Automation
 * Automatically optimizes LinkedIn profiles for all employees
 */

import { invokeLLM } from "./llm";
import { getCompanyEmployees, getLinkedInPerson, analyzeLinkedInProfile } from "./linkedin";

export interface ProfileOptimization {
  userId: string;
  username: string;
  currentProfile: {
    headline: string;
    summary: string;
    experience: any[];
    skills: string[];
    recommendations: number;
  };
  optimizations: {
    headline: {
      current: string;
      optimized: string;
      reasoning: string;
      impact: string;
    };
    summary: {
      current: string;
      optimized: string;
      reasoning: string;
      impact: string;
    };
    experience: Array<{
      title: string;
      company: string;
      currentDescription: string;
      optimizedDescription: string;
      reasoning: string;
    }>;
    skills: {
      currentSkills: string[];
      recommendedSkills: string[];
      skillsToRemove: string[];
      reasoning: string;
    };
    recommendations: {
      currentCount: number;
      targetCount: number;
      strategy: string;
    };
  };
  metrics: {
    currentScore: number;
    projectedScore: number;
    improvementPercentage: number;
    estimatedViewIncrease: string;
    estimatedConnectionIncrease: string;
  };
  actionPlan: Array<{
    step: number;
    action: string;
    priority: "critical" | "high" | "medium" | "low";
    estimatedTime: string;
    impact: string;
  }>;
}

/**
 * Optimize a single LinkedIn profile
 */
export async function optimizeLinkedInProfile(
  username: string,
  industry: string,
  companyName: string
): Promise<ProfileOptimization> {
  // Fetch current profile
  const profile = await getLinkedInPerson(username);
  const analysis = analyzeLinkedInProfile(profile);

  // Generate optimizations using AI
  const optimizationPrompt = `You are a LinkedIn optimization expert. Analyze this profile and provide specific, actionable improvements.

**Current Profile:**
- Headline: ${profile.headline || "Not set"}
- Summary: ${profile.summary || "Not set"}
- Experience: ${JSON.stringify(profile.experience?.slice(0, 3) || [])}
- Skills: ${profile.skills?.join(", ") || "None listed"}
- Industry: ${industry}
- Company: ${companyName}

**Analysis:**
- Profile Score: ${analysis.score}/100
- Improvements: ${analysis.improvements.join(", ")}

**Task:**
Generate optimized versions of:
1. Headline (max 220 characters, include keywords, value proposition)
2. Summary (max 2600 characters, storytelling, achievements, CTA)
3. Top 3 experience descriptions (achievement-focused, quantified results)
4. Recommended skills to add (10-15 industry-relevant skills)
5. Skills to remove (outdated or irrelevant)
6. Recommendation strategy (how to get more recommendations)

Return as JSON with this structure:
{
  "headline": { "optimized": "...", "reasoning": "...", "impact": "..." },
  "summary": { "optimized": "...", "reasoning": "...", "impact": "..." },
  "experience": [{ "title": "...", "optimizedDescription": "...", "reasoning": "..." }],
  "skills": { "toAdd": ["..."], "toRemove": ["..."], "reasoning": "..." },
  "recommendations": { "strategy": "...", "targetCount": 15 }
}`;

  const response: any = await invokeLLM({
    messages: [
      {
        role: "system",
        content:
          "You are a LinkedIn optimization expert with 10+ years of experience. Provide specific, actionable, and high-impact recommendations.",
      },
      { role: "user", content: optimizationPrompt },
    ],
    response_format: {
      type: "json_schema",
      json_schema: {
        name: "linkedin_optimization",
        strict: true,
        schema: {
          type: "object",
          properties: {
            headline: {
              type: "object",
              properties: {
                optimized: { type: "string" },
                reasoning: { type: "string" },
                impact: { type: "string" },
              },
              required: ["optimized", "reasoning", "impact"],
              additionalProperties: false,
            },
            summary: {
              type: "object",
              properties: {
                optimized: { type: "string" },
                reasoning: { type: "string" },
                impact: { type: "string" },
              },
              required: ["optimized", "reasoning", "impact"],
              additionalProperties: false,
            },
            experience: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  title: { type: "string" },
                  optimizedDescription: { type: "string" },
                  reasoning: { type: "string" },
                },
                required: ["title", "optimizedDescription", "reasoning"],
                additionalProperties: false,
              },
            },
            skills: {
              type: "object",
              properties: {
                toAdd: { type: "array", items: { type: "string" } },
                toRemove: { type: "array", items: { type: "string" } },
                reasoning: { type: "string" },
              },
              required: ["toAdd", "toRemove", "reasoning"],
              additionalProperties: false,
            },
            recommendations: {
              type: "object",
              properties: {
                strategy: { type: "string" },
                targetCount: { type: "number" },
              },
              required: ["strategy", "targetCount"],
              additionalProperties: false,
            },
          },
          required: ["headline", "summary", "experience", "skills", "recommendations"],
          additionalProperties: false,
        },
      },
    },
  });

  const optimizations = JSON.parse(response.choices[0].message.content || "{}");

  // Calculate metrics
  const currentScore = analysis.score;
  const projectedScore = Math.min(100, currentScore + 25); // Estimated improvement
  const improvementPercentage = ((projectedScore - currentScore) / currentScore) * 100;

  // Build action plan
  const actionPlan = [
    {
      step: 1,
      action: "Update headline with optimized version",
      priority: "critical" as const,
      estimatedTime: "5 minutes",
      impact: "Immediate visibility boost in search results",
    },
    {
      step: 2,
      action: "Rewrite summary with storytelling approach",
      priority: "critical" as const,
      estimatedTime: "15 minutes",
      impact: "Increases profile views by 30-50%",
    },
    {
      step: 3,
      action: "Update top 3 experience descriptions",
      priority: "high" as const,
      estimatedTime: "30 minutes",
      impact: "Demonstrates value and achievements",
    },
    {
      step: 4,
      action: `Add ${optimizations.skills.toAdd.length} recommended skills`,
      priority: "high" as const,
      estimatedTime: "10 minutes",
      impact: "Improves discoverability in recruiter searches",
    },
    {
      step: 5,
      action: `Remove ${optimizations.skills.toRemove.length} outdated skills`,
      priority: "medium" as const,
      estimatedTime: "5 minutes",
      impact: "Focuses profile on current expertise",
    },
    {
      step: 6,
      action: "Request recommendations from colleagues",
      priority: "medium" as const,
      estimatedTime: "20 minutes",
      impact: "Builds social proof and credibility",
    },
  ];

  return {
    userId: profile.userId || "",
    username,
    currentProfile: {
      headline: profile.headline || "",
      summary: profile.summary || "",
      experience: profile.experience || [],
      skills: profile.skills || [],
      recommendations: profile.recommendations || 0,
    },
    optimizations: {
      headline: {
        current: profile.headline || "",
        optimized: optimizations.headline.optimized,
        reasoning: optimizations.headline.reasoning,
        impact: optimizations.headline.impact,
      },
      summary: {
        current: profile.summary || "",
        optimized: optimizations.summary.optimized,
        reasoning: optimizations.summary.reasoning,
        impact: optimizations.summary.impact,
      },
      experience: optimizations.experience.map((exp: any, idx: number) => ({
        title: exp.title,
        company: profile.experience?.[idx]?.company || "",
        currentDescription: profile.experience?.[idx]?.description || "",
        optimizedDescription: exp.optimizedDescription,
        reasoning: exp.reasoning,
      })),
      skills: {
        currentSkills: profile.skills || [],
        recommendedSkills: optimizations.skills.toAdd,
        skillsToRemove: optimizations.skills.toRemove,
        reasoning: optimizations.skills.reasoning,
      },
      recommendations: {
        currentCount: profile.recommendations || 0,
        targetCount: optimizations.recommendations.targetCount,
        strategy: optimizations.recommendations.strategy,
      },
    },
    metrics: {
      currentScore,
      projectedScore,
      improvementPercentage: Math.round(improvementPercentage),
      estimatedViewIncrease: "30-50%",
      estimatedConnectionIncrease: "20-40%",
    },
    actionPlan,
  };
}

/**
 * Optimize all employee profiles for a company
 */
export async function optimizeAllEmployeeProfiles(
  companyLinkedInUrl: string,
  industry: string,
  companyName: string,
  maxEmployees: number = 50
): Promise<{
  companyName: string;
  totalEmployees: number;
  optimizedCount: number;
  optimizations: ProfileOptimization[];
  aggregateMetrics: {
    averageCurrentScore: number;
    averageProjectedScore: number;
    totalImprovementPercentage: number;
    estimatedCompanyVisibilityIncrease: string;
  };
}> {
  // Fetch all employees
  const employees = await getCompanyEmployees(companyLinkedInUrl, maxEmployees);

  // Optimize each profile
  const optimizations: ProfileOptimization[] = [];
  for (const employee of employees.slice(0, maxEmployees)) {
    try {
      const optimization = await optimizeLinkedInProfile(
        employee.username,
        industry,
        companyName
      );
      optimizations.push(optimization);
    } catch (error) {
      console.error(`Failed to optimize profile for ${employee.username}:`, error);
    }
  }

  // Calculate aggregate metrics
  const averageCurrentScore =
    optimizations.reduce((sum, opt) => sum + opt.metrics.currentScore, 0) /
    optimizations.length;
  const averageProjectedScore =
    optimizations.reduce((sum, opt) => sum + opt.metrics.projectedScore, 0) /
    optimizations.length;
  const totalImprovementPercentage =
    ((averageProjectedScore - averageCurrentScore) / averageCurrentScore) * 100;

  return {
    companyName,
    totalEmployees: employees.length,
    optimizedCount: optimizations.length,
    optimizations,
    aggregateMetrics: {
      averageCurrentScore: Math.round(averageCurrentScore),
      averageProjectedScore: Math.round(averageProjectedScore),
      totalImprovementPercentage: Math.round(totalImprovementPercentage),
      estimatedCompanyVisibilityIncrease: "40-70%",
    },
  };
}

/**
 * Execute approved optimizations (update LinkedIn profiles)
 * Note: This requires LinkedIn API access or manual updates
 */
export async function executeProfileOptimizations(
  optimizations: ProfileOptimization[],
  approvedUsernames: string[]
): Promise<{
  executed: number;
  failed: number;
  results: Array<{
    username: string;
    status: "success" | "failed";
    message: string;
  }>;
}> {
  const results: Array<{
    username: string;
    status: "success" | "failed";
    message: string;
  }> = [];

  for (const optimization of optimizations) {
    if (!approvedUsernames.includes(optimization.username)) {
      continue;
    }

    try {
      // TODO: Implement actual LinkedIn profile update via API
      // For now, we'll simulate success
      results.push({
        username: optimization.username,
        status: "success",
        message: "Profile optimization instructions generated. Manual update required.",
      });
    } catch (error) {
      results.push({
        username: optimization.username,
        status: "failed",
        message: error instanceof Error ? error.message : "Unknown error",
      });
    }
  }

  const executed = results.filter((r) => r.status === "success").length;
  const failed = results.filter((r) => r.status === "failed").length;

  return { executed, failed, results };
}
