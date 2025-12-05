/**
 * AUTOMATED RECOMMENDATION ENGINE
 * 
 * This module generates specific, actionable recommendations based on:
 * - Complete business intelligence data
 * - Industry best practices
 * - Competitor analysis
 * - AI-powered gap analysis
 * 
 * Provides:
 * - Prioritized recommendations (high/medium/low ROI)
 * - Detailed implementation plans
 * - Cost and time estimates
 * - Expected ROI calculations
 * - Automated workflow mapping to platforms
 */

import { invokeAI } from './backend_integration';
import { CompleteBusinessIntelligence } from './enhanced_scraper';
import { getRecommendedPlatforms } from './industry_deeplinks';
import { UniversalAutomationEngine, AutomationWorkflow } from './automation_engine';

// ============================================================================
// TYPES
// ============================================================================

export interface Recommendation {
  id: string;
  category: 'website' | 'linkedin' | 'marketing' | 'operations' | 'sales' | 'customer-service';
  title: string;
  description: string;
  priority: 'high' | 'medium' | 'low';
  estimatedCost: number; // USD
  estimatedTime: string; // e.g., "2 weeks"
  expectedROI: number; // percentage
  confidence: number; // 0-100
  
  // Implementation details
  steps: Array<{
    stepNumber: number;
    action: string;
    details: string;
  }>;
  
  // Automation mapping
  automatable: boolean;
  automationPlatforms: string[];
  
  // Metrics
  metrics: {
    before: Record<string, number>;
    after: Record<string, number>;
    improvement: Record<string, number>;
  };
}

export interface GapAnalysis {
  websiteGaps: string[];
  linkedinGaps: string[];
  marketingGaps: string[];
  operationalGaps: string[];
  salesGaps: string[];
  customerServiceGaps: string[];
}

export interface RecommendationReport {
  businessId: string;
  organizationNumber: string;
  companyName: string;
  industryCode: string;
  
  // Analysis
  gapAnalysis: GapAnalysis;
  overallScore: number; // 0-100
  competitivePosition: 'leader' | 'challenger' | 'follower' | 'niche';
  
  // Recommendations
  recommendations: Recommendation[];
  totalEstimatedCost: number;
  totalExpectedROI: number;
  implementationTimeline: string;
  
  // Automation
  automationWorkflow: AutomationWorkflow | null;
  
  // Metadata
  generatedAt: Date;
  dataCompleteness: number;
}

// ============================================================================
// GAP ANALYSIS
// ============================================================================

/**
 * Perform comprehensive gap analysis using AI
 */
export async function performGapAnalysis(
  intelligence: CompleteBusinessIntelligence
): Promise<GapAnalysis> {
  const prompt = `Perform comprehensive gap analysis for this business:

Company: ${intelligence.name}
Industry: ${intelligence.industryName}

Current State:
- Website: ${JSON.stringify(intelligence.website, null, 2)}
- LinkedIn: ${JSON.stringify(intelligence.linkedin, null, 2)}
- Social Media: ${JSON.stringify(intelligence.socialMedia, null, 2)}
- Reviews: ${JSON.stringify(intelligence.reviews, null, 2)}
- Competitors: ${JSON.stringify(intelligence.competitors, null, 2)}
- Advertising: ${JSON.stringify(intelligence.advertising, null, 2)}

Identify specific gaps in:
1. Website (SEO, performance, content, UX, mobile, accessibility)
2. LinkedIn (company page, employee profiles, content, engagement)
3. Marketing (social media, content marketing, email, ads)
4. Operations (automation, tools, processes, efficiency)
5. Sales (CRM, lead generation, sales funnel, conversion)
6. Customer Service (support tools, response time, satisfaction)

Return detailed JSON with specific, actionable gaps.`;

  const response = await invokeAI({
    prompt,
    systemPrompt: 'You are an expert business consultant specializing in digital transformation and growth strategies.',
    maxTokens: 3000
  });
  
  // Parse AI response
  let gaps: GapAnalysis;
  try {
    const parsed = JSON.parse(response.content);
    gaps = {
      websiteGaps: parsed.websiteGaps || [],
      linkedinGaps: parsed.linkedinGaps || [],
      marketingGaps: parsed.marketingGaps || [],
      operationalGaps: parsed.operationalGaps || [],
      salesGaps: parsed.salesGaps || [],
      customerServiceGaps: parsed.customerServiceGaps || []
    };
  } catch (error) {
    console.error('[Recommendation Engine] Failed to parse gap analysis:', error);
    gaps = {
      websiteGaps: [],
      linkedinGaps: [],
      marketingGaps: [],
      operationalGaps: [],
      salesGaps: [],
      customerServiceGaps: []
    };
  }
  
  return gaps;
}

// ============================================================================
// RECOMMENDATION GENERATION
// ============================================================================

/**
 * Generate specific recommendations based on gap analysis
 */
export async function generateRecommendations(
  intelligence: CompleteBusinessIntelligence,
  gaps: GapAnalysis
): Promise<Recommendation[]> {
  const allGaps = [
    ...gaps.websiteGaps,
    ...gaps.linkedinGaps,
    ...gaps.marketingGaps,
    ...gaps.operationalGaps,
    ...gaps.salesGaps,
    ...gaps.customerServiceGaps
  ];
  
  const prompt = `Generate specific, actionable recommendations to address these gaps:

Company: ${intelligence.name}
Industry: ${intelligence.industryName}

Gaps:
${allGaps.map((gap, i) => `${i + 1}. ${gap}`).join('\n')}

For each gap, create a recommendation with:
1. Category (website, linkedin, marketing, operations, sales, customer-service)
2. Title (concise, actionable)
3. Description (detailed explanation)
4. Priority (high, medium, low) based on ROI potential
5. Estimated cost (USD)
6. Estimated time (e.g., "2 weeks", "1 month")
7. Expected ROI (percentage)
8. Confidence (0-100)
9. Implementation steps (detailed, numbered)
10. Automatable (true/false)
11. Automation platforms (if automatable)
12. Metrics (before, after, improvement)

Return detailed JSON array of recommendations.`;

  const response = await invokeAI({
    prompt,
    systemPrompt: 'You are an expert business consultant who creates detailed, actionable recommendations with realistic cost and ROI estimates.',
    maxTokens: 4000
  });
  
  // Parse AI response
  let recommendations: Recommendation[];
  try {
    const parsed = JSON.parse(response.content);
    recommendations = (parsed.recommendations || parsed || []).map((rec: any, index: number) => ({
      id: `rec_${Date.now()}_${index}`,
      category: rec.category || 'operations',
      title: rec.title || 'Untitled Recommendation',
      description: rec.description || '',
      priority: rec.priority || 'medium',
      estimatedCost: rec.estimatedCost || 0,
      estimatedTime: rec.estimatedTime || '1 week',
      expectedROI: rec.expectedROI || 0,
      confidence: rec.confidence || 70,
      steps: rec.steps || [],
      automatable: rec.automatable || false,
      automationPlatforms: rec.automationPlatforms || [],
      metrics: rec.metrics || {
        before: {},
        after: {},
        improvement: {}
      }
    }));
  } catch (error) {
    console.error('[Recommendation Engine] Failed to parse recommendations:', error);
    recommendations = [];
  }
  
  return recommendations;
}

// ============================================================================
// OVERALL SCORE CALCULATION
// ============================================================================

/**
 * Calculate overall digital maturity score (0-100)
 */
export function calculateOverallScore(intelligence: CompleteBusinessIntelligence): number {
  let score = 0;
  
  // Website (30 points)
  if (intelligence.website.seo.hasMetaTags) score += 3;
  if (intelligence.website.seo.hasStructuredData) score += 3;
  if (intelligence.website.seo.hasSitemap) score += 2;
  if (intelligence.website.seo.mobileOptimized) score += 5;
  score += (intelligence.website.seo.pageSpeed / 100) * 7;
  score += (intelligence.website.seo.accessibility / 100) * 5;
  if (intelligence.website.content.hasContactForm) score += 2;
  if (intelligence.website.content.hasBlog) score += 3;
  
  // LinkedIn (20 points)
  if (intelligence.linkedin.companyPage.followers > 100) score += 5;
  if (intelligence.linkedin.companyPage.followers > 1000) score += 5;
  if (intelligence.linkedin.employees.length > 5) score += 5;
  if (intelligence.linkedin.companyPage.recentPosts > 4) score += 5;
  
  // Social Media (15 points)
  if (intelligence.socialMedia.facebook.url) score += 3;
  if (intelligence.socialMedia.instagram.url) score += 3;
  if (intelligence.socialMedia.twitter.url) score += 2;
  if (intelligence.socialMedia.youtube.url) score += 2;
  if (intelligence.socialMedia.facebook.followers > 500) score += 2.5;
  if (intelligence.socialMedia.instagram.followers > 500) score += 2.5;
  
  // Reviews (15 points)
  if (intelligence.reviews.google.reviewCount > 10) score += 5;
  if (intelligence.reviews.google.rating >= 4.0) score += 5;
  if (intelligence.reviews.overallSentiment === 'positive') score += 5;
  
  // Advertising (10 points)
  if (intelligence.advertising.googleAds.isActive) score += 5;
  if (intelligence.advertising.facebookAds.isActive) score += 3;
  if (intelligence.advertising.linkedinAds.isActive) score += 2;
  
  // Competitors (10 points)
  if (intelligence.competitors.competitivePosition === 'leader') score += 10;
  else if (intelligence.competitors.competitivePosition === 'challenger') score += 7;
  else if (intelligence.competitors.competitivePosition === 'follower') score += 4;
  else score += 2;
  
  return Math.min(Math.round(score), 100);
}

// ============================================================================
// MAIN ORCHESTRATION
// ============================================================================

/**
 * Generate complete recommendation report
 */
export async function generateRecommendationReport(
  intelligence: CompleteBusinessIntelligence
): Promise<RecommendationReport> {
  console.log(`[Recommendation Engine] Generating report for ${intelligence.name}`);
  
  // 1. Perform gap analysis
  const gapAnalysis = await performGapAnalysis(intelligence);
  
  // 2. Calculate overall score
  const overallScore = calculateOverallScore(intelligence);
  
  // 3. Generate recommendations
  const recommendations = await generateRecommendations(intelligence, gapAnalysis);
  
  // 4. Sort recommendations by priority and ROI
  const sortedRecommendations = recommendations.sort((a, b) => {
    const priorityWeight = { high: 3, medium: 2, low: 1 };
    const aPriority = priorityWeight[a.priority];
    const bPriority = priorityWeight[b.priority];
    
    if (aPriority !== bPriority) {
      return bPriority - aPriority;
    }
    
    return b.expectedROI - a.expectedROI;
  });
  
  // 5. Calculate totals
  const totalEstimatedCost = sortedRecommendations.reduce((sum, rec) => sum + rec.estimatedCost, 0);
  const totalExpectedROI = sortedRecommendations.length > 0
    ? sortedRecommendations.reduce((sum, rec) => sum + rec.expectedROI, 0) / sortedRecommendations.length
    : 0;
  
  // 6. Estimate implementation timeline
  const highPriorityCount = sortedRecommendations.filter(r => r.priority === 'high').length;
  const mediumPriorityCount = sortedRecommendations.filter(r => r.priority === 'medium').length;
  const lowPriorityCount = sortedRecommendations.filter(r => r.priority === 'low').length;
  
  const totalWeeks = (highPriorityCount * 2) + (mediumPriorityCount * 1) + (lowPriorityCount * 0.5);
  const implementationTimeline = totalWeeks < 4 
    ? `${Math.ceil(totalWeeks)} weeks`
    : `${Math.ceil(totalWeeks / 4)} months`;
  
  // 7. Generate automation workflow
  let automationWorkflow: AutomationWorkflow | null = null;
  try {
    const businessNeeds = {
      websiteImprovements: gapAnalysis.websiteGaps,
      linkedinImprovements: gapAnalysis.linkedinGaps,
      marketingImprovements: gapAnalysis.marketingGaps,
      operationalImprovements: gapAnalysis.operationalGaps,
      priority: overallScore < 50 ? 'high' as const : overallScore < 75 ? 'medium' as const : 'low' as const
    };
    
    automationWorkflow = await UniversalAutomationEngine.generateWorkflow(
      intelligence.organizationNumber,
      intelligence.industryCode,
      businessNeeds
    );
  } catch (error) {
    console.error('[Recommendation Engine] Failed to generate automation workflow:', error);
  }
  
  // 8. Assemble report
  const report: RecommendationReport = {
    businessId: intelligence.organizationNumber,
    organizationNumber: intelligence.organizationNumber,
    companyName: intelligence.name,
    industryCode: intelligence.industryCode,
    gapAnalysis,
    overallScore,
    competitivePosition: intelligence.competitors.competitivePosition,
    recommendations: sortedRecommendations,
    totalEstimatedCost,
    totalExpectedROI,
    implementationTimeline,
    automationWorkflow,
    generatedAt: new Date(),
    dataCompleteness: intelligence.dataCompleteness
  };
  
  console.log(`[Recommendation Engine] Report generated. Score: ${overallScore}/100, Recommendations: ${sortedRecommendations.length}`);
  
  return report;
}

/**
 * Generate quick recommendations (without full intelligence gathering)
 */
export async function generateQuickRecommendations(
  companyName: string,
  industryCode: string,
  currentState: {
    hasWebsite: boolean;
    hasLinkedIn: boolean;
    hasSocialMedia: boolean;
    hasReviews: boolean;
  }
): Promise<Recommendation[]> {
  const prompt = `Generate quick recommendations for: ${companyName} in industry: ${industryCode}

Current state:
- Website: ${currentState.hasWebsite ? 'Yes' : 'No'}
- LinkedIn: ${currentState.hasLinkedIn ? 'Yes' : 'No'}
- Social Media: ${currentState.hasSocialMedia ? 'Yes' : 'No'}
- Online Reviews: ${currentState.hasReviews ? 'Yes' : 'No'}

Generate 5-10 high-priority recommendations to improve digital presence.

Return detailed JSON array of recommendations.`;

  const response = await invokeAI({
    prompt,
    systemPrompt: 'You are an expert business consultant who creates actionable recommendations.',
    maxTokens: 2000
  });
  
  // Parse AI response
  let recommendations: Recommendation[];
  try {
    const parsed = JSON.parse(response.content);
    recommendations = (parsed.recommendations || parsed || []).map((rec: any, index: number) => ({
      id: `rec_quick_${Date.now()}_${index}`,
      category: rec.category || 'operations',
      title: rec.title || 'Untitled Recommendation',
      description: rec.description || '',
      priority: rec.priority || 'medium',
      estimatedCost: rec.estimatedCost || 0,
      estimatedTime: rec.estimatedTime || '1 week',
      expectedROI: rec.expectedROI || 0,
      confidence: rec.confidence || 70,
      steps: rec.steps || [],
      automatable: rec.automatable || false,
      automationPlatforms: rec.automationPlatforms || [],
      metrics: rec.metrics || {
        before: {},
        after: {},
        improvement: {}
      }
    }));
  } catch (error) {
    console.error('[Recommendation Engine] Failed to parse quick recommendations:', error);
    recommendations = [];
  }
  
  return recommendations;
}

// ============================================================================
// EXPORT
// ============================================================================

export default {
  generateRecommendationReport,
  generateQuickRecommendations,
  performGapAnalysis,
  generateRecommendations,
  calculateOverallScore
};
