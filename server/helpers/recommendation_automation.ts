/**
 * Recommendation Automation System
 * 
 * Converts "No automated execution available" recommendations into
 * executable deeplinks with automatic platform integration.
 * 
 * Uses AI-powered matching to map recommendations to optimal platforms.
 */

import { DeeplinkPlatform, searchPlatforms } from './deeplink_database';

/**
 * Recommendation category types
 */
export type RecommendationCategory =
  | 'pricing'
  | 'marketing'
  | 'operations'
  | 'technology'
  | 'hr'
  | 'sales'
  | 'customer_service'
  | 'finance'
  | 'legal'
  | 'product';

/**
 * Recommendation difficulty levels
 */
export type DifficultyLevel = 'easy' | 'medium' | 'hard';

/**
 * Recommendation impact levels
 */
export type ImpactLevel = 'low' | 'medium' | 'high';

/**
 * Recommendation structure
 */
export interface Recommendation {
  id: string;
  category: RecommendationCategory;
  impact: ImpactLevel;
  difficulty: DifficultyLevel;
  title: string;
  description: string;
  expectedROI: string;
  priority: number; // 1-10
  cost: string;
  timeframe: string;
  isAutomated: boolean;
}

/**
 * Execution plan for a recommendation
 */
export interface ExecutionPlan {
  recommendationId: string;
  platforms: DeeplinkPlatform[];
  steps: ExecutionStep[];
  totalCost: string;
  estimatedTime: string;
  expectedROI: string;
  automationLevel: 'full' | 'partial' | 'manual';
}

/**
 * Individual execution step
 */
export interface ExecutionStep {
  stepNumber: number;
  title: string;
  description: string;
  platform?: DeeplinkPlatform;
  estimatedTime: string;
  cost: string;
  isAutomated: boolean;
  instructions: string[];
}

/**
 * Recommendation-to-platform mapping rules
 */
const RECOMMENDATION_MAPPINGS: Record<string, string[]> = {
  // Pricing & Finance
  'pricing strategy': ['stripe', 'quickbooks', 'profitwell', 'chargebee', 'paddle'],
  'value-based pricing': ['stripe', 'profitwell', 'chargebee'],
  
  // Marketing & SEO
  'seo': ['semrush', 'ahrefs', 'moz', 'google-search-console'],
  'website': ['webflow', 'wordpress', 'wix', 'squarespace'],
  'linkedin advertising': ['linkedin-ads', 'hootsuite', 'buffer'],
  'google ads': ['google-ads', 'google-analytics'],
  'social media': ['hootsuite', 'buffer', 'sprout-social'],
  'content marketing': ['hubspot', 'contentful', 'wordpress'],
  'email marketing': ['mailchimp', 'sendgrid', 'hubspot'],
  
  // Sales & CRM
  'crm': ['salesforce', 'hubspot', 'pipedrive', 'zoho-crm'],
  'sales automation': ['salesforce', 'hubspot', 'outreach'],
  'lead generation': ['hubspot', 'salesforce', 'linkedin-ads'],
  
  // Operations & Project Management
  'project management': ['asana', 'monday', 'jira', 'trello', 'clickup'],
  'agile': ['jira', 'azure-devops', 'github'],
  'process automation': ['zapier', 'make', 'n8n'],
  'workflow': ['asana', 'monday', 'airtable'],
  
  // Technology & IT
  'cybersecurity': ['cloudflare', 'auth0', 'okta', 'aws-security'],
  'cloud computing': ['aws', 'google-cloud', 'azure', 'digitalocean'],
  'analytics': ['google-analytics', 'mixpanel', 'amplitude', 'segment'],
  'ai': ['openai', 'anthropic', 'google-ai', 'huggingface'],
  'collaboration': ['slack', 'microsoft-teams', 'zoom', 'google-workspace'],
  
  // HR & Talent
  'recruitment': ['linkedin-recruiter', 'greenhouse', 'lever', 'workday'],
  'employee engagement': ['culture-amp', 'lattice', '15five'],
  'learning': ['udemy-business', 'coursera', 'linkedin-learning'],
  
  // Customer Service
  'customer support': ['zendesk', 'intercom', 'freshdesk', 'help-scout'],
  'live chat': ['intercom', 'drift', 'zendesk-chat'],
  'customer feedback': ['typeform', 'surveymonkey', 'qualtrics'],
};

/**
 * Analyze recommendation text and extract key actions
 * 
 * @param recommendation - Recommendation object
 * @returns Extracted keywords for platform matching
 */
export function extractRecommendationKeywords(recommendation: Recommendation): string[] {
  const text = `${recommendation.title} ${recommendation.description}`.toLowerCase();
  const keywords: string[] = [];

  // Extract keywords from mapping rules
  for (const [keyword, platforms] of Object.entries(RECOMMENDATION_MAPPINGS)) {
    if (text.includes(keyword)) {
      keywords.push(keyword);
    }
  }

  // Add category-specific keywords
  keywords.push(recommendation.category);

  // Remove duplicates
  const uniqueKeywords = Array.from(new Set(keywords));
  return uniqueKeywords;
}

/**
 * Find matching platforms for a recommendation
 * 
 * @param recommendation - Recommendation object
 * @returns Ranked list of matching platforms
 */
export function findMatchingPlatforms(recommendation: Recommendation): DeeplinkPlatform[] {
  const keywords = extractRecommendationKeywords(recommendation);
  const platformIds = new Set<string>();

  // Collect all platform IDs from mappings
  for (const keyword of keywords) {
    const mappedPlatforms = RECOMMENDATION_MAPPINGS[keyword] || [];
    mappedPlatforms.forEach(id => platformIds.add(id));
  }

  // Search deeplink database
  const searchResults = searchPlatforms(keywords);

  // Combine mapped platforms with search results
  const platformIdArray = Array.from(platformIds);
  const allPlatforms = [
    ...searchResults,
    ...platformIdArray.map(id => searchResults.find(p => p.id === id)).filter(Boolean) as DeeplinkPlatform[]
  ];

  // Remove duplicates and rank by relevance
  const uniquePlatforms = Array.from(
    new Map(allPlatforms.map(p => [p.id, p])).values()
  );

  // Sort by keyword match count (simple relevance scoring)
  return uniquePlatforms.sort((a, b) => {
    const aMatches = keywords.filter(k => 
      a.keywords.some(pk => pk.includes(k) || k.includes(pk))
    ).length;
    const bMatches = keywords.filter(k => 
      b.keywords.some(pk => pk.includes(k) || k.includes(pk))
    ).length;
    return bMatches - aMatches;
  }).slice(0, 5); // Return top 5 platforms
}

/**
 * Generate execution plan for a recommendation
 * 
 * @param recommendation - Recommendation object
 * @returns Complete execution plan with steps and platforms
 */
export function generateExecutionPlan(recommendation: Recommendation): ExecutionPlan {
  const platforms = findMatchingPlatforms(recommendation);
  const steps: ExecutionStep[] = [];

  if (platforms.length === 0) {
    // No platforms found - manual execution required
    return {
      recommendationId: recommendation.id,
      platforms: [],
      steps: [{
        stepNumber: 1,
        title: 'Manual Implementation Required',
        description: recommendation.description,
        estimatedTime: recommendation.timeframe,
        cost: recommendation.cost,
        isAutomated: false,
        instructions: [
          'This recommendation requires custom implementation',
          'Consult with relevant stakeholders',
          'Create detailed implementation plan',
          'Allocate resources and budget',
          'Execute and monitor progress',
        ],
      }],
      totalCost: recommendation.cost,
      estimatedTime: recommendation.timeframe,
      expectedROI: recommendation.expectedROI,
      automationLevel: 'manual',
    };
  }

  // Generate steps for each platform
  platforms.forEach((platform, index) => {
    steps.push({
      stepNumber: index + 1,
      title: `Set up ${platform.name}`,
      description: platform.description,
      platform,
      estimatedTime: platform.setupTime,
      cost: platform.cost,
      isAutomated: true,
      instructions: [
        `Visit ${platform.url}`,
        `Sign up for ${platform.name} account`,
        platform.authType === 'oauth' ? 'Connect via OAuth' : 'Create API key',
        'Configure settings based on company needs',
        'Integrate with existing systems',
        'Test functionality',
        'Monitor performance and ROI',
      ],
    });
  });

  // Add final review step
  steps.push({
    stepNumber: steps.length + 1,
    title: 'Review and Optimize',
    description: 'Monitor implementation and optimize based on results',
    estimatedTime: '1-2 weeks',
    cost: 'Included',
    isAutomated: false,
    instructions: [
      'Monitor key metrics',
      'Compare actual ROI vs expected ROI',
      'Identify optimization opportunities',
      'Adjust strategy as needed',
      'Document learnings for future recommendations',
    ],
  });

  return {
    recommendationId: recommendation.id,
    platforms,
    steps,
    totalCost: platforms.length > 0 ? platforms[0].cost : recommendation.cost,
    estimatedTime: platforms.length > 0 ? platforms[0].setupTime : recommendation.timeframe,
    expectedROI: recommendation.expectedROI,
    automationLevel: platforms.length > 0 ? 'partial' : 'manual',
  };
}

/**
 * Parse Capgemini-style recommendation text
 * 
 * @param text - Raw recommendation text
 * @returns Structured recommendation object
 */
export function parseRecommendation(text: string): Recommendation {
  const lines = text.trim().split('\n').map(l => l.trim()).filter(Boolean);
  
  const category = lines[0]?.toLowerCase() as RecommendationCategory || 'operations';
  const impact = lines[1]?.toLowerCase().includes('high') ? 'high' :
                 lines[1]?.toLowerCase().includes('medium') ? 'medium' : 'low';
  const difficulty = lines[2]?.toLowerCase() as DifficultyLevel || 'medium';
  const title = lines[3] || '';
  const expectedROI = lines[4] || '';
  const priority = parseInt(lines[6]?.split('/')[0] || '5');
  const cost = lines[8] || '';
  const timeframe = lines[10] || '';

  return {
    id: `rec_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    category,
    impact,
    difficulty,
    title,
    description: title,
    expectedROI,
    priority,
    cost,
    timeframe,
    isAutomated: false,
  };
}

/**
 * Batch process multiple recommendations
 * 
 * @param recommendations - Array of recommendations
 * @returns Array of execution plans
 */
export function batchGenerateExecutionPlans(
  recommendations: Recommendation[]
): ExecutionPlan[] {
  return recommendations.map(rec => generateExecutionPlan(rec));
}

/**
 * Calculate total automation coverage
 * 
 * @param plans - Array of execution plans
 * @returns Percentage of recommendations that can be automated
 */
export function calculateAutomationCoverage(plans: ExecutionPlan[]): number {
  const automatedCount = plans.filter(p => p.automationLevel !== 'manual').length;
  return (automatedCount / plans.length) * 100;
}

/**
 * Get automation statistics
 * 
 * @param plans - Array of execution plans
 * @returns Statistics object
 */
export function getAutomationStats(plans: ExecutionPlan[]): {
  total: number;
  fullyAutomated: number;
  partiallyAutomated: number;
  manual: number;
  coveragePercentage: number;
  totalPlatforms: number;
} {
  const fullyAutomated = plans.filter(p => p.automationLevel === 'full').length;
  const partiallyAutomated = plans.filter(p => p.automationLevel === 'partial').length;
  const manual = plans.filter(p => p.automationLevel === 'manual').length;
  const totalPlatforms = plans.reduce((sum, p) => sum + p.platforms.length, 0);

  return {
    total: plans.length,
    fullyAutomated,
    partiallyAutomated,
    manual,
    coveragePercentage: calculateAutomationCoverage(plans),
    totalPlatforms,
  };
}
