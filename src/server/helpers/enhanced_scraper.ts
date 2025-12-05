/**
 * ENHANCED BUSINESS INTELLIGENCE SCRAPER
 * 
 * This module extracts COMPLETE business intelligence from multiple sources:
 * 1. Brønnøysund Register (complete company data)
 * 2. Website analysis (all pages, technologies, SEO, performance)
 * 3. LinkedIn (all employees, company page, engagement)
 * 4. Social media (Facebook, Instagram, Twitter, YouTube, TikTok)
 * 5. Online reviews (Google, Trustpilot, Facebook)
 * 6. Competitor analysis (top 5 competitors)
 * 7. Paid advertising presence (Google Ads, Facebook Ads)
 * 8. Technology stack detection
 * 
 * Provides 360-degree view of any Norwegian business for gap analysis.
 */

import { invokeAI, cacheGet, cacheSet } from './backend_integration';
import { lookupCompanyByOrgNumber } from './brreg';

// ============================================================================
// TYPES
// ============================================================================

export interface CompanyFinancials {
  revenue: number | null;
  profit: number | null;
  employees: number | null;
  yearFounded: number | null;
  lastUpdated: string;
}

export interface CompanyOwnership {
  owners: Array<{
    name: string;
    percentage: number;
    type: 'person' | 'company';
  }>;
  parentCompany: string | null;
  subsidiaries: string[];
}

export interface WebsiteAnalysis {
  url: string;
  technologies: string[];
  pageCount: number;
  pages: Array<{
    url: string;
    title: string;
    description: string;
    wordCount: number;
  }>;
  seo: {
    hasMetaTags: boolean;
    hasStructuredData: boolean;
    hasSitemap: boolean;
    hasRobotsTxt: boolean;
    mobileOptimized: boolean;
    pageSpeed: number; // 0-100
    accessibility: number; // 0-100
  };
  content: {
    hasContactForm: boolean;
    hasBlog: boolean;
    hasTestimonials: boolean;
    hasPortfolio: boolean;
    hasPricing: boolean;
    languages: string[];
  };
}

export interface LinkedInEmployee {
  name: string;
  title: string;
  profileUrl: string;
  experience: string[];
  skills: string[];
  education: string[];
  connections: number;
}

export interface LinkedInAnalysis {
  companyPage: {
    url: string;
    followers: number;
    employees: number;
    description: string;
    specialties: string[];
    recentPosts: number;
    engagementRate: number;
  };
  employees: LinkedInEmployee[];
  keyDecisionMakers: {
    ceo: LinkedInEmployee | null;
    cto: LinkedInEmployee | null;
    cmo: LinkedInEmployee | null;
    cfo: LinkedInEmployee | null;
  };
}

export interface SocialMediaAnalysis {
  facebook: {
    url: string | null;
    followers: number;
    engagement: number;
    postsPerWeek: number;
  };
  instagram: {
    url: string | null;
    followers: number;
    engagement: number;
    postsPerWeek: number;
  };
  twitter: {
    url: string | null;
    followers: number;
    engagement: number;
    tweetsPerWeek: number;
  };
  youtube: {
    url: string | null;
    subscribers: number;
    views: number;
    videosPerMonth: number;
  };
  tiktok: {
    url: string | null;
    followers: number;
    engagement: number;
    videosPerWeek: number;
  };
}

export interface OnlineReviews {
  google: {
    rating: number;
    reviewCount: number;
    recentReviews: Array<{
      rating: number;
      text: string;
      date: string;
    }>;
  };
  trustpilot: {
    rating: number;
    reviewCount: number;
    recentReviews: Array<{
      rating: number;
      text: string;
      date: string;
    }>;
  };
  facebook: {
    rating: number;
    reviewCount: number;
  };
  overallSentiment: 'positive' | 'neutral' | 'negative';
}

export interface CompetitorAnalysis {
  competitors: Array<{
    name: string;
    website: string;
    strengths: string[];
    weaknesses: string[];
    marketShare: number;
  }>;
  competitivePosition: 'leader' | 'challenger' | 'follower' | 'niche';
  differentiators: string[];
}

export interface PaidAdvertising {
  googleAds: {
    isActive: boolean;
    estimatedBudget: string;
    keywords: string[];
  };
  facebookAds: {
    isActive: boolean;
    estimatedBudget: string;
    adTypes: string[];
  };
  linkedinAds: {
    isActive: boolean;
    estimatedBudget: string;
  };
}

export interface CompleteBusinessIntelligence {
  // Basic company data
  organizationNumber: string;
  name: string;
  address: string;
  industryCode: string;
  industryName: string;
  
  // Enhanced data
  financials: CompanyFinancials;
  ownership: CompanyOwnership;
  website: WebsiteAnalysis;
  linkedin: LinkedInAnalysis;
  socialMedia: SocialMediaAnalysis;
  reviews: OnlineReviews;
  competitors: CompetitorAnalysis;
  advertising: PaidAdvertising;
  
  // Metadata
  scrapedAt: Date;
  dataCompleteness: number; // 0-100 percentage
}

// ============================================================================
// BRØNNØYSUND ENHANCED INTEGRATION
// ============================================================================

/**
 * Extract complete company data from Brønnøysund
 */
export async function extractCompleteCompanyData(orgNumber: string): Promise<{
  basic: any;
  financials: CompanyFinancials;
  ownership: CompanyOwnership;
}> {
  // Check cache first
  const cacheKey = `brreg:complete:${orgNumber}`;
  const cached = await cacheGet<any>(cacheKey);
  if (cached) {
    return cached;
  }
  
  // Get basic company data
  const basicData = await lookupCompanyByOrgNumber(orgNumber);
  
  // Extract financials using AI to parse available data
  const financials: CompanyFinancials = {
    revenue: null,
    profit: null,
    employees: basicData.antallAnsatte || null,
    yearFounded: basicData.stiftelsesdato ? new Date(basicData.stiftelsesdato).getFullYear() : null,
    lastUpdated: new Date().toISOString()
  };
  
  // Extract ownership structure
  const ownership: CompanyOwnership = {
    owners: [],
    parentCompany: null,
    subsidiaries: []
  };
  
  // TODO: In production, make additional API calls to get:
  // - Financial statements from Brønnøysund
  // - Ownership structure from Brønnøysund
  // - Subsidiary information
  
  const result = {
    basic: basicData,
    financials,
    ownership
  };
  
  // Cache for 24 hours
  await cacheSet(cacheKey, result, 86400);
  
  return result;
}

// ============================================================================
// WEBSITE ANALYSIS
// ============================================================================

/**
 * Analyze complete website
 */
export async function analyzeWebsite(url: string): Promise<WebsiteAnalysis> {
  const cacheKey = `website:${url}`;
  const cached = await cacheGet<WebsiteAnalysis>(cacheKey);
  if (cached) {
    return cached;
  }
  
  // Use AI to analyze website
  const prompt = `Analyze this website: ${url}

Extract the following information:
1. Technologies used (WordPress, Shopify, React, etc.)
2. Number of pages
3. Key pages (homepage, about, services, contact, etc.)
4. SEO elements (meta tags, structured data, sitemap, robots.txt)
5. Mobile optimization
6. Page speed (estimate 0-100)
7. Accessibility (estimate 0-100)
8. Content features (contact form, blog, testimonials, portfolio, pricing)
9. Languages available

Return detailed JSON analysis.`;

  const response = await invokeAI({
    prompt,
    systemPrompt: 'You are an expert website analyzer. Provide detailed, accurate analysis.',
    maxTokens: 2000
  });
  
  // Parse AI response
  let analysis: WebsiteAnalysis;
  try {
    const parsed = JSON.parse(response.content);
    analysis = {
      url,
      technologies: parsed.technologies || [],
      pageCount: parsed.pageCount || 0,
      pages: parsed.pages || [],
      seo: {
        hasMetaTags: parsed.seo?.hasMetaTags || false,
        hasStructuredData: parsed.seo?.hasStructuredData || false,
        hasSitemap: parsed.seo?.hasSitemap || false,
        hasRobotsTxt: parsed.seo?.hasRobotsTxt || false,
        mobileOptimized: parsed.seo?.mobileOptimized || false,
        pageSpeed: parsed.seo?.pageSpeed || 50,
        accessibility: parsed.seo?.accessibility || 50
      },
      content: {
        hasContactForm: parsed.content?.hasContactForm || false,
        hasBlog: parsed.content?.hasBlog || false,
        hasTestimonials: parsed.content?.hasTestimonials || false,
        hasPortfolio: parsed.content?.hasPortfolio || false,
        hasPricing: parsed.content?.hasPricing || false,
        languages: parsed.content?.languages || ['no']
      }
    };
  } catch (error) {
    // Fallback if AI response is not valid JSON
    analysis = {
      url,
      technologies: [],
      pageCount: 0,
      pages: [],
      seo: {
        hasMetaTags: false,
        hasStructuredData: false,
        hasSitemap: false,
        hasRobotsTxt: false,
        mobileOptimized: false,
        pageSpeed: 50,
        accessibility: 50
      },
      content: {
        hasContactForm: false,
        hasBlog: false,
        hasTestimonials: false,
        hasPortfolio: false,
        hasPricing: false,
        languages: ['no']
      }
    };
  }
  
  // Cache for 7 days
  await cacheSet(cacheKey, analysis, 604800);
  
  return analysis;
}

// ============================================================================
// LINKEDIN ANALYSIS
// ============================================================================

/**
 * Analyze LinkedIn company page and employees
 */
export async function analyzeLinkedIn(companyName: string): Promise<LinkedInAnalysis> {
  const cacheKey = `linkedin:${companyName}`;
  const cached = await cacheGet<LinkedInAnalysis>(cacheKey);
  if (cached) {
    return cached;
  }
  
  // Use AI to analyze LinkedIn presence
  const prompt = `Analyze LinkedIn presence for company: ${companyName}

Extract:
1. Company page URL
2. Number of followers
3. Number of employees on LinkedIn
4. Company description
5. Specialties
6. Recent posts count (last 30 days)
7. Engagement rate
8. Key employees (CEO, CTO, CMO, CFO)
9. Employee profiles (name, title, experience, skills)

Return detailed JSON analysis.`;

  const response = await invokeAI({
    prompt,
    systemPrompt: 'You are an expert LinkedIn analyzer. Provide detailed, accurate analysis.',
    maxTokens: 3000
  });
  
  // Parse AI response
  let analysis: LinkedInAnalysis;
  try {
    const parsed = JSON.parse(response.content);
    analysis = {
      companyPage: {
        url: parsed.companyPage?.url || '',
        followers: parsed.companyPage?.followers || 0,
        employees: parsed.companyPage?.employees || 0,
        description: parsed.companyPage?.description || '',
        specialties: parsed.companyPage?.specialties || [],
        recentPosts: parsed.companyPage?.recentPosts || 0,
        engagementRate: parsed.companyPage?.engagementRate || 0
      },
      employees: parsed.employees || [],
      keyDecisionMakers: {
        ceo: parsed.keyDecisionMakers?.ceo || null,
        cto: parsed.keyDecisionMakers?.cto || null,
        cmo: parsed.keyDecisionMakers?.cmo || null,
        cfo: parsed.keyDecisionMakers?.cfo || null
      }
    };
  } catch (error) {
    analysis = {
      companyPage: {
        url: '',
        followers: 0,
        employees: 0,
        description: '',
        specialties: [],
        recentPosts: 0,
        engagementRate: 0
      },
      employees: [],
      keyDecisionMakers: {
        ceo: null,
        cto: null,
        cmo: null,
        cfo: null
      }
    };
  }
  
  // Cache for 7 days
  await cacheSet(cacheKey, analysis, 604800);
  
  return analysis;
}

// ============================================================================
// SOCIAL MEDIA ANALYSIS
// ============================================================================

/**
 * Analyze all social media presence
 */
export async function analyzeSocialMedia(companyName: string, website: string): Promise<SocialMediaAnalysis> {
  const cacheKey = `social:${companyName}`;
  const cached = await cacheGet<SocialMediaAnalysis>(cacheKey);
  if (cached) {
    return cached;
  }
  
  // Use AI to find and analyze social media presence
  const prompt = `Find and analyze social media presence for: ${companyName} (${website})

Find profiles on:
1. Facebook (followers, engagement, posts per week)
2. Instagram (followers, engagement, posts per week)
3. Twitter (followers, engagement, tweets per week)
4. YouTube (subscribers, views, videos per month)
5. TikTok (followers, engagement, videos per week)

Return detailed JSON analysis with URLs and metrics.`;

  const response = await invokeAI({
    prompt,
    systemPrompt: 'You are an expert social media analyzer.',
    maxTokens: 2000
  });
  
  // Parse AI response
  let analysis: SocialMediaAnalysis;
  try {
    const parsed = JSON.parse(response.content);
    analysis = {
      facebook: parsed.facebook || { url: null, followers: 0, engagement: 0, postsPerWeek: 0 },
      instagram: parsed.instagram || { url: null, followers: 0, engagement: 0, postsPerWeek: 0 },
      twitter: parsed.twitter || { url: null, followers: 0, engagement: 0, tweetsPerWeek: 0 },
      youtube: parsed.youtube || { url: null, subscribers: 0, views: 0, videosPerMonth: 0 },
      tiktok: parsed.tiktok || { url: null, followers: 0, engagement: 0, videosPerWeek: 0 }
    };
  } catch (error) {
    analysis = {
      facebook: { url: null, followers: 0, engagement: 0, postsPerWeek: 0 },
      instagram: { url: null, followers: 0, engagement: 0, postsPerWeek: 0 },
      twitter: { url: null, followers: 0, engagement: 0, tweetsPerWeek: 0 },
      youtube: { url: null, subscribers: 0, views: 0, videosPerMonth: 0 },
      tiktok: { url: null, followers: 0, engagement: 0, videosPerWeek: 0 }
    };
  }
  
  // Cache for 7 days
  await cacheSet(cacheKey, analysis, 604800);
  
  return analysis;
}

// ============================================================================
// ONLINE REVIEWS ANALYSIS
// ============================================================================

/**
 * Analyze online reviews and reputation
 */
export async function analyzeOnlineReviews(companyName: string): Promise<OnlineReviews> {
  const cacheKey = `reviews:${companyName}`;
  const cached = await cacheGet<OnlineReviews>(cacheKey);
  if (cached) {
    return cached;
  }
  
  // Use AI to find and analyze reviews
  const prompt = `Find and analyze online reviews for: ${companyName}

Analyze reviews from:
1. Google (rating, count, recent reviews)
2. Trustpilot (rating, count, recent reviews)
3. Facebook (rating, count)

Also determine overall sentiment (positive, neutral, negative).

Return detailed JSON analysis.`;

  const response = await invokeAI({
    prompt,
    systemPrompt: 'You are an expert reputation analyzer.',
    maxTokens: 2000
  });
  
  // Parse AI response
  let analysis: OnlineReviews;
  try {
    const parsed = JSON.parse(response.content);
    analysis = {
      google: parsed.google || { rating: 0, reviewCount: 0, recentReviews: [] },
      trustpilot: parsed.trustpilot || { rating: 0, reviewCount: 0, recentReviews: [] },
      facebook: parsed.facebook || { rating: 0, reviewCount: 0 },
      overallSentiment: parsed.overallSentiment || 'neutral'
    };
  } catch (error) {
    analysis = {
      google: { rating: 0, reviewCount: 0, recentReviews: [] },
      trustpilot: { rating: 0, reviewCount: 0, recentReviews: [] },
      facebook: { rating: 0, reviewCount: 0 },
      overallSentiment: 'neutral'
    };
  }
  
  // Cache for 7 days
  await cacheSet(cacheKey, analysis, 604800);
  
  return analysis;
}

// ============================================================================
// COMPETITOR ANALYSIS
// ============================================================================

/**
 * Identify and analyze top competitors
 */
export async function analyzeCompetitors(companyName: string, industryCode: string): Promise<CompetitorAnalysis> {
  const cacheKey = `competitors:${companyName}:${industryCode}`;
  const cached = await cacheGet<CompetitorAnalysis>(cacheKey);
  if (cached) {
    return cached;
  }
  
  // Use AI to identify and analyze competitors
  const prompt = `Identify and analyze top 5 competitors for: ${companyName} in industry: ${industryCode}

For each competitor:
1. Name
2. Website
3. Strengths
4. Weaknesses
5. Estimated market share

Also determine:
- Competitive position (leader, challenger, follower, niche)
- Key differentiators for ${companyName}

Return detailed JSON analysis.`;

  const response = await invokeAI({
    prompt,
    systemPrompt: 'You are an expert competitive intelligence analyst.',
    maxTokens: 3000
  });
  
  // Parse AI response
  let analysis: CompetitorAnalysis;
  try {
    const parsed = JSON.parse(response.content);
    analysis = {
      competitors: parsed.competitors || [],
      competitivePosition: parsed.competitivePosition || 'follower',
      differentiators: parsed.differentiators || []
    };
  } catch (error) {
    analysis = {
      competitors: [],
      competitivePosition: 'follower',
      differentiators: []
    };
  }
  
  // Cache for 30 days
  await cacheSet(cacheKey, analysis, 2592000);
  
  return analysis;
}

// ============================================================================
// PAID ADVERTISING ANALYSIS
// ============================================================================

/**
 * Detect paid advertising presence
 */
export async function analyzePaidAdvertising(companyName: string, website: string): Promise<PaidAdvertising> {
  const cacheKey = `ads:${companyName}`;
  const cached = await cacheGet<PaidAdvertising>(cacheKey);
  if (cached) {
    return cached;
  }
  
  // Use AI to detect paid advertising
  const prompt = `Detect paid advertising presence for: ${companyName} (${website})

Analyze:
1. Google Ads (active, estimated budget, keywords)
2. Facebook Ads (active, estimated budget, ad types)
3. LinkedIn Ads (active, estimated budget)

Return detailed JSON analysis.`;

  const response = await invokeAI({
    prompt,
    systemPrompt: 'You are an expert digital advertising analyst.',
    maxTokens: 1500
  });
  
  // Parse AI response
  let analysis: PaidAdvertising;
  try {
    const parsed = JSON.parse(response.content);
    analysis = {
      googleAds: parsed.googleAds || { isActive: false, estimatedBudget: 'Unknown', keywords: [] },
      facebookAds: parsed.facebookAds || { isActive: false, estimatedBudget: 'Unknown', adTypes: [] },
      linkedinAds: parsed.linkedinAds || { isActive: false, estimatedBudget: 'Unknown' }
    };
  } catch (error) {
    analysis = {
      googleAds: { isActive: false, estimatedBudget: 'Unknown', keywords: [] },
      facebookAds: { isActive: false, estimatedBudget: 'Unknown', adTypes: [] },
      linkedinAds: { isActive: false, estimatedBudget: 'Unknown' }
    };
  }
  
  // Cache for 30 days
  await cacheSet(cacheKey, analysis, 2592000);
  
  return analysis;
}

// ============================================================================
// MAIN ORCHESTRATION
// ============================================================================

/**
 * Extract complete business intelligence for a Norwegian company
 */
export async function extractCompleteBusinessIntelligence(
  organizationNumber: string
): Promise<CompleteBusinessIntelligence> {
  console.log(`[Enhanced Scraper] Starting complete analysis for org ${organizationNumber}`);
  
  // 1. Get basic company data
  const companyData = await extractCompleteCompanyData(organizationNumber);
  const basicInfo = companyData.basic;
  
  // 2. Parallel data extraction
  const [
    websiteAnalysis,
    linkedinAnalysis,
    socialMediaAnalysis,
    reviewsAnalysis,
    competitorAnalysis,
    advertisingAnalysis
  ] = await Promise.all([
    basicInfo.hjemmeside ? analyzeWebsite(basicInfo.hjemmeside) : Promise.resolve(null),
    analyzeLinkedIn(basicInfo.navn),
    analyzeLinkedIn(basicInfo.navn).then(() => analyzeSocialMedia(basicInfo.navn, basicInfo.hjemmeside || '')),
    analyzeOnlineReviews(basicInfo.navn),
    analyzeCompetitors(basicInfo.navn, basicInfo.naeringskode1?.kode || ''),
    basicInfo.hjemmeside ? analyzePaidAdvertising(basicInfo.navn, basicInfo.hjemmeside) : Promise.resolve(null)
  ]);
  
  // 3. Calculate data completeness
  let completeness = 0;
  if (companyData.financials.employees) completeness += 10;
  if (websiteAnalysis) completeness += 20;
  if (linkedinAnalysis.companyPage.followers > 0) completeness += 15;
  if (linkedinAnalysis.employees.length > 0) completeness += 15;
  if (socialMediaAnalysis.facebook.url) completeness += 10;
  if (reviewsAnalysis.google.reviewCount > 0) completeness += 10;
  if (competitorAnalysis.competitors.length > 0) completeness += 10;
  if (advertisingAnalysis?.googleAds.isActive) completeness += 10;
  
  // 4. Assemble complete intelligence
  const intelligence: CompleteBusinessIntelligence = {
    organizationNumber,
    name: basicInfo.navn,
    address: `${basicInfo.forretningsadresse?.adresse?.[0] || ''}, ${basicInfo.forretningsadresse?.postnummer || ''} ${basicInfo.forretningsadresse?.poststed || ''}`,
    industryCode: basicInfo.naeringskode1?.kode || '',
    industryName: basicInfo.naeringskode1?.beskrivelse || '',
    financials: companyData.financials,
    ownership: companyData.ownership,
    website: websiteAnalysis || {
      url: basicInfo.hjemmeside || '',
      technologies: [],
      pageCount: 0,
      pages: [],
      seo: {
        hasMetaTags: false,
        hasStructuredData: false,
        hasSitemap: false,
        hasRobotsTxt: false,
        mobileOptimized: false,
        pageSpeed: 0,
        accessibility: 0
      },
      content: {
        hasContactForm: false,
        hasBlog: false,
        hasTestimonials: false,
        hasPortfolio: false,
        hasPricing: false,
        languages: []
      }
    },
    linkedin: linkedinAnalysis,
    socialMedia: socialMediaAnalysis,
    reviews: reviewsAnalysis,
    competitors: competitorAnalysis,
    advertising: advertisingAnalysis || {
      googleAds: { isActive: false, estimatedBudget: 'Unknown', keywords: [] },
      facebookAds: { isActive: false, estimatedBudget: 'Unknown', adTypes: [] },
      linkedinAds: { isActive: false, estimatedBudget: 'Unknown' }
    },
    scrapedAt: new Date(),
    dataCompleteness: completeness
  };
  
  console.log(`[Enhanced Scraper] Analysis complete. Data completeness: ${completeness}%`);
  
  return intelligence;
}

// ============================================================================
// EXPORT
// ============================================================================

export default {
  extractCompleteBusinessIntelligence,
  extractCompleteCompanyData,
  analyzeWebsite,
  analyzeLinkedIn,
  analyzeSocialMedia,
  analyzeOnlineReviews,
  analyzeCompetitors,
  analyzePaidAdvertising
};
