/**
 * LinkedIn Integration
 * Company pages and employee data
 * Uses deeplinks and public data access
 */

import { cacheGet, cacheSet } from './cache';

const LINKEDIN_BASE = 'https://www.linkedin.com';
const CACHE_TTL = 3 * 24 * 60 * 60; // 3 days

export interface LinkedInCompanyData {
  companyName: string;
  linkedInUrl?: string;
  followers?: number;
  employees?: number;
  description?: string;
  industry?: string;
  headquarters?: string;
  founded?: number;
  specialties?: string[];
  website?: string;
  posts?: {
    totalPosts: number;
    avgLikes: number;
    avgComments: number;
    avgShares: number;
    engagementRate: number;
  };
}

export interface LinkedInEmployee {
  name: string;
  title: string;
  profileUrl: string;
  location?: string;
  experience?: {
    company: string;
    title: string;
    duration: string;
  }[];
  education?: {
    school: string;
    degree?: string;
    field?: string;
  }[];
  skills?: string[];
  isDecisionMaker: boolean;
  seniorityLevel: 'executive' | 'senior' | 'mid' | 'junior' | 'entry';
}

/**
 * Generate LinkedIn company page deeplink
 */
export function getLinkedInCompanyDeeplink(companyName: string): string {
  // Format company name for LinkedIn URL
  const formattedName = companyName
    .toLowerCase()
    .replace(/\s+/g, '-')
    .replace(/[^a-z0-9-]/g, '')
    .replace(/^-+|-+$/g, ''); // Remove leading/trailing hyphens
  
  return `${LINKEDIN_BASE}/company/${formattedName}`;
}

/**
 * Generate LinkedIn search deeplink for company employees
 */
export function getLinkedInEmployeeSearchDeeplink(companyName: string): string {
  const encodedCompany = encodeURIComponent(companyName);
  return `${LINKEDIN_BASE}/search/results/people/?currentCompany=%5B%22${encodedCompany}%22%5D`;
}

/**
 * Generate LinkedIn sales navigator deeplink (premium feature)
 */
export function getLinkedInSalesNavigatorDeeplink(companyName: string): string {
  const encodedCompany = encodeURIComponent(companyName);
  return `${LINKEDIN_BASE}/sales/search/people?companyIncluded=${encodedCompany}`;
}

/**
 * Get LinkedIn company data
 * Note: This returns structure with deeplinks. Actual scraping requires authentication.
 */
export async function getLinkedInCompanyData(companyName: string): Promise<LinkedInCompanyData> {
  const cacheKey = `linkedin:company:${companyName}`;
  
  // Check cache first
  const cached = await cacheGet<LinkedInCompanyData>(cacheKey);
  if (cached) {
    console.log(`[LinkedIn] Cache hit for ${companyName}`);
    return cached;
  }

  console.log(`[LinkedIn] Creating data structure for ${companyName}`);
  
  const data: LinkedInCompanyData = {
    companyName,
    linkedInUrl: getLinkedInCompanyDeeplink(companyName),
    followers: undefined, // Would be scraped from page
    employees: undefined, // Would be scraped from page
    description: undefined,
    industry: undefined,
    headquarters: undefined,
    founded: undefined,
    specialties: [],
    website: undefined,
    posts: {
      totalPosts: 0,
      avgLikes: 0,
      avgComments: 0,
      avgShares: 0,
      engagementRate: 0
    }
  };
  
  // Cache the result
  await cacheSet(cacheKey, data, CACHE_TTL);
  
  console.log(`[LinkedIn] Company deeplink: ${data.linkedInUrl}`);
  
  return data;
}

/**
 * Identify decision makers from employee list
 */
export function identifyDecisionMakers(employees: LinkedInEmployee[]): LinkedInEmployee[] {
  const decisionMakerTitles = [
    'ceo', 'chief executive', 'president', 'founder',
    'cto', 'chief technology', 'vp technology', 'head of technology',
    'cfo', 'chief financial', 'vp finance', 'head of finance',
    'cmo', 'chief marketing', 'vp marketing', 'head of marketing',
    'coo', 'chief operating', 'vp operations', 'head of operations',
    'cio', 'chief information', 'vp it', 'head of it',
    'managing director', 'general manager', 'director'
  ];
  
  return employees.filter(emp => {
    const title = emp.title.toLowerCase();
    return decisionMakerTitles.some(dm => title.includes(dm));
  });
}

/**
 * Determine seniority level from title
 */
export function determineSeniorityLevel(title: string): LinkedInEmployee['seniorityLevel'] {
  const titleLower = title.toLowerCase();
  
  if (titleLower.includes('chief') || 
      titleLower.includes('ceo') || 
      titleLower.includes('president') || 
      titleLower.includes('founder')) {
    return 'executive';
  }
  
  if (titleLower.includes('vp') || 
      titleLower.includes('vice president') || 
      titleLower.includes('director') || 
      titleLower.includes('head of')) {
    return 'senior';
  }
  
  if (titleLower.includes('manager') || 
      titleLower.includes('lead') || 
      titleLower.includes('senior')) {
    return 'mid';
  }
  
  if (titleLower.includes('junior') || 
      titleLower.includes('associate') || 
      titleLower.includes('assistant')) {
    return 'entry';
  }
  
  return 'junior';
}

/**
 * Calculate LinkedIn engagement score
 */
export function calculateEngagementScore(companyData: LinkedInCompanyData): {
  score: number;
  level: 'excellent' | 'good' | 'fair' | 'poor';
  recommendations: string[];
} {
  const recommendations: string[] = [];
  let score = 0;
  
  // Followers score (max 25 points)
  if (companyData.followers) {
    if (companyData.followers > 10000) {
      score += 25;
    } else if (companyData.followers > 5000) {
      score += 20;
    } else if (companyData.followers > 1000) {
      score += 15;
    } else if (companyData.followers > 500) {
      score += 10;
    } else {
      score += 5;
      recommendations.push('Increase follower count through targeted content and employee advocacy');
    }
  } else {
    recommendations.push('Set up LinkedIn company page to build brand presence');
  }
  
  // Engagement rate score (max 35 points)
  if (companyData.posts) {
    const engagementRate = companyData.posts.engagementRate;
    if (engagementRate > 5) {
      score += 35;
    } else if (engagementRate > 3) {
      score += 25;
    } else if (engagementRate > 1) {
      score += 15;
    } else if (engagementRate > 0) {
      score += 5;
      recommendations.push('Improve content quality to increase engagement rate');
    } else {
      recommendations.push('Start posting regular content (3-4 times per week)');
    }
  }
  
  // Post frequency score (max 20 points)
  if (companyData.posts && companyData.posts.totalPosts > 0) {
    if (companyData.posts.totalPosts > 100) {
      score += 20;
    } else if (companyData.posts.totalPosts > 50) {
      score += 15;
    } else if (companyData.posts.totalPosts > 20) {
      score += 10;
    } else {
      score += 5;
      recommendations.push('Increase posting frequency for better visibility');
    }
  }
  
  // Company page completeness score (max 20 points)
  let completenessScore = 0;
  if (companyData.description) completenessScore += 5;
  if (companyData.specialties && companyData.specialties.length > 0) completenessScore += 5;
  if (companyData.website) completenessScore += 5;
  if (companyData.headquarters) completenessScore += 5;
  
  score += completenessScore;
  
  if (completenessScore < 20) {
    recommendations.push('Complete company page with description, specialties, and contact information');
  }
  
  let level: 'excellent' | 'good' | 'fair' | 'poor';
  if (score >= 80) level = 'excellent';
  else if (score >= 60) level = 'good';
  else if (score >= 40) level = 'fair';
  else level = 'poor';
  
  return { score, level, recommendations };
}

/**
 * Generate employee advocacy recommendations
 */
export function generateEmployeeAdvocacyPlan(employees: LinkedInEmployee[]): {
  activeEmployees: number;
  potentialReach: number;
  recommendations: string[];
} {
  const recommendations: string[] = [];
  
  // Estimate active LinkedIn users (assume 60% of employees)
  const activeEmployees = Math.floor(employees.length * 0.6);
  
  // Estimate potential reach (assume 500 connections per employee)
  const potentialReach = activeEmployees * 500;
  
  if (activeEmployees < 10) {
    recommendations.push('Encourage employees to create and optimize LinkedIn profiles');
    recommendations.push('Provide LinkedIn training for employees');
  }
  
  recommendations.push('Create employee advocacy program with content sharing guidelines');
  recommendations.push('Recognize and reward employees who actively share company content');
  recommendations.push('Provide ready-to-share content templates for employees');
  
  if (employees.length > 50) {
    recommendations.push('Consider LinkedIn Sales Navigator for team to expand reach');
  }
  
  return {
    activeEmployees,
    potentialReach,
    recommendations
  };
}

/**
 * Get LinkedIn deeplinks for all relevant pages
 */
export function getAllLinkedInDeeplinks(companyName: string, organizationNumber?: string) {
  return {
    companyPage: getLinkedInCompanyDeeplink(companyName),
    employeeSearch: getLinkedInEmployeeSearchDeeplink(companyName),
    salesNavigator: getLinkedInSalesNavigatorDeeplink(companyName),
    jobsPage: `${getLinkedInCompanyDeeplink(companyName)}/jobs`,
    lifePage: `${getLinkedInCompanyDeeplink(companyName)}/life`,
    aboutPage: `${getLinkedInCompanyDeeplink(companyName)}/about`,
    postsPage: `${getLinkedInCompanyDeeplink(companyName)}/posts`
  };
}
