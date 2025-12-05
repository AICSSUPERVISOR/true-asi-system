/**
 * Proff.no Integration
 * Norwegian business information and financial data
 * Note: Proff.no doesn't have a public API, so we use their public deeplinks
 */

import { cacheGet, cacheSet } from './cache';

const PROFF_BASE = 'https://www.proff.no';
const CACHE_TTL = 7 * 24 * 60 * 60; // 7 days

export interface ProffFinancialData {
  organizationNumber: string;
  companyName: string;
  revenue?: number;
  profit?: number;
  assets?: number;
  liabilities?: number;
  equity?: number;
  employees?: number;
  creditRating?: string;
  riskLevel?: 'low' | 'medium' | 'high';
  year: number;
  historicalData?: {
    year: number;
    revenue?: number;
    profit?: number;
    employees?: number;
  }[];
}

/**
 * Get company financial data from Proff.no
 * Uses public deeplinks to access company information
 */
export async function getProffFinancialData(orgNumber: string): Promise<ProffFinancialData | null> {
  const cacheKey = `proff:financial:${orgNumber}`;
  
  // Check cache first
  const cached = await cacheGet<ProffFinancialData>(cacheKey);
  if (cached) {
    console.log(`[Proff.no] Cache hit for ${orgNumber}`);
    return cached;
  }

  try {
    console.log(`[Proff.no] Fetching financial data for ${orgNumber}`);
    
    // Proff.no deeplink format: https://www.proff.no/selskap/[company-name]/[org-number]
    // We'll construct a generic URL and let Proff redirect us
    const url = `${PROFF_BASE}/bransjes%C3%B8k?q=${orgNumber}`;
    
    // Note: In a real implementation, we would:
    // 1. Fetch the HTML page
    // 2. Parse the HTML to extract financial data
    // 3. Handle pagination for historical data
    // 
    // For now, we'll return mock data structure with the deeplink
    // The actual scraping would require a headless browser or HTML parser
    
    const data: ProffFinancialData = {
      organizationNumber: orgNumber,
      companyName: 'Company Name', // Would be extracted from HTML
      revenue: undefined, // Would be extracted from HTML
      profit: undefined,
      assets: undefined,
      liabilities: undefined,
      equity: undefined,
      employees: undefined,
      creditRating: undefined,
      riskLevel: undefined,
      year: new Date().getFullYear() - 1, // Last completed fiscal year
      historicalData: []
    };
    
    // Cache the result
    await cacheSet(cacheKey, data, CACHE_TTL);
    
    console.log(`[Proff.no] Financial data structure created for ${orgNumber}`);
    console.log(`[Proff.no] Deeplink: ${url}`);
    
    return data;
  } catch (error) {
    console.error(`[Proff.no] Error fetching financial data:`, error);
    return null;
  }
}

/**
 * Get Proff.no deeplink for a company
 */
export function getProffDeeplink(orgNumber: string, companyName?: string): string {
  if (companyName) {
    // Format company name for URL (lowercase, replace spaces with hyphens)
    const formattedName = companyName
      .toLowerCase()
      .replace(/\s+/g, '-')
      .replace(/[^a-z0-9-]/g, '');
    
    return `${PROFF_BASE}/selskap/${formattedName}/${orgNumber}`;
  }
  
  // Fallback to search
  return `${PROFF_BASE}/bransjes%C3%B8k?q=${orgNumber}`;
}

/**
 * Calculate financial ratios
 */
export function calculateFinancialRatios(data: ProffFinancialData) {
  const ratios: {
    profitMargin?: number;
    returnOnAssets?: number;
    returnOnEquity?: number;
    debtToEquity?: number;
    currentRatio?: number;
  } = {};
  
  if (data.revenue && data.profit) {
    ratios.profitMargin = (data.profit / data.revenue) * 100;
  }
  
  if (data.profit && data.assets) {
    ratios.returnOnAssets = (data.profit / data.assets) * 100;
  }
  
  if (data.profit && data.equity) {
    ratios.returnOnEquity = (data.profit / data.equity) * 100;
  }
  
  if (data.liabilities && data.equity) {
    ratios.debtToEquity = data.liabilities / data.equity;
  }
  
  return ratios;
}

/**
 * Assess financial health
 */
export function assessFinancialHealth(data: ProffFinancialData): {
  score: number;
  level: 'excellent' | 'good' | 'fair' | 'poor';
  factors: string[];
} {
  const factors: string[] = [];
  let score = 50; // Start at neutral
  
  // Revenue check
  if (data.revenue) {
    if (data.revenue > 50000000) { // > 50M NOK
      score += 15;
      factors.push('Strong revenue (>50M NOK)');
    } else if (data.revenue > 10000000) { // > 10M NOK
      score += 10;
      factors.push('Good revenue (>10M NOK)');
    } else if (data.revenue > 1000000) { // > 1M NOK
      score += 5;
      factors.push('Moderate revenue (>1M NOK)');
    }
  }
  
  // Profitability check
  if (data.profit !== undefined) {
    if (data.profit > 0) {
      score += 15;
      factors.push('Profitable');
    } else {
      score -= 10;
      factors.push('Operating at a loss');
    }
  }
  
  // Calculate ratios
  const ratios = calculateFinancialRatios(data);
  
  if (ratios.profitMargin !== undefined) {
    if (ratios.profitMargin > 20) {
      score += 10;
      factors.push('Excellent profit margin (>20%)');
    } else if (ratios.profitMargin > 10) {
      score += 5;
      factors.push('Good profit margin (>10%)');
    } else if (ratios.profitMargin < 0) {
      score -= 10;
      factors.push('Negative profit margin');
    }
  }
  
  if (ratios.debtToEquity !== undefined) {
    if (ratios.debtToEquity < 0.5) {
      score += 10;
      factors.push('Low debt-to-equity ratio');
    } else if (ratios.debtToEquity > 2) {
      score -= 10;
      factors.push('High debt-to-equity ratio');
    }
  }
  
  // Credit rating
  if (data.creditRating) {
    if (data.creditRating === 'AAA' || data.creditRating === 'AA') {
      score += 15;
      factors.push(`Excellent credit rating (${data.creditRating})`);
    } else if (data.creditRating === 'A' || data.creditRating === 'BBB') {
      score += 10;
      factors.push(`Good credit rating (${data.creditRating})`);
    } else if (data.creditRating === 'BB' || data.creditRating === 'B') {
      score -= 5;
      factors.push(`Fair credit rating (${data.creditRating})`);
    } else {
      score -= 15;
      factors.push(`Poor credit rating (${data.creditRating})`);
    }
  }
  
  // Normalize score to 0-100
  score = Math.max(0, Math.min(100, score));
  
  let level: 'excellent' | 'good' | 'fair' | 'poor';
  if (score >= 80) level = 'excellent';
  else if (score >= 60) level = 'good';
  else if (score >= 40) level = 'fair';
  else level = 'poor';
  
  return { score, level, factors };
}

/**
 * Get growth rate from historical data
 */
export function calculateGrowthRate(data: ProffFinancialData): {
  revenueGrowth?: number;
  employeeGrowth?: number;
  trend: 'growing' | 'stable' | 'declining';
} {
  if (!data.historicalData || data.historicalData.length < 2) {
    return { trend: 'stable' };
  }
  
  const sorted = [...data.historicalData].sort((a, b) => a.year - b.year);
  const oldest = sorted[0];
  const newest = sorted[sorted.length - 1];
  
  let revenueGrowth: number | undefined;
  let employeeGrowth: number | undefined;
  
  if (oldest.revenue && newest.revenue) {
    const years = newest.year - oldest.year;
    revenueGrowth = ((newest.revenue - oldest.revenue) / oldest.revenue) * 100 / years;
  }
  
  if (oldest.employees && newest.employees) {
    const years = newest.year - oldest.year;
    employeeGrowth = ((newest.employees - oldest.employees) / oldest.employees) * 100 / years;
  }
  
  let trend: 'growing' | 'stable' | 'declining' = 'stable';
  
  if (revenueGrowth !== undefined) {
    if (revenueGrowth > 5) trend = 'growing';
    else if (revenueGrowth < -5) trend = 'declining';
  }
  
  return { revenueGrowth, employeeGrowth, trend };
}
