/**
 * Forvalt.no Premium Web Scraper
 * 
 * Authenticated scraping of Norwegian company credit ratings, financial data,
 * bankruptcy probability, and risk assessments from Forvalt.no premium platform.
 * 
 * Credentials: LL2020365@gmail.com / S8LRXdWk
 * Customer Number: 90347
 */

import puppeteer, { Browser, Page } from 'puppeteer';
import { getCachedForvaltData, setCachedForvaltData } from './redis_cache';

// Forvalt.no credentials
const FORVALT_EMAIL = 'LL2020365@gmail.com';
const FORVALT_PASSWORD = 'S8LRXdWk';
const FORVALT_BASE_URL = 'https://forvalt.no';

// Cache for browser instance (reuse across requests)
let browserInstance: Browser | null = null;

/**
 * Credit rating levels from Forvalt.no
 */
export type CreditRating = 'A+' | 'A' | 'A-' | 'B+' | 'B' | 'B-' | 'C+' | 'C' | 'C-' | 'D' | 'E';

/**
 * Risk level classification
 */
export type RiskLevel = 'very_low' | 'low' | 'moderate' | 'high' | 'very_high';

/**
 * Complete Forvalt.no company data structure
 */
export interface ForvaltCompanyData {
  // Credit Rating
  creditRating: CreditRating | null;
  creditScore: number | null; // 0-100
  bankruptcyProbability: number | null; // 0-100%
  creditLimit: number | null; // NOK
  riskLevel: RiskLevel | null;
  riskDescription: string | null;
  
  // Rating Components (1-5 scale)
  leadershipScore: number | null;
  economyScore: number | null;
  paymentHistoryScore: number | null;
  generalScore: number | null;
  
  // Financial Metrics (in company's currency)
  revenue: number | null;
  ebitda: number | null;
  operatingResult: number | null;
  totalAssets: number | null;
  profitability: number | null; // %
  liquidity: number | null;
  solidity: number | null; // %
  ebitdaMargin: number | null; // %
  currency: string | null; // USD, NOK, EUR, etc.
  
  // Payment Remarks
  voluntaryLiens: number | null;
  factoringAgreements: number | null;
  forcedLiens: number | null;
  hasPaymentRemarks: boolean;
  
  // Company Info
  companyName: string | null;
  orgNumber: string | null;
  organizationForm: string | null;
  shareCapital: number | null;
  founded: string | null;
  employees: number | null;
  website: string | null;
  phone: string | null;
  
  // Leadership
  ceo: string | null;
  boardChairman: string | null;
  auditor: string | null;
  
  // Metadata
  lastUpdated: Date;
  forvaltUrl: string;
}

/**
 * Get or create browser instance
 */
async function getBrowser(): Promise<Browser> {
  if (browserInstance && browserInstance.connected) {
    return browserInstance;
  }
  
  browserInstance = await puppeteer.launch({
    headless: true,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
      '--disable-gpu',
    ],
  });
  
  return browserInstance;
}

/**
 * Login to Forvalt.no and return authenticated page
 */
async function loginToForvalt(browser: Browser): Promise<Page> {
  const page = await browser.newPage();
  
  // Set viewport
  await page.setViewport({ width: 1920, height: 1080 });
  
  // Navigate to login page
  await page.goto(`${FORVALT_BASE_URL}/Account/LogOn`, {
    waitUntil: 'networkidle2',
    timeout: 30000,
  });
  
  // Fill in credentials
  await page.type('#UserName', FORVALT_EMAIL);
  await page.type('#Password', FORVALT_PASSWORD);
  
  // Click login button
  await Promise.all([
    page.waitForNavigation({ waitUntil: 'networkidle2' }),
    page.click('input[type="submit"]'),
  ]);
  
  // Verify login success
  const isLoggedIn = await page.evaluate(() => {
    return document.body.textContent?.includes('Lucas') || 
           document.body.textContent?.includes('Kundenr: 90347');
  });
  
  if (!isLoggedIn) {
    throw new Error('Failed to login to Forvalt.no - credentials may be incorrect');
  }
  
  return page;
}

/**
 * Extract credit rating from Forvalt company page
 */
async function extractCreditRating(page: Page): Promise<{
  rating: CreditRating | null;
  score: number | null;
  riskLevel: RiskLevel | null;
  riskDescription: string | null;
}> {
  return await page.evaluate(() => {
    // Find "Proff Premium rating" section
    const ratingSection = document.querySelector('.proff-premium-rating, [class*="rating"]');
    if (!ratingSection) {
      return { rating: null, score: null, riskLevel: null, riskDescription: null };
    }
    
    // Extract rating (A+, A, B, etc.)
    const ratingText = ratingSection.textContent || '';
    const ratingMatch = ratingText.match(/([A-E][+-]?)/);
    const rating = ratingMatch ? ratingMatch[1] as any : null;
    
    // Extract score (0-100)
    const scoreMatch = ratingText.match(/Score:\s*(\d+)\/100/i) || 
                       ratingText.match(/(\d+)\/100/);
    const score = scoreMatch ? parseInt(scoreMatch[1]) : null;
    
    // Extract risk description
    const riskMatch = ratingText.match(/(Meget lav risiko|Lav risiko|Moderat risiko|Høy risiko|Særdeles høy risiko)/i);
    const riskDescription = riskMatch ? riskMatch[1] : null;
    
    // Map to risk level
    let riskLevel: any = null;
    if (riskDescription) {
      if (riskDescription.includes('Meget lav') || riskDescription.includes('Særdeles lav')) {
        riskLevel = 'very_low';
      } else if (riskDescription.includes('Lav')) {
        riskLevel = 'low';
      } else if (riskDescription.includes('Moderat')) {
        riskLevel = 'moderate';
      } else if (riskDescription.includes('Høy')) {
        riskLevel = 'high';
      } else if (riskDescription.includes('Særdeles høy')) {
        riskLevel = 'very_high';
      }
    }
    
    return { rating, score, riskLevel, riskDescription };
  });
}

/**
 * Extract bankruptcy probability
 */
async function extractBankruptcyProbability(page: Page): Promise<number | null> {
  return await page.evaluate(() => {
    const text = document.body.textContent || '';
    const match = text.match(/sannsynligheten for konkurs er\s*([\d,\.]+)%/i);
    if (match) {
      return parseFloat(match[1].replace(',', '.'));
    }
    return null;
  });
}

/**
 * Extract credit limit
 */
async function extractCreditLimit(page: Page): Promise<number | null> {
  return await page.evaluate(() => {
    const text = document.body.textContent || '';
    const match = text.match(/Kredittramme[:\s]*([\d\s]+)/i);
    if (match) {
      return parseInt(match[1].replace(/\s/g, ''));
    }
    return null;
  });
}

/**
 * Extract rating components (Leadership, Economy, Payment History, General)
 */
async function extractRatingComponents(page: Page): Promise<{
  leadershipScore: number | null;
  economyScore: number | null;
  paymentHistoryScore: number | null;
  generalScore: number | null;
}> {
  return await page.evaluate(() => {
    const text = document.body.textContent || '';
    
    const leadershipMatch = text.match(/Ledelse og eierskap[:\s]*Vurdering:\s*(\d+)/i);
    const economyMatch = text.match(/Økonomi[:\s]*Vurdering:\s*(\d+)/i);
    const paymentMatch = text.match(/Betalingshistorikk[:\s]*Vurdering:\s*(\d+)/i);
    const generalMatch = text.match(/Generelt[:\s]*Vurdering:\s*(\d+)/i);
    
    return {
      leadershipScore: leadershipMatch ? parseInt(leadershipMatch[1]) : null,
      economyScore: economyMatch ? parseInt(economyMatch[1]) : null,
      paymentHistoryScore: paymentMatch ? parseInt(paymentMatch[1]) : null,
      generalScore: generalMatch ? parseInt(generalMatch[1]) : null,
    };
  });
}

/**
 * Extract financial metrics from company page
 */
async function extractFinancialMetrics(page: Page): Promise<{
  revenue: number | null;
  ebitda: number | null;
  operatingResult: number | null;
  totalAssets: number | null;
  profitability: number | null;
  liquidity: number | null;
  solidity: number | null;
  ebitdaMargin: number | null;
  currency: string | null;
}> {
  return await page.evaluate(() => {
    const text = document.body.textContent || '';
    
    // Extract currency
    const currencyMatch = text.match(/Valutakode[:\s]*(USD|NOK|EUR|GBP|SEK|DKK)/i);
    const currency = currencyMatch ? currencyMatch[1] : 'NOK';
    
    // Helper to parse numbers (handles spaces and commas)
    const parseNum = (str: string | null): number | null => {
      if (!str) return null;
      const cleaned = str.replace(/\s/g, '').replace(',', '.');
      const num = parseFloat(cleaned);
      return isNaN(num) ? null : num;
    };
    
    // Extract revenue (Driftsinnt.)
    const revenueMatch = text.match(/Driftsinnt\.[:\s]*([\d\s,\.]+)/i);
    const revenue = revenueMatch ? parseNum(revenueMatch[1]) : null;
    
    // Extract EBITDA
    const ebitdaMatch = text.match(/EBITDA[:\s]*([\d\s,\.]+)/i);
    const ebitda = ebitdaMatch ? parseNum(ebitdaMatch[1]) : null;
    
    // Extract operating result (Driftsresultat)
    const opResultMatch = text.match(/Driftsresultat[:\s]*([\d\s,\.]+)/i);
    const operatingResult = opResultMatch ? parseNum(opResultMatch[1]) : null;
    
    // Extract total assets (Sum eiend.)
    const assetsMatch = text.match(/Sum eiend\.[:\s]*([\d\s,\.]+)/i);
    const totalAssets = assetsMatch ? parseNum(assetsMatch[1]) : null;
    
    // Extract profitability (Lønnsomhet)
    const profitMatch = text.match(/Lønnsomhet[:\s]*([\d,\.]+)%/i);
    const profitability = profitMatch ? parseNum(profitMatch[1]) : null;
    
    // Extract liquidity (Likviditet)
    const liquidityMatch = text.match(/Likviditet[:\s]*([\d,\.]+)/i);
    const liquidity = liquidityMatch ? parseNum(liquidityMatch[1]) : null;
    
    // Extract solidity (Soliditet)
    const solidityMatch = text.match(/Soliditet[:\s]*([\d,\.]+)%/i);
    const solidity = solidityMatch ? parseNum(solidityMatch[1]) : null;
    
    // Extract EBITDA margin
    const ebitdaMarginMatch = text.match(/EBITDA[:\s]*[\d\s,\.]+.*?([\d,\.]+)%/i);
    const ebitdaMargin = ebitdaMarginMatch ? parseNum(ebitdaMarginMatch[1]) : null;
    
    return {
      revenue,
      ebitda,
      operatingResult,
      totalAssets,
      profitability,
      liquidity,
      solidity,
      ebitdaMargin,
      currency,
    };
  });
}

/**
 * Extract payment remarks
 */
async function extractPaymentRemarks(page: Page): Promise<{
  voluntaryLiens: number | null;
  factoringAgreements: number | null;
  forcedLiens: number | null;
  hasPaymentRemarks: boolean;
}> {
  return await page.evaluate(() => {
    const text = document.body.textContent || '';
    
    const voluntaryMatch = text.match(/Frivillig pant[:\s]*(\d+)/i);
    const factoringMatch = text.match(/Factoringavtaler[:\s]*(\d+)/i);
    const forcedMatch = text.match(/Tvungen pant[:\s]*(\d+)/i);
    const remarksMatch = text.match(/Betalingsanmerkninger[:\s]*(Ja|Nei)/i);
    
    return {
      voluntaryLiens: voluntaryMatch ? parseInt(voluntaryMatch[1]) : null,
      factoringAgreements: factoringMatch ? parseInt(factoringMatch[1]) : null,
      forcedLiens: forcedMatch ? parseInt(forcedMatch[1]) : null,
      hasPaymentRemarks: remarksMatch ? remarksMatch[1].toLowerCase() === 'ja' : false,
    };
  });
}

/**
 * Extract company information
 */
async function extractCompanyInfo(page: Page): Promise<{
  companyName: string | null;
  orgNumber: string | null;
  organizationForm: string | null;
  shareCapital: number | null;
  founded: string | null;
  employees: number | null;
  website: string | null;
  phone: string | null;
}> {
  return await page.evaluate(() => {
    const text = document.body.textContent || '';
    
    const nameMatch = text.match(/Selskapsnavn[:\s]*([^\n]+)/i);
    const orgMatch = text.match(/Organisasjonsnr[:\s]*([\d\s]+)/i);
    const formMatch = text.match(/Organisasjonsform[:\s]*([^\n]+)/i);
    const capitalMatch = text.match(/Aksjekapital[:\s]*([\d\s]+)/i);
    const foundedMatch = text.match(/Stiftelsedato[:\s]*([\d\.]+)/i);
    const employeesMatch = text.match(/Antall ansatte[:\s]*([\d\s]+)/i);
    const websiteMatch = text.match(/Internett[:\s]*([^\n]+)/i);
    const phoneMatch = text.match(/Telefon[:\s]*([\d\s]+)/i);
    
    return {
      companyName: nameMatch ? nameMatch[1].trim() : null,
      orgNumber: orgMatch ? orgMatch[1].replace(/\s/g, '') : null,
      organizationForm: formMatch ? formMatch[1].trim() : null,
      shareCapital: capitalMatch ? parseInt(capitalMatch[1].replace(/\s/g, '')) : null,
      founded: foundedMatch ? foundedMatch[1] : null,
      employees: employeesMatch ? parseInt(employeesMatch[1].replace(/\s/g, '')) : null,
      website: websiteMatch ? websiteMatch[1].trim() : null,
      phone: phoneMatch ? phoneMatch[1].replace(/\s/g, '') : null,
    };
  });
}

/**
 * Extract leadership information
 */
async function extractLeadership(page: Page): Promise<{
  ceo: string | null;
  boardChairman: string | null;
  auditor: string | null;
}> {
  return await page.evaluate(() => {
    const text = document.body.textContent || '';
    
    const ceoMatch = text.match(/Daglig leder[:\s]*([^\n]+)/i) ||
                     text.match(/DAGL[:\s]*([^\n]+)/i);
    const chairmanMatch = text.match(/Styrets leder[:\s]*([^\n]+)/i);
    const auditorMatch = text.match(/Revisor[:\s]*([^\n]+)/i);
    
    return {
      ceo: ceoMatch ? ceoMatch[1].trim() : null,
      boardChairman: chairmanMatch ? chairmanMatch[1].trim() : null,
      auditor: auditorMatch ? auditorMatch[1].trim() : null,
    };
  });
}

/**
 * Main function: Scrape complete Forvalt.no data for a Norwegian company
 * 
 * @param orgNumber - Norwegian organization number (9 digits)
 * @returns Complete Forvalt company data
 */
export async function scrapeForvaltData(orgNumber: string): Promise<ForvaltCompanyData> {
  // Check cache first
  const cached = await getCachedForvaltData(orgNumber);
  if (cached) {
    console.log(`[Forvalt] Cache hit for ${orgNumber}`);
    return cached as ForvaltCompanyData;
  }
  
  console.log(`[Forvalt] Cache miss for ${orgNumber}, scraping...`);
  let page: Page | null = null;
  
  try {
    // Get browser instance
    const browser = await getBrowser();
    
    // Login to Forvalt
    page = await loginToForvalt(browser);
    
    // Navigate to company page
    const companyUrl = `${FORVALT_BASE_URL}/ForetaksIndex/Firma/FirmaSide/${orgNumber}`;
    await page.goto(companyUrl, {
      waitUntil: 'networkidle2',
      timeout: 30000,
    });
    
    // Wait for content to load
    await page.waitForSelector('body', { timeout: 10000 });
    
    // Extract all data in parallel
    const [
      creditRating,
      bankruptcyProbability,
      creditLimit,
      ratingComponents,
      financialMetrics,
      paymentRemarks,
      companyInfo,
      leadership,
    ] = await Promise.all([
      extractCreditRating(page),
      extractBankruptcyProbability(page),
      extractCreditLimit(page),
      extractRatingComponents(page),
      extractFinancialMetrics(page),
      extractPaymentRemarks(page),
      extractCompanyInfo(page),
      extractLeadership(page),
    ]);
    
    // Combine all data
    const forvaltData: ForvaltCompanyData = {
      // Credit Rating
      creditRating: creditRating.rating,
      creditScore: creditRating.score,
      bankruptcyProbability,
      creditLimit,
      riskLevel: creditRating.riskLevel,
      riskDescription: creditRating.riskDescription,
      
      // Rating Components
      leadershipScore: ratingComponents.leadershipScore,
      economyScore: ratingComponents.economyScore,
      paymentHistoryScore: ratingComponents.paymentHistoryScore,
      generalScore: ratingComponents.generalScore,
      
      // Financial Metrics
      revenue: financialMetrics.revenue,
      ebitda: financialMetrics.ebitda,
      operatingResult: financialMetrics.operatingResult,
      totalAssets: financialMetrics.totalAssets,
      profitability: financialMetrics.profitability,
      liquidity: financialMetrics.liquidity,
      solidity: financialMetrics.solidity,
      ebitdaMargin: financialMetrics.ebitdaMargin,
      currency: financialMetrics.currency,
      
      // Payment Remarks
      voluntaryLiens: paymentRemarks.voluntaryLiens,
      factoringAgreements: paymentRemarks.factoringAgreements,
      forcedLiens: paymentRemarks.forcedLiens,
      hasPaymentRemarks: paymentRemarks.hasPaymentRemarks,
      
      // Company Info
      companyName: companyInfo.companyName,
      orgNumber: companyInfo.orgNumber,
      organizationForm: companyInfo.organizationForm,
      shareCapital: companyInfo.shareCapital,
      founded: companyInfo.founded,
      employees: companyInfo.employees,
      website: companyInfo.website,
      phone: companyInfo.phone,
      
      // Leadership
      ceo: leadership.ceo,
      boardChairman: leadership.boardChairman,
      auditor: leadership.auditor,
      
      // Metadata
      lastUpdated: new Date(),
      forvaltUrl: companyUrl,
    };
    
    // Cache the result (24 hour TTL)
    await setCachedForvaltData(orgNumber, forvaltData, 86400);
    console.log(`[Forvalt] Cached data for ${orgNumber}`);
    
    return forvaltData;
    
  } catch (error) {
    console.error('[Forvalt Scraper] Error scraping data:', error);
    throw error;
  } finally {
    // Close page but keep browser instance for reuse
    if (page) {
      await page.close();
    }
  }
}

/**
 * Close browser instance (call on server shutdown)
 */
export async function closeBrowser(): Promise<void> {
  if (browserInstance) {
    await browserInstance.close();
    browserInstance = null;
  }
}

/**
 * Health check: Verify Forvalt.no credentials and connection
 */
export async function healthCheck(): Promise<boolean> {
  try {
    const browser = await getBrowser();
    const page = await loginToForvalt(browser);
    await page.close();
    return true;
  } catch (error) {
    console.error('[Forvalt Scraper] Health check failed:', error);
    return false;
  }
}

/**
 * Search companies using Forvalt segmentation API (1.2M+ companies)
 */
export async function searchCompaniesBySegmentation(
  filters: {
    industry?: string;
    region?: string;
    employeeRange?: string;
    revenueRange?: string;
    creditRating?: string;
  },
  limit: number = 100
): Promise<Array<{ orgNumber: string; name: string; industry: string }>> {
  try {
    const browser = await getBrowser();
    const page = await loginToForvalt(browser);
    
    // Navigate to Forvalt search page
    await page.goto('https://www.forvalt.no/search', { waitUntil: 'networkidle2' });
    
    // Apply filters (simplified - actual implementation would use Forvalt's search API)
    const companies: Array<{ orgNumber: string; name: string; industry: string }> = [];
    
    // Placeholder for actual segmentation search
    console.log('[Forvalt] Segmentation search with filters:', filters);
    
    await page.close();
    return companies;
  } catch (error) {
    console.error('[Forvalt] Error in segmentation search:', error);
    return [];
  }
}

/**
 * Search international companies (Belgium, etc.)
 */
export async function searchInternationalCompany(
  companyName: string,
  country: string = 'Belgium'
): Promise<{
  name: string;
  registrationNumber: string;
  country: string;
  address: string;
  website?: string;
} | null> {
  try {
    console.log(`[Forvalt] Searching international company: ${companyName} in ${country}`);
    
    // Placeholder for international search
    // In production, this would integrate with European business registries
    return {
      name: companyName,
      registrationNumber: 'INT-' + Date.now(),
      country,
      address: 'International address',
      website: undefined,
    };
  } catch (error) {
    console.error('[Forvalt] Error in international search:', error);
    return null;
  }
}

/**
 * Get competitor analysis for a company
 */
export async function getCompetitorAnalysis(
  orgNumber: string,
  topN: number = 5
): Promise<Array<{
  orgNumber: string;
  name: string;
  creditRating: string;
  revenue: number;
  employees: number;
  similarityScore: number;
}>> {
  try {
    console.log(`[Forvalt] Getting competitor analysis for ${orgNumber}`);
    
    // Placeholder for competitor analysis
    // In production, this would use Forvalt's industry comparison features
    return [];
  } catch (error) {
    console.error('[Forvalt] Error in competitor analysis:', error);
    return [];
  }
}

/**
 * Export company data to Excel
 */
export async function exportToExcel(
  companies: Array<{ orgNumber: string; name: string }>
): Promise<Buffer> {
  try {
    console.log(`[Forvalt] Exporting ${companies.length} companies to Excel`);
    
    // Placeholder for Excel export
    // In production, this would use a library like exceljs
    return Buffer.from('Excel data placeholder');
  } catch (error) {
    console.error('[Forvalt] Error exporting to Excel:', error);
    throw error;
  }
}
