import { z } from 'zod';
import { publicProcedure, router } from '../_core/trpc';
import { scrapeForvaltData, healthCheck } from '../helpers/forvalt_scraper';

/**
 * Forvalt.no tRPC Router
 * 
 * Provides procedures for fetching Norwegian company credit ratings,
 * financial data, and risk assessments from Forvalt.no premium platform.
 * 
 * Uses authenticated web scraping with Puppeteer.
 * Credentials: LL2020365@gmail.com / S8LRXdWk
 */

export const forvaltRouter = router({
  /**
   * Get complete financial data and credit rating for a Norwegian company
   */
  getFinancialData: publicProcedure
    .input(z.object({
      orgNumber: z.string().regex(/^\d{9}$/, 'Organization number must be 9 digits'),
    }))
    .query(async ({ input }) => {
      try {
        const data = await scrapeForvaltData(input.orgNumber);
        return {
          success: true,
          data,
        };
      } catch (error) {
        console.error('[Forvalt Router] Error fetching financial data:', error);
        return {
          success: false,
          error: error instanceof Error ? error.message : 'Failed to fetch Forvalt data',
          data: null,
        };
      }
    }),

  /**
   * Get credit rating and risk assessment only (faster)
   */
  getCreditRating: publicProcedure
    .input(z.object({
      orgNumber: z.string().regex(/^\d{9}$/, 'Organization number must be 9 digits'),
    }))
    .query(async ({ input }) => {
      try {
        const data = await scrapeForvaltData(input.orgNumber);
        return {
          success: true,
          creditRating: data.creditRating,
          creditScore: data.creditScore,
          bankruptcyProbability: data.bankruptcyProbability,
          riskLevel: data.riskLevel,
          riskDescription: data.riskDescription,
          creditLimit: data.creditLimit,
        };
      } catch (error) {
        console.error('[Forvalt Router] Error fetching credit rating:', error);
        return {
          success: false,
          error: error instanceof Error ? error.message : 'Failed to fetch credit rating',
          creditRating: null,
          creditScore: null,
          bankruptcyProbability: null,
          riskLevel: null,
          riskDescription: null,
          creditLimit: null,
        };
      }
    }),

  /**
   * Get compliance check (payment remarks, liens, etc.)
   */
  getComplianceCheck: publicProcedure
    .input(z.object({
      orgNumber: z.string().regex(/^\d{9}$/, 'Organization number must be 9 digits'),
    }))
    .query(async ({ input }) => {
      try {
        const data = await scrapeForvaltData(input.orgNumber);
        return {
          success: true,
          voluntaryLiens: data.voluntaryLiens,
          factoringAgreements: data.factoringAgreements,
          forcedLiens: data.forcedLiens,
          hasPaymentRemarks: data.hasPaymentRemarks,
          companyName: data.companyName,
          orgNumber: data.orgNumber,
        };
      } catch (error) {
        console.error('[Forvalt Router] Error fetching compliance check:', error);
        return {
          success: false,
          error: error instanceof Error ? error.message : 'Failed to fetch compliance data',
          voluntaryLiens: null,
          factoringAgreements: null,
          forcedLiens: null,
          hasPaymentRemarks: false,
          companyName: null,
          orgNumber: null,
        };
      }
    }),

  /**
   * Health check for Forvalt.no connection
   */
  healthCheck: publicProcedure
    .query(async () => {
      const isHealthy = await healthCheck();
      return {
        success: isHealthy,
        message: isHealthy 
          ? 'Forvalt.no connection is healthy' 
          : 'Forvalt.no connection failed - check credentials',
      };
    }),
});
