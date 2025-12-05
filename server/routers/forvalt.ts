import { z } from "zod";
import { TRPCError } from "@trpc/server";
import { protectedProcedure, publicProcedure, router } from "../_core/trpc";
import { getDb } from "../db";
import { companyFinancials } from "../../drizzle/schema";
import { eq } from "drizzle-orm";
import axios from "axios";

/**
 * Forvalt.no Premium Integration
 * 
 * Provides access to premium Norwegian company financial data:
 * - Credit ratings (AAA, AA, A, BBB, BB, B, CCC, CC, C, D)
 * - Financial statements (revenue, profit, assets, liabilities)
 * - Bankruptcy risk scores (0-100)
 * - Payment history and behavior
 * - AML/KYC compliance checks
 * 
 * Note: Forvalt.no requires premium subscription and API key
 * For demo purposes, this implementation uses mock data
 */

const FORVALT_API_KEY = process.env.FORVALT_API_KEY || "demo_key";
const FORVALT_API_URL = "https://api.forvalt.no/v1"; // Placeholder URL

export const forvaltRouter = router({
  /**
   * Get comprehensive financial data for a Norwegian company
   */
  getFinancialData: publicProcedure
    .input(
      z.object({
        orgnr: z.string().length(9),
      })
    )
    .query(async ({ input }) => {
      try {
        // TODO: Replace with real Forvalt.no API call when API key is available
        // const response = await axios.get(`${FORVALT_API_URL}/companies/${input.orgnr}/financial`, {
        //   headers: {
        //     Authorization: `Bearer ${FORVALT_API_KEY}`,
        //   },
        // });

        // Mock data for demonstration
        const mockFinancialData = {
          orgnr: input.orgnr,
          revenue: 25000000, // 25M NOK
          profit: 3500000, // 3.5M NOK
          assets: 18000000, // 18M NOK
          liabilities: 12000000, // 12M NOK
          equity: 6000000, // 6M NOK
          profitMargin: 14.0, // 14%
          roe: 58.3, // Return on Equity 58.3%
          debtRatio: 66.7, // 66.7%
          currentRatio: 1.5, // 1.5
          quickRatio: 1.2, // 1.2
          year: 2023,
          currency: "NOK",
        };

        return {
          success: true,
          data: mockFinancialData,
        };
      } catch (error) {
        console.error("[Forvalt] Error fetching financial data:", error);
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: "Failed to fetch financial data from Forvalt.no",
        });
      }
    }),

  /**
   * Get credit rating for a Norwegian company
   */
  getCreditRating: publicProcedure
    .input(
      z.object({
        orgnr: z.string().length(9),
      })
    )
    .query(async ({ input }) => {
      try {
        // TODO: Replace with real Forvalt.no API call
        // const response = await axios.get(`${FORVALT_API_URL}/companies/${input.orgnr}/credit-rating`, {
        //   headers: {
        //     Authorization: `Bearer ${FORVALT_API_KEY}`,
        //   },
        // });

        // Mock credit rating data
        const mockCreditRating = {
          orgnr: input.orgnr,
          rating: "A", // AAA, AA, A, BBB, BB, B, CCC, CC, C, D
          ratingNumeric: 7, // 1-10 scale (10 = AAA, 1 = D)
          outlook: "Stable", // Positive, Stable, Negative
          riskLevel: "Low", // Very Low, Low, Medium, High, Very High
          bankruptcyRisk: 5, // 0-100 (0 = no risk, 100 = imminent bankruptcy)
          paymentBehavior: "Good", // Excellent, Good, Fair, Poor, Very Poor
          paymentDelays: 2, // Average days of payment delay
          creditLimit: 5000000, // Recommended credit limit in NOK
          lastUpdated: new Date().toISOString(),
        };

        return {
          success: true,
          data: mockCreditRating,
        };
      } catch (error) {
        console.error("[Forvalt] Error fetching credit rating:", error);
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: "Failed to fetch credit rating from Forvalt.no",
        });
      }
    }),

  /**
   * Get AML/KYC compliance check for a Norwegian company
   */
  getComplianceCheck: publicProcedure
    .input(
      z.object({
        orgnr: z.string().length(9),
      })
    )
    .query(async ({ input }) => {
      try {
        // TODO: Replace with real Forvalt.no API call
        // const response = await axios.get(`${FORVALT_API_URL}/companies/${input.orgnr}/compliance`, {
        //   headers: {
        //     Authorization: `Bearer ${FORVALT_API_KEY}`,
        //   },
        // });

        // Mock compliance data
        const mockComplianceData = {
          orgnr: input.orgnr,
          amlStatus: "Clear", // Clear, Warning, Alert
          kycStatus: "Verified", // Verified, Pending, Failed
          sanctionsList: false, // On sanctions list?
          pepExposure: false, // Politically Exposed Person?
          adverseMedia: false, // Negative media coverage?
          riskScore: 15, // 0-100 (0 = no risk, 100 = high risk)
          riskCategory: "Low", // Low, Medium, High
          lastChecked: new Date().toISOString(),
          checks: [
            { type: "Sanctions List", status: "Clear", date: new Date().toISOString() },
            { type: "PEP Screening", status: "Clear", date: new Date().toISOString() },
            { type: "Adverse Media", status: "Clear", date: new Date().toISOString() },
            { type: "UBO Verification", status: "Verified", date: new Date().toISOString() },
          ],
        };

        return {
          success: true,
          data: mockComplianceData,
        };
      } catch (error) {
        console.error("[Forvalt] Error fetching compliance check:", error);
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: "Failed to fetch compliance check from Forvalt.no",
        });
      }
    }),

  /**
   * Get payment history for a Norwegian company
   */
  getPaymentHistory: publicProcedure
    .input(
      z.object({
        orgnr: z.string().length(9),
      })
    )
    .query(async ({ input }) => {
      try {
        // Mock payment history data
        const mockPaymentHistory = {
          orgnr: input.orgnr,
          averagePaymentTime: 32, // Days
          onTimePaymentRate: 85, // Percentage
          latePayments: 15, // Percentage
          totalInvoices: 245,
          totalValue: 12500000, // NOK
          longestDelay: 45, // Days
          recentPayments: [
            { date: "2024-01-15", amount: 125000, daysLate: 0, status: "On Time" },
            { date: "2024-01-10", amount: 85000, daysLate: 5, status: "Late" },
            { date: "2024-01-05", amount: 200000, daysLate: 0, status: "On Time" },
            { date: "2023-12-28", amount: 150000, daysLate: 2, status: "Late" },
            { date: "2023-12-20", amount: 95000, daysLate: 0, status: "On Time" },
          ],
        };

        return {
          success: true,
          data: mockPaymentHistory,
        };
      } catch (error) {
        console.error("[Forvalt] Error fetching payment history:", error);
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: "Failed to fetch payment history from Forvalt.no",
        });
      }
    }),

  // Note: getFullReport removed due to circular reference issues
  // TODO: Implement in separate service layer
});
