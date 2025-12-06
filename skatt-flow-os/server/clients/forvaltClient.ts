import axios, { AxiosInstance, AxiosError } from "axios";
import { nanoid } from "nanoid";
import type { ForvaltCompanyInfo, ForvaltCreditInfo, ForvaltFinancials } from "@shared/types";

// ============================================================================
// FORVALT/PROFF API CLIENT
// For Norwegian company data, credit scores, and financial information
// ============================================================================

export class ForvaltApiError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public correlationId?: string
  ) {
    super(message);
    this.name = "ForvaltApiError";
  }
}

interface ForvaltClientConfig {
  baseUrl: string;
  apiKey: string;
  timeout?: number;
}

export class ForvaltClient {
  private client: AxiosInstance;
  private apiKey: string;

  constructor(config: ForvaltClientConfig) {
    this.apiKey = config.apiKey;
    this.client = axios.create({
      baseURL: config.baseUrl,
      timeout: config.timeout || 30000,
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
    });
  }

  private getCorrelationId(): string {
    return nanoid(16);
  }

  private getAuthHeaders(correlationId: string) {
    return {
      Authorization: `Bearer ${this.apiKey}`,
      "X-Correlation-Id": correlationId,
    };
  }

  /**
   * Fetch core company info including name, address, and industry
   */
  async getCompany(orgNumber: string): Promise<ForvaltCompanyInfo> {
    const correlationId = this.getCorrelationId();
    const cleanOrgNumber = orgNumber.replace(/\s/g, "");

    try {
      const response = await this.client.get(`/companies/${cleanOrgNumber}`, {
        headers: this.getAuthHeaders(correlationId),
      });

      const data = response.data;
      return {
        orgNumber: data.organisasjonsnummer || data.orgNumber || cleanOrgNumber,
        name: data.navn || data.name || "",
        address: data.forretningsadresse?.adresse?.[0] || data.address || "",
        city: data.forretningsadresse?.poststed || data.city || "",
        postalCode: data.forretningsadresse?.postnummer || data.postalCode || "",
        industryCode: data.naeringskode1?.kode || data.industryCode || "",
        industryDescription: data.naeringskode1?.beskrivelse || data.industryDescription || "",
        registrationDate: data.stiftelsesdato || data.registrationDate || "",
        employees: data.antallAnsatte || data.employees || 0,
        status: data.status || "ACTIVE",
      };
    } catch (error) {
      if (error instanceof AxiosError) {
        throw new ForvaltApiError(
          `Failed to fetch company ${cleanOrgNumber}: ${error.message}`,
          error.response?.status,
          correlationId
        );
      }
      throw error;
    }
  }

  /**
   * Fetch credit score, rating, and risk information
   */
  async getCredit(orgNumber: string): Promise<ForvaltCreditInfo> {
    const correlationId = this.getCorrelationId();
    const cleanOrgNumber = orgNumber.replace(/\s/g, "");

    try {
      const response = await this.client.get(`/credit/${cleanOrgNumber}`, {
        headers: this.getAuthHeaders(correlationId),
      });

      const data = response.data;
      return {
        orgNumber: cleanOrgNumber,
        creditScore: data.creditScore || data.score || null,
        rating: data.rating || data.grade || null,
        riskClass: data.riskClass || data.risikoklasse || null,
        paymentRemarks: data.paymentRemarks || data.betalingsanmerkninger || 0,
        creditLimit: data.creditLimit || data.kredittgrense || null,
        lastUpdated: data.lastUpdated || data.sistOppdatert || new Date().toISOString(),
      };
    } catch (error) {
      if (error instanceof AxiosError) {
        throw new ForvaltApiError(
          `Failed to fetch credit for ${cleanOrgNumber}: ${error.message}`,
          error.response?.status,
          correlationId
        );
      }
      throw error;
    }
  }

  /**
   * Fetch key financial figures (revenue, result, equity, debt)
   */
  async getFinancials(orgNumber: string, year?: number): Promise<ForvaltFinancials> {
    const correlationId = this.getCorrelationId();
    const cleanOrgNumber = orgNumber.replace(/\s/g, "");
    const targetYear = year || new Date().getFullYear() - 1;

    try {
      const response = await this.client.get(`/financials/${cleanOrgNumber}`, {
        headers: this.getAuthHeaders(correlationId),
        params: { year: targetYear },
      });

      const data = response.data;
      return {
        orgNumber: cleanOrgNumber,
        year: data.year || targetYear,
        revenue: data.revenue || data.driftsinntekter || null,
        operatingResult: data.operatingResult || data.driftsresultat || null,
        netResult: data.netResult || data.arsresultat || null,
        equity: data.equity || data.egenkapital || null,
        totalDebt: data.totalDebt || data.gjeld || null,
        totalAssets: data.totalAssets || data.sumEiendeler || null,
      };
    } catch (error) {
      if (error instanceof AxiosError) {
        throw new ForvaltApiError(
          `Failed to fetch financials for ${cleanOrgNumber}: ${error.message}`,
          error.response?.status,
          correlationId
        );
      }
      throw error;
    }
  }

  /**
   * Fetch all company data in one call (company info + credit + financials)
   */
  async getFullProfile(orgNumber: string): Promise<{
    company: ForvaltCompanyInfo;
    credit: ForvaltCreditInfo;
    financials: ForvaltFinancials;
  }> {
    const [company, credit, financials] = await Promise.all([
      this.getCompany(orgNumber),
      this.getCredit(orgNumber),
      this.getFinancials(orgNumber),
    ]);

    return { company, credit, financials };
  }
}

// Factory function to create client from environment variables
export function createForvaltClient(): ForvaltClient {
  const baseUrl = process.env.FORVALT_BASE_URL || "https://api.forvalt.no/v1";
  const apiKey = process.env.PROFF_API_KEY || "";

  if (!apiKey) {
    console.warn("[ForvaltClient] PROFF_API_KEY not set - API calls will fail");
  }

  return new ForvaltClient({ baseUrl, apiKey });
}
