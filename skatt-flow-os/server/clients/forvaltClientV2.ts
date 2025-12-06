import axios, { AxiosInstance, AxiosError } from "axios";
import * as db from "../db";

// ============================================================================
// FORVALT/PROFF API CLIENT V2 - PRODUCTION READY
// Norwegian company data, credit ratings, and financial information
// Supports both Forvalt API and Proff API as fallback
// ============================================================================

export interface ForvaltConfig {
  apiKey: string;
  baseUrl?: string;
  proffApiKey?: string;
  proffBaseUrl?: string;
  timeout?: number;
  retryAttempts?: number;
}

export interface CompanyProfile {
  orgNumber: string;
  name: string;
  organizationForm: string;
  registrationDate: string;
  address: {
    street: string;
    postalCode: string;
    city: string;
    country: string;
  };
  businessAddress?: {
    street: string;
    postalCode: string;
    city: string;
  };
  industry: {
    code: string;
    description: string;
  };
  employees?: number;
  shareCapital?: number;
  isActive: boolean;
  isBankrupt: boolean;
  isUnderLiquidation: boolean;
  roles?: CompanyRole[];
}

export interface CompanyRole {
  type: string;
  name: string;
  personalId?: string;
  orgNumber?: string;
  from: string;
}

export interface CreditRating {
  orgNumber: string;
  rating: string; // A, B, C, D, E
  ratingDescription: string;
  creditScore: number; // 0-100
  creditLimit: number; // in NOK
  riskClass: "LOW" | "MEDIUM" | "HIGH" | "VERY_HIGH";
  paymentRemarks: number;
  paymentIndex?: number;
  lastUpdated: string;
  factors: CreditFactor[];
}

export interface CreditFactor {
  factor: string;
  impact: "POSITIVE" | "NEGATIVE" | "NEUTRAL";
  description: string;
}

export interface FinancialStatement {
  orgNumber: string;
  year: number;
  revenue: number;
  operatingProfit: number;
  netProfit: number;
  totalAssets: number;
  equity: number;
  debt: number;
  employees: number;
  currency: string;
  ratios: {
    profitMargin: number;
    equityRatio: number;
    currentRatio: number;
    debtToEquity: number;
    returnOnEquity: number;
  };
}

export interface OwnershipInfo {
  orgNumber: string;
  shareholders: Shareholder[];
  ultimateOwner?: {
    name: string;
    country: string;
    ownershipPercentage: number;
  };
}

export interface Shareholder {
  name: string;
  orgNumber?: string;
  personalId?: string;
  shares: number;
  percentage: number;
  shareClass?: string;
}

export interface ForvaltFullProfile {
  company: CompanyProfile;
  credit: CreditRating;
  financials: FinancialStatement[];
  ownership: OwnershipInfo;
  lastUpdated: string;
}

// Rate limiter for API calls
class RateLimiter {
  private tokens: number;
  private maxTokens: number;
  private refillRate: number;
  private lastRefill: number;

  constructor(maxTokens: number = 10, refillRate: number = 1000) {
    this.tokens = maxTokens;
    this.maxTokens = maxTokens;
    this.refillRate = refillRate;
    this.lastRefill = Date.now();
  }

  async acquire(): Promise<void> {
    const now = Date.now();
    const elapsed = now - this.lastRefill;
    const refillTokens = Math.floor(elapsed / this.refillRate);
    
    if (refillTokens > 0) {
      this.tokens = Math.min(this.maxTokens, this.tokens + refillTokens);
      this.lastRefill = now;
    }

    if (this.tokens <= 0) {
      const waitTime = this.refillRate - (now - this.lastRefill);
      await new Promise(resolve => setTimeout(resolve, waitTime));
      return this.acquire();
    }

    this.tokens--;
  }
}

export class ForvaltClientV2 {
  private client: AxiosInstance;
  private proffClient?: AxiosInstance;
  private rateLimiter: RateLimiter;
  private retryAttempts: number;
  private cache: Map<string, { data: unknown; expiry: number }> = new Map();
  private cacheTimeout: number = 3600000; // 1 hour

  constructor(config: ForvaltConfig) {
    this.retryAttempts = config.retryAttempts || 3;
    this.rateLimiter = new RateLimiter(10, 1000);

    // Primary Forvalt API client
    this.client = axios.create({
      baseURL: config.baseUrl || "https://api.forvalt.no/v2",
      timeout: config.timeout || 30000,
      headers: {
        "Authorization": `Bearer ${config.apiKey}`,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Client-Id": "skatt-flow-os",
      },
    });

    // Fallback Proff API client
    if (config.proffApiKey) {
      this.proffClient = axios.create({
        baseURL: config.proffBaseUrl || "https://api.proff.no/api",
        timeout: config.timeout || 30000,
        headers: {
          "Authorization": `Token ${config.proffApiKey}`,
          "Content-Type": "application/json",
        },
      });
    }

    // Request interceptor for logging
    this.client.interceptors.request.use((config) => {
      console.log(`[ForvaltV2] Request: ${config.method?.toUpperCase()} ${config.url}`);
      return config;
    });

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        if (error.response?.status === 429) {
          // Rate limited - wait and retry
          const retryAfter = parseInt(error.response.headers["retry-after"] || "5");
          await new Promise(resolve => setTimeout(resolve, retryAfter * 1000));
          return this.client.request(error.config!);
        }
        throw error;
      }
    );
  }

  /**
   * Get cached data or fetch from API
   */
  private async getCached<T>(key: string, fetcher: () => Promise<T>): Promise<T> {
    const cached = this.cache.get(key);
    if (cached && cached.expiry > Date.now()) {
      return cached.data as T;
    }

    const data = await fetcher();
    this.cache.set(key, { data, expiry: Date.now() + this.cacheTimeout });
    return data;
  }

  /**
   * Retry wrapper with exponential backoff
   */
  private async withRetry<T>(operation: () => Promise<T>, attempts: number = this.retryAttempts): Promise<T> {
    let lastError: Error | undefined;

    for (let i = 0; i < attempts; i++) {
      try {
        await this.rateLimiter.acquire();
        return await operation();
      } catch (error) {
        lastError = error as Error;
        
        if (error instanceof AxiosError) {
          // Don't retry on client errors (except rate limiting)
          if (error.response?.status && error.response.status >= 400 && error.response.status < 500 && error.response.status !== 429) {
            throw error;
          }
        }

        // Exponential backoff
        const delay = Math.pow(2, i) * 1000;
        console.log(`[ForvaltV2] Retry ${i + 1}/${attempts} after ${delay}ms`);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }

    throw lastError;
  }

  /**
   * Validate Norwegian organization number (9 digits with checksum)
   */
  validateOrgNumber(orgNumber: string): boolean {
    const cleaned = orgNumber.replace(/\s/g, "");
    if (!/^\d{9}$/.test(cleaned)) return false;

    const weights = [3, 2, 7, 6, 5, 4, 3, 2];
    const digits = cleaned.split("").map(Number);
    
    let sum = 0;
    for (let i = 0; i < 8; i++) {
      sum += digits[i] * weights[i];
    }
    
    const remainder = sum % 11;
    const checkDigit = remainder === 0 ? 0 : 11 - remainder;
    
    return checkDigit === digits[8] && checkDigit !== 10;
  }

  /**
   * Get company profile from Brønnøysund/Forvalt
   */
  async getCompany(orgNumber: string): Promise<CompanyProfile> {
    if (!this.validateOrgNumber(orgNumber)) {
      throw new Error(`Invalid organization number: ${orgNumber}`);
    }

    return this.getCached(`company:${orgNumber}`, async () => {
      try {
        const response = await this.withRetry(() =>
          this.client.get(`/companies/${orgNumber}`)
        );
        return this.mapCompanyResponse(response.data);
      } catch (error) {
        // Fallback to Proff API
        if (this.proffClient) {
          console.log("[ForvaltV2] Falling back to Proff API");
          const response = await this.proffClient.get(`/companies/register/NO/${orgNumber}`);
          return this.mapProffCompanyResponse(response.data);
        }
        
        // Fallback to Brønnøysund open API
        console.log("[ForvaltV2] Falling back to Brønnøysund API");
        const brreg = await axios.get(`https://data.brreg.no/enhetsregisteret/api/enheter/${orgNumber}`);
        return this.mapBrregResponse(brreg.data);
      }
    });
  }

  /**
   * Get credit rating and risk assessment
   */
  async getCredit(orgNumber: string): Promise<CreditRating> {
    if (!this.validateOrgNumber(orgNumber)) {
      throw new Error(`Invalid organization number: ${orgNumber}`);
    }

    return this.getCached(`credit:${orgNumber}`, async () => {
      try {
        const response = await this.withRetry(() =>
          this.client.get(`/credit/${orgNumber}`)
        );
        return this.mapCreditResponse(response.data);
      } catch (error) {
        // Return estimated credit based on available data
        console.log("[ForvaltV2] Credit API unavailable, using estimation");
        return this.estimateCreditRating(orgNumber);
      }
    });
  }

  /**
   * Get financial statements
   */
  async getFinancials(orgNumber: string, years: number = 3): Promise<FinancialStatement[]> {
    if (!this.validateOrgNumber(orgNumber)) {
      throw new Error(`Invalid organization number: ${orgNumber}`);
    }

    return this.getCached(`financials:${orgNumber}:${years}`, async () => {
      try {
        const response = await this.withRetry(() =>
          this.client.get(`/financials/${orgNumber}`, { params: { years } })
        );
        return response.data.statements.map(this.mapFinancialResponse);
      } catch (error) {
        console.log("[ForvaltV2] Financials API unavailable");
        return [];
      }
    });
  }

  /**
   * Get ownership information
   */
  async getOwnership(orgNumber: string): Promise<OwnershipInfo> {
    if (!this.validateOrgNumber(orgNumber)) {
      throw new Error(`Invalid organization number: ${orgNumber}`);
    }

    return this.getCached(`ownership:${orgNumber}`, async () => {
      try {
        const response = await this.withRetry(() =>
          this.client.get(`/ownership/${orgNumber}`)
        );
        return this.mapOwnershipResponse(response.data);
      } catch (error) {
        console.log("[ForvaltV2] Ownership API unavailable");
        return { orgNumber, shareholders: [] };
      }
    });
  }

  /**
   * Get full company profile with all data
   */
  async getFullProfile(orgNumber: string): Promise<ForvaltFullProfile> {
    const [company, credit, financials, ownership] = await Promise.all([
      this.getCompany(orgNumber),
      this.getCredit(orgNumber),
      this.getFinancials(orgNumber),
      this.getOwnership(orgNumber),
    ]);

    return {
      company,
      credit,
      financials,
      ownership,
      lastUpdated: new Date().toISOString(),
    };
  }

  /**
   * Search companies by name or industry
   */
  async searchCompanies(query: string, limit: number = 20): Promise<CompanyProfile[]> {
    try {
      const response = await this.withRetry(() =>
        this.client.get("/companies/search", { params: { q: query, limit } })
      );
      return response.data.companies.map(this.mapCompanyResponse);
    } catch (error) {
      // Fallback to Brønnøysund search
      const brreg = await axios.get("https://data.brreg.no/enhetsregisteret/api/enheter", {
        params: { navn: query, size: limit },
      });
      return brreg.data._embedded?.enheter?.map(this.mapBrregResponse) || [];
    }
  }

  /**
   * Save snapshot to database for audit trail
   */
  async saveSnapshot(companyId: number, profile: ForvaltFullProfile): Promise<number> {
    return db.createForvaltSnapshot({
      companyId,
      rawJson: profile,
      rating: profile.credit.rating,
      creditScore: profile.credit.creditScore,
      riskClass: profile.credit.riskClass,
    });
  }

  // ============================================================================
  // RESPONSE MAPPERS
  // ============================================================================

  private mapCompanyResponse(data: Record<string, unknown>): CompanyProfile {
    return {
      orgNumber: String(data.organizationNumber || data.orgNumber || ""),
      name: String(data.name || data.companyName || ""),
      organizationForm: String(data.organizationForm || data.orgForm || ""),
      registrationDate: String(data.registrationDate || data.founded || ""),
      address: {
        street: String((data.address as Record<string, unknown>)?.street || ""),
        postalCode: String((data.address as Record<string, unknown>)?.postalCode || ""),
        city: String((data.address as Record<string, unknown>)?.city || ""),
        country: "Norway",
      },
      industry: {
        code: String((data.industry as Record<string, unknown>)?.code || ""),
        description: String((data.industry as Record<string, unknown>)?.description || ""),
      },
      employees: Number(data.employees) || undefined,
      shareCapital: Number(data.shareCapital) || undefined,
      isActive: Boolean(data.isActive ?? true),
      isBankrupt: Boolean(data.isBankrupt ?? false),
      isUnderLiquidation: Boolean(data.isUnderLiquidation ?? false),
    };
  }

  private mapProffCompanyResponse(data: Record<string, unknown>): CompanyProfile {
    return {
      orgNumber: String(data.organisasjonsnummer || ""),
      name: String(data.navn || ""),
      organizationForm: String(data.organisasjonsform || ""),
      registrationDate: String(data.stiftelsesdato || ""),
      address: {
        street: String(((data.forretningsadresse as Record<string, unknown>)?.adresse as unknown[] | undefined)?.[0] || ""),
        postalCode: String((data.forretningsadresse as Record<string, unknown>)?.postnummer || ""),
        city: String((data.forretningsadresse as Record<string, unknown>)?.poststed || ""),
        country: "Norway",
      },
      industry: {
        code: String((data.naeringskode1 as Record<string, unknown>)?.kode || ""),
        description: String((data.naeringskode1 as Record<string, unknown>)?.beskrivelse || ""),
      },
      employees: Number(data.antallAnsatte) || undefined,
      isActive: !data.konkurs && !data.underAvvikling,
      isBankrupt: Boolean(data.konkurs),
      isUnderLiquidation: Boolean(data.underAvvikling),
    };
  }

  private mapBrregResponse(data: Record<string, unknown>): CompanyProfile {
    const forretningsadresse = data.forretningsadresse as Record<string, unknown> | undefined;
    const naeringskode1 = data.naeringskode1 as Record<string, unknown> | undefined;
    const adresseArray = forretningsadresse?.adresse as unknown[] | undefined;
    
    return {
      orgNumber: String(data.organisasjonsnummer || ""),
      name: String(data.navn || ""),
      organizationForm: String((data.organisasjonsform as Record<string, unknown>)?.kode || ""),
      registrationDate: String(data.stiftelsesdato || data.registreringsdatoEnhetsregisteret || ""),
      address: {
        street: Array.isArray(adresseArray) && adresseArray.length > 0 ? String(adresseArray[0]) : "",
        postalCode: String(forretningsadresse?.postnummer || ""),
        city: String(forretningsadresse?.poststed || ""),
        country: "Norway",
      },
      industry: {
        code: String(naeringskode1?.kode || ""),
        description: String(naeringskode1?.beskrivelse || ""),
      },
      employees: Number(data.antallAnsatte) || undefined,
      isActive: !data.konkurs && !data.underAvvikling,
      isBankrupt: Boolean(data.konkurs),
      isUnderLiquidation: Boolean(data.underAvvikling),
    };
  }

  private mapCreditResponse(data: Record<string, unknown>): CreditRating {
    return {
      orgNumber: String(data.orgNumber || ""),
      rating: String(data.rating || "C"),
      ratingDescription: String(data.ratingDescription || ""),
      creditScore: Number(data.creditScore) || 50,
      creditLimit: Number(data.creditLimit) || 0,
      riskClass: this.mapRiskClass(data.riskClass),
      paymentRemarks: Number(data.paymentRemarks) || 0,
      paymentIndex: Number(data.paymentIndex) || undefined,
      lastUpdated: String(data.lastUpdated || new Date().toISOString()),
      factors: Array.isArray(data.factors) ? data.factors.map((f: Record<string, unknown>) => ({
        factor: String(f.factor || ""),
        impact: f.impact as "POSITIVE" | "NEGATIVE" | "NEUTRAL" || "NEUTRAL",
        description: String(f.description || ""),
      })) : [],
    };
  }

  private mapRiskClass(value: unknown): "LOW" | "MEDIUM" | "HIGH" | "VERY_HIGH" {
    const str = String(value).toUpperCase();
    if (str === "LOW" || str === "MEDIUM" || str === "HIGH" || str === "VERY_HIGH") {
      return str;
    }
    return "MEDIUM";
  }

  private mapFinancialResponse(data: Record<string, unknown>): FinancialStatement {
    return {
      orgNumber: String(data.orgNumber || ""),
      year: Number(data.year) || new Date().getFullYear(),
      revenue: Number(data.revenue) || 0,
      operatingProfit: Number(data.operatingProfit) || 0,
      netProfit: Number(data.netProfit) || 0,
      totalAssets: Number(data.totalAssets) || 0,
      equity: Number(data.equity) || 0,
      debt: Number(data.debt) || 0,
      employees: Number(data.employees) || 0,
      currency: "NOK",
      ratios: {
        profitMargin: Number((data.ratios as Record<string, unknown>)?.profitMargin) || 0,
        equityRatio: Number((data.ratios as Record<string, unknown>)?.equityRatio) || 0,
        currentRatio: Number((data.ratios as Record<string, unknown>)?.currentRatio) || 0,
        debtToEquity: Number((data.ratios as Record<string, unknown>)?.debtToEquity) || 0,
        returnOnEquity: Number((data.ratios as Record<string, unknown>)?.returnOnEquity) || 0,
      },
    };
  }

  private mapOwnershipResponse(data: Record<string, unknown>): OwnershipInfo {
    return {
      orgNumber: String(data.orgNumber || ""),
      shareholders: Array.isArray(data.shareholders) ? data.shareholders.map((s: Record<string, unknown>) => ({
        name: String(s.name || ""),
        orgNumber: s.orgNumber ? String(s.orgNumber) : undefined,
        shares: Number(s.shares) || 0,
        percentage: Number(s.percentage) || 0,
        shareClass: s.shareClass ? String(s.shareClass) : undefined,
      })) : [],
      ultimateOwner: data.ultimateOwner ? {
        name: String((data.ultimateOwner as Record<string, unknown>).name || ""),
        country: String((data.ultimateOwner as Record<string, unknown>).country || "Norway"),
        ownershipPercentage: Number((data.ultimateOwner as Record<string, unknown>).percentage) || 0,
      } : undefined,
    };
  }

  /**
   * Estimate credit rating when API is unavailable
   */
  private async estimateCreditRating(orgNumber: string): Promise<CreditRating> {
    // Get basic company info from Brønnøysund
    try {
      const company = await this.getCompany(orgNumber);
      
      let score = 50; // Base score
      const factors: CreditFactor[] = [];

      // Adjust based on company status
      if (company.isBankrupt) {
        score = 0;
        factors.push({ factor: "Konkurs", impact: "NEGATIVE", description: "Selskapet er konkurs" });
      } else if (company.isUnderLiquidation) {
        score = 10;
        factors.push({ factor: "Avvikling", impact: "NEGATIVE", description: "Selskapet er under avvikling" });
      } else if (company.isActive) {
        score += 10;
        factors.push({ factor: "Aktiv", impact: "POSITIVE", description: "Selskapet er aktivt" });
      }

      // Adjust based on organization form
      if (company.organizationForm === "AS" || company.organizationForm === "ASA") {
        score += 10;
        factors.push({ factor: "Aksjeselskap", impact: "POSITIVE", description: "Begrenset ansvar" });
      }

      // Adjust based on employees
      if (company.employees && company.employees > 10) {
        score += 10;
        factors.push({ factor: "Ansatte", impact: "POSITIVE", description: `${company.employees} ansatte` });
      }

      // Determine rating and risk class
      let rating: string;
      let riskClass: "LOW" | "MEDIUM" | "HIGH" | "VERY_HIGH";

      if (score >= 80) { rating = "A"; riskClass = "LOW"; }
      else if (score >= 60) { rating = "B"; riskClass = "LOW"; }
      else if (score >= 40) { rating = "C"; riskClass = "MEDIUM"; }
      else if (score >= 20) { rating = "D"; riskClass = "HIGH"; }
      else { rating = "E"; riskClass = "VERY_HIGH"; }

      return {
        orgNumber,
        rating,
        ratingDescription: `Estimert rating basert på offentlig informasjon`,
        creditScore: score,
        creditLimit: score * 10000,
        riskClass,
        paymentRemarks: 0,
        lastUpdated: new Date().toISOString(),
        factors,
      };
    } catch (error) {
      // Return default low-confidence rating
      return {
        orgNumber,
        rating: "C",
        ratingDescription: "Ingen data tilgjengelig",
        creditScore: 50,
        creditLimit: 0,
        riskClass: "MEDIUM",
        paymentRemarks: 0,
        lastUpdated: new Date().toISOString(),
        factors: [{ factor: "Manglende data", impact: "NEUTRAL", description: "Kunne ikke hente kredittinformasjon" }],
      };
    }
  }
}

// Factory function
export function createForvaltClientV2(config?: Partial<ForvaltConfig>): ForvaltClientV2 {
  const apiKey = config?.apiKey || process.env.FORVALT_API_KEY || "";
  const proffApiKey = config?.proffApiKey || process.env.PROFF_API_KEY;

  return new ForvaltClientV2({
    apiKey,
    proffApiKey,
    baseUrl: config?.baseUrl || process.env.FORVALT_BASE_URL,
    proffBaseUrl: config?.proffBaseUrl || process.env.PROFF_BASE_URL,
    timeout: config?.timeout || 30000,
    retryAttempts: config?.retryAttempts || 3,
  });
}
