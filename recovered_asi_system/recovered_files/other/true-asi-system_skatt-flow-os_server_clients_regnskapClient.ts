import axios, { AxiosInstance, AxiosError } from "axios";
import { nanoid } from "nanoid";
import type {
  ChartOfAccountsEntry,
  VatCode,
  VoucherPayload,
  VoucherResponse,
  OpenInvoice,
  LedgerPeriodData,
  LedgerEntryData,
} from "@shared/types";

// ============================================================================
// GENERIC REGNSKAP (ACCOUNTING SYSTEM) API CLIENT
// Supports Tripletex, PowerOffice, Fiken, Visma eAccounting
// ============================================================================

export class RegnskapApiError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public correlationId?: string
  ) {
    super(message);
    this.name = "RegnskapApiError";
  }
}

interface RegnskapClientConfig {
  baseUrl: string;
  apiKey: string;
  timeout?: number;
}

export class RegnskapClient {
  private client: AxiosInstance;
  private apiKey: string;

  constructor(config: RegnskapClientConfig) {
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
   * Get chart of accounts for a company
   */
  async getChartOfAccounts(companyId: string): Promise<ChartOfAccountsEntry[]> {
    const correlationId = this.getCorrelationId();

    try {
      const response = await this.client.get(`/companies/${companyId}/accounts`, {
        headers: this.getAuthHeaders(correlationId),
      });

      const data = response.data;
      const accounts = Array.isArray(data) ? data : data.accounts || data.items || [];

      return accounts.map((acc: Record<string, unknown>) => ({
        accountNumber: String(acc.accountNumber || acc.number || acc.konto || ""),
        name: String(acc.name || acc.navn || acc.description || ""),
        type: this.mapAccountType(acc.type || acc.kontotype),
        vatCode: acc.vatCode ? String(acc.vatCode) : undefined,
        isActive: acc.isActive !== false && acc.active !== false,
      }));
    } catch (error) {
      if (error instanceof AxiosError) {
        throw new RegnskapApiError(
          `Failed to fetch chart of accounts: ${error.message}`,
          error.response?.status,
          correlationId
        );
      }
      throw error;
    }
  }

  private mapAccountType(type: unknown): "ASSET" | "LIABILITY" | "EQUITY" | "REVENUE" | "EXPENSE" {
    const typeStr = String(type).toUpperCase();
    if (typeStr.includes("ASSET") || typeStr.includes("EIENDEL")) return "ASSET";
    if (typeStr.includes("LIABILITY") || typeStr.includes("GJELD")) return "LIABILITY";
    if (typeStr.includes("EQUITY") || typeStr.includes("EGENKAPITAL")) return "EQUITY";
    if (typeStr.includes("REVENUE") || typeStr.includes("INNTEKT")) return "REVENUE";
    if (typeStr.includes("EXPENSE") || typeStr.includes("KOSTNAD")) return "EXPENSE";
    return "EXPENSE"; // Default
  }

  /**
   * Get VAT codes for a company
   */
  async getVatCodes(companyId: string): Promise<VatCode[]> {
    const correlationId = this.getCorrelationId();

    try {
      const response = await this.client.get(`/companies/${companyId}/vat-codes`, {
        headers: this.getAuthHeaders(correlationId),
      });

      const data = response.data;
      const codes = Array.isArray(data) ? data : data.vatCodes || data.items || [];

      return codes.map((code: Record<string, unknown>) => ({
        code: String(code.code || code.kode || ""),
        description: String(code.description || code.beskrivelse || code.name || ""),
        rate: Number(code.rate || code.sats || 0),
        isActive: code.isActive !== false && code.active !== false,
      }));
    } catch (error) {
      if (error instanceof AxiosError) {
        throw new RegnskapApiError(
          `Failed to fetch VAT codes: ${error.message}`,
          error.response?.status,
          correlationId
        );
      }
      throw error;
    }
  }

  /**
   * Create a voucher (journal entry) in the accounting system
   */
  async createVoucher(companyId: string, payload: VoucherPayload): Promise<VoucherResponse> {
    const correlationId = this.getCorrelationId();

    try {
      const response = await this.client.post(
        `/companies/${companyId}/vouchers`,
        {
          date: payload.date,
          description: payload.description,
          lines: payload.lines.map((line) => ({
            accountNumber: line.accountNumber,
            debit: line.debit || 0,
            credit: line.credit || 0,
            vatCode: line.vatCode,
            description: line.description,
          })),
          attachmentUrl: payload.attachmentUrl,
        },
        { headers: this.getAuthHeaders(correlationId) }
      );

      const data = response.data;
      return {
        voucherId: String(data.id || data.voucherId || data.bilagsId || ""),
        voucherNumber: String(data.voucherNumber || data.bilagsnummer || data.number || ""),
        createdAt: data.createdAt || data.opprettet || new Date().toISOString(),
      };
    } catch (error) {
      if (error instanceof AxiosError) {
        throw new RegnskapApiError(
          `Failed to create voucher: ${error.message}`,
          error.response?.status,
          correlationId
        );
      }
      throw error;
    }
  }

  /**
   * List open (unpaid) invoices for a company
   */
  async listOpenInvoices(companyId: string): Promise<OpenInvoice[]> {
    const correlationId = this.getCorrelationId();

    try {
      const response = await this.client.get(`/companies/${companyId}/invoices/open`, {
        headers: this.getAuthHeaders(correlationId),
      });

      const data = response.data;
      const invoices = Array.isArray(data) ? data : data.invoices || data.items || [];

      return invoices.map((inv: Record<string, unknown>) => ({
        invoiceId: String(inv.id || inv.invoiceId || ""),
        invoiceNumber: String(inv.invoiceNumber || inv.fakturanummer || inv.number || ""),
        customerName: String(inv.customerName || inv.kundenavn || inv.customer || ""),
        amount: Number(inv.amount || inv.belop || 0),
        currency: String(inv.currency || inv.valuta || "NOK"),
        dueDate: String(inv.dueDate || inv.forfallsdato || ""),
        isOverdue: Boolean(inv.isOverdue || inv.forfalt || false),
      }));
    } catch (error) {
      if (error instanceof AxiosError) {
        throw new RegnskapApiError(
          `Failed to fetch open invoices: ${error.message}`,
          error.response?.status,
          correlationId
        );
      }
      throw error;
    }
  }

  /**
   * Get ledger entries for a period
   */
  async getLedger(companyId: string, periodStart: string, periodEnd: string): Promise<LedgerPeriodData> {
    const correlationId = this.getCorrelationId();

    try {
      const response = await this.client.get(`/companies/${companyId}/ledger`, {
        headers: this.getAuthHeaders(correlationId),
        params: { from: periodStart, to: periodEnd },
      });

      const data = response.data;
      const entries = Array.isArray(data) ? data : data.entries || data.items || [];

      const mappedEntries: LedgerEntryData[] = entries.map((entry: Record<string, unknown>) => ({
        id: String(entry.id || ""),
        date: String(entry.date || entry.dato || ""),
        description: String(entry.description || entry.beskrivelse || entry.text || ""),
        debitAccount: String(entry.debitAccount || entry.debetkonto || ""),
        creditAccount: String(entry.creditAccount || entry.kreditkonto || ""),
        amount: Number(entry.amount || entry.belop || 0),
        vatCode: entry.vatCode ? String(entry.vatCode) : undefined,
        voucherNumber: entry.voucherNumber ? String(entry.voucherNumber) : undefined,
      }));

      const totalDebit = mappedEntries.reduce((sum, e) => sum + e.amount, 0);
      const totalCredit = totalDebit; // In double-entry bookkeeping, these should be equal

      return {
        entries: mappedEntries,
        periodStart,
        periodEnd,
        totalDebit,
        totalCredit,
      };
    } catch (error) {
      if (error instanceof AxiosError) {
        throw new RegnskapApiError(
          `Failed to fetch ledger: ${error.message}`,
          error.response?.status,
          correlationId
        );
      }
      throw error;
    }
  }

  /**
   * Export SAF-T file for a period
   */
  async exportSaft(
    companyId: string,
    periodStart: string,
    periodEnd: string
  ): Promise<{ fileUrl: string; format: string }> {
    const correlationId = this.getCorrelationId();

    try {
      const response = await this.client.post(
        `/companies/${companyId}/saft/export`,
        { periodStart, periodEnd, format: "XML" },
        { headers: this.getAuthHeaders(correlationId) }
      );

      const data = response.data;
      return {
        fileUrl: String(data.fileUrl || data.url || data.downloadUrl || ""),
        format: "XML",
      };
    } catch (error) {
      if (error instanceof AxiosError) {
        throw new RegnskapApiError(
          `Failed to export SAF-T: ${error.message}`,
          error.response?.status,
          correlationId
        );
      }
      throw error;
    }
  }

  /**
   * Get account balance for a specific account
   */
  async getAccountBalance(
    companyId: string,
    accountNumber: string,
    asOfDate?: string
  ): Promise<{ balance: number; currency: string }> {
    const correlationId = this.getCorrelationId();

    try {
      const response = await this.client.get(
        `/companies/${companyId}/accounts/${accountNumber}/balance`,
        {
          headers: this.getAuthHeaders(correlationId),
          params: { asOfDate: asOfDate || new Date().toISOString().split("T")[0] },
        }
      );

      const data = response.data;
      return {
        balance: Number(data.balance || data.saldo || 0),
        currency: String(data.currency || data.valuta || "NOK"),
      };
    } catch (error) {
      if (error instanceof AxiosError) {
        throw new RegnskapApiError(
          `Failed to fetch account balance: ${error.message}`,
          error.response?.status,
          correlationId
        );
      }
      throw error;
    }
  }
}

// Factory function to create client from environment variables
export function createRegnskapClient(): RegnskapClient {
  const baseUrl = process.env.REGNSKAP_BASE_URL || "https://api.tripletex.io/v2";
  const apiKey = process.env.REGNSKAP_API_KEY || "";

  if (!apiKey) {
    console.warn("[RegnskapClient] REGNSKAP_API_KEY not set - API calls will fail");
  }

  return new RegnskapClient({ baseUrl, apiKey });
}
