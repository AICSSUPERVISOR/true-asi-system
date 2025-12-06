/**
 * Unified type exports
 * Import shared types from this single entry point.
 */

export type * from "../drizzle/schema";
export * from "./_core/errors";

// ============================================================================
// SHARED TYPES FOR SKATT-FLOW OS
// ============================================================================

// Accounting roles for RBAC
export type AccountingRole = "OWNER" | "ADMIN" | "ACCOUNTANT" | "VIEWER";

// External accounting systems
export type ExternalRegnskapSystem = "TRIPLETEX" | "POWEROFFICE" | "FIKEN" | "VISMA_EACCOUNTING" | "OTHER";

// Document source types
export type DocumentSourceType = "INVOICE_SUPPLIER" | "INVOICE_CUSTOMER" | "RECEIPT" | "CONTRACT" | "OTHER";

// Document processing status
export type DocumentStatus = "NEW" | "PROCESSED" | "POSTED" | "REJECTED";

// Filing types
export type FilingType = "MVA_MELDING" | "A_MELDING_SUMMARY" | "SAF_T" | "ARSREGNSKAP" | "OTHER";

// Filing status
export type FilingStatus = "DRAFT" | "READY_FOR_REVIEW" | "SUBMITTED" | "ERROR";

// Document template categories
export type TemplateCategory = "CONTRACT" | "HR" | "LEGAL" | "FINANCIAL" | "GOVERNANCE" | "OTHER";

// Template sources
export type TemplateSource = "BUSINESS_IN_A_BOX" | "CUSTOM";

// ============================================================================
// API CLIENT TYPES
// ============================================================================

// Forvalt/Proff API types
export interface ForvaltCompanyInfo {
  orgNumber: string;
  name: string;
  address?: string;
  city?: string;
  postalCode?: string;
  industryCode?: string;
  industryDescription?: string;
  registrationDate?: string;
  employees?: number;
  status?: string;
}

export interface ForvaltCreditInfo {
  orgNumber: string;
  creditScore?: number;
  rating?: string;
  riskClass?: string;
  paymentRemarks?: number;
  creditLimit?: number;
  lastUpdated?: string;
}

export interface ForvaltFinancials {
  orgNumber: string;
  year: number;
  revenue?: number;
  operatingResult?: number;
  netResult?: number;
  equity?: number;
  totalDebt?: number;
  totalAssets?: number;
}

// Regnskap API types
export interface ChartOfAccountsEntry {
  accountNumber: string;
  name: string;
  type: "ASSET" | "LIABILITY" | "EQUITY" | "REVENUE" | "EXPENSE";
  vatCode?: string;
  isActive: boolean;
}

export interface VatCode {
  code: string;
  description: string;
  rate: number;
  isActive: boolean;
}

export interface VoucherPayload {
  date: string;
  description: string;
  lines: VoucherLine[];
  attachmentUrl?: string;
}

export interface VoucherLine {
  accountNumber: string;
  debit?: number;
  credit?: number;
  vatCode?: string;
  description?: string;
}

export interface VoucherResponse {
  voucherId: string;
  voucherNumber: string;
  createdAt: string;
}

export interface OpenInvoice {
  invoiceId: string;
  invoiceNumber: string;
  customerName: string;
  amount: number;
  currency: string;
  dueDate: string;
  isOverdue: boolean;
}

export interface LedgerPeriodData {
  entries: LedgerEntryData[];
  periodStart: string;
  periodEnd: string;
  totalDebit: number;
  totalCredit: number;
}

export interface LedgerEntryData {
  id: string;
  date: string;
  description: string;
  debitAccount: string;
  creditAccount: string;
  amount: number;
  vatCode?: string;
  voucherNumber?: string;
}

// Altinn API types
export interface AltinnTokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}

export interface AltinnDraftResponse {
  draftId: string;
  status: string;
  createdAt: string;
}

export interface AltinnSubmitResponse {
  reference: string;
  status: string;
  submittedAt: string;
}

// AIML API types
export interface AIMLRequest {
  model?: string;
  systemPrompt: string;
  userPrompt: string;
  inputs?: Record<string, unknown>;
  temperature?: number;
  maxTokens?: number;
}

export interface AIMLResponse {
  outputText: string;
  raw?: unknown;
  model?: string;
  usage?: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
}

// Document extraction types
export interface ExtractedInvoiceFields {
  supplierName?: string;
  supplierOrgNumber?: string;
  invoiceNumber?: string;
  invoiceDate?: string;
  dueDate?: string;
  totalAmount?: number;
  vatAmount?: number;
  currency?: string;
  kid?: string;
  bankAccount?: string;
  lines?: ExtractedInvoiceLine[];
  confidence: number;
}

export interface ExtractedInvoiceLine {
  description: string;
  quantity?: number;
  unitPrice?: number;
  amount: number;
  vatCode?: string;
}

// MVA-melding types
export interface MVAMeldingPayload {
  orgNumber: string;
  period: {
    year: number;
    term: number;
    type: "BIMONTHLY" | "QUARTERLY" | "YEARLY";
  };
  totals: {
    totalSales: number;
    totalPurchases: number;
    outputVat: number;
    inputVat: number;
    vatPayable: number;
  };
  details: MVAMeldingLine[];
}

export interface MVAMeldingLine {
  code: string;
  description: string;
  basis: number;
  vatAmount: number;
}

// SAF-T types
export interface SAFTExportRequest {
  companyId: string;
  periodStart: string;
  periodEnd: string;
  format: "XML" | "JSON";
}

export interface SAFTValidationResult {
  isValid: boolean;
  errors: SAFTValidationError[];
  warnings: SAFTValidationWarning[];
}

export interface SAFTValidationError {
  code: string;
  message: string;
  field?: string;
}

export interface SAFTValidationWarning {
  code: string;
  message: string;
  field?: string;
}

// ============================================================================
// UI TYPES
// ============================================================================

export interface DashboardStats {
  unpostedDocuments: number;
  pendingFilings: number;
  highRiskCustomers: number;
  upcomingDeadlines: DeadlineInfo[];
}

export interface DeadlineInfo {
  type: FilingType;
  dueDate: string;
  status: FilingStatus;
  companyName: string;
  companyId: number;
}

export interface CompanyWithStats extends ForvaltCompanyInfo {
  id: number;
  documentsCount: number;
  filingsCount: number;
  lastActivity?: string;
}

// Chat types
export interface ChatContext {
  companyId?: number;
  documentId?: number;
  filingId?: number;
}

export interface ChatAction {
  type: "SUGGEST_POSTING" | "GENERATE_MVA" | "RISK_REPORT" | "GENERATE_DOCUMENT";
  label: string;
  params?: Record<string, unknown>;
}
