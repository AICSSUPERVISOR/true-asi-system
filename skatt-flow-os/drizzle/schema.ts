import { int, mysqlEnum, mysqlTable, text, timestamp, varchar, json, bigint, boolean } from "drizzle-orm/mysql-core";

// ============================================================================
// USER TABLE - Extended with RBAC roles for accounting platform
// ============================================================================
export const userRoleEnum = mysqlEnum("userRole", ["OWNER", "ADMIN", "ACCOUNTANT", "VIEWER"]);

export const users = mysqlTable("users", {
  id: int("id").autoincrement().primaryKey(),
  openId: varchar("openId", { length: 64 }).notNull().unique(),
  name: text("name"),
  email: varchar("email", { length: 320 }),
  loginMethod: varchar("loginMethod", { length: 64 }),
  role: mysqlEnum("role", ["user", "admin"]).default("user").notNull(),
  // Extended role for accounting platform RBAC
  accountingRole: mysqlEnum("accountingRole", ["OWNER", "ADMIN", "ACCOUNTANT", "VIEWER"]).default("VIEWER").notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
  lastSignedIn: timestamp("lastSignedIn").defaultNow().notNull(),
});

export type User = typeof users.$inferSelect;
export type InsertUser = typeof users.$inferInsert;

// ============================================================================
// COMPANY TABLE - Core entity for Norwegian businesses
// ============================================================================
export const externalRegnskapSystemEnum = mysqlEnum("externalRegnskapSystem", [
  "TRIPLETEX", "POWEROFFICE", "FIKEN", "VISMA_EACCOUNTING", "OTHER"
]);

export const companies = mysqlTable("companies", {
  id: int("id").autoincrement().primaryKey(),
  orgNumber: varchar("orgNumber", { length: 20 }).notNull().unique(),
  name: varchar("name", { length: 255 }).notNull(),
  country: varchar("country", { length: 2 }).default("NO").notNull(),
  address: text("address"),
  city: varchar("city", { length: 100 }),
  postalCode: varchar("postalCode", { length: 10 }),
  industryCode: varchar("industryCode", { length: 20 }), // NACE code
  // Forvalt data fields
  forvaltRating: varchar("forvaltRating", { length: 10 }),
  forvaltCreditScore: int("forvaltCreditScore"),
  forvaltRiskClass: varchar("forvaltRiskClass", { length: 20 }),
  // External accounting system connection
  externalRegnskapSystem: mysqlEnum("externalRegnskapSystem", [
    "TRIPLETEX", "POWEROFFICE", "FIKEN", "VISMA_EACCOUNTING", "OTHER"
  ]),
  externalRegnskapCompanyId: varchar("externalRegnskapCompanyId", { length: 100 }),
  // Settings
  autoPostEnabled: boolean("autoPostEnabled").default(false).notNull(),
  createdById: int("createdById"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

export type Company = typeof companies.$inferSelect;
export type InsertCompany = typeof companies.$inferInsert;

// ============================================================================
// BANK ACCOUNT TABLE
// ============================================================================
export const bankAccounts = mysqlTable("bankAccounts", {
  id: int("id").autoincrement().primaryKey(),
  companyId: int("companyId").notNull(),
  bankName: varchar("bankName", { length: 100 }).notNull(),
  ibanOrAccountNo: varchar("ibanOrAccountNo", { length: 50 }).notNull(),
  currency: varchar("currency", { length: 3 }).default("NOK").notNull(),
  openBankingProvider: varchar("openBankingProvider", { length: 100 }),
  isActive: boolean("isActive").default(true).notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

export type BankAccount = typeof bankAccounts.$inferSelect;
export type InsertBankAccount = typeof bankAccounts.$inferInsert;

// ============================================================================
// ACCOUNTING DOCUMENT TABLE - Invoices, receipts, contracts
// ============================================================================
export const accountingDocuments = mysqlTable("accountingDocuments", {
  id: int("id").autoincrement().primaryKey(),
  companyId: int("companyId").notNull(),
  sourceType: mysqlEnum("sourceType", [
    "INVOICE_SUPPLIER", "INVOICE_CUSTOMER", "RECEIPT", "CONTRACT", "OTHER"
  ]).notNull(),
  originalFileUrl: text("originalFileUrl"),
  originalFileName: varchar("originalFileName", { length: 255 }),
  // AI extraction results
  parsedJson: json("parsedJson"),
  // Workflow status
  status: mysqlEnum("status", ["NEW", "PROCESSED", "POSTED", "REJECTED"]).default("NEW").notNull(),
  // AI suggestions
  suggestedAccount: varchar("suggestedAccount", { length: 20 }),
  suggestedVatCode: varchar("suggestedVatCode", { length: 10 }),
  suggestedDescription: text("suggestedDescription"),
  suggestedAmount: bigint("suggestedAmount", { mode: "number" }), // Amount in øre (cents)
  // Posted voucher reference
  postedVoucherId: varchar("postedVoucherId", { length: 100 }),
  postedAt: timestamp("postedAt"),
  postedById: int("postedById"),
  // Rejection info
  rejectedReason: text("rejectedReason"),
  rejectedAt: timestamp("rejectedAt"),
  rejectedById: int("rejectedById"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

export type AccountingDocument = typeof accountingDocuments.$inferSelect;
export type InsertAccountingDocument = typeof accountingDocuments.$inferInsert;

// ============================================================================
// LEDGER ENTRY TABLE - Journal entries from external systems
// ============================================================================
export const ledgerEntries = mysqlTable("ledgerEntries", {
  id: int("id").autoincrement().primaryKey(),
  companyId: int("companyId").notNull(),
  externalSystem: mysqlEnum("externalSystem", [
    "TRIPLETEX", "POWEROFFICE", "FIKEN", "VISMA_EACCOUNTING", "OTHER"
  ]),
  externalId: varchar("externalId", { length: 100 }),
  entryDate: timestamp("entryDate").notNull(),
  description: text("description"),
  debitAccount: varchar("debitAccount", { length: 20 }).notNull(),
  creditAccount: varchar("creditAccount", { length: 20 }).notNull(),
  amount: bigint("amount", { mode: "number" }).notNull(), // Amount in øre (cents)
  currency: varchar("currency", { length: 3 }).default("NOK").notNull(),
  vatCode: varchar("vatCode", { length: 10 }),
  voucherNumber: varchar("voucherNumber", { length: 50 }),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

export type LedgerEntry = typeof ledgerEntries.$inferSelect;
export type InsertLedgerEntry = typeof ledgerEntries.$inferInsert;

// ============================================================================
// FILING TABLE - MVA, SAF-T, A-melding submissions
// ============================================================================
export const filings = mysqlTable("filings", {
  id: int("id").autoincrement().primaryKey(),
  companyId: int("companyId").notNull(),
  filingType: mysqlEnum("filingType", [
    "MVA_MELDING", "A_MELDING_SUMMARY", "SAF_T", "ARSREGNSKAP", "OTHER"
  ]).notNull(),
  periodStart: timestamp("periodStart").notNull(),
  periodEnd: timestamp("periodEnd").notNull(),
  status: mysqlEnum("status", ["DRAFT", "READY_FOR_REVIEW", "SUBMITTED", "ERROR"]).default("DRAFT").notNull(),
  // Altinn integration
  altinnServiceCode: varchar("altinnServiceCode", { length: 50 }),
  altinnReference: varchar("altinnReference", { length: 100 }),
  altinnDraftId: varchar("altinnDraftId", { length: 100 }),
  // Filing payload
  payloadJson: json("payloadJson"),
  summaryJson: json("summaryJson"), // Human-readable summary
  // Error tracking
  errorMessage: text("errorMessage"),
  // Audit trail
  submittedAt: timestamp("submittedAt"),
  submittedById: int("submittedById"),
  createdById: int("createdById"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

export type Filing = typeof filings.$inferSelect;
export type InsertFiling = typeof filings.$inferInsert;

// ============================================================================
// FORVALT SNAPSHOT TABLE - Historical credit/rating data
// ============================================================================
export const forvaltSnapshots = mysqlTable("forvaltSnapshots", {
  id: int("id").autoincrement().primaryKey(),
  companyId: int("companyId").notNull(),
  rawJson: json("rawJson").notNull(),
  rating: varchar("rating", { length: 10 }),
  creditScore: int("creditScore"),
  riskClass: varchar("riskClass", { length: 20 }),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

export type ForvaltSnapshot = typeof forvaltSnapshots.$inferSelect;
export type InsertForvaltSnapshot = typeof forvaltSnapshots.$inferInsert;

// ============================================================================
// DOCUMENT TEMPLATE TABLE - Business-in-a-Box style templates
// ============================================================================
export const documentTemplates = mysqlTable("documentTemplates", {
  id: int("id").autoincrement().primaryKey(),
  name: varchar("name", { length: 255 }).notNull(),
  category: mysqlEnum("category", [
    "CONTRACT", "HR", "LEGAL", "FINANCIAL", "GOVERNANCE", "OTHER"
  ]).notNull(),
  source: mysqlEnum("source", ["BUSINESS_IN_A_BOX", "CUSTOM"]).default("CUSTOM").notNull(),
  language: varchar("language", { length: 5 }).default("no").notNull(),
  description: text("description"),
  bodyMarkdown: text("bodyMarkdown").notNull(),
  variablesJson: json("variablesJson"), // List of required variables
  isActive: boolean("isActive").default(true).notNull(),
  createdById: int("createdById"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

export type DocumentTemplate = typeof documentTemplates.$inferSelect;
export type InsertDocumentTemplate = typeof documentTemplates.$inferInsert;

// ============================================================================
// GENERATED DOCUMENT TABLE - Documents created from templates
// ============================================================================
export const generatedDocuments = mysqlTable("generatedDocuments", {
  id: int("id").autoincrement().primaryKey(),
  companyId: int("companyId").notNull(),
  templateId: int("templateId").notNull(),
  title: varchar("title", { length: 255 }).notNull(),
  filledVariablesJson: json("filledVariablesJson"),
  outputMarkdown: text("outputMarkdown").notNull(),
  outputFileUrl: text("outputFileUrl"), // PDF/DOCX export URL
  createdById: int("createdById"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

export type GeneratedDocument = typeof generatedDocuments.$inferSelect;
export type InsertGeneratedDocument = typeof generatedDocuments.$inferInsert;

// ============================================================================
// CHAT MESSAGE TABLE - AI conversation history
// ============================================================================
export const chatMessages = mysqlTable("chatMessages", {
  id: int("id").autoincrement().primaryKey(),
  userId: int("userId").notNull(),
  companyId: int("companyId"),
  sessionId: varchar("sessionId", { length: 64 }).notNull(),
  role: mysqlEnum("role", ["user", "assistant", "system"]).notNull(),
  content: text("content").notNull(),
  attachedDocumentId: int("attachedDocumentId"),
  attachedFilingId: int("attachedFilingId"),
  metadata: json("metadata"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

export type ChatMessage = typeof chatMessages.$inferSelect;
export type InsertChatMessage = typeof chatMessages.$inferInsert;

// ============================================================================
// API LOG TABLE - Audit trail for external API calls
// ============================================================================
export const apiLogs = mysqlTable("apiLogs", {
  id: int("id").autoincrement().primaryKey(),
  companyId: int("companyId"),
  userId: int("userId"),
  endpoint: varchar("endpoint", { length: 500 }).notNull(),
  method: varchar("method", { length: 10 }).notNull(),
  statusCode: int("statusCode"),
  correlationId: varchar("correlationId", { length: 64 }).notNull(),
  durationMs: int("durationMs"),
  errorMessage: text("errorMessage"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

export type ApiLog = typeof apiLogs.$inferSelect;
export type InsertApiLog = typeof apiLogs.$inferInsert;

// ============================================================================
// USER COMPANY ACCESS TABLE - Many-to-many relationship
// ============================================================================
export const userCompanyAccess = mysqlTable("userCompanyAccess", {
  id: int("id").autoincrement().primaryKey(),
  userId: int("userId").notNull(),
  companyId: int("companyId").notNull(),
  accessRole: mysqlEnum("accessRole", ["OWNER", "ADMIN", "ACCOUNTANT", "VIEWER"]).default("VIEWER").notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

export type UserCompanyAccess = typeof userCompanyAccess.$inferSelect;
export type InsertUserCompanyAccess = typeof userCompanyAccess.$inferInsert;
