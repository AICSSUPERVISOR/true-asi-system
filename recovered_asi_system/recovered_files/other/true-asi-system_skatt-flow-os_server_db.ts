import { eq, and, desc, gte, lte, sql, inArray } from "drizzle-orm";
import { drizzle } from "drizzle-orm/mysql2";
import {
  InsertUser,
  users,
  companies,
  InsertCompany,
  Company,
  bankAccounts,
  InsertBankAccount,
  BankAccount,
  accountingDocuments,
  InsertAccountingDocument,
  AccountingDocument,
  ledgerEntries,
  InsertLedgerEntry,
  LedgerEntry,
  filings,
  InsertFiling,
  Filing,
  forvaltSnapshots,
  InsertForvaltSnapshot,
  ForvaltSnapshot,
  documentTemplates,
  InsertDocumentTemplate,
  DocumentTemplate,
  generatedDocuments,
  InsertGeneratedDocument,
  GeneratedDocument,
  chatMessages,
  InsertChatMessage,
  ChatMessage,
  apiLogs,
  InsertApiLog,
  userCompanyAccess,
  InsertUserCompanyAccess,
  UserCompanyAccess,
} from "../drizzle/schema";
import { ENV } from "./_core/env";

let _db: ReturnType<typeof drizzle> | null = null;

export async function getDb() {
  if (!_db && process.env.DATABASE_URL) {
    try {
      _db = drizzle(process.env.DATABASE_URL);
    } catch (error) {
      console.warn("[Database] Failed to connect:", error);
      _db = null;
    }
  }
  return _db;
}

// ============================================================================
// USER QUERIES
// ============================================================================

export async function upsertUser(user: InsertUser): Promise<void> {
  if (!user.openId) {
    throw new Error("User openId is required for upsert");
  }

  const db = await getDb();
  if (!db) {
    console.warn("[Database] Cannot upsert user: database not available");
    return;
  }

  try {
    const values: InsertUser = { openId: user.openId };
    const updateSet: Record<string, unknown> = {};

    const textFields = ["name", "email", "loginMethod"] as const;
    type TextField = (typeof textFields)[number];

    const assignNullable = (field: TextField) => {
      const value = user[field];
      if (value === undefined) return;
      const normalized = value ?? null;
      values[field] = normalized;
      updateSet[field] = normalized;
    };

    textFields.forEach(assignNullable);

    if (user.lastSignedIn !== undefined) {
      values.lastSignedIn = user.lastSignedIn;
      updateSet.lastSignedIn = user.lastSignedIn;
    }
    if (user.role !== undefined) {
      values.role = user.role;
      updateSet.role = user.role;
    } else if (user.openId === ENV.ownerOpenId) {
      values.role = "admin";
      updateSet.role = "admin";
    }

    if (!values.lastSignedIn) {
      values.lastSignedIn = new Date();
    }

    if (Object.keys(updateSet).length === 0) {
      updateSet.lastSignedIn = new Date();
    }

    await db.insert(users).values(values).onDuplicateKeyUpdate({ set: updateSet });
  } catch (error) {
    console.error("[Database] Failed to upsert user:", error);
    throw error;
  }
}

export async function getUserByOpenId(openId: string) {
  const db = await getDb();
  if (!db) return undefined;
  const result = await db.select().from(users).where(eq(users.openId, openId)).limit(1);
  return result.length > 0 ? result[0] : undefined;
}

export async function getUserById(id: number) {
  const db = await getDb();
  if (!db) return undefined;
  const result = await db.select().from(users).where(eq(users.id, id)).limit(1);
  return result.length > 0 ? result[0] : undefined;
}

// ============================================================================
// COMPANY QUERIES
// ============================================================================

export async function createCompany(data: InsertCompany): Promise<number> {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  const result = await db.insert(companies).values(data);
  return Number(result[0].insertId);
}

export async function getCompanyById(id: number): Promise<Company | undefined> {
  const db = await getDb();
  if (!db) return undefined;
  const result = await db.select().from(companies).where(eq(companies.id, id)).limit(1);
  return result[0];
}

export async function getCompanyByOrgNumber(orgNumber: string): Promise<Company | undefined> {
  const db = await getDb();
  if (!db) return undefined;
  const result = await db.select().from(companies).where(eq(companies.orgNumber, orgNumber)).limit(1);
  return result[0];
}

export async function listCompanies(userId?: number): Promise<Company[]> {
  const db = await getDb();
  if (!db) return [];

  if (userId) {
    const accessList = await db.select().from(userCompanyAccess).where(eq(userCompanyAccess.userId, userId));
    const companyIds = accessList.map((a) => a.companyId);
    if (companyIds.length === 0) return [];
    return db.select().from(companies).where(inArray(companies.id, companyIds)).orderBy(desc(companies.updatedAt));
  }

  return db.select().from(companies).orderBy(desc(companies.updatedAt));
}

export async function updateCompany(id: number, data: Partial<InsertCompany>): Promise<void> {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  await db.update(companies).set(data).where(eq(companies.id, id));
}

export async function deleteCompany(id: number): Promise<void> {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  await db.delete(companies).where(eq(companies.id, id));
}

// ============================================================================
// USER COMPANY ACCESS QUERIES
// ============================================================================

export async function grantCompanyAccess(data: InsertUserCompanyAccess): Promise<void> {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  await db.insert(userCompanyAccess).values(data);
}

export async function getUserCompanyAccess(userId: number, companyId: number): Promise<UserCompanyAccess | undefined> {
  const db = await getDb();
  if (!db) return undefined;
  const result = await db
    .select()
    .from(userCompanyAccess)
    .where(and(eq(userCompanyAccess.userId, userId), eq(userCompanyAccess.companyId, companyId)))
    .limit(1);
  return result[0];
}

export async function checkUserCompanyRole(
  userId: number,
  companyId: number,
  requiredRoles: string[]
): Promise<boolean> {
  const access = await getUserCompanyAccess(userId, companyId);
  if (!access) return false;
  return requiredRoles.includes(access.accessRole);
}

// ============================================================================
// BANK ACCOUNT QUERIES
// ============================================================================

export async function createBankAccount(data: InsertBankAccount): Promise<number> {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  const result = await db.insert(bankAccounts).values(data);
  return Number(result[0].insertId);
}

export async function listBankAccounts(companyId: number): Promise<BankAccount[]> {
  const db = await getDb();
  if (!db) return [];
  return db.select().from(bankAccounts).where(eq(bankAccounts.companyId, companyId));
}

// ============================================================================
// ACCOUNTING DOCUMENT QUERIES
// ============================================================================

export async function createAccountingDocument(data: InsertAccountingDocument): Promise<number> {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  const result = await db.insert(accountingDocuments).values(data);
  return Number(result[0].insertId);
}

export async function getAccountingDocumentById(id: number): Promise<AccountingDocument | undefined> {
  const db = await getDb();
  if (!db) return undefined;
  const result = await db.select().from(accountingDocuments).where(eq(accountingDocuments.id, id)).limit(1);
  return result[0];
}

export async function listAccountingDocuments(
  companyId: number,
  status?: "NEW" | "PROCESSED" | "POSTED" | "REJECTED"
): Promise<AccountingDocument[]> {
  const db = await getDb();
  if (!db) return [];

  if (status) {
    return db
      .select()
      .from(accountingDocuments)
      .where(and(eq(accountingDocuments.companyId, companyId), eq(accountingDocuments.status, status)))
      .orderBy(desc(accountingDocuments.createdAt));
  }

  return db
    .select()
    .from(accountingDocuments)
    .where(eq(accountingDocuments.companyId, companyId))
    .orderBy(desc(accountingDocuments.createdAt));
}

export async function updateAccountingDocument(id: number, data: Partial<InsertAccountingDocument>): Promise<void> {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  await db.update(accountingDocuments).set(data).where(eq(accountingDocuments.id, id));
}

export async function countUnpostedDocuments(companyId?: number): Promise<number> {
  const db = await getDb();
  if (!db) return 0;

  const condition = companyId
    ? and(eq(accountingDocuments.companyId, companyId), eq(accountingDocuments.status, "NEW"))
    : eq(accountingDocuments.status, "NEW");

  const result = await db.select({ count: sql<number>`count(*)` }).from(accountingDocuments).where(condition);
  return result[0]?.count || 0;
}

// ============================================================================
// LEDGER ENTRY QUERIES
// ============================================================================

export async function createLedgerEntry(data: InsertLedgerEntry): Promise<number> {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  const result = await db.insert(ledgerEntries).values(data);
  return Number(result[0].insertId);
}

export async function listLedgerEntries(
  companyId: number,
  periodStart?: Date,
  periodEnd?: Date
): Promise<LedgerEntry[]> {
  const db = await getDb();
  if (!db) return [];

  let query = db.select().from(ledgerEntries).where(eq(ledgerEntries.companyId, companyId));

  if (periodStart && periodEnd) {
    query = db
      .select()
      .from(ledgerEntries)
      .where(
        and(
          eq(ledgerEntries.companyId, companyId),
          gte(ledgerEntries.entryDate, periodStart),
          lte(ledgerEntries.entryDate, periodEnd)
        )
      );
  }

  return query.orderBy(desc(ledgerEntries.entryDate));
}

// ============================================================================
// FILING QUERIES
// ============================================================================

export async function createFiling(data: InsertFiling): Promise<number> {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  const result = await db.insert(filings).values(data);
  return Number(result[0].insertId);
}

export async function getFilingById(id: number): Promise<Filing | undefined> {
  const db = await getDb();
  if (!db) return undefined;
  const result = await db.select().from(filings).where(eq(filings.id, id)).limit(1);
  return result[0];
}

export async function listFilings(
  companyId: number,
  filingType?: "MVA_MELDING" | "A_MELDING_SUMMARY" | "SAF_T" | "ARSREGNSKAP" | "OTHER"
): Promise<Filing[]> {
  const db = await getDb();
  if (!db) return [];

  if (filingType) {
    return db
      .select()
      .from(filings)
      .where(and(eq(filings.companyId, companyId), eq(filings.filingType, filingType)))
      .orderBy(desc(filings.createdAt));
  }

  return db.select().from(filings).where(eq(filings.companyId, companyId)).orderBy(desc(filings.createdAt));
}

export async function updateFiling(id: number, data: Partial<InsertFiling>): Promise<void> {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  await db.update(filings).set(data).where(eq(filings.id, id));
}

export async function countPendingFilings(companyId?: number): Promise<number> {
  const db = await getDb();
  if (!db) return 0;

  const condition = companyId
    ? and(eq(filings.companyId, companyId), eq(filings.status, "DRAFT"))
    : eq(filings.status, "DRAFT");

  const result = await db.select({ count: sql<number>`count(*)` }).from(filings).where(condition);
  return result[0]?.count || 0;
}

// ============================================================================
// FORVALT SNAPSHOT QUERIES
// ============================================================================

export async function createForvaltSnapshot(data: InsertForvaltSnapshot): Promise<number> {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  const result = await db.insert(forvaltSnapshots).values(data);
  return Number(result[0].insertId);
}

export async function getLatestForvaltSnapshot(companyId: number): Promise<ForvaltSnapshot | undefined> {
  const db = await getDb();
  if (!db) return undefined;
  const result = await db
    .select()
    .from(forvaltSnapshots)
    .where(eq(forvaltSnapshots.companyId, companyId))
    .orderBy(desc(forvaltSnapshots.createdAt))
    .limit(1);
  return result[0];
}

// ============================================================================
// DOCUMENT TEMPLATE QUERIES
// ============================================================================

export async function createDocumentTemplate(data: InsertDocumentTemplate): Promise<number> {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  const result = await db.insert(documentTemplates).values(data);
  return Number(result[0].insertId);
}

export async function getDocumentTemplateById(id: number): Promise<DocumentTemplate | undefined> {
  const db = await getDb();
  if (!db) return undefined;
  const result = await db.select().from(documentTemplates).where(eq(documentTemplates.id, id)).limit(1);
  return result[0];
}

export async function listDocumentTemplates(
  category?: "CONTRACT" | "HR" | "LEGAL" | "FINANCIAL" | "GOVERNANCE" | "OTHER"
): Promise<DocumentTemplate[]> {
  const db = await getDb();
  if (!db) return [];

  if (category) {
    return db
      .select()
      .from(documentTemplates)
      .where(and(eq(documentTemplates.category, category), eq(documentTemplates.isActive, true)))
      .orderBy(desc(documentTemplates.updatedAt));
  }

  return db
    .select()
    .from(documentTemplates)
    .where(eq(documentTemplates.isActive, true))
    .orderBy(desc(documentTemplates.updatedAt));
}

export async function updateDocumentTemplate(id: number, data: Partial<InsertDocumentTemplate>): Promise<void> {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  await db.update(documentTemplates).set(data).where(eq(documentTemplates.id, id));
}

// ============================================================================
// GENERATED DOCUMENT QUERIES
// ============================================================================

export async function createGeneratedDocument(data: InsertGeneratedDocument): Promise<number> {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  const result = await db.insert(generatedDocuments).values(data);
  return Number(result[0].insertId);
}

export async function getGeneratedDocumentById(id: number): Promise<GeneratedDocument | undefined> {
  const db = await getDb();
  if (!db) return undefined;
  const result = await db.select().from(generatedDocuments).where(eq(generatedDocuments.id, id)).limit(1);
  return result[0];
}

export async function listGeneratedDocuments(companyId: number): Promise<GeneratedDocument[]> {
  const db = await getDb();
  if (!db) return [];
  return db
    .select()
    .from(generatedDocuments)
    .where(eq(generatedDocuments.companyId, companyId))
    .orderBy(desc(generatedDocuments.createdAt));
}

// ============================================================================
// CHAT MESSAGE QUERIES
// ============================================================================

export async function createChatMessage(data: InsertChatMessage): Promise<number> {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  const result = await db.insert(chatMessages).values(data);
  return Number(result[0].insertId);
}

export async function listChatMessages(sessionId: string, limit: number = 50): Promise<ChatMessage[]> {
  const db = await getDb();
  if (!db) return [];
  return db
    .select()
    .from(chatMessages)
    .where(eq(chatMessages.sessionId, sessionId))
    .orderBy(desc(chatMessages.createdAt))
    .limit(limit);
}

// ============================================================================
// API LOG QUERIES
// ============================================================================

export async function createApiLog(data: InsertApiLog): Promise<void> {
  const db = await getDb();
  if (!db) return;
  await db.insert(apiLogs).values(data);
}

// ============================================================================
// DASHBOARD STATS
// ============================================================================

export async function getDashboardStats(userId: number) {
  const db = await getDb();
  if (!db) return { unpostedDocuments: 0, pendingFilings: 0, highRiskCompanies: 0 };

  const accessList = await db.select().from(userCompanyAccess).where(eq(userCompanyAccess.userId, userId));
  const companyIds = accessList.map((a) => a.companyId);

  if (companyIds.length === 0) {
    return { unpostedDocuments: 0, pendingFilings: 0, highRiskCompanies: 0 };
  }

  const [unpostedResult, pendingResult, highRiskResult] = await Promise.all([
    db
      .select({ count: sql<number>`count(*)` })
      .from(accountingDocuments)
      .where(and(inArray(accountingDocuments.companyId, companyIds), eq(accountingDocuments.status, "NEW"))),
    db
      .select({ count: sql<number>`count(*)` })
      .from(filings)
      .where(and(inArray(filings.companyId, companyIds), eq(filings.status, "DRAFT"))),
    db
      .select({ count: sql<number>`count(*)` })
      .from(companies)
      .where(and(inArray(companies.id, companyIds), eq(companies.forvaltRiskClass, "HIGH"))),
  ]);

  return {
    unpostedDocuments: unpostedResult[0]?.count || 0,
    pendingFilings: pendingResult[0]?.count || 0,
    highRiskCompanies: highRiskResult[0]?.count || 0,
  };
}
