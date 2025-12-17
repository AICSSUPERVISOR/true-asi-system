import { describe, expect, it, vi, beforeEach } from "vitest";
import { appRouter } from "./routers";
import type { TrpcContext } from "./_core/context";

// ============================================================================
// INTEGRATION TESTS FOR SKATT-FLOW OS
// Comprehensive tests for all major features
// ============================================================================

// Mock database module
vi.mock("./db", () => ({
  getDb: vi.fn().mockResolvedValue({}),
  listCompanies: vi.fn().mockResolvedValue([
    { id: 1, name: "Test AS", orgNumber: "123456789", status: "ACTIVE" },
  ]),
  getCompanyById: vi.fn().mockResolvedValue({
    id: 1,
    name: "Test AS",
    orgNumber: "123456789",
    status: "ACTIVE",
    forvaltRating: "A",
  }),
  getCompanyByOrgNumber: vi.fn().mockResolvedValue(null),
  createCompany: vi.fn().mockResolvedValue(1),
  updateCompany: vi.fn().mockResolvedValue(undefined),
  grantCompanyAccess: vi.fn().mockResolvedValue(undefined),
  listAccountingDocuments: vi.fn().mockResolvedValue([
    { id: 1, companyId: 1, status: "NEW", sourceType: "INVOICE_SUPPLIER" },
  ]),
  getAccountingDocumentById: vi.fn().mockResolvedValue({
    id: 1,
    companyId: 1,
    status: "NEW",
    sourceType: "INVOICE_SUPPLIER",
  }),
  listDocuments: vi.fn().mockResolvedValue([
    { id: 1, companyId: 1, status: "NEW", sourceType: "INVOICE_SUPPLIER" },
  ]),
  getDocumentById: vi.fn().mockResolvedValue({
    id: 1,
    companyId: 1,
    status: "NEW",
    sourceType: "INVOICE_SUPPLIER",
  }),
  createDocument: vi.fn().mockResolvedValue(1),
  updateDocument: vi.fn().mockResolvedValue(undefined),
  listLedgerEntries: vi.fn().mockResolvedValue([
    { id: 1, companyId: 1, amount: 10000, debitAccount: "4000", creditAccount: "2400" },
  ]),
  createLedgerEntry: vi.fn().mockResolvedValue(1),
  listFilings: vi.fn().mockResolvedValue([
    { id: 1, companyId: 1, filingType: "MVA_MELDING", status: "DRAFT" },
  ]),
  getFilingById: vi.fn().mockResolvedValue({
    id: 1,
    companyId: 1,
    filingType: "MVA_MELDING",
    status: "DRAFT",
  }),
  createFiling: vi.fn().mockResolvedValue(1),
  updateFiling: vi.fn().mockResolvedValue(undefined),
  listTemplates: vi.fn().mockResolvedValue([
    { id: 1, name: "Invoice Template", templateType: "INVOICE", isActive: true },
  ]),
  getTemplateById: vi.fn().mockResolvedValue({
    id: 1,
    name: "Invoice Template",
    templateType: "INVOICE",
    content: "Template content",
    isActive: true,
  }),
  createTemplate: vi.fn().mockResolvedValue(1),
  getUserCompanyAccess: vi.fn().mockResolvedValue({ role: "OWNER" }),
  createApiLog: vi.fn().mockResolvedValue(1),
  createChatMessage: vi.fn().mockResolvedValue(1),
  listChatMessages: vi.fn().mockResolvedValue([]),
  getDashboardStats: vi.fn().mockResolvedValue({
    unpostedDocuments: 5,
    pendingFilings: 2,
    highRiskCompanies: 1,
    activeCompanies: 10,
  }),
  getRecentActivities: vi.fn().mockResolvedValue([]),
  getUpcomingDeadlines: vi.fn().mockResolvedValue([]),
  createGeneratedDocument: vi.fn().mockResolvedValue(1),
  listDocumentTemplates: vi.fn().mockResolvedValue([
    { id: 1, name: "Invoice Template", templateType: "INVOICE", isActive: true },
  ]),
  countCompanies: vi.fn().mockResolvedValue(10),
  countDocumentsByStatus: vi.fn().mockResolvedValue(5),
  countFilingsByStatus: vi.fn().mockResolvedValue(2),
  countHighRiskCompanies: vi.fn().mockResolvedValue(1),
}));

// Mock middleware
vi.mock("./middleware/auditLog", () => ({
  logAudit: vi.fn().mockResolvedValue(undefined),
  requireCompanyAccess: vi.fn().mockResolvedValue({ role: "OWNER" }),
}));

// Mock external clients
vi.mock("./clients/forvaltClient", () => ({
  createForvaltClient: vi.fn().mockReturnValue({
    getCompany: vi.fn().mockResolvedValue({ name: "Test AS", orgNumber: "123456789" }),
    getCredit: vi.fn().mockResolvedValue({ creditScore: 85, riskClass: "LOW" }),
    getFinancials: vi.fn().mockResolvedValue({ revenue: 1000000, profit: 100000 }),
    getFullProfile: vi.fn().mockResolvedValue({
      company: { name: "Test AS" },
      credit: { creditScore: 85 },
      financials: { revenue: 1000000 },
    }),
  }),
}));

vi.mock("./agents/skattFlowAgent", () => ({
  getSkattFlowAgent: vi.fn().mockReturnValue({
    chat: vi.fn().mockResolvedValue({
      message: "This is a test response",
      actions: [],
    }),
  }),
}));

vi.mock("./agents/skattFlowAgentV2", () => ({
  getSkattFlowAgentV2: vi.fn().mockReturnValue({
    chat: vi.fn().mockResolvedValue({
      message: "This is a test response from V2",
      actions: [],
      toolResults: [],
    }),
  }),
}));

// Helper to create authenticated context with accountingRole
function createAuthContext(role: "admin" | "user" = "admin"): { ctx: TrpcContext } {
  const ctx = {
    user: {
      id: 1,
      openId: "test-user",
      email: "test@example.com",
      name: "Test User",
      loginMethod: "manus",
      role,
      accountingRole: role === "admin" ? "OWNER" : "VIEWER",
      createdAt: new Date(),
      updatedAt: new Date(),
      lastSignedIn: new Date(),
    },
    req: {
      protocol: "https",
      headers: {},
    } as TrpcContext["req"],
    res: {
      clearCookie: vi.fn(),
    } as unknown as TrpcContext["res"],
  } as TrpcContext;

  return { ctx };
}

// ============================================================================
// COMPANY ROUTER TESTS
// ============================================================================

describe("company router", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("lists companies for authenticated user", async () => {
    const { ctx } = createAuthContext();
    const caller = appRouter.createCaller(ctx);

    const result = await caller.company.list();

    expect(result).toBeDefined();
    expect(Array.isArray(result)).toBe(true);
  });

  it("gets a single company by id", async () => {
    const { ctx } = createAuthContext();
    const caller = appRouter.createCaller(ctx);

    const result = await caller.company.get({ id: 1 });

    expect(result).toBeDefined();
    expect(result?.name).toBe("Test AS");
  });

  it("creates a new company", async () => {
    const { ctx } = createAuthContext();
    const caller = appRouter.createCaller(ctx);

    const result = await caller.company.create({
      name: "New Company AS",
      orgNumber: "987654321",
    });

    expect(result).toBeDefined();
    expect(result.id).toBe(1);
  });
});

// ============================================================================
// DOCUMENT ROUTER TESTS (correct name: document, not accountingDocument)
// ============================================================================

describe("document router", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("lists documents for a company", async () => {
    const { ctx } = createAuthContext();
    const caller = appRouter.createCaller(ctx);

    const result = await caller.document.list({ companyId: 1 });

    expect(result).toBeDefined();
    expect(Array.isArray(result)).toBe(true);
  });
});

// ============================================================================
// FILING ROUTER TESTS
// ============================================================================

describe("filing router", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("lists filings for a company", async () => {
    const { ctx } = createAuthContext();
    const caller = appRouter.createCaller(ctx);

    const result = await caller.filing.list({ companyId: 1 });

    expect(result).toBeDefined();
    expect(Array.isArray(result)).toBe(true);
  });
});

// ============================================================================
// DASHBOARD ROUTER TESTS
// ============================================================================

describe("dashboard router", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("returns dashboard stats", async () => {
    const { ctx } = createAuthContext();
    const caller = appRouter.createCaller(ctx);

    const result = await caller.dashboard.stats({});

    expect(result).toBeDefined();
    expect(typeof result.unpostedDocuments).toBe("number");
    expect(typeof result.pendingFilings).toBe("number");
  });
});

// ============================================================================
// CHAT ROUTER TESTS
// ============================================================================

describe("chat router", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("sends a message and receives a response", async () => {
    const { ctx } = createAuthContext();
    const caller = appRouter.createCaller(ctx);

    const result = await caller.chat.send({
      companyId: 1,
      message: "What is the VAT rate in Norway?",
      sessionId: "test-session",
    });

    expect(result).toBeDefined();
    expect(typeof result.message).toBe("string");
  });
});

// ============================================================================
// AUTH ROUTER TESTS
// ============================================================================

describe("auth router", () => {
  it("returns current user for authenticated request", async () => {
    const { ctx } = createAuthContext();
    const caller = appRouter.createCaller(ctx);

    const result = await caller.auth.me();

    expect(result).toBeDefined();
    expect(result?.email).toBe("test@example.com");
  });

  it("returns null for unauthenticated request", async () => {
    const ctx: TrpcContext = {
      user: null,
      req: { protocol: "https", headers: {} } as TrpcContext["req"],
      res: { clearCookie: vi.fn() } as unknown as TrpcContext["res"],
    };
    const caller = appRouter.createCaller(ctx);

    const result = await caller.auth.me();

    expect(result).toBeNull();
  });

  it("logs out and clears cookie", async () => {
    const { ctx } = createAuthContext();
    const caller = appRouter.createCaller(ctx);

    const result = await caller.auth.logout();

    expect(result).toEqual({ success: true });
    expect(ctx.res.clearCookie).toHaveBeenCalled();
  });
});

// ============================================================================
// TEMPLATE ROUTER TESTS (correct name: template, not documentTemplate)
// ============================================================================

describe("template router", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("lists document templates", async () => {
    const { ctx } = createAuthContext();
    const caller = appRouter.createCaller(ctx);

    const result = await caller.template.list({});

    expect(result).toBeDefined();
    expect(Array.isArray(result)).toBe(true);
  });
});

// ============================================================================
// LEDGER ROUTER TESTS
// ============================================================================

describe("ledger router", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("lists ledger entries for a company", async () => {
    const { ctx } = createAuthContext();
    const caller = appRouter.createCaller(ctx);

    const result = await caller.ledger.list({ companyId: 1 });

    expect(result).toBeDefined();
    expect(Array.isArray(result)).toBe(true);
  });
});
