import { describe, expect, it, vi, beforeEach } from "vitest";
import { appRouter } from "./routers";
import type { TrpcContext } from "./_core/context";

// Mock database functions
vi.mock("./db", () => ({
  listCompanies: vi.fn().mockResolvedValue([
    {
      id: 1,
      name: "Test Company AS",
      orgNumber: "123456789",
      city: "Oslo",
      forvaltRating: "A",
      forvaltCreditScore: 85,
      forvaltRiskClass: "LOW",
      externalRegnskapSystem: "TRIPLETEX",
      createdAt: new Date(),
      updatedAt: new Date(),
    },
  ]),
  getCompanyById: vi.fn().mockResolvedValue({
    id: 1,
    name: "Test Company AS",
    orgNumber: "123456789",
    address: "Testveien 1",
    postalCode: "0123",
    city: "Oslo",
    forvaltRating: "A",
    forvaltCreditScore: 85,
    forvaltRiskClass: "LOW",
    externalRegnskapSystem: "TRIPLETEX",
    externalRegnskapCompanyId: "12345",
    createdAt: new Date(),
    updatedAt: new Date(),
    createdById: 1,
  }),
  createCompany: vi.fn().mockResolvedValue(1),
  updateCompany: vi.fn().mockResolvedValue(undefined),
  getUserCompanyAccess: vi.fn().mockResolvedValue({
    id: 1,
    userId: 1,
    companyId: 1,
    accessRole: "OWNER",
    createdAt: new Date(),
    updatedAt: new Date(),
  }),
}));

type AuthenticatedUser = NonNullable<TrpcContext["user"]>;

function createAuthContext(): TrpcContext {
  const user: AuthenticatedUser = {
    id: 1,
    openId: "test-user-123",
    email: "test@example.com",
    name: "Test User",
    loginMethod: "manus",
    role: "admin",
    accountingRole: "OWNER",
    createdAt: new Date(),
    updatedAt: new Date(),
    lastSignedIn: new Date(),
  };

  return {
    user,
    req: {
      protocol: "https",
      headers: {},
    } as TrpcContext["req"],
    res: {
      clearCookie: vi.fn(),
    } as unknown as TrpcContext["res"],
  };
}

describe("company router", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("company.list", () => {
    it("returns list of companies for authenticated user", async () => {
      const ctx = createAuthContext();
      const caller = appRouter.createCaller(ctx);

      const result = await caller.company.list();

      expect(result).toHaveLength(1);
      expect(result[0].name).toBe("Test Company AS");
      expect(result[0].orgNumber).toBe("123456789");
      expect(result[0].forvaltRating).toBe("A");
    });
  });

  describe("company.get", () => {
    it("returns company details by id", async () => {
      const ctx = createAuthContext();
      const caller = appRouter.createCaller(ctx);

      const result = await caller.company.get({ id: 1 });

      expect(result.id).toBe(1);
      expect(result.name).toBe("Test Company AS");
      expect(result.orgNumber).toBe("123456789");
      expect(result.address).toBe("Testveien 1");
      expect(result.forvaltCreditScore).toBe(85);
    });

    it("throws NOT_FOUND for non-existent company", async () => {
      const ctx = createAuthContext();
      const caller = appRouter.createCaller(ctx);

      // Mock getCompanyById to return null
      const db = await import("./db");
      vi.mocked(db.getCompanyById).mockResolvedValueOnce(null);

      await expect(caller.company.get({ id: 999 })).rejects.toThrow("Company not found");
    });
  });
});

describe("auth.logout", () => {
  it("clears the session cookie and reports success", async () => {
    const ctx = createAuthContext();
    const caller = appRouter.createCaller(ctx);

    const result = await caller.auth.logout();

    expect(result).toEqual({ success: true });
    expect(ctx.res.clearCookie).toHaveBeenCalled();
  });
});
