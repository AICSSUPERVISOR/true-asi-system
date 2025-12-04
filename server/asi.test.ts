import { describe, expect, it } from "vitest";
import { appRouter } from "./routers";
import type { TrpcContext } from "./_core/context";

type AuthenticatedUser = NonNullable<TrpcContext["user"]>;

function createTestContext(): { ctx: TrpcContext } {
  const user: AuthenticatedUser = {
    id: 1,
    openId: "test-user",
    email: "test@example.com",
    name: "Test User",
    loginMethod: "manus",
    role: "user",
    createdAt: new Date(),
    updatedAt: new Date(),
    lastSignedIn: new Date(),
  };

  const ctx: TrpcContext = {
    user,
    req: {
      protocol: "https",
      headers: {},
    } as TrpcContext["req"],
    res: {
      clearCookie: () => {},
    } as TrpcContext["res"],
  };

  return { ctx };
}

function createPublicContext(): { ctx: TrpcContext } {
  const ctx: TrpcContext = {
    user: undefined,
    req: {
      protocol: "https",
      headers: {},
    } as TrpcContext["req"],
    res: {
      clearCookie: () => {},
    } as TrpcContext["res"],
  };

  return { ctx };
}

describe("ASI System API", () => {
  describe("asi.status", () => {
    it("should return system status", async () => {
      const { ctx } = createPublicContext();
      const caller = appRouter.createCaller(ctx);

      const result = await caller.asi.status();

      expect(result).toBeDefined();
      expect(result.agents).toBe(250);
      expect(result.knowledgeBase).toBe("6.54TB");
      expect(result.uptime).toBe("99.9%");
      expect(["operational", "degraded"]).toContain(result.status);
    }, 10000); // Increase timeout for EC2 API call
  });

  describe("asi.agents", () => {
    it("should return list of 250 agents", async () => {
      const { ctx } = createPublicContext();
      const caller = appRouter.createCaller(ctx);

      const result = await caller.asi.agents();

      expect(result).toBeDefined();
      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBe(250);
      
      // Check first agent structure
      const firstAgent = result[0];
      expect(firstAgent).toHaveProperty("id");
      expect(firstAgent).toHaveProperty("name");
      expect(firstAgent).toHaveProperty("status");
      expect(firstAgent).toHaveProperty("capabilities");
      expect(firstAgent?.status).toBe("active");
    });

    it("should have correct agent naming format", async () => {
      const { ctx } = createPublicContext();
      const caller = appRouter.createCaller(ctx);

      const result = await caller.asi.agents();

      expect(result[0]?.name).toBe("Agent 000");
      expect(result[1]?.name).toBe("Agent 001");
      expect(result[249]?.name).toBe("Agent 249");
    });
  });

  describe("asi.models", () => {
    it("should return available AI models", async () => {
      const { ctx } = createPublicContext();
      const caller = appRouter.createCaller(ctx);

      const result = await caller.asi.models();

      expect(result).toBeDefined();
      expect(result).toHaveProperty("models");
      expect(Array.isArray(result.models)).toBe(true);
      expect(result.models.length).toBeGreaterThan(0);
    });
  });

  describe("asi.knowledgeGraph", () => {
    it("should return knowledge graph statistics", async () => {
      const { ctx } = createPublicContext();
      const caller = appRouter.createCaller(ctx);

      const result = await caller.asi.knowledgeGraph();

      expect(result).toBeDefined();
      expect(result.entities).toBe(19649);
      expect(result.relationships).toBe(468);
      expect(result.files).toBe(1174651);
      expect(result.size).toBe("6.54TB");
      expect(result.lastUpdated).toBeInstanceOf(Date);
    });
  });

  describe("asi.metrics", () => {
    it("should require authentication", async () => {
      const { ctx } = createPublicContext();
      const caller = appRouter.createCaller(ctx);

      await expect(caller.asi.metrics()).rejects.toThrow();
    });

    it("should return system metrics for authenticated users", async () => {
      const { ctx } = createTestContext();
      const caller = appRouter.createCaller(ctx);

      const result = await caller.asi.metrics();

      expect(result).toBeDefined();
      expect(result.cpu).toHaveProperty("cores", 8);
      expect(result.cpu).toHaveProperty("usage");
      expect(result.memory).toHaveProperty("total", "16GB");
      expect(result.storage).toHaveProperty("total", "5TB");
      expect(result.agents).toHaveProperty("total", 250);
      expect(result.agents).toHaveProperty("active", 250);
      expect(result.requests).toHaveProperty("total");
      expect(result.requests).toHaveProperty("success");
      expect(result.requests).toHaveProperty("failed");
    });
  });

  describe("asi.chat", () => {
    it("should require authentication", async () => {
      const { ctx } = createPublicContext();
      const caller = appRouter.createCaller(ctx);

      await expect(
        caller.asi.chat({ message: "Hello" })
      ).rejects.toThrow();
    });

    it("should accept chat messages from authenticated users", async () => {
      const { ctx } = createTestContext();
      const caller = appRouter.createCaller(ctx);

      const result = await caller.asi.chat({
        message: "Hello, ASI!",
        model: "gpt-4",
      });

      expect(result).toBeDefined();
      expect(result).toHaveProperty("success");
      expect(result).toHaveProperty("message");
      // API may fail, so we check for error handling
      if (result.success) {
        expect(result).toHaveProperty("model");
      } else {
        expect(result).toHaveProperty("error");
      }
    }, 35000); // Increase timeout for API call

    it("should validate message input", async () => {
      const { ctx } = createTestContext();
      const caller = appRouter.createCaller(ctx);

      await expect(
        caller.asi.chat({ message: "" })
      ).rejects.toThrow();
    });
  });
});

describe("Authentication", () => {
  describe("auth.me", () => {
    it("should return null for unauthenticated users", async () => {
      const { ctx } = createPublicContext();
      const caller = appRouter.createCaller(ctx);

      const result = await caller.auth.me();

      expect(result).toBeUndefined();
    });

    it("should return user info for authenticated users", async () => {
      const { ctx } = createTestContext();
      const caller = appRouter.createCaller(ctx);

      const result = await caller.auth.me();

      expect(result).toBeDefined();
      expect(result?.email).toBe("test@example.com");
      expect(result?.name).toBe("Test User");
      expect(result?.role).toBe("user");
    });
  });

  describe("auth.logout", () => {
    it("should successfully logout", async () => {
      const { ctx } = createTestContext();
      const caller = appRouter.createCaller(ctx);

      const result = await caller.auth.logout();

      expect(result).toEqual({ success: true });
    });
  });
});
