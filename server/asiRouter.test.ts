import { describe, it, expect } from "vitest";
import { appRouter } from "./routers";
import type { TrpcContext } from "./_core/context";

function createTestContext(): TrpcContext {
  return {
    user: null,
    req: {
      protocol: "https",
      headers: {},
    } as TrpcContext["req"],
    res: {
      clearCookie: () => {},
    } as TrpcContext["res"],
  };
}

describe("ASI Symbiosis Router", () => {
  const ctx = createTestContext();
  const caller = appRouter.createCaller(ctx);

  describe("asiSymbiosis.getModels", () => {
    it("should return all models with stats", async () => {
      const result = await caller.asiSymbiosis.getModels();
      
      expect(result).toHaveProperty("models");
      expect(result).toHaveProperty("stats");
      expect(Array.isArray(result.models)).toBe(true);
      expect(result.models.length).toBeGreaterThan(0);
      
      // Check stats
      expect(result.stats).toHaveProperty("total");
      expect(result.stats).toHaveProperty("downloaded");
      expect(result.stats).toHaveProperty("pending");
      expect(result.stats).toHaveProperty("downloadedSize");
      expect(result.stats).toHaveProperty("totalSize");
      expect(result.stats).toHaveProperty("downloadProgress");
      
      // Verify downloaded models exist
      expect(result.stats.downloaded).toBeGreaterThan(0);
    });

    it("should have correct model structure", async () => {
      const result = await caller.asiSymbiosis.getModels();
      const model = result.models[0];
      
      expect(model).toHaveProperty("id");
      expect(model).toHaveProperty("category");
      expect(model).toHaveProperty("size_gb");
      expect(model).toHaveProperty("status");
      expect(model).toHaveProperty("capabilities");
      expect(Array.isArray(model.capabilities)).toBe(true);
    });
  });

  describe("asiSymbiosis.getModelsByCategory", () => {
    it("should return embedding models", async () => {
      const result = await caller.asiSymbiosis.getModelsByCategory({ category: "embedding" });
      
      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBeGreaterThan(0);
      result.forEach(model => {
        expect(model.category).toBe("embedding");
      });
    });

    it("should return foundation models", async () => {
      const result = await caller.asiSymbiosis.getModelsByCategory({ category: "foundation" });
      
      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBeGreaterThan(0);
      result.forEach(model => {
        expect(model.category).toBe("foundation");
      });
    });
  });

  describe("asiSymbiosis.getAgents", () => {
    it("should return all 9 specialized agents", async () => {
      const result = await caller.asiSymbiosis.getAgents();
      
      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBe(9);
    });

    it("should have correct agent structure", async () => {
      const result = await caller.asiSymbiosis.getAgents();
      const agent = result[0];
      
      expect(agent).toHaveProperty("id");
      expect(agent).toHaveProperty("name");
      expect(agent).toHaveProperty("description");
      expect(agent).toHaveProperty("models");
      expect(agent).toHaveProperty("capabilities");
      expect(agent).toHaveProperty("status");
      expect(Array.isArray(agent.models)).toBe(true);
      expect(Array.isArray(agent.capabilities)).toBe(true);
    });

    it("should include all expected agent types", async () => {
      const result = await caller.asiSymbiosis.getAgents();
      const agentIds = result.map(a => a.id);
      
      expect(agentIds).toContain("reasoning_agent");
      expect(agentIds).toContain("code_agent");
      expect(agentIds).toContain("math_agent");
      expect(agentIds).toContain("research_agent");
      expect(agentIds).toContain("embedding_agent");
      expect(agentIds).toContain("multimodal_agent");
      expect(agentIds).toContain("audio_agent");
      expect(agentIds).toContain("finance_agent");
      expect(agentIds).toContain("science_agent");
    });
  });

  describe("asiSymbiosis.getAgent", () => {
    it("should return specific agent by ID", async () => {
      const result = await caller.asiSymbiosis.getAgent({ id: "reasoning_agent" });
      
      expect(result).toBeDefined();
      expect(result?.id).toBe("reasoning_agent");
      expect(result?.name).toBe("Reasoning Agent");
    });

    it("should return undefined for non-existent agent", async () => {
      const result = await caller.asiSymbiosis.getAgent({ id: "non_existent_agent" });
      
      expect(result).toBeUndefined();
    });
  });

  describe("asiSymbiosis.getSystemStatus", () => {
    it("should return operational system status", async () => {
      const result = await caller.asiSymbiosis.getSystemStatus();
      
      expect(result).toHaveProperty("status");
      expect(result.status).toBe("operational");
      expect(result).toHaveProperty("models");
      expect(result).toHaveProperty("agents");
      expect(result).toHaveProperty("capabilities");
      expect(result).toHaveProperty("version");
      expect(result).toHaveProperty("lastUpdated");
    });

    it("should have correct model counts", async () => {
      const result = await caller.asiSymbiosis.getSystemStatus();
      
      expect(result.models.total).toBeGreaterThan(0);
      expect(result.models.downloaded).toBeGreaterThan(0);
      expect(result.models.ready).toBeGreaterThan(0);
    });

    it("should have 9 agents ready", async () => {
      const result = await caller.asiSymbiosis.getSystemStatus();
      
      expect(result.agents.total).toBe(9);
      expect(result.agents.ready).toBe(9);
    });

    it("should list all capabilities", async () => {
      const result = await caller.asiSymbiosis.getSystemStatus();
      
      expect(result.capabilities).toContain("chat");
      expect(result.capabilities).toContain("code");
      expect(result.capabilities).toContain("math");
      expect(result.capabilities).toContain("reasoning");
      expect(result.capabilities).toContain("embedding");
    });
  });

  describe("asiSymbiosis.consensusQuery", () => {
    it("should handle chat task type", async () => {
      const result = await caller.asiSymbiosis.consensusQuery({
        prompt: "Hello, world!",
        taskType: "chat",
        modelCount: 3
      });
      
      expect(result).toHaveProperty("taskType");
      expect(result.taskType).toBe("chat");
      expect(result).toHaveProperty("prompt");
      expect(result).toHaveProperty("selectedModels");
      expect(result).toHaveProperty("consensusMethod");
      expect(result.consensusMethod).toBe("weighted_voting");
    });

    it("should handle embedding task type with available models", async () => {
      const result = await caller.asiSymbiosis.consensusQuery({
        prompt: "Test embedding",
        taskType: "embedding",
        modelCount: 3
      });
      
      expect(result.taskType).toBe("embedding");
      expect(result.availableModels.length).toBeGreaterThan(0);
      expect(result.status).toBe("ready");
    });

    it("should indicate pending status for unavailable models", async () => {
      const result = await caller.asiSymbiosis.consensusQuery({
        prompt: "Complex reasoning task",
        taskType: "reasoning",
        modelCount: 3
      });
      
      expect(result.taskType).toBe("reasoning");
      expect(result.selectedModels.length).toBeGreaterThan(0);
      // Reasoning models are pending download
      expect(result.status).toBe("models_pending");
    });

    it("should respect modelCount parameter", async () => {
      const result = await caller.asiSymbiosis.consensusQuery({
        prompt: "Test",
        taskType: "embedding",
        modelCount: 2
      });
      
      expect(result.selectedModels.length).toBeLessThanOrEqual(2);
    });
  });
});
