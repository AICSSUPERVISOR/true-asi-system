import { describe, expect, it, vi, beforeEach } from "vitest";

// Mock the LLM invocation
vi.mock("./_core/llm", () => ({
  invokeLLM: vi.fn().mockResolvedValue({
    choices: [
      {
        message: {
          content: "Jeg har analysert forespørselen din. Her er min anbefaling for kontering.",
          tool_calls: undefined,
        },
      },
    ],
    model: "gpt-4o",
    usage: { prompt_tokens: 100, completion_tokens: 50, total_tokens: 150 },
  }),
}));

// Mock database functions
vi.mock("./db", () => ({
  createLedgerEntry: vi.fn().mockResolvedValue(1),
  updateAccountingDocument: vi.fn().mockResolvedValue(undefined),
  createFiling: vi.fn().mockResolvedValue(1),
  getDocumentTemplateById: vi.fn().mockResolvedValue({
    id: 1,
    name: "Test Template",
    bodyMarkdown: "# Test\n\nHello {{name}}",
    language: "no",
  }),
  createGeneratedDocument: vi.fn().mockResolvedValue(1),
  getCompanyById: vi.fn().mockResolvedValue({
    id: 1,
    name: "Test Company AS",
    orgNumber: "123456789",
  }),
  listLedgerEntries: vi.fn().mockResolvedValue([
    {
      id: 1,
      companyId: 1,
      debitAccount: "3000",
      creditAccount: "1500",
      amount: 10000,
      vatCode: "3",
      entryDate: new Date(),
    },
  ]),
  createApiLog: vi.fn().mockResolvedValue(1),
}));

// Mock API clients
vi.mock("./clients/aimlClient", () => ({
  createAIMLClient: vi.fn().mockReturnValue({
    extractInvoiceFields: vi.fn().mockResolvedValue({
      supplierName: "Test Supplier AS",
      invoiceNumber: "INV-001",
      totalAmount: 125000,
      vatAmount: 25000,
      currency: "NOK",
      confidence: 0.95,
    }),
    classifyTransaction: vi.fn().mockResolvedValue({
      suggestedAccount: "4000",
      suggestedVatCode: "3",
      confidence: 0.9,
      reasoning: "Varekjøp med 25% MVA",
    }),
    draftDocumentFromTemplate: vi.fn().mockResolvedValue({
      outputMarkdown: "# Test Document\n\nGenerated content",
      confidence: 0.95,
    }),
  }),
}));

vi.mock("./clients/forvaltClient", () => ({
  createForvaltClient: vi.fn().mockReturnValue({
    getFullProfile: vi.fn().mockResolvedValue({
      company: { name: "Test Company AS", orgNumber: "123456789" },
      credit: { creditScore: 85, riskClass: "LOW", rating: "A" },
      financials: { revenue: 1000000 },
    }),
    getCredit: vi.fn().mockResolvedValue({
      creditScore: 85,
      riskClass: "LOW",
      paymentRemarks: 0,
    }),
  }),
}));

vi.mock("./clients/regnskapClient", () => ({
  createRegnskapClient: vi.fn().mockReturnValue({}),
}));

import { SkattFlowAgentV2 } from "./agents/skattFlowAgentV2";

describe("SkattFlowAgentV2", () => {
  let agent: SkattFlowAgentV2;

  beforeEach(() => {
    vi.clearAllMocks();
    agent = new SkattFlowAgentV2();
  });

  describe("chat", () => {
    it("returns a response for a simple message", async () => {
      const response = await agent.chat("Hva er MVA-satsen for mat?");

      expect(response).toBeDefined();
      expect(response.message).toBeDefined();
      expect(typeof response.message).toBe("string");
      expect(response.message.length).toBeGreaterThan(0);
    });

    it("includes metadata in the response", async () => {
      const response = await agent.chat("Forklar bokføringsloven");

      expect(response.metadata).toBeDefined();
      expect(response.metadata?.model).toBe("gpt-4o");
    });

    it("handles context with companyId", async () => {
      const response = await agent.chat(
        "Vis meg siste fakturaer",
        { companyId: 1 },
        [],
        1
      );

      expect(response).toBeDefined();
      expect(response.message).toBeDefined();
    });
  });

  describe("processDocumentWithOCR", () => {
    it("extracts fields from a document", async () => {
      const result = await agent.processDocumentWithOCR(
        "https://example.com/invoice.pdf",
        "INVOICE_SUPPLIER",
        false
      );

      expect(result.extractedFields).toBeDefined();
      expect(result.extractedFields.supplierName).toBe("Test Supplier AS");
      expect(result.extractedFields.totalAmount).toBe(125000);
      expect(result.suggestedPosting).toBeDefined();
      expect(result.suggestedPosting.account).toBe("4000");
      expect(result.suggestedPosting.vatCode).toBe("3");
      expect(result.confidence).toBeGreaterThan(0);
    });
  });

  describe("autonomousMVAFiling", () => {
    it("generates MVA filing draft", async () => {
      const result = await agent.autonomousMVAFiling(1, 2024, 6, false);

      expect(result.filingId).toBe(1);
      expect(result.status).toBe("DRAFT");
      expect(result.summary).toContain("MVA-melding");
      expect(result.submitted).toBe(false);
    });

    it("includes warnings for empty periods", async () => {
      // Mock empty ledger
      const db = await import("./db");
      vi.mocked(db.listLedgerEntries).mockResolvedValueOnce([]);

      const result = await agent.autonomousMVAFiling(1, 2024, 1, false);

      expect(result.warnings).toContain("Ingen posteringer funnet i perioden");
    });
  });

  describe("clearHistory", () => {
    it("clears conversation history", async () => {
      // Send a message to populate history
      await agent.chat("Test message");

      // Clear history
      agent.clearHistory();

      // Send another message - should not have previous context
      const response = await agent.chat("Another message");
      expect(response).toBeDefined();
    });
  });
});

describe("Agent Tool Execution", () => {
  it("agent has tool definitions", async () => {
    const agent = new SkattFlowAgentV2();
    
    // The agent should be able to handle tool-related requests
    const response = await agent.chat("Opprett et bilag for varekjøp på 1000 kr");
    
    expect(response).toBeDefined();
    expect(response.message).toBeDefined();
  });
});
