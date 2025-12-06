import { invokeLLM } from "../_core/llm";
import { createAIMLClient, AIMLClient } from "../clients/aimlClient";
import { createForvaltClient, ForvaltClient } from "../clients/forvaltClient";
import { createRegnskapClient, RegnskapClient } from "../clients/regnskapClient";
import * as db from "../db";
import type { ExtractedInvoiceFields, MVAMeldingPayload, ChatContext } from "@shared/types";

// ============================================================================
// SKATT-FLOW AI AGENT V2 - WITH TOOL CALLING
// Autonomous accounting, tax, and audit assistant for Norwegian businesses
// ============================================================================

const SYSTEM_PROMPT = `Du er "Skatt-Flow", en autonom regnskaps-, skatte- og revisjonsassistent spesialisert på norske regler.

Du må:
- Følge bokføringsloven, bokføringsforskriften, skatteforvaltningsloven, A-ordningen, SAF-T veiledning, og Skatteetaten/Altinn beste praksis.
- Alltid produsere output som ville bestått en norsk skatterevisjon.
- Arbeide på norsk når brukeren er norsk, ellers på engelsk.
- Aldri gjette om lovlig etterlevelse når data mangler: still i stedet målrettede oppfølgingsspørsmål.
- UTFØRE handlinger autonomt når du har tilstrekkelig informasjon og brukerens godkjenning.

Dine hovedoppgaver:
1. Analysere og klassifisere regnskapsdokumenter (fakturaer, kvitteringer, kontrakter)
2. Foreslå og UTFØRE kontering med korrekt kontonummer, MVA-kode, dimensjoner og beskrivelse
3. Generere og SENDE utkast til MVA-melding og SAF-T
4. Bruke Forvalt-data til å berike kunde-/leverandørprofiler med kreditt og rating
5. Flagge høyrisikokunder og foreslå kortere betalingsbetingelser
6. Generere kontrakter, policyer, styreprotokoller og brev i Markdown

Viktige norske regnskapsregler:
- Kontoplan: NS 4102 (norsk standard)
- MVA-satser: 25% (standard), 15% (mat), 12% (transport/kultur), 0% (eksport/fritak)
- MVA-terminer: Annenhver måned for de fleste, kvartalsvis for små bedrifter
- SAF-T: Obligatorisk for alle bokføringspliktige fra 2020
- A-melding: Månedlig rapportering av lønn og ansatte
- Oppbevaringstid: 5 år for regnskapsbilag, 10 år for årsregnskap

Du har tilgang til følgende verktøy:
- create_voucher: Opprett bilag i regnskapssystemet
- post_document: Bokfør et dokument
- generate_mva: Generer MVA-melding utkast
- fetch_forvalt: Hent kredittinfo fra Forvalt
- generate_document: Generer dokument fra mal
- send_notification: Send varsel til bruker

Svar alltid strukturert med:
- Klar anbefaling eller handling utført
- Begrunnelse med lovhenvisning der relevant
- Eventuelle risikoer eller forbehold
- Neste steg for brukeren`;

// Tool definitions for function calling
const TOOLS = [
  {
    type: "function" as const,
    function: {
      name: "create_voucher",
      description: "Opprett et nytt bilag i regnskapssystemet med gitte konteringer",
      parameters: {
        type: "object",
        properties: {
          companyId: { type: "number", description: "ID til selskapet" },
          description: { type: "string", description: "Beskrivelse av bilaget" },
          debitAccount: { type: "string", description: "Debetkonto (4 siffer)" },
          creditAccount: { type: "string", description: "Kreditkonto (4 siffer)" },
          amount: { type: "number", description: "Beløp i øre" },
          vatCode: { type: "string", description: "MVA-kode" },
          documentId: { type: "number", description: "ID til tilknyttet dokument" },
        },
        required: ["companyId", "description", "debitAccount", "creditAccount", "amount"],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "post_document",
      description: "Bokfør et dokument med foreslått kontering",
      parameters: {
        type: "object",
        properties: {
          documentId: { type: "number", description: "ID til dokumentet" },
          account: { type: "string", description: "Kontonummer" },
          vatCode: { type: "string", description: "MVA-kode" },
          description: { type: "string", description: "Beskrivelse" },
        },
        required: ["documentId", "account", "vatCode"],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "generate_mva",
      description: "Generer MVA-melding utkast for en periode",
      parameters: {
        type: "object",
        properties: {
          companyId: { type: "number", description: "ID til selskapet" },
          year: { type: "number", description: "År" },
          term: { type: "number", description: "Termin (1-6 for annenhver måned)" },
        },
        required: ["companyId", "year", "term"],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "fetch_forvalt",
      description: "Hent kredittinformasjon fra Forvalt for et organisasjonsnummer",
      parameters: {
        type: "object",
        properties: {
          orgNumber: { type: "string", description: "Organisasjonsnummer (9 siffer)" },
        },
        required: ["orgNumber"],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "generate_document",
      description: "Generer et dokument fra en mal med variabler",
      parameters: {
        type: "object",
        properties: {
          templateId: { type: "number", description: "ID til malen" },
          companyId: { type: "number", description: "ID til selskapet" },
          variables: { type: "object", description: "Variabler for malen" },
          title: { type: "string", description: "Tittel på dokumentet" },
        },
        required: ["templateId", "companyId", "variables", "title"],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "assess_risk",
      description: "Vurder kredittrisiko for en kunde eller leverandør",
      parameters: {
        type: "object",
        properties: {
          orgNumber: { type: "string", description: "Organisasjonsnummer" },
          outstandingAmount: { type: "number", description: "Utestående beløp i øre" },
        },
        required: ["orgNumber"],
      },
    },
  },
];

export interface AgentMessage {
  role: "user" | "assistant" | "system" | "tool";
  content: string;
  tool_call_id?: string;
  name?: string;
}

export interface ToolCall {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
}

export interface AgentResponse {
  message: string;
  toolCalls?: ToolCall[];
  toolResults?: Array<{ name: string; result: unknown; success: boolean }>;
  actions?: AgentAction[];
  metadata?: Record<string, unknown>;
}

export interface AgentAction {
  type: "VOUCHER_CREATED" | "DOCUMENT_POSTED" | "MVA_GENERATED" | "FORVALT_FETCHED" | "DOCUMENT_GENERATED" | "RISK_ASSESSED";
  label: string;
  params?: Record<string, unknown>;
  result?: unknown;
  success: boolean;
}

export class SkattFlowAgentV2 {
  private aimlClient: AIMLClient;
  private forvaltClient: ForvaltClient;
  private regnskapClient: RegnskapClient;
  private conversationHistory: AgentMessage[] = [];

  constructor() {
    this.aimlClient = createAIMLClient();
    this.forvaltClient = createForvaltClient();
    this.regnskapClient = createRegnskapClient();
  }

  /**
   * Execute a tool call
   */
  private async executeTool(name: string, args: Record<string, unknown>): Promise<{ result: unknown; success: boolean }> {
    console.log(`[SkattFlowAgentV2] Executing tool: ${name}`, args);

    try {
      switch (name) {
        case "create_voucher": {
          const entryId = await db.createLedgerEntry({
            companyId: args.companyId as number,
            description: args.description as string,
            debitAccount: args.debitAccount as string,
            creditAccount: args.creditAccount as string,
            amount: args.amount as number,
            vatCode: args.vatCode as string,
            entryDate: new Date(),
          });
          return { result: { entryId, message: `Bilag opprettet med ID ${entryId}` }, success: true };
        }

        case "post_document": {
          await db.updateAccountingDocument(args.documentId as number, {
            status: "POSTED",
            suggestedAccount: args.account as string,
            suggestedVatCode: args.vatCode as string,
            suggestedDescription: args.description as string,
            postedAt: new Date(),
          });
          return { result: { documentId: args.documentId, message: "Dokument bokført" }, success: true };
        }

        case "generate_mva": {
          const year = args.year as number;
          const term = args.term as number;
          const periodStart = new Date(year, (term - 1) * 2, 1);
          const periodEnd = new Date(year, term * 2, 0);

          const filingId = await db.createFiling({
            companyId: args.companyId as number,
            filingType: "MVA_MELDING",
            periodStart,
            periodEnd,
            status: "DRAFT",
            createdById: 1, // System user
          });

          return { result: { filingId, message: `MVA-melding utkast opprettet for termin ${term}/${year}` }, success: true };
        }

        case "fetch_forvalt": {
          const profile = await this.forvaltClient.getFullProfile(args.orgNumber as string);
          return { result: profile, success: true };
        }

        case "generate_document": {
          const template = await db.getDocumentTemplateById(args.templateId as number);
          if (!template) {
            return { result: { error: "Mal ikke funnet" }, success: false };
          }

          const variables = args.variables as Record<string, string>;
          const generated = await this.aimlClient.draftDocumentFromTemplate(
            template.bodyMarkdown,
            variables,
            template.language
          );

          const docId = await db.createGeneratedDocument({
            companyId: args.companyId as number,
            templateId: args.templateId as number,
            title: args.title as string,
            filledVariablesJson: variables,
            outputMarkdown: generated.outputMarkdown,
          });

          return { result: { documentId: docId, message: `Dokument generert: ${args.title}` }, success: true };
        }

        case "assess_risk": {
          const credit = await this.forvaltClient.getCredit(args.orgNumber as string);
          let riskLevel = "Lav";
          let recommendation = "Normale betalingsbetingelser";

          if (credit.riskClass === "HIGH" || (credit.creditScore && credit.creditScore < 30)) {
            riskLevel = "Høy";
            recommendation = "Forskuddsbetaling eller redusert kreditt";
          } else if (credit.riskClass === "MEDIUM" || (credit.creditScore && credit.creditScore < 60)) {
            riskLevel = "Middels";
            recommendation = "Kortere betalingsfrist (14 dager)";
          }

          return {
            result: {
              orgNumber: args.orgNumber,
              riskLevel,
              creditScore: credit.creditScore,
              riskClass: credit.riskClass,
              paymentRemarks: credit.paymentRemarks,
              recommendation,
            },
            success: true,
          };
        }

        default:
          return { result: { error: `Ukjent verktøy: ${name}` }, success: false };
      }
    } catch (error) {
      console.error(`[SkattFlowAgentV2] Tool execution error:`, error);
      return { result: { error: error instanceof Error ? error.message : "Ukjent feil" }, success: false };
    }
  }

  /**
   * Process a user message with tool calling capability
   */
  async chat(
    userMessage: string,
    context?: ChatContext,
    sessionHistory?: AgentMessage[],
    userId?: number
  ): Promise<AgentResponse> {
    const history = sessionHistory || this.conversationHistory;

    // Build context information
    let contextInfo = "";
    if (context?.companyId) {
      contextInfo += `\nAktivt selskap ID: ${context.companyId}`;
    }
    if (context?.documentId) {
      contextInfo += `\nTilknyttet dokument ID: ${context.documentId}`;
    }
    if (context?.filingId) {
      contextInfo += `\nTilknyttet innlevering ID: ${context.filingId}`;
    }

    // Build messages for LLM
    const messages: Array<{ role: "system" | "user" | "assistant" | "tool"; content: string; tool_call_id?: string; name?: string }> = [
      { role: "system", content: SYSTEM_PROMPT + contextInfo },
    ];

    // Add conversation history (last 10 messages)
    for (const msg of history.slice(-10)) {
      if (msg.role === "tool") {
        messages.push({ role: "tool", content: msg.content, tool_call_id: msg.tool_call_id, name: msg.name });
      } else {
        messages.push({ role: msg.role as "user" | "assistant", content: msg.content });
      }
    }

    // Add current user message
    messages.push({ role: "user", content: userMessage });

    try {
      // Call LLM with tools
      const response = await invokeLLM({
        messages,
        tools: TOOLS,
        tool_choice: "auto",
      });

      const choice = response.choices?.[0];
      const assistantMessage = choice?.message;

      // Check if the model wants to call tools
      if (assistantMessage?.tool_calls && assistantMessage.tool_calls.length > 0) {
        const toolResults: Array<{ name: string; result: unknown; success: boolean }> = [];
        const actions: AgentAction[] = [];

        // Execute each tool call
        for (const toolCall of assistantMessage.tool_calls) {
          const funcName = toolCall.function.name;
          const funcArgs = JSON.parse(toolCall.function.arguments);

          const { result, success } = await this.executeTool(funcName, funcArgs);
          toolResults.push({ name: funcName, result, success });

          // Map tool name to action type
          const actionTypeMap: Record<string, AgentAction["type"]> = {
            create_voucher: "VOUCHER_CREATED",
            post_document: "DOCUMENT_POSTED",
            generate_mva: "MVA_GENERATED",
            fetch_forvalt: "FORVALT_FETCHED",
            generate_document: "DOCUMENT_GENERATED",
            assess_risk: "RISK_ASSESSED",
          };

          actions.push({
            type: actionTypeMap[funcName] || "VOUCHER_CREATED",
            label: `${funcName}: ${success ? "Vellykket" : "Feilet"}`,
            params: funcArgs,
            result,
            success,
          });

          // Add tool result to history for follow-up
          this.conversationHistory.push({
            role: "tool",
            content: JSON.stringify(result),
            tool_call_id: toolCall.id,
            name: funcName,
          });
        }

        // Get follow-up response from LLM with tool results
        const followUpMessages = [
          ...messages,
          {
            role: "assistant" as const,
            content: assistantMessage.content || "",
          },
          ...toolResults.map((tr, i) => ({
            role: "tool" as const,
            content: JSON.stringify(tr.result),
            tool_call_id: assistantMessage.tool_calls![i].id,
            name: tr.name,
          })),
        ];

        const followUpResponse = await invokeLLM({ messages: followUpMessages });
        const followUpContent = followUpResponse.choices?.[0]?.message?.content;
        const finalMessage = typeof followUpContent === "string" ? followUpContent : "Handlinger utført.";

        // Update history
        this.conversationHistory.push({ role: "user", content: userMessage });
        this.conversationHistory.push({ role: "assistant", content: finalMessage });

        // Log to audit
        if (userId && context?.companyId) {
          await db.createApiLog({
            companyId: context.companyId,
            userId,
            endpoint: "agent/tool_calls",
            method: "POST",
            statusCode: 200,
            correlationId: `agent-${Date.now()}`,
            durationMs: 0,
          });
        }

        return {
          message: finalMessage,
          toolCalls: assistantMessage.tool_calls,
          toolResults,
          actions,
          metadata: {
            model: response.model,
            usage: response.usage,
            toolsExecuted: toolResults.length,
          },
        };
      }

      // No tool calls - regular response
      const rawContent = assistantMessage?.content;
      const responseMessage = typeof rawContent === "string" ? rawContent : "";

      // Update history
      this.conversationHistory.push({ role: "user", content: userMessage });
      this.conversationHistory.push({ role: "assistant", content: responseMessage });

      return {
        message: responseMessage,
        actions: [],
        metadata: {
          model: response.model,
          usage: response.usage,
        },
      };
    } catch (error) {
      console.error("[SkattFlowAgentV2] Chat error:", error);
      return {
        message: "Beklager, jeg kunne ikke behandle forespørselen din. Vennligst prøv igjen.",
        actions: [],
      };
    }
  }

  /**
   * Process an uploaded document with OCR support
   */
  async processDocumentWithOCR(
    documentUrl: string,
    documentType: "INVOICE_SUPPLIER" | "INVOICE_CUSTOMER" | "RECEIPT" | "CONTRACT" | "OTHER",
    isScanned: boolean = false
  ): Promise<{
    extractedFields: ExtractedInvoiceFields;
    suggestedPosting: {
      account: string;
      vatCode: string;
      description: string;
    };
    confidence: number;
  }> {
    let documentContent: string;

    if (isScanned) {
      // Use vision model for scanned documents
      const ocrResponse = await invokeLLM({
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "Ekstraher all tekst og informasjon fra dette dokumentet. Returner strukturert JSON med fakturainformasjon." },
              { type: "image_url", image_url: { url: documentUrl } },
            ],
          },
        ],
      });
      const rawContent = ocrResponse.choices?.[0]?.message?.content;
      documentContent = typeof rawContent === "string" ? rawContent : "";
    } else {
      // For text-based documents, fetch content
      documentContent = `Document URL: ${documentUrl}`;
    }

    // Extract fields using AIML client
    const extractedFields = await this.aimlClient.extractInvoiceFields(documentContent);

    // Get posting suggestion
    const description = extractedFields.supplierName
      ? `Faktura fra ${extractedFields.supplierName}`
      : `${documentType} dokument`;

    const classification = await this.aimlClient.classifyTransaction(
      description,
      extractedFields.totalAmount || 0,
      extractedFields.currency || "NOK"
    );

    return {
      extractedFields,
      suggestedPosting: {
        account: classification.suggestedAccount,
        vatCode: classification.suggestedVatCode,
        description: classification.reasoning,
      },
      confidence: (extractedFields.confidence + classification.confidence) / 2,
    };
  }

  /**
   * Autonomous MVA filing workflow
   */
  async autonomousMVAFiling(
    companyId: number,
    year: number,
    term: number,
    autoSubmit: boolean = false
  ): Promise<{
    filingId: number;
    status: string;
    summary: string;
    warnings: string[];
    submitted: boolean;
  }> {
    // Step 1: Get company
    const company = await db.getCompanyById(companyId);
    if (!company) {
      throw new Error("Selskap ikke funnet");
    }

    // Step 2: Calculate period
    const periodStart = new Date(year, (term - 1) * 2, 1);
    const periodEnd = new Date(year, term * 2, 0);

    // Step 3: Fetch ledger entries
    const entries = await db.listLedgerEntries(companyId, periodStart, periodEnd);

    // Step 4: Calculate MVA totals
    let totalSales = 0;
    let totalPurchases = 0;
    let outputVat = 0;
    let inputVat = 0;

    for (const entry of entries) {
      const amount = entry.amount;
      const account = entry.debitAccount;

      // Sales accounts (3xxx)
      if (account.startsWith("3")) {
        totalSales += amount;
        if (entry.vatCode === "3") outputVat += Math.round(amount * 0.25);
        else if (entry.vatCode === "31") outputVat += Math.round(amount * 0.15);
        else if (entry.vatCode === "32") outputVat += Math.round(amount * 0.12);
      }

      // Purchase accounts (4xxx-6xxx)
      if (account.startsWith("4") || account.startsWith("5") || account.startsWith("6")) {
        totalPurchases += amount;
        if (entry.vatCode === "1") inputVat += Math.round(amount * 0.25);
      }
    }

    const vatPayable = outputVat - inputVat;

    // Step 5: Create filing
    const filingId = await db.createFiling({
      companyId,
      filingType: "MVA_MELDING",
      periodStart,
      periodEnd,
      status: autoSubmit ? "READY_FOR_REVIEW" : "DRAFT",
      payloadJson: {
        orgNumber: company.orgNumber,
        period: { year, term, type: term <= 6 ? "BIMONTHLY" : "QUARTERLY" },
        totals: { totalSales, totalPurchases, outputVat, inputVat, vatPayable },
      },
      summaryJson: {
        totalSales: totalSales / 100,
        totalPurchases: totalPurchases / 100,
        outputVat: outputVat / 100,
        inputVat: inputVat / 100,
        vatPayable: vatPayable / 100,
        currency: "NOK",
      },
    });

    // Step 6: Generate warnings
    const warnings: string[] = [];
    if (entries.length === 0) {
      warnings.push("Ingen posteringer funnet i perioden");
    }
    if (vatPayable < 0) {
      warnings.push("MVA til gode - sjekk at dette er korrekt");
    }
    if (totalSales === 0 && totalPurchases > 0) {
      warnings.push("Ingen omsetning registrert - kun utgifter");
    }

    const summary = `MVA-melding for ${company.name} termin ${term}/${year}:
- Omsetning: ${(totalSales / 100).toLocaleString("nb-NO")} NOK
- Innkjøp: ${(totalPurchases / 100).toLocaleString("nb-NO")} NOK
- Utgående MVA: ${(outputVat / 100).toLocaleString("nb-NO")} NOK
- Inngående MVA: ${(inputVat / 100).toLocaleString("nb-NO")} NOK
- MVA til betaling: ${(vatPayable / 100).toLocaleString("nb-NO")} NOK`;

    return {
      filingId,
      status: autoSubmit ? "READY_FOR_REVIEW" : "DRAFT",
      summary,
      warnings,
      submitted: false,
    };
  }

  /**
   * Clear conversation history
   */
  clearHistory(): void {
    this.conversationHistory = [];
  }
}

// Singleton instance
let agentInstance: SkattFlowAgentV2 | null = null;

export function getSkattFlowAgentV2(): SkattFlowAgentV2 {
  if (!agentInstance) {
    agentInstance = new SkattFlowAgentV2();
  }
  return agentInstance;
}
