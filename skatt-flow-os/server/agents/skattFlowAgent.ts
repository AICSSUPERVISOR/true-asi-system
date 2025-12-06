import { invokeLLM } from "../_core/llm";
import { createAIMLClient, AIMLClient } from "../clients/aimlClient";
import { createForvaltClient, ForvaltClient } from "../clients/forvaltClient";
import { createRegnskapClient, RegnskapClient } from "../clients/regnskapClient";
import type { ExtractedInvoiceFields, MVAMeldingPayload, ChatContext } from "@shared/types";

// ============================================================================
// SKATT-FLOW AI AGENT
// Autonomous accounting, tax, and audit assistant for Norwegian businesses
// ============================================================================

const SYSTEM_PROMPT = `Du er "Skatt-Flow", en autonom regnskaps-, skatte- og revisjonsassistent spesialisert på norske regler.

Du må:
- Følge bokføringsloven, bokføringsforskriften, skatteforvaltningsloven, A-ordningen, SAF-T veiledning, og Skatteetaten/Altinn beste praksis.
- Alltid produsere output som ville bestått en norsk skatterevisjon.
- Arbeide på norsk når brukeren er norsk, ellers på engelsk.
- Aldri gjette om lovlig etterlevelse når data mangler: still i stedet målrettede oppfølgingsspørsmål.

Dine hovedoppgaver:
1. Analysere og klassifisere regnskapsdokumenter (fakturaer, kvitteringer, kontrakter)
2. Foreslå korrekt kontonummer, MVA-kode, dimensjoner og beskrivelse
3. Generere utkast til MVA-melding og SAF-T fra hovedbokdata
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

Svar alltid strukturert med:
- Klar anbefaling
- Begrunnelse med lovhenvisning der relevant
- Eventuelle risikoer eller forbehold
- Neste steg for brukeren`;

export interface AgentMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

export interface AgentResponse {
  message: string;
  actions?: AgentAction[];
  metadata?: Record<string, unknown>;
}

export interface AgentAction {
  type: "SUGGEST_POSTING" | "GENERATE_MVA" | "RISK_REPORT" | "GENERATE_DOCUMENT" | "FETCH_FORVALT" | "CREATE_VOUCHER";
  label: string;
  params?: Record<string, unknown>;
  completed?: boolean;
}

export class SkattFlowAgent {
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
   * Process a user message and generate a response
   */
  async chat(
    userMessage: string,
    context?: ChatContext,
    sessionHistory?: AgentMessage[]
  ): Promise<AgentResponse> {
    // Use provided history or internal history
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
    const messages: Array<{ role: "system" | "user" | "assistant"; content: string }> = [
      { role: "system", content: SYSTEM_PROMPT + contextInfo },
    ];

    // Add conversation history
    for (const msg of history.slice(-10)) {
      messages.push({ role: msg.role as "user" | "assistant", content: msg.content });
    }

    // Add current user message
    messages.push({ role: "user", content: userMessage });

    try {
      // Call LLM
      const response = await invokeLLM({ messages });
      const rawContent = response.choices?.[0]?.message?.content;
      const assistantMessage = typeof rawContent === 'string' ? rawContent : '';

      // Update history
      this.conversationHistory.push({ role: "user", content: userMessage });
      this.conversationHistory.push({ role: "assistant", content: assistantMessage });

      // Detect suggested actions from the response
      const actions = this.detectActions(userMessage, assistantMessage);

      return {
        message: assistantMessage,
        actions,
        metadata: {
          model: response.model,
          usage: response.usage,
        },
      };
    } catch (error) {
      console.error("[SkattFlowAgent] Chat error:", error);
      return {
        message: "Beklager, jeg kunne ikke behandle forespørselen din. Vennligst prøv igjen.",
        actions: [],
      };
    }
  }

  /**
   * Process an uploaded document and extract fields
   */
  async processDocument(
    documentContent: string,
    documentType: "INVOICE_SUPPLIER" | "INVOICE_CUSTOMER" | "RECEIPT" | "CONTRACT" | "OTHER"
  ): Promise<{
    extractedFields: ExtractedInvoiceFields;
    suggestedPosting: {
      account: string;
      vatCode: string;
      description: string;
    };
    confidence: number;
  }> {
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
   * Generate MVA-melding draft from ledger data
   */
  async generateMVADraft(
    companyId: string,
    periodStart: string,
    periodEnd: string,
    year: number,
    term: number
  ): Promise<{
    payload: MVAMeldingPayload;
    summary: string;
    warnings: string[];
  }> {
    try {
      // Fetch ledger data
      const ledgerData = await this.regnskapClient.getLedger(companyId, periodStart, periodEnd);

      // Convert to string for AI analysis
      const ledgerSummary = ledgerData.entries
        .map((e) => `${e.date}: ${e.description} - ${e.debitAccount}/${e.creditAccount} - ${e.amount} ${e.vatCode || ""}`)
        .join("\n");

      // Generate MVA summary using AI
      const mvaResult = await this.aimlClient.generateMVASummary(ledgerSummary, { year, term });

      const payload: MVAMeldingPayload = {
        orgNumber: "", // Will be filled by caller
        period: {
          year,
          term,
          type: term <= 6 ? "BIMONTHLY" : "QUARTERLY",
        },
        totals: mvaResult.totals,
        details: [],
      };

      return {
        payload,
        summary: mvaResult.summary,
        warnings: [],
      };
    } catch (error) {
      console.error("[SkattFlowAgent] MVA generation error:", error);
      throw error;
    }
  }

  /**
   * Enrich company data with Forvalt information
   */
  async enrichCompanyData(orgNumber: string): Promise<{
    company: Record<string, unknown>;
    credit: Record<string, unknown>;
    financials: Record<string, unknown>;
    riskAssessment: string;
  }> {
    const profile = await this.forvaltClient.getFullProfile(orgNumber);

    // Generate risk assessment
    let riskAssessment = "Lav risiko";
    if (profile.credit.riskClass === "HIGH" || (profile.credit.creditScore && profile.credit.creditScore < 30)) {
      riskAssessment = "Høy risiko - Anbefaler kortere betalingsbetingelser eller forskuddsbetaling";
    } else if (profile.credit.riskClass === "MEDIUM" || (profile.credit.creditScore && profile.credit.creditScore < 60)) {
      riskAssessment = "Middels risiko - Vurder kredittgrense og oppfølging";
    }

    if (profile.credit.paymentRemarks && profile.credit.paymentRemarks > 0) {
      riskAssessment += ` (${profile.credit.paymentRemarks} betalingsanmerkninger registrert)`;
    }

    return {
      company: profile.company as unknown as Record<string, unknown>,
      credit: profile.credit as unknown as Record<string, unknown>,
      financials: profile.financials as unknown as Record<string, unknown>,
      riskAssessment,
    };
  }

  /**
   * Generate a document from template
   */
  async generateDocument(
    templateMarkdown: string,
    variables: Record<string, string>,
    language: string = "no"
  ): Promise<{ outputMarkdown: string; confidence: number }> {
    return this.aimlClient.draftDocumentFromTemplate(templateMarkdown, variables, language);
  }

  /**
   * Answer compliance questions
   */
  async askCompliance(
    context: string,
    question: string
  ): Promise<{ answer: string; references: string[]; confidence: number }> {
    return this.aimlClient.reasonAboutCompliance(context, question);
  }

  /**
   * Validate SAF-T export
   */
  async validateSAFT(saftSummary: string): Promise<{
    isValid: boolean;
    issues: { type: "error" | "warning"; message: string }[];
  }> {
    return this.aimlClient.validateSAFT(saftSummary);
  }

  /**
   * Generate risk report for top customers
   */
  async generateRiskReport(
    customers: Array<{ name: string; orgNumber: string; outstandingAmount: number }>
  ): Promise<{
    report: string;
    highRiskCustomers: Array<{ name: string; orgNumber: string; riskLevel: string; recommendation: string }>;
  }> {
    const highRiskCustomers: Array<{ name: string; orgNumber: string; riskLevel: string; recommendation: string }> = [];

    for (const customer of customers.slice(0, 20)) {
      try {
        const credit = await this.forvaltClient.getCredit(customer.orgNumber);
        let riskLevel = "Lav";
        let recommendation = "Normale betalingsbetingelser";

        if (credit.riskClass === "HIGH" || (credit.creditScore && credit.creditScore < 30)) {
          riskLevel = "Høy";
          recommendation = "Forskuddsbetaling eller redusert kreditt";
        } else if (credit.riskClass === "MEDIUM" || (credit.creditScore && credit.creditScore < 60)) {
          riskLevel = "Middels";
          recommendation = "Kortere betalingsfrist (14 dager)";
        }

        if (credit.paymentRemarks && credit.paymentRemarks > 0) {
          riskLevel = "Høy";
          recommendation = `${credit.paymentRemarks} betalingsanmerkninger - Krever forskuddsbetaling`;
        }

        highRiskCustomers.push({
          name: customer.name,
          orgNumber: customer.orgNumber,
          riskLevel,
          recommendation,
        });
      } catch (error) {
        console.warn(`[SkattFlowAgent] Could not fetch credit for ${customer.orgNumber}:`, error);
      }
    }

    const highRiskCount = highRiskCustomers.filter((c) => c.riskLevel === "Høy").length;
    const report = `# Risikorapport for kunder\n\n` +
      `Analysert ${highRiskCustomers.length} kunder.\n` +
      `**${highRiskCount} kunder med høy risiko.**\n\n` +
      `## Anbefalinger\n` +
      highRiskCustomers
        .filter((c) => c.riskLevel !== "Lav")
        .map((c) => `- **${c.name}** (${c.orgNumber}): ${c.riskLevel} risiko - ${c.recommendation}`)
        .join("\n");

    return { report, highRiskCustomers };
  }

  /**
   * Detect suggested actions from conversation
   */
  private detectActions(userMessage: string, assistantMessage: string): AgentAction[] {
    const actions: AgentAction[] = [];
    const lowerUser = userMessage.toLowerCase();
    const lowerAssistant = assistantMessage.toLowerCase();

    if (lowerUser.includes("bokfør") || lowerUser.includes("poster") || lowerAssistant.includes("foreslår bokføring")) {
      actions.push({
        type: "SUGGEST_POSTING",
        label: "Foreslå bokføring for dokumenter",
      });
    }

    if (lowerUser.includes("mva") || lowerUser.includes("merverdiavgift") || lowerAssistant.includes("mva-melding")) {
      actions.push({
        type: "GENERATE_MVA",
        label: "Generer MVA-melding utkast",
      });
    }

    if (lowerUser.includes("risiko") || lowerUser.includes("kreditt") || lowerAssistant.includes("risikorapport")) {
      actions.push({
        type: "RISK_REPORT",
        label: "Generer risikorapport",
      });
    }

    if (lowerUser.includes("dokument") || lowerUser.includes("kontrakt") || lowerUser.includes("mal")) {
      actions.push({
        type: "GENERATE_DOCUMENT",
        label: "Generer dokument fra mal",
      });
    }

    return actions;
  }

  /**
   * Clear conversation history
   */
  clearHistory(): void {
    this.conversationHistory = [];
  }
}

// Singleton instance
let agentInstance: SkattFlowAgent | null = null;

export function getSkattFlowAgent(): SkattFlowAgent {
  if (!agentInstance) {
    agentInstance = new SkattFlowAgent();
  }
  return agentInstance;
}
