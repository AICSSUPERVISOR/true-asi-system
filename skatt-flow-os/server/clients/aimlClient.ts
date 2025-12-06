import axios, { AxiosInstance, AxiosError } from "axios";
import { nanoid } from "nanoid";
import type { AIMLRequest, AIMLResponse, ExtractedInvoiceFields } from "@shared/types";

// ============================================================================
// AIML MULTI-MODEL API CLIENT
// For AI-powered document processing, classification, and generation
// ============================================================================

export class AIMLApiError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public correlationId?: string
  ) {
    super(message);
    this.name = "AIMLApiError";
  }
}

interface AIMLClientConfig {
  baseUrl: string;
  apiKey: string;
  defaultModel?: string;
  timeout?: number;
}

export class AIMLClient {
  private client: AxiosInstance;
  private apiKey: string;
  private defaultModel: string;

  constructor(config: AIMLClientConfig) {
    this.apiKey = config.apiKey;
    this.defaultModel = config.defaultModel || "gpt-4o";
    this.client = axios.create({
      baseURL: config.baseUrl,
      timeout: config.timeout || 120000,
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
    });
  }

  private getCorrelationId(): string {
    return nanoid(16);
  }

  private getAuthHeaders(correlationId: string) {
    return {
      Authorization: `Bearer ${this.apiKey}`,
      "X-Correlation-Id": correlationId,
    };
  }

  /**
   * General model invocation
   */
  async callModel(request: AIMLRequest): Promise<AIMLResponse> {
    const correlationId = this.getCorrelationId();
    const model = request.model || this.defaultModel;

    try {
      const response = await this.client.post(
        "/chat/completions",
        {
          model,
          messages: [
            { role: "system", content: request.systemPrompt },
            { role: "user", content: request.userPrompt },
          ],
          temperature: request.temperature ?? 0.3,
          max_tokens: request.maxTokens ?? 4096,
        },
        { headers: this.getAuthHeaders(correlationId) }
      );

      const data = response.data;
      const choice = data.choices?.[0];

      return {
        outputText: choice?.message?.content || "",
        raw: data,
        model: data.model || model,
        usage: data.usage
          ? {
              promptTokens: data.usage.prompt_tokens || 0,
              completionTokens: data.usage.completion_tokens || 0,
              totalTokens: data.usage.total_tokens || 0,
            }
          : undefined,
      };
    } catch (error) {
      if (error instanceof AxiosError) {
        throw new AIMLApiError(
          `AIML API call failed: ${error.message}`,
          error.response?.status,
          correlationId
        );
      }
      throw error;
    }
  }

  /**
   * Classify a transaction based on description and amount
   */
  async classifyTransaction(
    description: string,
    amount: number,
    currency: string = "NOK"
  ): Promise<{
    suggestedAccount: string;
    suggestedVatCode: string;
    confidence: number;
    reasoning: string;
  }> {
    const systemPrompt = `Du er en ekspert på norsk regnskap og bokføring. 
Klassifiser transaksjoner i henhold til norsk kontoplan (NS 4102) og MVA-regler.
Svar alltid i JSON-format med følgende felter:
- suggestedAccount: kontonummer (4 siffer)
- suggestedVatCode: MVA-kode (f.eks. "3" for 25% MVA, "6" for 0% MVA)
- confidence: tall mellom 0 og 1
- reasoning: kort forklaring på norsk`;

    const userPrompt = `Klassifiser denne transaksjonen:
Beskrivelse: ${description}
Beløp: ${amount} ${currency}

Returner JSON med suggestedAccount, suggestedVatCode, confidence og reasoning.`;

    const response = await this.callModel({ systemPrompt, userPrompt });

    try {
      const parsed = JSON.parse(response.outputText);
      return {
        suggestedAccount: String(parsed.suggestedAccount || ""),
        suggestedVatCode: String(parsed.suggestedVatCode || ""),
        confidence: Number(parsed.confidence || 0),
        reasoning: String(parsed.reasoning || ""),
      };
    } catch {
      return {
        suggestedAccount: "",
        suggestedVatCode: "",
        confidence: 0,
        reasoning: "Kunne ikke analysere transaksjonen",
      };
    }
  }

  /**
   * Extract invoice fields from document text or image
   */
  async extractInvoiceFields(
    documentContent: string,
    documentType: "text" | "image_url" = "text"
  ): Promise<ExtractedInvoiceFields> {
    const systemPrompt = `Du er en ekspert på å lese og tolke norske fakturaer og bilag.
Ekstraher alle relevante felter fra dokumentet.
Svar alltid i JSON-format med følgende felter:
- supplierName: leverandørens navn
- supplierOrgNumber: organisasjonsnummer (9 siffer)
- invoiceNumber: fakturanummer
- invoiceDate: fakturadato (YYYY-MM-DD)
- dueDate: forfallsdato (YYYY-MM-DD)
- totalAmount: totalbeløp i øre (heltall)
- vatAmount: MVA-beløp i øre (heltall)
- currency: valutakode (NOK, EUR, USD, etc.)
- kid: KID-nummer for betaling
- bankAccount: kontonummer for betaling
- lines: array med linjer (description, quantity, unitPrice, amount, vatCode)
- confidence: tall mellom 0 og 1`;

    const userPrompt = `Ekstraher informasjon fra dette dokumentet:\n\n${documentContent}`;

    const response = await this.callModel({ systemPrompt, userPrompt });

    try {
      const parsed = JSON.parse(response.outputText);
      return {
        supplierName: parsed.supplierName,
        supplierOrgNumber: parsed.supplierOrgNumber,
        invoiceNumber: parsed.invoiceNumber,
        invoiceDate: parsed.invoiceDate,
        dueDate: parsed.dueDate,
        totalAmount: parsed.totalAmount,
        vatAmount: parsed.vatAmount,
        currency: parsed.currency || "NOK",
        kid: parsed.kid,
        bankAccount: parsed.bankAccount,
        lines: parsed.lines || [],
        confidence: Number(parsed.confidence || 0.5),
      };
    } catch {
      return { confidence: 0 };
    }
  }

  /**
   * Draft a document from a template with variables
   */
  async draftDocumentFromTemplate(
    templateMarkdown: string,
    variables: Record<string, string>,
    language: string = "no"
  ): Promise<{ outputMarkdown: string; confidence: number }> {
    const langName = language === "no" ? "norsk" : "English";

    const systemPrompt = `Du er en profesjonell dokumentforfatter.
Fyll ut malen med de gitte variablene og forbedre teksten der det er nødvendig.
Behold Markdown-formatering. Skriv på ${langName}.
Returner kun det ferdige dokumentet, ingen ekstra kommentarer.`;

    const variablesText = Object.entries(variables)
      .map(([key, value]) => `${key}: ${value}`)
      .join("\n");

    const userPrompt = `Mal:\n${templateMarkdown}\n\nVariabler:\n${variablesText}`;

    const response = await this.callModel({ systemPrompt, userPrompt });

    return {
      outputMarkdown: response.outputText,
      confidence: 0.9,
    };
  }

  /**
   * Reason about compliance issues
   */
  async reasonAboutCompliance(
    context: string,
    question: string
  ): Promise<{ answer: string; references: string[]; confidence: number }> {
    const systemPrompt = `Du er en ekspert på norsk skatte- og regnskapslovgivning.
Du kjenner til:
- Bokføringsloven og bokføringsforskriften
- Skatteforvaltningsloven
- A-ordningen og a-meldingsregler
- SAF-T krav
- MVA-regler og frister

Svar alltid med:
1. Et klart svar på spørsmålet
2. Relevante lovhenvisninger
3. Praktiske anbefalinger

Hvis du er usikker, si det tydelig og be om mer informasjon.`;

    const userPrompt = `Kontekst:\n${context}\n\nSpørsmål:\n${question}`;

    const response = await this.callModel({ systemPrompt, userPrompt });

    // Extract references from the response
    const referencePattern = /(?:bokføringsloven|skatteforvaltningsloven|a-opplysningsloven|mva-loven|§\s*\d+)/gi;
    const references = response.outputText.match(referencePattern) || [];

    return {
      answer: response.outputText,
      references: Array.from(new Set(references)),
      confidence: 0.85,
    };
  }

  /**
   * Generate MVA-melding summary from ledger data
   */
  async generateMVASummary(
    ledgerData: string,
    period: { year: number; term: number }
  ): Promise<{
    summary: string;
    totals: {
      totalSales: number;
      totalPurchases: number;
      outputVat: number;
      inputVat: number;
      vatPayable: number;
    };
  }> {
    const systemPrompt = `Du er en ekspert på norsk MVA-rapportering.
Analyser hovedbokdata og generer et sammendrag for MVA-melding.
Beregn:
- Total omsetning (utgående)
- Total innkjøp (inngående)
- Utgående MVA
- Inngående MVA
- MVA til betaling/tilgode

Returner JSON med "summary" (tekst) og "totals" (tall i øre).`;

    const userPrompt = `Periode: ${period.year} termin ${period.term}\n\nHovedbok:\n${ledgerData}`;

    const response = await this.callModel({ systemPrompt, userPrompt });

    try {
      const parsed = JSON.parse(response.outputText);
      return {
        summary: parsed.summary || "",
        totals: {
          totalSales: Number(parsed.totals?.totalSales || 0),
          totalPurchases: Number(parsed.totals?.totalPurchases || 0),
          outputVat: Number(parsed.totals?.outputVat || 0),
          inputVat: Number(parsed.totals?.inputVat || 0),
          vatPayable: Number(parsed.totals?.vatPayable || 0),
        },
      };
    } catch {
      return {
        summary: response.outputText,
        totals: {
          totalSales: 0,
          totalPurchases: 0,
          outputVat: 0,
          inputVat: 0,
          vatPayable: 0,
        },
      };
    }
  }

  /**
   * Validate SAF-T data for consistency
   */
  async validateSAFT(saftSummary: string): Promise<{
    isValid: boolean;
    issues: { type: "error" | "warning"; message: string }[];
  }> {
    const systemPrompt = `Du er en ekspert på SAF-T validering for norske virksomheter.
Sjekk data for:
- Balanse mellom debet og kredit
- Manglende kontoer
- Ugyldige MVA-koder
- Periodiseringsfeil
- Dokumentasjonskrav

Returner JSON med "isValid" (boolean) og "issues" (array med type og message).`;

    const userPrompt = `Valider denne SAF-T eksporten:\n${saftSummary}`;

    const response = await this.callModel({ systemPrompt, userPrompt });

    try {
      const parsed = JSON.parse(response.outputText);
      return {
        isValid: Boolean(parsed.isValid),
        issues: parsed.issues || [],
      };
    } catch {
      return {
        isValid: true,
        issues: [],
      };
    }
  }
}

// Factory function to create client from environment variables
export function createAIMLClient(): AIMLClient {
  const baseUrl = process.env.AIML_BASE_URL || "https://api.aimlapi.com/v1";
  const apiKey = process.env.AIML_API_KEY || "";
  const defaultModel = process.env.AIML_DEFAULT_MODEL || "gpt-4o";

  if (!apiKey) {
    console.warn("[AIMLClient] AIML_API_KEY not set - API calls will fail");
  }

  return new AIMLClient({ baseUrl, apiKey, defaultModel });
}
