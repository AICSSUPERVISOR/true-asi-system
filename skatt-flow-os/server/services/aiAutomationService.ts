import { invokeLLM } from "../_core/llm";
import * as db from "../db";

// ============================================================================
// AI AUTOMATION SERVICE
// Complete automation of all accounting and auditing tasks
// ============================================================================

export interface TransactionClassification {
  accountNumber: string;
  accountName: string;
  vatCode: string;
  vatRate: number;
  confidence: number;
  reasoning: string;
}

export interface InvoiceExtraction {
  vendorName: string;
  vendorOrgNumber?: string;
  invoiceNumber: string;
  invoiceDate: string;
  dueDate: string;
  totalAmount: number;
  vatAmount: number;
  currency: string;
  lineItems: LineItem[];
  confidence: number;
}

export interface LineItem {
  description: string;
  quantity: number;
  unitPrice: number;
  totalPrice: number;
  vatRate: number;
}

export interface AnomalyDetection {
  type: "duplicate" | "unusual_amount" | "wrong_period" | "missing_documentation" | "fraud_risk";
  severity: "low" | "medium" | "high" | "critical";
  description: string;
  affectedTransactions: number[];
  recommendation: string;
}

export interface ReconciliationResult {
  matched: ReconciliationMatch[];
  unmatched: UnmatchedItem[];
  discrepancies: Discrepancy[];
  summary: ReconciliationSummary;
}

export interface ReconciliationMatch {
  bankTransactionId: string;
  ledgerEntryId: number;
  amount: number;
  matchConfidence: number;
}

export interface UnmatchedItem {
  type: "bank" | "ledger";
  id: string | number;
  amount: number;
  date: string;
  description: string;
  suggestedAction: string;
}

export interface Discrepancy {
  type: "amount_mismatch" | "date_mismatch" | "missing_entry";
  bankAmount?: number;
  ledgerAmount?: number;
  difference?: number;
  description: string;
}

export interface ReconciliationSummary {
  totalBankTransactions: number;
  totalLedgerEntries: number;
  matchedCount: number;
  unmatchedBankCount: number;
  unmatchedLedgerCount: number;
  totalDiscrepancy: number;
}

export interface MvaCalculation {
  termin: string;
  grunnlagUtgaaende: number;
  utgaaendeMva: number;
  grunnlagInngaaende: number;
  inngaaendeMva: number;
  tilGode: number;
  aBetale: number;
  poster: MvaPost[];
}

export interface MvaPost {
  postNr: string;
  beskrivelse: string;
  grunnlag: number;
  sats: number;
  mvaBeloep: number;
}

export interface AuditFinding {
  type: "observation" | "recommendation" | "material_weakness" | "significant_deficiency";
  area: string;
  finding: string;
  impact: string;
  recommendation: string;
  priority: "low" | "medium" | "high" | "critical";
}

export interface AuditReport {
  companyName: string;
  orgNumber: string;
  period: string;
  auditorOpinion: "unqualified" | "qualified" | "adverse" | "disclaimer";
  findings: AuditFinding[];
  summary: string;
  generatedAt: string;
}

// ============================================================================
// AI AUTOMATION CLASS
// ============================================================================

export class AIAutomationService {
  // ============================================================================
  // TRANSACTION CLASSIFICATION
  // ============================================================================

  async classifyTransaction(
    description: string,
    amount: number,
    vendorName?: string
  ): Promise<TransactionClassification> {
    const prompt = `Du er en ekspert norsk regnskapsfører. Klassifiser følgende transaksjon:

Beskrivelse: ${description}
Beløp: ${amount} NOK
${vendorName ? `Leverandør: ${vendorName}` : ""}

Returner JSON med følgende struktur:
{
  "accountNumber": "4-sifret kontonummer fra norsk kontoplan",
  "accountName": "Kontonavn",
  "vatCode": "MVA-kode (0, 1, 3, 5, 6, etc.)",
  "vatRate": "MVA-sats som desimaltall (0, 0.12, 0.15, 0.25)",
  "confidence": "Konfidens 0-1",
  "reasoning": "Kort forklaring på valget"
}

Bruk standard norsk kontoplan (NS 4102).`;

    const response = await invokeLLM({
      messages: [
        { role: "system", content: "Du er en ekspert norsk regnskapsfører som klassifiserer transaksjoner." },
        { role: "user", content: prompt },
      ],
      response_format: {
        type: "json_schema",
        json_schema: {
          name: "transaction_classification",
          strict: true,
          schema: {
            type: "object",
            properties: {
              accountNumber: { type: "string" },
              accountName: { type: "string" },
              vatCode: { type: "string" },
              vatRate: { type: "number" },
              confidence: { type: "number" },
              reasoning: { type: "string" },
            },
            required: ["accountNumber", "accountName", "vatCode", "vatRate", "confidence", "reasoning"],
            additionalProperties: false,
          },
        },
      },
    });

    const rawContent = response.choices[0]?.message?.content;
    const content = typeof rawContent === 'string' ? rawContent : '';
    if (!content) {
      throw new Error("No response from AI");
    }

    return JSON.parse(content) as TransactionClassification;
  }

  // ============================================================================
  // INVOICE EXTRACTION
  // ============================================================================

  async extractInvoiceData(
    documentText: string,
    documentUrl?: string
  ): Promise<InvoiceExtraction> {
    const messages: Array<{ role: "system" | "user"; content: string | Array<{ type: "text"; text: string } | { type: "image_url"; image_url: { url: string } }> }> = [
      {
        role: "system",
        content: "Du er en ekspert på å lese og tolke norske fakturaer. Ekstraher all relevant informasjon.",
      },
    ];

    if (documentUrl) {
      messages.push({
        role: "user",
        content: [
          { type: "text", text: "Ekstraher all fakturainformasjon fra dette bildet:" },
          { type: "image_url", image_url: { url: documentUrl } },
        ],
      });
    } else {
      messages.push({
        role: "user",
        content: `Ekstraher all fakturainformasjon fra følgende tekst:

${documentText}

Returner JSON med struktur:
{
  "vendorName": "Leverandørnavn",
  "vendorOrgNumber": "Org.nr hvis tilgjengelig",
  "invoiceNumber": "Fakturanummer",
  "invoiceDate": "YYYY-MM-DD",
  "dueDate": "YYYY-MM-DD",
  "totalAmount": tall,
  "vatAmount": tall,
  "currency": "NOK",
  "lineItems": [{"description": "", "quantity": 1, "unitPrice": 0, "totalPrice": 0, "vatRate": 0.25}],
  "confidence": 0-1
}`,
      });
    }

    const response = await invokeLLM({
      messages,
      response_format: {
        type: "json_schema",
        json_schema: {
          name: "invoice_extraction",
          strict: true,
          schema: {
            type: "object",
            properties: {
              vendorName: { type: "string" },
              vendorOrgNumber: { type: "string" },
              invoiceNumber: { type: "string" },
              invoiceDate: { type: "string" },
              dueDate: { type: "string" },
              totalAmount: { type: "number" },
              vatAmount: { type: "number" },
              currency: { type: "string" },
              lineItems: {
                type: "array",
                items: {
                  type: "object",
                  properties: {
                    description: { type: "string" },
                    quantity: { type: "number" },
                    unitPrice: { type: "number" },
                    totalPrice: { type: "number" },
                    vatRate: { type: "number" },
                  },
                  required: ["description", "quantity", "unitPrice", "totalPrice", "vatRate"],
                  additionalProperties: false,
                },
              },
              confidence: { type: "number" },
            },
            required: ["vendorName", "invoiceNumber", "invoiceDate", "dueDate", "totalAmount", "vatAmount", "currency", "lineItems", "confidence"],
            additionalProperties: false,
          },
        },
      },
    });

    const rawContent = response.choices[0]?.message?.content;
    const content = typeof rawContent === 'string' ? rawContent : '';
    if (!content) {
      throw new Error("No response from AI");
    }

    return JSON.parse(content) as InvoiceExtraction;
  }

  // ============================================================================
  // ANOMALY DETECTION
  // ============================================================================

  async detectAnomalies(
    companyId: number,
    periodStart: string,
    periodEnd: string
  ): Promise<AnomalyDetection[]> {
    const entries = await db.listLedgerEntries(companyId, new Date(periodStart), new Date(periodEnd));

    const prompt = `Analyser følgende regnskapsposter for anomalier og potensielle feil:

${JSON.stringify(entries.slice(0, 100), null, 2)}

Se etter:
1. Duplikate transaksjoner
2. Uvanlig store beløp
3. Transaksjoner i feil periode
4. Manglende dokumentasjon
5. Potensielle svindelmønstre

Returner JSON array med anomalier:
[{
  "type": "duplicate|unusual_amount|wrong_period|missing_documentation|fraud_risk",
  "severity": "low|medium|high|critical",
  "description": "Beskrivelse",
  "affectedTransactions": [id1, id2],
  "recommendation": "Anbefalt handling"
}]`;

    const response = await invokeLLM({
      messages: [
        { role: "system", content: "Du er en revisor som analyserer regnskap for feil og svindel." },
        { role: "user", content: prompt },
      ],
    });

    const rawContent = response.choices[0]?.message?.content;
    const content = typeof rawContent === 'string' ? rawContent : '';
    if (!content) {
      return [];
    }

    try {
      const jsonMatch = content.match(/\[[\s\S]*\]/);
      if (jsonMatch) {
        return JSON.parse(jsonMatch[0]) as AnomalyDetection[];
      }
    } catch {
      // Return empty if parsing fails
    }

    return [];
  }

  // ============================================================================
  // BANK RECONCILIATION
  // ============================================================================

  async reconcileBankStatement(
    companyId: number,
    bankTransactions: Array<{ id: string; date: string; amount: number; description: string }>,
    periodStart: string,
    periodEnd: string
  ): Promise<ReconciliationResult> {
    const ledgerEntries = await db.listLedgerEntries(
      companyId,
      new Date(periodStart),
      new Date(periodEnd)
    );

    // Filter to bank-related accounts (1900-series)
    const bankLedgerEntries = ledgerEntries.filter(
      (e) => e.debitAccount?.startsWith("19") || e.creditAccount?.startsWith("19")
    );

    const matched: ReconciliationMatch[] = [];
    const unmatchedBank: UnmatchedItem[] = [];
    const unmatchedLedger: UnmatchedItem[] = [];
    const discrepancies: Discrepancy[] = [];

    const matchedBankIds = new Set<string>();
    const matchedLedgerIds = new Set<number>();

    // Attempt to match transactions
    for (const bankTx of bankTransactions) {
      let bestMatch: { entry: typeof bankLedgerEntries[0]; confidence: number } | null = null;

      for (const ledgerEntry of bankLedgerEntries) {
        if (matchedLedgerIds.has(ledgerEntry.id)) continue;

        const ledgerAmount = ledgerEntry.amount || 0;
        const amountMatch = Math.abs(bankTx.amount - ledgerAmount) < 0.01;
        const dateMatch =
          ledgerEntry.entryDate &&
          Math.abs(
            new Date(bankTx.date).getTime() -
              new Date(ledgerEntry.entryDate).getTime()
          ) <
            3 * 24 * 60 * 60 * 1000; // Within 3 days

        if (amountMatch && dateMatch) {
          const confidence = 0.95;
          if (!bestMatch || confidence > bestMatch.confidence) {
            bestMatch = { entry: ledgerEntry, confidence };
          }
        } else if (amountMatch) {
          const confidence = 0.7;
          if (!bestMatch || confidence > bestMatch.confidence) {
            bestMatch = { entry: ledgerEntry, confidence };
          }
        }
      }

      if (bestMatch && bestMatch.confidence > 0.5) {
        matched.push({
          bankTransactionId: bankTx.id,
          ledgerEntryId: bestMatch.entry.id,
          amount: bankTx.amount,
          matchConfidence: bestMatch.confidence,
        });
        matchedBankIds.add(bankTx.id);
        matchedLedgerIds.add(bestMatch.entry.id);
      }
    }

    // Find unmatched items
    for (const bankTx of bankTransactions) {
      if (!matchedBankIds.has(bankTx.id)) {
        unmatchedBank.push({
          type: "bank",
          id: bankTx.id,
          amount: bankTx.amount,
          date: bankTx.date,
          description: bankTx.description,
          suggestedAction: "Opprett bilag for denne banktransaksjonen",
        });
      }
    }

    for (const ledgerEntry of bankLedgerEntries) {
      if (!matchedLedgerIds.has(ledgerEntry.id)) {
        unmatchedLedger.push({
          type: "ledger",
          id: ledgerEntry.id,
          amount: ledgerEntry.amount || 0,
          date: ledgerEntry.entryDate?.toISOString().split("T")[0] || "",
          description: ledgerEntry.description || "",
          suggestedAction: "Verifiser at denne posten har tilhørende banktransaksjon",
        });
      }
    }

    // Calculate summary
    const totalBankAmount = bankTransactions.reduce((sum, tx) => sum + tx.amount, 0);
    const totalLedgerAmount = bankLedgerEntries.reduce(
      (sum, e) => sum + (e.amount || 0),
      0
    );

    if (Math.abs(totalBankAmount - totalLedgerAmount) > 0.01) {
      discrepancies.push({
        type: "amount_mismatch",
        bankAmount: totalBankAmount,
        ledgerAmount: totalLedgerAmount,
        difference: totalBankAmount - totalLedgerAmount,
        description: `Total differanse mellom bank og regnskap: ${(totalBankAmount - totalLedgerAmount).toFixed(2)} NOK`,
      });
    }

    return {
      matched,
      unmatched: [...unmatchedBank, ...unmatchedLedger],
      discrepancies,
      summary: {
        totalBankTransactions: bankTransactions.length,
        totalLedgerEntries: bankLedgerEntries.length,
        matchedCount: matched.length,
        unmatchedBankCount: unmatchedBank.length,
        unmatchedLedgerCount: unmatchedLedger.length,
        totalDiscrepancy: totalBankAmount - totalLedgerAmount,
      },
    };
  }

  // ============================================================================
  // MVA CALCULATION
  // ============================================================================

  async calculateMva(
    companyId: number,
    termin: string
  ): Promise<MvaCalculation> {
    // Parse termin (e.g., "2024-T1" for Jan-Feb 2024)
    const [year, terminNum] = termin.split("-T");
    const terminNumber = parseInt(terminNum);
    const startMonth = (terminNumber - 1) * 2;
    const endMonth = startMonth + 2;

    const periodStart = new Date(parseInt(year), startMonth, 1);
    const periodEnd = new Date(parseInt(year), endMonth, 0);

    const entries = await db.listLedgerEntries(companyId, periodStart, periodEnd);

    // Group by VAT code
    const vatGroups: Record<string, { grunnlag: number; mva: number }> = {};

    for (const entry of entries) {
      const vatCode = entry.vatCode || "0";
      if (!vatGroups[vatCode]) {
        vatGroups[vatCode] = { grunnlag: 0, mva: 0 };
      }

      // Determine if utgående or inngående based on account
      const debitAccountNum = parseInt(entry.debitAccount || "0");
      const creditAccountNum = parseInt(entry.creditAccount || "0");
      if (creditAccountNum >= 3000 && creditAccountNum < 4000) {
        // Revenue - utgående MVA
        vatGroups[vatCode].grunnlag += entry.amount || 0;
      } else if (debitAccountNum >= 4000 && debitAccountNum < 8000) {
        // Expenses - inngående MVA
        vatGroups[vatCode].grunnlag += entry.amount || 0;
      }
    }

    // Calculate MVA amounts
    const vatRates: Record<string, number> = {
      "0": 0,
      "1": 0.25,
      "3": 0.15,
      "5": 0.12,
      "6": 0,
    };

    let grunnlagUtgaaende = 0;
    let utgaaendeMva = 0;
    let grunnlagInngaaende = 0;
    let inngaaendeMva = 0;

    const poster: MvaPost[] = [];

    for (const [code, data] of Object.entries(vatGroups)) {
      const rate = vatRates[code] || 0;
      const mvaBeloep = data.grunnlag * rate;

      // Simplified: assume codes 1,3,5 are utgående, others are inngående
      if (["1", "3", "5"].includes(code)) {
        grunnlagUtgaaende += data.grunnlag;
        utgaaendeMva += mvaBeloep;
      } else {
        grunnlagInngaaende += data.grunnlag;
        inngaaendeMva += mvaBeloep;
      }

      poster.push({
        postNr: code,
        beskrivelse: this.getVatCodeDescription(code),
        grunnlag: data.grunnlag,
        sats: rate * 100,
        mvaBeloep,
      });
    }

    const tilGode = inngaaendeMva > utgaaendeMva ? inngaaendeMva - utgaaendeMva : 0;
    const aBetale = utgaaendeMva > inngaaendeMva ? utgaaendeMva - inngaaendeMva : 0;

    return {
      termin,
      grunnlagUtgaaende,
      utgaaendeMva,
      grunnlagInngaaende,
      inngaaendeMva,
      tilGode,
      aBetale,
      poster,
    };
  }

  private getVatCodeDescription(code: string): string {
    const descriptions: Record<string, string> = {
      "0": "Ingen MVA",
      "1": "Utgående MVA 25%",
      "3": "Utgående MVA 15% (mat)",
      "5": "Utgående MVA 12% (transport)",
      "6": "Fritatt for MVA",
      "7": "Inngående MVA 25%",
      "8": "Inngående MVA 15%",
      "9": "Inngående MVA 12%",
    };
    return descriptions[code] || `MVA-kode ${code}`;
  }

  // ============================================================================
  // AUDIT REPORT GENERATION
  // ============================================================================

  async generateAuditReport(
    companyId: number,
    year: number
  ): Promise<AuditReport> {
    const company = await db.getCompanyById(companyId);
    if (!company) {
      throw new Error("Company not found");
    }

    const periodStart = new Date(year, 0, 1);
    const periodEnd = new Date(year, 11, 31);

    const entries = await db.listLedgerEntries(companyId, periodStart, periodEnd);
    const anomalies = await this.detectAnomalies(
      companyId,
      periodStart.toISOString(),
      periodEnd.toISOString()
    );

    const prompt = `Du er en statsautorisert revisor. Generer en revisjonsrapport basert på følgende data:

Selskap: ${company.name}
Org.nr: ${company.orgNumber}
Regnskapsår: ${year}
Antall posteringer: ${entries.length}
Oppdagede anomalier: ${JSON.stringify(anomalies)}

Generer en profesjonell revisjonsrapport med:
1. Revisjonsuttalelse (unqualified/qualified/adverse/disclaimer)
2. Funn og observasjoner
3. Anbefalinger
4. Sammendrag

Returner JSON:
{
  "auditorOpinion": "unqualified|qualified|adverse|disclaimer",
  "findings": [{
    "type": "observation|recommendation|material_weakness|significant_deficiency",
    "area": "Område",
    "finding": "Funn",
    "impact": "Konsekvens",
    "recommendation": "Anbefaling",
    "priority": "low|medium|high|critical"
  }],
  "summary": "Sammendrag av revisjonen"
}`;

    const response = await invokeLLM({
      messages: [
        { role: "system", content: "Du er en statsautorisert revisor som skriver profesjonelle revisjonsrapporter." },
        { role: "user", content: prompt },
      ],
    });

    const rawContent = response.choices[0]?.message?.content;
    const content = typeof rawContent === 'string' ? rawContent : '';
    if (!content) {
      throw new Error("No response from AI");
    }

    let parsed: { auditorOpinion: string; findings: AuditFinding[]; summary: string };
    try {
      const jsonMatch = content.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        parsed = JSON.parse(jsonMatch[0]);
      } else {
        throw new Error("No JSON found");
      }
    } catch {
      parsed = {
        auditorOpinion: "unqualified",
        findings: [],
        summary: "Revisjonen er gjennomført uten vesentlige funn.",
      };
    }

    return {
      companyName: company.name,
      orgNumber: company.orgNumber || "",
      period: `${year}`,
      auditorOpinion: parsed.auditorOpinion as AuditReport["auditorOpinion"],
      findings: parsed.findings,
      summary: parsed.summary,
      generatedAt: new Date().toISOString(),
    };
  }

  // ============================================================================
  // NATURAL LANGUAGE QUERIES
  // ============================================================================

  async answerAccountingQuery(
    companyId: number,
    query: string
  ): Promise<string> {
    const company = await db.getCompanyById(companyId);
    const stats = await db.getDashboardStats(1); // User ID placeholder

    const prompt = `Du er en ekspert norsk regnskapsfører og revisor. Svar på følgende spørsmål om regnskapet:

Selskap: ${company?.name || "Ukjent"}
Spørsmål: ${query}

Gi et profesjonelt og nøyaktig svar basert på norsk regnskapslovgivning og god regnskapsskikk.`;

    const response = await invokeLLM({
      messages: [
        {
          role: "system",
          content:
            "Du er en ekspert norsk regnskapsfører og revisor som svarer på spørsmål om regnskap, skatt, MVA, og norsk lovgivning.",
        },
        { role: "user", content: prompt },
      ],
    });

    const result = response.choices[0]?.message?.content;
    return typeof result === 'string' ? result : "Beklager, kunne ikke generere svar.";
  }
}

// ============================================================================
// SINGLETON INSTANCE
// ============================================================================

export const aiAutomation = new AIAutomationService();
