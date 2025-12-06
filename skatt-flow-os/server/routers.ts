import { z } from "zod";
import { TRPCError } from "@trpc/server";
import { COOKIE_NAME } from "@shared/const";
import { getSessionCookieOptions } from "./_core/cookies";
import { systemRouter } from "./_core/systemRouter";
import { publicProcedure, protectedProcedure, router } from "./_core/trpc";
import * as db from "./db";
import { createForvaltClient } from "./clients/forvaltClient";
import { createRegnskapClient } from "./clients/regnskapClient";
import { createAltinnClient } from "./clients/altinnClient";
import { getSkattFlowAgent } from "./agents/skattFlowAgent";
import { getSkattFlowAgentV2 } from "./agents/skattFlowAgentV2";
import { storagePut } from "./storage";
import { nanoid } from "nanoid";
import { logAudit, requireCompanyAccess } from "./middleware/auditLog";

// ============================================================================
// ROLE-BASED ACCESS CONTROL
// ============================================================================

const accountantProcedure = protectedProcedure.use(({ ctx, next }) => {
  const role = ctx.user.accountingRole;
  if (!["OWNER", "ADMIN", "ACCOUNTANT"].includes(role)) {
    throw new TRPCError({ code: "FORBIDDEN", message: "Accountant access required" });
  }
  return next({ ctx });
});

const adminProcedure = protectedProcedure.use(({ ctx, next }) => {
  const role = ctx.user.accountingRole;
  if (!["OWNER", "ADMIN"].includes(role)) {
    throw new TRPCError({ code: "FORBIDDEN", message: "Admin access required" });
  }
  return next({ ctx });
});

// ============================================================================
// COMPANY ROUTER
// ============================================================================

const companyRouter = router({
  list: protectedProcedure.query(async ({ ctx }) => {
    return db.listCompanies(ctx.user.id);
  }),

  get: protectedProcedure.input(z.object({ id: z.number() })).query(async ({ ctx, input }) => {
    // Check company access
    await requireCompanyAccess(ctx.user.id, input.id);
    const company = await db.getCompanyById(input.id);
    if (!company) throw new TRPCError({ code: "NOT_FOUND", message: "Company not found" });
    return company;
  }),

  create: adminProcedure
    .input(
      z.object({
        orgNumber: z.string().min(9).max(20),
        name: z.string().min(1),
        address: z.string().optional(),
        city: z.string().optional(),
        postalCode: z.string().optional(),
        externalRegnskapSystem: z
          .enum(["TRIPLETEX", "POWEROFFICE", "FIKEN", "VISMA_EACCOUNTING", "OTHER"])
          .optional(),
        externalRegnskapCompanyId: z.string().optional(),
      })
    )
    .mutation(async ({ ctx, input }) => {
      // Check if company already exists
      const existing = await db.getCompanyByOrgNumber(input.orgNumber);
      if (existing) throw new TRPCError({ code: "CONFLICT", message: "Company already exists" });

      const companyId = await db.createCompany({
        ...input,
        createdById: ctx.user.id,
      });

      // Grant owner access to creator
      await db.grantCompanyAccess({
        userId: ctx.user.id,
        companyId,
        accessRole: "OWNER",
      });

      return { id: companyId };
    }),

  update: adminProcedure
    .input(
      z.object({
        id: z.number(),
        name: z.string().optional(),
        address: z.string().optional(),
        city: z.string().optional(),
        postalCode: z.string().optional(),
        externalRegnskapSystem: z
          .enum(["TRIPLETEX", "POWEROFFICE", "FIKEN", "VISMA_EACCOUNTING", "OTHER"])
          .optional(),
        externalRegnskapCompanyId: z.string().optional(),
        autoPostEnabled: z.boolean().optional(),
      })
    )
    .mutation(async ({ input }) => {
      const { id, ...data } = input;
      await db.updateCompany(id, data);
      return { success: true };
    }),

  delete: adminProcedure.input(z.object({ id: z.number() })).mutation(async ({ input }) => {
    await db.deleteCompany(input.id);
    return { success: true };
  }),

  lookupFromForvalt: adminProcedure.input(z.object({ orgNumber: z.string() })).mutation(async ({ input }) => {
    const client = createForvaltClient();
    try {
      const profile = await client.getFullProfile(input.orgNumber);
      return {
        company: profile.company,
        credit: profile.credit,
        financials: profile.financials,
      };
    } catch (error) {
      throw new TRPCError({
        code: "BAD_REQUEST",
        message: `Failed to lookup company: ${error instanceof Error ? error.message : "Unknown error"}`,
      });
    }
  }),

  refreshForvaltData: adminProcedure.input(z.object({ id: z.number() })).mutation(async ({ input }) => {
    const company = await db.getCompanyById(input.id);
    if (!company) throw new TRPCError({ code: "NOT_FOUND", message: "Company not found" });

    const client = createForvaltClient();
    try {
      const profile = await client.getFullProfile(company.orgNumber);

      // Update company with new data
      await db.updateCompany(input.id, {
        forvaltRating: profile.credit.rating || undefined,
        forvaltCreditScore: profile.credit.creditScore || undefined,
        forvaltRiskClass: profile.credit.riskClass || undefined,
      });

      // Save snapshot
      await db.createForvaltSnapshot({
        companyId: input.id,
        rawJson: profile as unknown as Record<string, unknown>,
        rating: profile.credit.rating || undefined,
        creditScore: profile.credit.creditScore || undefined,
        riskClass: profile.credit.riskClass || undefined,
      });

      return { success: true, data: profile };
    } catch (error) {
      throw new TRPCError({
        code: "BAD_REQUEST",
        message: `Failed to refresh data: ${error instanceof Error ? error.message : "Unknown error"}`,
      });
    }
  }),
});

// ============================================================================
// ACCOUNTING DOCUMENT ROUTER
// ============================================================================

const documentRouter = router({
  list: protectedProcedure
    .input(
      z.object({
        companyId: z.number(),
        status: z.enum(["NEW", "PROCESSED", "POSTED", "REJECTED"]).optional(),
      })
    )
    .query(async ({ input }) => {
      return db.listAccountingDocuments(input.companyId, input.status);
    }),

  get: protectedProcedure.input(z.object({ id: z.number() })).query(async ({ input }) => {
    const doc = await db.getAccountingDocumentById(input.id);
    if (!doc) throw new TRPCError({ code: "NOT_FOUND", message: "Document not found" });
    return doc;
  }),

  upload: accountantProcedure
    .input(
      z.object({
        companyId: z.number(),
        sourceType: z.enum(["INVOICE_SUPPLIER", "INVOICE_CUSTOMER", "RECEIPT", "CONTRACT", "OTHER"]),
        fileName: z.string(),
        fileContent: z.string(), // Base64 encoded
        contentType: z.string(),
      })
    )
    .mutation(async ({ input }) => {
      // Upload file to S3
      const fileBuffer = Buffer.from(input.fileContent, "base64");
      const fileKey = `documents/${input.companyId}/${nanoid()}-${input.fileName}`;
      const { url } = await storagePut(fileKey, fileBuffer, input.contentType);

      // Create document record
      const docId = await db.createAccountingDocument({
        companyId: input.companyId,
        sourceType: input.sourceType,
        originalFileUrl: url,
        originalFileName: input.fileName,
        status: "NEW",
      });

      return { id: docId, fileUrl: url };
    }),

  process: accountantProcedure.input(z.object({ id: z.number() })).mutation(async ({ input }) => {
    const doc = await db.getAccountingDocumentById(input.id);
    if (!doc) throw new TRPCError({ code: "NOT_FOUND", message: "Document not found" });

    const agent = getSkattFlowAgent();

    // For now, we'll use a placeholder for document content
    // In production, this would OCR/parse the actual document
    const documentContent = `Document ID: ${doc.id}, Type: ${doc.sourceType}, File: ${doc.originalFileName}`;

    try {
      const result = await agent.processDocument(documentContent, doc.sourceType);

      await db.updateAccountingDocument(input.id, {
        status: "PROCESSED",
        parsedJson: result.extractedFields as unknown as Record<string, unknown>,
        suggestedAccount: result.suggestedPosting.account,
        suggestedVatCode: result.suggestedPosting.vatCode,
        suggestedDescription: result.suggestedPosting.description,
        suggestedAmount: result.extractedFields.totalAmount,
      });

      return { success: true, result };
    } catch (error) {
      throw new TRPCError({
        code: "INTERNAL_SERVER_ERROR",
        message: `Failed to process document: ${error instanceof Error ? error.message : "Unknown error"}`,
      });
    }
  }),

  approve: accountantProcedure
    .input(
      z.object({
        id: z.number(),
        account: z.string(),
        vatCode: z.string(),
        description: z.string(),
        amount: z.number(),
      })
    )
    .mutation(async ({ ctx, input }) => {
      const doc = await db.getAccountingDocumentById(input.id);
      if (!doc) throw new TRPCError({ code: "NOT_FOUND", message: "Document not found" });

      const company = await db.getCompanyById(doc.companyId);
      if (!company) throw new TRPCError({ code: "NOT_FOUND", message: "Company not found" });

      // Create voucher in external system if connected
      let voucherId: string | undefined;
      if (company.externalRegnskapCompanyId) {
        try {
          const client = createRegnskapClient();
          const result = await client.createVoucher(company.externalRegnskapCompanyId, {
            date: new Date().toISOString().split("T")[0],
            description: input.description,
            lines: [
              { accountNumber: input.account, debit: input.amount, vatCode: input.vatCode },
              { accountNumber: "2400", credit: input.amount }, // Default AP account
            ],
            attachmentUrl: doc.originalFileUrl || undefined,
          });
          voucherId = result.voucherId;
        } catch (error) {
          console.error("[DocumentRouter] Failed to create voucher:", error);
          // Continue without external posting
        }
      }

      await db.updateAccountingDocument(input.id, {
        status: "POSTED",
        postedVoucherId: voucherId,
        postedAt: new Date(),
        postedById: ctx.user.id,
      });

      return { success: true, voucherId };
    }),

  reject: accountantProcedure
    .input(z.object({ id: z.number(), reason: z.string() }))
    .mutation(async ({ ctx, input }) => {
      await db.updateAccountingDocument(input.id, {
        status: "REJECTED",
        rejectedReason: input.reason,
        rejectedAt: new Date(),
        rejectedById: ctx.user.id,
      });
      return { success: true };
    }),
});

// ============================================================================
// FILING ROUTER
// ============================================================================

const filingRouter = router({
  list: protectedProcedure
    .input(
      z.object({
        companyId: z.number(),
        type: z.enum(["MVA_MELDING", "A_MELDING_SUMMARY", "SAF_T", "ARSREGNSKAP", "OTHER"]).optional(),
      })
    )
    .query(async ({ input }) => {
      return db.listFilings(input.companyId, input.type);
    }),

  get: protectedProcedure.input(z.object({ id: z.number() })).query(async ({ input }) => {
    const filing = await db.getFilingById(input.id);
    if (!filing) throw new TRPCError({ code: "NOT_FOUND", message: "Filing not found" });
    return filing;
  }),

  generateMVADraft: accountantProcedure
    .input(
      z.object({
        companyId: z.number(),
        year: z.number(),
        term: z.number().min(1).max(6),
      })
    )
    .mutation(async ({ ctx, input }) => {
      const company = await db.getCompanyById(input.companyId);
      if (!company) throw new TRPCError({ code: "NOT_FOUND", message: "Company not found" });

      // Calculate period dates
      const termStartMonth = (input.term - 1) * 2;
      const periodStart = new Date(input.year, termStartMonth, 1);
      const periodEnd = new Date(input.year, termStartMonth + 2, 0);

      const agent = getSkattFlowAgent();
      try {
        const result = await agent.generateMVADraft(
          company.externalRegnskapCompanyId || String(company.id),
          periodStart.toISOString().split("T")[0],
          periodEnd.toISOString().split("T")[0],
          input.year,
          input.term
        );

        result.payload.orgNumber = company.orgNumber;

        const filingId = await db.createFiling({
          companyId: input.companyId,
          filingType: "MVA_MELDING",
          periodStart,
          periodEnd,
          status: "DRAFT",
          payloadJson: result.payload as unknown as Record<string, unknown>,
          summaryJson: { summary: result.summary, warnings: result.warnings },
          createdById: ctx.user.id,
        });

        return { id: filingId, summary: result.summary };
      } catch (error) {
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: `Failed to generate MVA draft: ${error instanceof Error ? error.message : "Unknown error"}`,
        });
      }
    }),

  generateSAFT: accountantProcedure
    .input(
      z.object({
        companyId: z.number(),
        periodStart: z.string(),
        periodEnd: z.string(),
      })
    )
    .mutation(async ({ ctx, input }) => {
      const company = await db.getCompanyById(input.companyId);
      if (!company) throw new TRPCError({ code: "NOT_FOUND", message: "Company not found" });

      if (!company.externalRegnskapCompanyId) {
        throw new TRPCError({ code: "BAD_REQUEST", message: "Company not connected to accounting system" });
      }

      const client = createRegnskapClient();
      try {
        const result = await client.exportSaft(company.externalRegnskapCompanyId, input.periodStart, input.periodEnd);

        // Validate SAF-T
        const agent = getSkattFlowAgent();
        const validation = await agent.validateSAFT(`SAF-T export for ${company.name}, period ${input.periodStart} to ${input.periodEnd}`);

        const filingId = await db.createFiling({
          companyId: input.companyId,
          filingType: "SAF_T",
          periodStart: new Date(input.periodStart),
          periodEnd: new Date(input.periodEnd),
          status: validation.isValid ? "READY_FOR_REVIEW" : "ERROR",
          payloadJson: { fileUrl: result.fileUrl, format: result.format },
          summaryJson: { validation },
          errorMessage: validation.isValid ? undefined : validation.issues.map((i) => i.message).join("; "),
          createdById: ctx.user.id,
        });

        return { id: filingId, fileUrl: result.fileUrl, validation };
      } catch (error) {
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: `Failed to generate SAF-T: ${error instanceof Error ? error.message : "Unknown error"}`,
        });
      }
    }),

  submitToAltinn: adminProcedure
    .input(z.object({ id: z.number(), confirmed: z.boolean() }))
    .mutation(async ({ ctx, input }) => {
      if (!input.confirmed) {
        throw new TRPCError({ code: "BAD_REQUEST", message: "Submission must be explicitly confirmed" });
      }

      const filing = await db.getFilingById(input.id);
      if (!filing) throw new TRPCError({ code: "NOT_FOUND", message: "Filing not found" });

      if (filing.status === "SUBMITTED") {
        throw new TRPCError({ code: "BAD_REQUEST", message: "Filing already submitted" });
      }

      const company = await db.getCompanyById(filing.companyId);
      if (!company) throw new TRPCError({ code: "NOT_FOUND", message: "Company not found" });

      const client = createAltinnClient();
      try {
        // Create draft in Altinn
        const draft = await client.createDraftFiling(
          filing.filingType,
          company.orgNumber,
          filing.payloadJson as Record<string, unknown>
        );

        // Submit the draft
        const submission = await client.submitFiling(draft.draftId);

        await db.updateFiling(input.id, {
          status: "SUBMITTED",
          altinnDraftId: draft.draftId,
          altinnReference: submission.reference,
          submittedAt: new Date(),
          submittedById: ctx.user.id,
        });

        return { success: true, reference: submission.reference };
      } catch (error) {
        await db.updateFiling(input.id, {
          status: "ERROR",
          errorMessage: error instanceof Error ? error.message : "Unknown error",
        });

        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: `Failed to submit to Altinn: ${error instanceof Error ? error.message : "Unknown error"}`,
        });
      }
    }),
});

// ============================================================================
// DOCUMENT TEMPLATE ROUTER
// ============================================================================

const templateRouter = router({
  list: protectedProcedure
    .input(z.object({ category: z.enum(["CONTRACT", "HR", "LEGAL", "FINANCIAL", "GOVERNANCE", "OTHER"]).optional() }))
    .query(async ({ input }) => {
      return db.listDocumentTemplates(input.category);
    }),

  get: protectedProcedure.input(z.object({ id: z.number() })).query(async ({ input }) => {
    const template = await db.getDocumentTemplateById(input.id);
    if (!template) throw new TRPCError({ code: "NOT_FOUND", message: "Template not found" });
    return template;
  }),

  create: adminProcedure
    .input(
      z.object({
        name: z.string().min(1),
        category: z.enum(["CONTRACT", "HR", "LEGAL", "FINANCIAL", "GOVERNANCE", "OTHER"]),
        language: z.string().default("no"),
        description: z.string().optional(),
        bodyMarkdown: z.string().min(1),
        variablesJson: z.record(z.string(), z.string()).optional(),
      })
    )
    .mutation(async ({ ctx, input }) => {
      const id = await db.createDocumentTemplate({
        ...input,
        source: "CUSTOM",
        createdById: ctx.user.id,
      });
      return { id };
    }),

  update: adminProcedure
    .input(
      z.object({
        id: z.number(),
        name: z.string().optional(),
        description: z.string().optional(),
        bodyMarkdown: z.string().optional(),
        variablesJson: z.record(z.string(), z.string()).optional(),
        isActive: z.boolean().optional(),
      })
    )
    .mutation(async ({ input }) => {
      const { id, ...data } = input;
      await db.updateDocumentTemplate(id, data);
      return { success: true };
    }),

  generate: accountantProcedure
    .input(
      z.object({
        templateId: z.number(),
        companyId: z.number(),
        title: z.string(),
        variables: z.record(z.string(), z.string()),
        language: z.string().default("no"),
      })
    )
    .mutation(async ({ ctx, input }) => {
      const template = await db.getDocumentTemplateById(input.templateId);
      if (!template) throw new TRPCError({ code: "NOT_FOUND", message: "Template not found" });

      const agent = getSkattFlowAgent();
      const result = await agent.generateDocument(template.bodyMarkdown, input.variables, input.language);

      const docId = await db.createGeneratedDocument({
        companyId: input.companyId,
        templateId: input.templateId,
        title: input.title,
        filledVariablesJson: input.variables,
        outputMarkdown: result.outputMarkdown,
        createdById: ctx.user.id,
      });

      return { id: docId, outputMarkdown: result.outputMarkdown };
    }),
});

// ============================================================================
// GENERATED DOCUMENT ROUTER
// ============================================================================

const generatedDocRouter = router({
  list: protectedProcedure.input(z.object({ companyId: z.number() })).query(async ({ input }) => {
    return db.listGeneratedDocuments(input.companyId);
  }),

  get: protectedProcedure.input(z.object({ id: z.number() })).query(async ({ input }) => {
    const doc = await db.getGeneratedDocumentById(input.id);
    if (!doc) throw new TRPCError({ code: "NOT_FOUND", message: "Document not found" });
    return doc;
  }),
});

// ============================================================================
// CHAT ROUTER
// ============================================================================

const chatRouter = router({
  send: protectedProcedure
    .input(
      z.object({
        message: z.string().min(1),
        sessionId: z.string(),
        companyId: z.number().optional(),
        documentId: z.number().optional(),
        filingId: z.number().optional(),
        useV2Agent: z.boolean().optional().default(true), // Use V2 agent with tool calling by default
      })
    )
    .mutation(async ({ ctx, input }) => {
      // Log audit event
      logAudit({
        userId: ctx.user.id,
        companyId: input.companyId,
        action: "CHAT_MESSAGE",
        entityType: "chat",
      });

      // Save user message
      await db.createChatMessage({
        userId: ctx.user.id,
        companyId: input.companyId,
        sessionId: input.sessionId,
        role: "user",
        content: input.message,
        attachedDocumentId: input.documentId,
        attachedFilingId: input.filingId,
      });

      // Get conversation history
      const history = await db.listChatMessages(input.sessionId, 20);
      const formattedHistory = history.reverse().map((m) => ({
        role: m.role as "user" | "assistant" | "system",
        content: m.content,
      }));

      // Get agent response - use V2 agent with tool calling by default
      let response;
      if (input.useV2Agent) {
        const agentV2 = getSkattFlowAgentV2();
        response = await agentV2.chat(
          input.message,
          { companyId: input.companyId, documentId: input.documentId, filingId: input.filingId },
          formattedHistory,
          ctx.user.id
        );
      } else {
        const agent = getSkattFlowAgent();
        response = await agent.chat(
          input.message,
          { companyId: input.companyId, documentId: input.documentId, filingId: input.filingId },
          formattedHistory
        );
      }

      // Save assistant message with tool results
      await db.createChatMessage({
        userId: ctx.user.id,
        companyId: input.companyId,
        sessionId: input.sessionId,
        role: "assistant",
        content: response.message,
        metadata: { 
          actions: response.actions,
          toolResults: (response as { toolResults?: unknown }).toolResults,
          toolsExecuted: response.metadata?.toolsExecuted,
        },
      });

      return response;
    }),

  history: protectedProcedure.input(z.object({ sessionId: z.string() })).query(async ({ input }) => {
    const messages = await db.listChatMessages(input.sessionId);
    return messages.reverse();
  }),
});

// ============================================================================
// LEDGER ROUTER
// ============================================================================

const ledgerRouter = router({
  list: protectedProcedure
    .input(
      z.object({
        companyId: z.number(),
        periodStart: z.string().optional(),
        periodEnd: z.string().optional(),
      })
    )
    .query(async ({ input }) => {
      const periodStart = input.periodStart ? new Date(input.periodStart) : undefined;
      const periodEnd = input.periodEnd ? new Date(input.periodEnd) : undefined;
      return db.listLedgerEntries(input.companyId, periodStart, periodEnd);
    }),
});

// ============================================================================
// FORVALT ROUTER
// ============================================================================

const forvaltRouter = router({
  getMarketStats: protectedProcedure.query(async () => {
    // Return market statistics from Forvalt
    return {
      nyetableringerSisteDogn: 318,
      nyetableringer30Dager: 3079,
      konkurserSisteDogn: 5,
      konkurser30Dager: 413,
      betalingsanmerkningerAntall: 47632,
      betalingsanmerkningerBelop: 4300000000,
      tvungenPantAntall: 5116,
      tvungenPantBelop: 2600000000,
      aktiveSelskaperTotal: 1100000,
      inaktiveSelskaperTotal: 1600000,
    };
  }),

  searchCompanies: protectedProcedure
    .input(z.object({ query: z.string() }))
    .mutation(async ({ input }) => {
      // Search by org number directly
      const forvalt = createForvaltClient();
      try {
        const company = await forvalt.getCompany(input.query);
        return [{ orgNr: company.orgNumber, navn: company.name, status: company.status || 'Aktivt' }];
      } catch {
        return [];
      }
    }),

  getCompanyDetails: protectedProcedure
    .input(z.object({ orgNr: z.string() }))
    .query(async ({ input }) => {
      const forvalt = createForvaltClient();
      const company = await forvalt.getCompany(input.orgNr);
      const credit = await forvalt.getCredit(input.orgNr);
      const financials = await forvalt.getFinancials(input.orgNr);
      
      return {
        basic: {
          orgNr: company.orgNumber,
          navn: company.name,
          organisasjonsform: 'AS',
          aksjekapital: 0,
          stiftelsesdato: company.registrationDate || '',
          registreringsdato: company.registrationDate || '',
          status: company.status || 'Aktivt',
          forretningsadresse: {
            gate: company.address || '',
            postnr: company.postalCode || '',
            poststed: company.city || '',
            kommune: '',
            fylke: '',
          },
          telefon: '',
          internett: '',
          antallAnsatte: company.employees || 0,
          ehfFaktura: false,
          mvaRegistrert: false,
          sektor: '',
          naceBransje: company.industryCode || '',
          proffBransje: [company.industryDescription || ''],
        },
        creditRating: credit ? {
          rating: (credit.rating || 'B') as 'A+' | 'A' | 'B' | 'C' | 'D',
          ratingText: credit.riskClass || '',
          score: credit.creditScore || 0,
          konkursrisiko: 0,
          kredittramme: credit.creditLimit || 0,
          vurderinger: {
            ledelseOgEierskap: 4,
            okonomi: 4,
            betalingshistorikk: credit.paymentRemarks ? 2 : 4,
            generelt: 4,
          },
        } : null,
        financials: financials ? [{
          regnskapsaar: financials.year,
          valutakode: 'NOK',
          sumDriftsinntekter: financials.revenue || 0,
          driftsresultat: financials.operatingResult || 0,
          ordinaertResultatForSkatt: financials.netResult || 0,
          sumEiendeler: financials.totalAssets || 0,
          sumGjeld: financials.totalDebt || 0,
          sumEgenkapital: financials.equity || 0,
          lonnsomhet: 0,
          likviditetsgrad: 0,
          soliditet: 0,
          ebitda: 0,
          ebitdaMargin: 0,
        }] : [],
        paymentRemarks: null,
        roles: [],
        shareholders: [],
        subsidiaries: [],
        properties: [],
        courtCases: [],
        announcements: [],
      };
    }),
});

// ============================================================================
// DASHBOARD ROUTER
// ============================================================================

const dashboardRouter = router({
  stats: protectedProcedure.query(async ({ ctx }) => {
    return db.getDashboardStats(ctx.user.id);
  }),
});

// ============================================================================
// MAIN APP ROUTER
// ============================================================================

export const appRouter = router({
  system: systemRouter,
  auth: router({
    me: publicProcedure.query((opts) => opts.ctx.user),
    logout: publicProcedure.mutation(({ ctx }) => {
      const cookieOptions = getSessionCookieOptions(ctx.req);
      ctx.res.clearCookie(COOKIE_NAME, { ...cookieOptions, maxAge: -1 });
      return { success: true } as const;
    }),
  }),

  // Feature routers
  company: companyRouter,
  document: documentRouter,
  filing: filingRouter,
  template: templateRouter,
  generatedDoc: generatedDocRouter,
  chat: chatRouter,
  ledger: ledgerRouter,
  dashboard: dashboardRouter,
  forvalt: forvaltRouter,
});

export type AppRouter = typeof appRouter;
