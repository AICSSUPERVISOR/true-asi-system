import { z } from "zod";
import { TRPCError } from "@trpc/server";
import { publicProcedure, protectedProcedure, router } from "../_core/trpc";
import { getDb } from "../db";
import { companies, companyRoles } from "../../drizzle/schema";
import { eq } from "drizzle-orm";

/**
 * Brreg.no API Integration
 * Fetches Norwegian company data from Brønnøysundregistrene
 */

const BRREG_API_BASE = "https://data.brreg.no/enhetsregisteret/api";

export const brregRouter = router({
  /**
   * Get company by organization number from Brreg.no API
   */
  getCompanyByOrgnr: publicProcedure
    .input(
      z.object({
        orgnr: z.string().length(9, "Organization number must be exactly 9 digits"),
      })
    )
    .query(async ({ input }) => {
      try {
        const response = await fetch(`${BRREG_API_BASE}/enheter/${input.orgnr}`, {
          headers: {
            Accept: "application/json",
          },
        });

        if (!response.ok) {
          if (response.status === 404) {
            throw new TRPCError({
              code: "NOT_FOUND",
              message: `Company with organization number ${input.orgnr} not found`,
            });
          }
          throw new TRPCError({
            code: "INTERNAL_SERVER_ERROR",
            message: "Failed to fetch company data from Brreg.no",
          });
        }

        const data = await response.json();
        return data;
      } catch (error) {
        if (error instanceof TRPCError) throw error;
        console.error("[Brreg] Error fetching company:", error);
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: "Failed to fetch company data",
        });
      }
    }),

  /**
   * Get company roles (board members, CEO, etc.) from Brreg.no API
   */
  getCompanyRoles: publicProcedure
    .input(
      z.object({
        orgnr: z.string().length(9),
      })
    )
    .query(async ({ input }) => {
      try {
        const response = await fetch(`${BRREG_API_BASE}/enheter/${input.orgnr}/roller`, {
          headers: {
            Accept: "application/json",
          },
        });

        if (!response.ok) {
          if (response.status === 404) {
            return { rollegrupper: [] }; // No roles found
          }
          throw new TRPCError({
            code: "INTERNAL_SERVER_ERROR",
            message: "Failed to fetch company roles from Brreg.no",
          });
        }

        const data = await response.json();
        return data;
      } catch (error) {
        if (error instanceof TRPCError) throw error;
        console.error("[Brreg] Error fetching roles:", error);
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: "Failed to fetch company roles",
        });
      }
    }),

  /**
   * Save company data to database (authenticated users only)
   */
  saveCompany: protectedProcedure
    .input(
      z.object({
        orgnr: z.string().length(9),
        brregData: z.any(), // Full JSON response from Brreg.no
      })
    )
    .mutation(async ({ ctx, input }) => {
      const db = await getDb();
      if (!db) {
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: "Database connection failed",
        });
      }

      try {
        const companyId = `company_${input.orgnr}_${Date.now()}`;

        // Extract relevant fields from Brreg.no response
        const brregData = input.brregData;

        await db.insert(companies).values({
          id: companyId,
          userId: parseInt(ctx.user.id.toString()) || 0,
          orgnr: input.orgnr,
          name: brregData.navn || "",
          organizationForm: brregData.organisasjonsform?.kode || null,
          organizationFormDescription: brregData.organisasjonsform?.beskrivelse || null,
          registrationDate: brregData.registreringsdatoEnhetsregisteret || null,
          industryCode: brregData.naeringskode1?.kode || null,
          industryDescription: brregData.naeringskode1?.beskrivelse || null,
          employees: brregData.antallAnsatte || null,
          businessAddress: brregData.forretningsadresse
            ? JSON.stringify(brregData.forretningsadresse)
            : null,
          postalAddress: brregData.postadresse
            ? JSON.stringify(brregData.postadresse)
            : null,
          municipality: brregData.forretningsadresse?.kommune || null,
          municipalityNumber: brregData.forretningsadresse?.kommunenummer || null,
          vatRegistered: brregData.registrertIMvaregisteret ? 1 : 0,
          registeredInBusinessRegistry: brregData.registrertIForetaksregisteret ? 1 : 0,
          bankrupt: brregData.konkurs ? 1 : 0,
          underLiquidation: brregData.underAvvikling ? 1 : 0,
          rawData: JSON.stringify(brregData),
        });

        return { success: true, companyId };
      } catch (error) {
        console.error("[Brreg] Error saving company:", error);
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: "Failed to save company data",
        });
      }
    }),

  /**
   * Save company roles to database
   */
  saveCompanyRoles: protectedProcedure
    .input(
      z.object({
        companyId: z.string(),
        rolesData: z.any(), // Full JSON response from Brreg.no roles endpoint
      })
    )
    .mutation(async ({ input }) => {
      const db = await getDb();
      if (!db) {
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: "Database connection failed",
        });
      }

      try {
        const rollegrupper = input.rolesData.rollegrupper || [];
        const rolesToInsert = [];

        for (const gruppe of rollegrupper) {
          const roller = gruppe.roller || [];
          for (const rolle of roller) {
            const roleId = `role_${input.companyId}_${rolle.type?.kode}_${Date.now()}_${Math.random()}`;

            rolesToInsert.push({
              id: roleId,
              companyId: input.companyId,
              roleType: rolle.type?.kode || "",
              roleTypeDescription: rolle.type?.beskrivelse || null,
              personName: rolle.person?.navn?.fornavn && rolle.person?.navn?.etternavn
                ? `${rolle.person.navn.fornavn} ${rolle.person.navn.etternavn}`
                : null,
              personBirthDate: rolle.person?.fodselsdato || null,
              organizationNumber: rolle.enhet?.organisasjonsnummer || null,
              organizationName: rolle.enhet?.navn || null,
            });
          }
        }

        if (rolesToInsert.length > 0) {
          await db.insert(companyRoles).values(rolesToInsert);
        }

        return { success: true, rolesCount: rolesToInsert.length };
      } catch (error) {
        console.error("[Brreg] Error saving roles:", error);
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: "Failed to save company roles",
        });
      }
    }),

  /**
   * Get saved companies for authenticated user
   */
  getMySavedCompanies: protectedProcedure.query(async ({ ctx }) => {
    const db = await getDb();
    if (!db) {
      throw new TRPCError({
        code: "INTERNAL_SERVER_ERROR",
        message: "Database connection failed",
      });
    }

    try {
      const userCompanies = await db
        .select()
        .from(companies)
        .where(eq(companies.userId, ctx.user.id));

      return userCompanies;
    } catch (error) {
      console.error("[Brreg] Error fetching saved companies:", error);
      throw new TRPCError({
        code: "INTERNAL_SERVER_ERROR",
        message: "Failed to fetch saved companies",
      });
    }
  }),

  /**
   * Get company details by ID (including roles)
   */
  getCompanyById: protectedProcedure
    .input(z.object({ companyId: z.string() }))
    .query(async ({ input }) => {
      const db = await getDb();
      if (!db) {
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: "Database connection failed",
        });
      }

      try {
        const company = await db
          .select()
          .from(companies)
          .where(eq(companies.id, input.companyId))
          .limit(1);

        if (company.length === 0) {
          throw new TRPCError({
            code: "NOT_FOUND",
            message: "Company not found",
          });
        }

        const roles = await db
          .select()
          .from(companyRoles)
          .where(eq(companyRoles.companyId, input.companyId));

        return {
          company: company[0],
          roles,
        };
      } catch (error) {
        if (error instanceof TRPCError) throw error;
        console.error("[Brreg] Error fetching company details:", error);
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: "Failed to fetch company details",
        });
      }
    }),
});
