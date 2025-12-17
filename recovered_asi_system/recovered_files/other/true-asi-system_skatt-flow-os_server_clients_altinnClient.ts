import axios, { AxiosInstance, AxiosError } from "axios";
import { nanoid } from "nanoid";
import type { AltinnTokenResponse, AltinnDraftResponse, AltinnSubmitResponse, FilingType } from "@shared/types";

// ============================================================================
// ALTINN API CLIENT
// For Norwegian government reporting (MVA, SAF-T, A-melding)
// ============================================================================

export class AltinnApiError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public correlationId?: string
  ) {
    super(message);
    this.name = "AltinnApiError";
  }
}

interface AltinnClientConfig {
  baseUrl: string;
  clientId: string;
  clientSecret: string;
  scope: string;
  timeout?: number;
}

interface CachedToken {
  accessToken: string;
  expiresAt: number;
}

export class AltinnClient {
  private client: AxiosInstance;
  private config: AltinnClientConfig;
  private cachedToken: CachedToken | null = null;

  constructor(config: AltinnClientConfig) {
    this.config = config;
    this.client = axios.create({
      baseURL: config.baseUrl,
      timeout: config.timeout || 60000,
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
    });
  }

  private getCorrelationId(): string {
    return nanoid(16);
  }

  /**
   * Authenticate using OAuth2 client credentials flow
   * Caches the token until expiry
   */
  async authenticate(): Promise<string> {
    // Return cached token if still valid (with 60s buffer)
    if (this.cachedToken && this.cachedToken.expiresAt > Date.now() + 60000) {
      return this.cachedToken.accessToken;
    }

    const correlationId = this.getCorrelationId();

    try {
      const tokenUrl = `${this.config.baseUrl}/token`;
      const params = new URLSearchParams({
        grant_type: "client_credentials",
        client_id: this.config.clientId,
        client_secret: this.config.clientSecret,
        scope: this.config.scope,
      });

      const response = await axios.post<AltinnTokenResponse>(tokenUrl, params.toString(), {
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
          "X-Correlation-Id": correlationId,
        },
      });

      const { access_token, expires_in } = response.data;

      this.cachedToken = {
        accessToken: access_token,
        expiresAt: Date.now() + expires_in * 1000,
      };

      return access_token;
    } catch (error) {
      if (error instanceof AxiosError) {
        throw new AltinnApiError(
          `Altinn authentication failed: ${error.message}`,
          error.response?.status,
          correlationId
        );
      }
      throw error;
    }
  }

  private async getAuthHeaders(correlationId: string) {
    const token = await this.authenticate();
    return {
      Authorization: `Bearer ${token}`,
      "X-Correlation-Id": correlationId,
    };
  }

  /**
   * Create a draft filing in Altinn
   */
  async createDraftFiling(
    filingType: FilingType,
    orgNumber: string,
    payloadJson: Record<string, unknown>
  ): Promise<AltinnDraftResponse> {
    const correlationId = this.getCorrelationId();

    try {
      const serviceCode = this.getServiceCode(filingType);
      const headers = await this.getAuthHeaders(correlationId);

      const response = await this.client.post(
        `/api/v1/instances`,
        {
          instanceOwner: {
            organisationNumber: orgNumber.replace(/\s/g, ""),
          },
          appId: serviceCode,
          dataType: this.getDataType(filingType),
          data: payloadJson,
        },
        { headers }
      );

      const data = response.data;
      return {
        draftId: String(data.id || data.instanceId || ""),
        status: String(data.status || "DRAFT"),
        createdAt: data.created || new Date().toISOString(),
      };
    } catch (error) {
      if (error instanceof AxiosError) {
        throw new AltinnApiError(
          `Failed to create draft filing: ${error.message}`,
          error.response?.status,
          correlationId
        );
      }
      throw error;
    }
  }

  /**
   * Submit a draft filing to Altinn
   * IMPORTANT: Only call after explicit user confirmation
   */
  async submitFiling(draftId: string): Promise<AltinnSubmitResponse> {
    const correlationId = this.getCorrelationId();

    try {
      const headers = await this.getAuthHeaders(correlationId);

      const response = await this.client.post(
        `/api/v1/instances/${draftId}/process/next`,
        {},
        { headers }
      );

      const data = response.data;
      return {
        reference: String(data.reference || data.receiptId || data.id || ""),
        status: String(data.status || "SUBMITTED"),
        submittedAt: data.submittedAt || data.ended || new Date().toISOString(),
      };
    } catch (error) {
      if (error instanceof AxiosError) {
        throw new AltinnApiError(
          `Failed to submit filing: ${error.message}`,
          error.response?.status,
          correlationId
        );
      }
      throw error;
    }
  }

  /**
   * Get the status of a filing
   */
  async getFilingStatus(draftId: string): Promise<{ status: string; reference?: string }> {
    const correlationId = this.getCorrelationId();

    try {
      const headers = await this.getAuthHeaders(correlationId);

      const response = await this.client.get(`/api/v1/instances/${draftId}`, { headers });

      const data = response.data;
      return {
        status: String(data.status?.isArchived ? "SUBMITTED" : data.status?.currentTask || "DRAFT"),
        reference: data.reference || data.receiptId,
      };
    } catch (error) {
      if (error instanceof AxiosError) {
        throw new AltinnApiError(
          `Failed to get filing status: ${error.message}`,
          error.response?.status,
          correlationId
        );
      }
      throw error;
    }
  }

  /**
   * Delete a draft filing (before submission)
   */
  async deleteDraft(draftId: string): Promise<boolean> {
    const correlationId = this.getCorrelationId();

    try {
      const headers = await this.getAuthHeaders(correlationId);

      await this.client.delete(`/api/v1/instances/${draftId}`, { headers });

      return true;
    } catch (error) {
      if (error instanceof AxiosError) {
        throw new AltinnApiError(
          `Failed to delete draft: ${error.message}`,
          error.response?.status,
          correlationId
        );
      }
      throw error;
    }
  }

  private getServiceCode(filingType: FilingType): string {
    const serviceCodes: Record<FilingType, string> = {
      MVA_MELDING: "skd/mva-melding",
      A_MELDING_SUMMARY: "skd/a-melding",
      SAF_T: "skd/saf-t-regnskap",
      ARSREGNSKAP: "brreg/arsregnskap",
      OTHER: "generic/filing",
    };
    return serviceCodes[filingType] || serviceCodes.OTHER;
  }

  private getDataType(filingType: FilingType): string {
    const dataTypes: Record<FilingType, string> = {
      MVA_MELDING: "mvamelding",
      A_MELDING_SUMMARY: "amelding",
      SAF_T: "saft",
      ARSREGNSKAP: "arsregnskap",
      OTHER: "generic",
    };
    return dataTypes[filingType] || dataTypes.OTHER;
  }
}

// Factory function to create client from environment variables
export function createAltinnClient(): AltinnClient {
  const baseUrl = process.env.ALTINN_BASE_URL || "https://platform.altinn.no";
  const clientId = process.env.ALTINN_CLIENT_ID || "";
  const clientSecret = process.env.ALTINN_CLIENT_SECRET || "";
  const scope = process.env.ALTINN_SCOPE || "altinn:instances.read altinn:instances.write";

  if (!clientId || !clientSecret) {
    console.warn("[AltinnClient] ALTINN_CLIENT_ID or ALTINN_CLIENT_SECRET not set - API calls will fail");
  }

  return new AltinnClient({ baseUrl, clientId, clientSecret, scope });
}
