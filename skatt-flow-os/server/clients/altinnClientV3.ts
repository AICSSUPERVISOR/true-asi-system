import axios, { AxiosInstance, AxiosError } from "axios";
import { nanoid } from "nanoid";
import * as crypto from "crypto";

// ============================================================================
// ALTINN V3 API CLIENT - TRIPLETEX-LEVEL INTEGRATION
// Full integration with all Altinn services matching Tripletex capabilities
// ============================================================================

export class AltinnApiError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public correlationId?: string,
    public altinnErrorCode?: string
  ) {
    super(message);
    this.name = "AltinnApiError";
  }
}

// ============================================================================
// TYPES
// ============================================================================

export interface MaskinportenConfig {
  clientId: string;
  privateKeyPem: string;
  scopes: string[];
  issuer?: string;
  audience?: string;
  tokenEndpoint?: string;
}

export interface AltinnClientConfig {
  environment: "test" | "production";
  maskinporten: MaskinportenConfig;
  orgNumber: string;
}

export interface MvaReturn {
  termin: string; // e.g., "2024-01" for January 2024
  grunnlagUtgaaendeMva: number;
  utgaaendeMva: number;
  grunnlagInngaaendeMva: number;
  inngaaendeMva: number;
  mvaATilgode: number;
  mvaABetale: number;
  poster: MvaPost[];
}

export interface MvaPost {
  postNr: string;
  beskrivelse: string;
  grunnlag?: number;
  sats?: number;
  mvaBeloep: number;
}

export interface AMelding {
  kalendermaaned: string; // e.g., "2024-01"
  arbeidsgiverId: string;
  inntektsmottakere: Inntektsmottaker[];
  oppsummering: AMeldingOppsummering;
}

export interface Inntektsmottaker {
  personId: string;
  navn: string;
  loenn: number;
  feriepenger: number;
  trekkpliktigLoenn: number;
  forskuddstrekk: number;
  arbeidsgiveravgiftGrunnlag: number;
}

export interface AMeldingOppsummering {
  sumLoenn: number;
  sumFeriepenger: number;
  sumForskuddstrekk: number;
  sumArbeidsgiveravgift: number;
  arbeidsgiveravgiftSats: number;
}

export interface SaftExport {
  companyId: string;
  orgNumber: string;
  periodStart: string;
  periodEnd: string;
  xmlContent: string;
  validationResult?: SaftValidationResult;
}

export interface SaftValidationResult {
  isValid: boolean;
  errors: SaftValidationError[];
  warnings: SaftValidationWarning[];
}

export interface SaftValidationError {
  code: string;
  message: string;
  path?: string;
}

export interface SaftValidationWarning {
  code: string;
  message: string;
  path?: string;
}

export interface AarsregnskapSubmission {
  orgNumber: string;
  regnskapsaar: number;
  balanse: BalanseData;
  resultat: ResultatData;
  noter: Note[];
  revisorBeretning?: string;
  styreBeretning: string;
}

export interface BalanseData {
  sumEiendeler: number;
  anleggsmidler: number;
  omloepsmidler: number;
  sumEgenkapitalOgGjeld: number;
  egenkapital: number;
  langsiktigGjeld: number;
  kortsiktigGjeld: number;
}

export interface ResultatData {
  driftsinntekter: number;
  driftskostnader: number;
  driftsresultat: number;
  finansinntekter: number;
  finanskostnader: number;
  ordinaertResultatFoerSkatt: number;
  skattekostnad: number;
  aarsresultat: number;
}

export interface Note {
  noteNr: number;
  tittel: string;
  innhold: string;
}

export interface AksjonaeroppgaveSubmission {
  orgNumber: string;
  regnskapsaar: number;
  aksjekapital: number;
  antallAksjer: number;
  aksjonaerer: Aksjonaer[];
}

export interface Aksjonaer {
  type: "person" | "selskap";
  identifikator: string; // Fødselsnummer or org number
  navn: string;
  antallAksjer: number;
  eierandel: number;
  stemmeandel: number;
}

export interface FilingStatus {
  id: string;
  type: string;
  status: "draft" | "submitted" | "processing" | "accepted" | "rejected" | "error";
  submittedAt?: string;
  processedAt?: string;
  receiptId?: string;
  errorMessage?: string;
  altinnReference?: string;
}

export interface FilingDeadline {
  type: string;
  termin: string;
  deadline: Date;
  daysRemaining: number;
  status: "upcoming" | "due_soon" | "overdue" | "submitted";
}

// ============================================================================
// MASKINPORTEN JWT GENERATOR
// ============================================================================

class MaskinportenAuth {
  private config: MaskinportenConfig;
  private tokenCache: { token: string; expiresAt: number } | null = null;

  constructor(config: MaskinportenConfig) {
    this.config = config;
  }

  private base64UrlEncode(data: string | Buffer): string {
    const base64 = Buffer.from(data).toString("base64");
    return base64.replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
  }

  private createJwtAssertion(): string {
    const now = Math.floor(Date.now() / 1000);
    const header = {
      alg: "RS256",
      typ: "JWT",
    };

    const payload = {
      iss: this.config.clientId,
      aud: this.config.audience || "https://maskinporten.no/",
      scope: this.config.scopes.join(" "),
      iat: now,
      exp: now + 120, // 2 minutes
      jti: nanoid(32),
    };

    const headerB64 = this.base64UrlEncode(JSON.stringify(header));
    const payloadB64 = this.base64UrlEncode(JSON.stringify(payload));
    const signatureInput = `${headerB64}.${payloadB64}`;

    const sign = crypto.createSign("RSA-SHA256");
    sign.update(signatureInput);
    const signature = sign.sign(this.config.privateKeyPem);
    const signatureB64 = this.base64UrlEncode(signature);

    return `${signatureInput}.${signatureB64}`;
  }

  async getAccessToken(): Promise<string> {
    // Check cache
    if (this.tokenCache && this.tokenCache.expiresAt > Date.now() + 60000) {
      return this.tokenCache.token;
    }

    const tokenEndpoint =
      this.config.tokenEndpoint || "https://maskinporten.no/token";

    const assertion = this.createJwtAssertion();

    const response = await axios.post(
      tokenEndpoint,
      new URLSearchParams({
        grant_type: "urn:ietf:params:oauth:grant-type:jwt-bearer",
        assertion: assertion,
      }),
      {
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
        },
      }
    );

    const { access_token, expires_in } = response.data;

    this.tokenCache = {
      token: access_token,
      expiresAt: Date.now() + expires_in * 1000,
    };

    return access_token;
  }
}

// ============================================================================
// ALTINN V3 CLIENT
// ============================================================================

export class AltinnClientV3 {
  private client: AxiosInstance;
  private auth: MaskinportenAuth;
  private orgNumber: string;
  private baseUrl: string;

  constructor(config: AltinnClientConfig) {
    this.orgNumber = config.orgNumber;
    this.baseUrl =
      config.environment === "production"
        ? "https://platform.altinn.no"
        : "https://platform.tt02.altinn.no";

    this.auth = new MaskinportenAuth(config.maskinporten);

    this.client = axios.create({
      baseURL: this.baseUrl,
      timeout: 60000,
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
    });
  }

  private getCorrelationId(): string {
    return nanoid(16);
  }

  private async getAuthHeaders(correlationId: string) {
    const token = await this.auth.getAccessToken();
    return {
      Authorization: `Bearer ${token}`,
      "X-Correlation-Id": correlationId,
      "Altinn-Party-Id": this.orgNumber,
    };
  }

  // ============================================================================
  // MVA-MELDING (VAT RETURN)
  // ============================================================================

  async createMvaMelding(mvaReturn: MvaReturn): Promise<{ id: string; status: string }> {
    const correlationId = this.getCorrelationId();

    try {
      const headers = await this.getAuthHeaders(correlationId);

      // Generate XML for MVA-melding
      const xmlContent = this.generateMvaXml(mvaReturn);

      const response = await this.client.post(
        "/skd/mva-melding/v1/innsending",
        {
          skjemainnhold: xmlContent,
          innsender: {
            organisasjonsnummer: this.orgNumber,
          },
          termin: mvaReturn.termin,
        },
        { headers }
      );

      return {
        id: response.data.id || response.data.referanse,
        status: "submitted",
      };
    } catch (error) {
      if (error instanceof AxiosError) {
        throw new AltinnApiError(
          `Failed to submit MVA-melding: ${error.message}`,
          error.response?.status,
          correlationId,
          error.response?.data?.errorCode
        );
      }
      throw error;
    }
  }

  private generateMvaXml(mvaReturn: MvaReturn): string {
    const posts = mvaReturn.poster
      .map(
        (p) => `
      <mvaSpesifikasjonslinje>
        <mvaKode>${p.postNr}</mvaKode>
        <grunnlag>${p.grunnlag || 0}</grunnlag>
        <sats>${p.sats || 0}</sats>
        <merverdiavgift>${p.mvaBeloep}</merverdiavgift>
      </mvaSpesifikasjonslinje>`
      )
      .join("");

    return `<?xml version="1.0" encoding="UTF-8"?>
<mvaMeldingDto xmlns="no:skatteetaten:fastsetting:avgift:mva:mvameldinginnsending:v1.0">
  <innsending>
    <regnskapssystemreferanse>SKATTFLOW-${nanoid(8)}</regnskapssystemreferanse>
    <regnskapssystem>
      <systemnavn>Skatt-Flow OS</systemnavn>
      <systemversjon>2.0</systemversjon>
    </regnskapssystem>
  </innsending>
  <skattegrunnlagOgBeregnetSkatt>
    <skattleggingsperiode>
      <periode>${mvaReturn.termin}</periode>
      <skattleggingsperiodeType>maaned</skattleggingsperiodeType>
    </skattleggingsperiode>
    <fastsattMerverdiavgift>${mvaReturn.mvaABetale - mvaReturn.mvaATilgode}</fastsattMerverdiavgift>
    ${posts}
  </skattegrunnlagOgBeregnetSkatt>
</mvaMeldingDto>`;
  }

  async validateMvaMelding(mvaReturn: MvaReturn): Promise<SaftValidationResult> {
    const correlationId = this.getCorrelationId();

    try {
      const headers = await this.getAuthHeaders(correlationId);
      const xmlContent = this.generateMvaXml(mvaReturn);

      const response = await this.client.post(
        "/skd/mva-melding/v1/validering",
        { skjemainnhold: xmlContent },
        { headers }
      );

      return {
        isValid: response.data.status === "GYLDIG",
        errors: response.data.feil || [],
        warnings: response.data.advarsler || [],
      };
    } catch (error) {
      return {
        isValid: false,
        errors: [{ code: "VALIDATION_ERROR", message: String(error) }],
        warnings: [],
      };
    }
  }

  // ============================================================================
  // A-MELDING (PAYROLL REPORTING)
  // ============================================================================

  async createAMelding(aMelding: AMelding): Promise<{ id: string; status: string }> {
    const correlationId = this.getCorrelationId();

    try {
      const headers = await this.getAuthHeaders(correlationId);

      const xmlContent = this.generateAMeldingXml(aMelding);

      const response = await this.client.post(
        "/edag/a-melding/v1/innsending",
        {
          skjemainnhold: xmlContent,
          arbeidsgiverId: aMelding.arbeidsgiverId,
          kalendermaaned: aMelding.kalendermaaned,
        },
        { headers }
      );

      return {
        id: response.data.id || response.data.referanse,
        status: "submitted",
      };
    } catch (error) {
      if (error instanceof AxiosError) {
        throw new AltinnApiError(
          `Failed to submit A-melding: ${error.message}`,
          error.response?.status,
          correlationId
        );
      }
      throw error;
    }
  }

  private generateAMeldingXml(aMelding: AMelding): string {
    const inntektsmottakere = aMelding.inntektsmottakere
      .map(
        (im) => `
      <inntektsmottaker>
        <norskIdentifikator>${im.personId}</norskIdentifikator>
        <inntekt>
          <loennsinntekt>
            <beloep>${im.loenn}</beloep>
            <fordel>kontantytelse</fordel>
          </loennsinntekt>
        </inntekt>
        <forskuddstrekk>
          <beloep>${im.forskuddstrekk}</beloep>
        </forskuddstrekk>
      </inntektsmottaker>`
      )
      .join("");

    return `<?xml version="1.0" encoding="UTF-8"?>
<melding xmlns="urn:no:skatteetaten:datasamarbeid:a-melding:v1.0">
  <leveranse>
    <kalendermaaned>${aMelding.kalendermaaned}</kalendermaaned>
    <opplysningspliktig>
      <norskIdentifikator>${aMelding.arbeidsgiverId}</norskIdentifikator>
    </opplysningspliktig>
    ${inntektsmottakere}
    <arbeidsgiveravgift>
      <grunnlag>${aMelding.oppsummering.sumLoenn}</grunnlag>
      <beloep>${aMelding.oppsummering.sumArbeidsgiveravgift}</beloep>
    </arbeidsgiveravgift>
  </leveranse>
</melding>`;
  }

  // ============================================================================
  // SAF-T EXPORT
  // ============================================================================

  async generateSaftExport(
    companyId: string,
    periodStart: string,
    periodEnd: string,
    ledgerData: unknown[]
  ): Promise<SaftExport> {
    const xmlContent = this.generateSaftXml(companyId, periodStart, periodEnd, ledgerData);

    const validationResult = await this.validateSaftExport(xmlContent);

    return {
      companyId,
      orgNumber: this.orgNumber,
      periodStart,
      periodEnd,
      xmlContent,
      validationResult,
    };
  }

  private generateSaftXml(
    companyId: string,
    periodStart: string,
    periodEnd: string,
    ledgerData: unknown[]
  ): string {
    const entries = Array.isArray(ledgerData)
      ? ledgerData
          .map(
            (entry: unknown, idx: number) => {
              const e = entry as { accountNumber?: string; description?: string; debit?: number; credit?: number; transactionDate?: string };
              return `
        <Transaction>
          <TransactionID>${idx + 1}</TransactionID>
          <Period>${new Date(periodStart).getMonth() + 1}</Period>
          <TransactionDate>${e.transactionDate || periodStart}</TransactionDate>
          <Description>${e.description || ""}</Description>
          <Line>
            <AccountID>${e.accountNumber || "0000"}</AccountID>
            <DebitAmount>${e.debit || 0}</DebitAmount>
            <CreditAmount>${e.credit || 0}</CreditAmount>
          </Line>
        </Transaction>`;
            }
          )
          .join("")
      : "";

    return `<?xml version="1.0" encoding="UTF-8"?>
<AuditFile xmlns="urn:StandardAuditFile-Taxation-Financial:NO">
  <Header>
    <AuditFileVersion>1.10</AuditFileVersion>
    <AuditFileCountry>NO</AuditFileCountry>
    <AuditFileDateCreated>${new Date().toISOString().split("T")[0]}</AuditFileDateCreated>
    <SoftwareCompanyName>Skatt-Flow OS</SoftwareCompanyName>
    <SoftwareID>SKATTFLOW</SoftwareID>
    <SoftwareVersion>2.0</SoftwareVersion>
    <Company>
      <RegistrationNumber>${this.orgNumber}</RegistrationNumber>
      <Name>Company ${companyId}</Name>
    </Company>
    <SelectionCriteria>
      <SelectionStartDate>${periodStart}</SelectionStartDate>
      <SelectionEndDate>${periodEnd}</SelectionEndDate>
    </SelectionCriteria>
  </Header>
  <GeneralLedgerEntries>
    <NumberOfEntries>${Array.isArray(ledgerData) ? ledgerData.length : 0}</NumberOfEntries>
    ${entries}
  </GeneralLedgerEntries>
</AuditFile>`;
  }

  async validateSaftExport(xmlContent: string): Promise<SaftValidationResult> {
    const correlationId = this.getCorrelationId();

    try {
      const headers = await this.getAuthHeaders(correlationId);

      const response = await this.client.post(
        "/skd/saft/v1/validering",
        { xmlContent },
        { headers }
      );

      return {
        isValid: response.data.status === "GYLDIG",
        errors: response.data.feil || [],
        warnings: response.data.advarsler || [],
      };
    } catch {
      // If validation endpoint fails, do basic XML validation
      const errors: SaftValidationError[] = [];
      const warnings: SaftValidationWarning[] = [];

      if (!xmlContent.includes("AuditFile")) {
        errors.push({ code: "MISSING_ROOT", message: "Missing AuditFile root element" });
      }
      if (!xmlContent.includes("Header")) {
        errors.push({ code: "MISSING_HEADER", message: "Missing Header element" });
      }

      return {
        isValid: errors.length === 0,
        errors,
        warnings,
      };
    }
  }

  // ============================================================================
  // ÅRSREGNSKAP (ANNUAL ACCOUNTS)
  // ============================================================================

  async submitAarsregnskap(
    submission: AarsregnskapSubmission
  ): Promise<{ id: string; status: string }> {
    const correlationId = this.getCorrelationId();

    try {
      const headers = await this.getAuthHeaders(correlationId);

      const response = await this.client.post(
        "/br/aarsregnskap/v1/innsending",
        {
          organisasjonsnummer: submission.orgNumber,
          regnskapsaar: submission.regnskapsaar,
          balanse: submission.balanse,
          resultat: submission.resultat,
          noter: submission.noter,
          styreBeretning: submission.styreBeretning,
          revisorBeretning: submission.revisorBeretning,
        },
        { headers }
      );

      return {
        id: response.data.id || response.data.referanse,
        status: "submitted",
      };
    } catch (error) {
      if (error instanceof AxiosError) {
        throw new AltinnApiError(
          `Failed to submit Årsregnskap: ${error.message}`,
          error.response?.status,
          correlationId
        );
      }
      throw error;
    }
  }

  // ============================================================================
  // AKSJONÆRREGISTEROPPGAVEN
  // ============================================================================

  async submitAksjonaeroppgave(
    submission: AksjonaeroppgaveSubmission
  ): Promise<{ id: string; status: string }> {
    const correlationId = this.getCorrelationId();

    try {
      const headers = await this.getAuthHeaders(correlationId);

      const response = await this.client.post(
        "/skd/aksjonaerregister/v1/innsending",
        {
          organisasjonsnummer: submission.orgNumber,
          regnskapsaar: submission.regnskapsaar,
          aksjekapital: submission.aksjekapital,
          antallAksjer: submission.antallAksjer,
          aksjonaerer: submission.aksjonaerer,
        },
        { headers }
      );

      return {
        id: response.data.id || response.data.referanse,
        status: "submitted",
      };
    } catch (error) {
      if (error instanceof AxiosError) {
        throw new AltinnApiError(
          `Failed to submit Aksjonærregisteroppgaven: ${error.message}`,
          error.response?.status,
          correlationId
        );
      }
      throw error;
    }
  }

  // ============================================================================
  // FILING STATUS
  // ============================================================================

  async getFilingStatus(filingId: string, filingType: string): Promise<FilingStatus> {
    const correlationId = this.getCorrelationId();

    try {
      const headers = await this.getAuthHeaders(correlationId);

      const endpoint = this.getStatusEndpoint(filingType);
      const response = await this.client.get(`${endpoint}/${filingId}/status`, { headers });

      return {
        id: filingId,
        type: filingType,
        status: this.mapAltinnStatus(response.data.status),
        submittedAt: response.data.innsendtTidspunkt,
        processedAt: response.data.behandletTidspunkt,
        receiptId: response.data.kvitteringsId,
        errorMessage: response.data.feilmelding,
        altinnReference: response.data.altinnReferanse,
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

  private getStatusEndpoint(filingType: string): string {
    const endpoints: Record<string, string> = {
      MVA_MELDING: "/skd/mva-melding/v1/innsending",
      A_MELDING: "/edag/a-melding/v1/innsending",
      SAF_T: "/skd/saft/v1/innsending",
      ARSREGNSKAP: "/br/aarsregnskap/v1/innsending",
      AKSJONAEROPPGAVE: "/skd/aksjonaerregister/v1/innsending",
    };
    return endpoints[filingType] || endpoints.MVA_MELDING;
  }

  private mapAltinnStatus(
    altinnStatus: string
  ): "draft" | "submitted" | "processing" | "accepted" | "rejected" | "error" {
    const statusMap: Record<string, "draft" | "submitted" | "processing" | "accepted" | "rejected" | "error"> = {
      MOTTATT: "submitted",
      UNDER_BEHANDLING: "processing",
      GODKJENT: "accepted",
      AVVIST: "rejected",
      FEIL: "error",
    };
    return statusMap[altinnStatus] || "submitted";
  }

  // ============================================================================
  // DEADLINE MANAGEMENT
  // ============================================================================

  getFilingDeadlines(year: number): FilingDeadline[] {
    const deadlines: FilingDeadline[] = [];
    const now = new Date();

    // MVA deadlines (10th of month after termin ends, bi-monthly for most)
    for (let termin = 1; termin <= 6; termin++) {
      const terminEnd = new Date(year, termin * 2, 0);
      const deadline = new Date(year, termin * 2, 10);

      deadlines.push({
        type: "MVA_MELDING",
        termin: `${year}-T${termin}`,
        deadline,
        daysRemaining: Math.ceil((deadline.getTime() - now.getTime()) / (1000 * 60 * 60 * 24)),
        status: deadline < now ? "overdue" : deadline.getTime() - now.getTime() < 7 * 24 * 60 * 60 * 1000 ? "due_soon" : "upcoming",
      });
    }

    // A-melding deadlines (5th of following month)
    for (let month = 1; month <= 12; month++) {
      const deadline = new Date(year, month, 5);

      deadlines.push({
        type: "A_MELDING",
        termin: `${year}-${month.toString().padStart(2, "0")}`,
        deadline,
        daysRemaining: Math.ceil((deadline.getTime() - now.getTime()) / (1000 * 60 * 60 * 24)),
        status: deadline < now ? "overdue" : deadline.getTime() - now.getTime() < 7 * 24 * 60 * 60 * 1000 ? "due_soon" : "upcoming",
      });
    }

    // Årsregnskap deadline (July 31st)
    const aarsregnskapDeadline = new Date(year + 1, 6, 31);
    deadlines.push({
      type: "ARSREGNSKAP",
      termin: `${year}`,
      deadline: aarsregnskapDeadline,
      daysRemaining: Math.ceil((aarsregnskapDeadline.getTime() - now.getTime()) / (1000 * 60 * 60 * 24)),
      status: aarsregnskapDeadline < now ? "overdue" : aarsregnskapDeadline.getTime() - now.getTime() < 30 * 24 * 60 * 60 * 1000 ? "due_soon" : "upcoming",
    });

    // Aksjonærregisteroppgaven deadline (January 31st)
    const aksjonaerDeadline = new Date(year + 1, 0, 31);
    deadlines.push({
      type: "AKSJONAEROPPGAVE",
      termin: `${year}`,
      deadline: aksjonaerDeadline,
      daysRemaining: Math.ceil((aksjonaerDeadline.getTime() - now.getTime()) / (1000 * 60 * 60 * 24)),
      status: aksjonaerDeadline < now ? "overdue" : aksjonaerDeadline.getTime() - now.getTime() < 30 * 24 * 60 * 60 * 1000 ? "due_soon" : "upcoming",
    });

    return deadlines.sort((a, b) => a.deadline.getTime() - b.deadline.getTime());
  }
}

// ============================================================================
// FACTORY FUNCTION
// ============================================================================

export function createAltinnClientV3(): AltinnClientV3 {
  const clientId = process.env.MASKINPORTEN_CLIENT_ID || "";
  const privateKey = process.env.MASKINPORTEN_PRIVATE_KEY || "";
  const orgNumber = process.env.ALTINN_ORG_NUMBER || "";
  const environment = (process.env.ALTINN_ENVIRONMENT || "test") as "test" | "production";

  return new AltinnClientV3({
    environment,
    orgNumber,
    maskinporten: {
      clientId,
      privateKeyPem: privateKey,
      scopes: [
        "skatteetaten:mvameldinginnsending",
        "skatteetaten:a-melding",
        "skatteetaten:saft",
        "brreg:aarsregnskap",
        "skatteetaten:aksjonaerregister",
      ],
    },
  });
}
