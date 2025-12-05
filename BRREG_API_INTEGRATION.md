# Brreg.no API Integration Documentation

## API Overview

**Base URL:** `https://data.brreg.no/enhetsregisteret/api`

**License:** Norsk lisens for offentlige data (NLOD) - No registration required

**Documentation:** https://data.brreg.no/enhetsregisteret/api/dokumentasjon/no/index.html

---

## Key Endpoints for TRUE ASI System

### 1. Search for Companies (Enheter)
```
GET /api/enheter?navn={companyName}
GET /api/enheter?organisasjonsnummer={orgnr}
```

**Response:** JSON (HAL format) with company details

### 2. Get Specific Company by Organization Number
```
GET /api/enheter/{orgnr}
```

**Example:** `GET /api/enheter/974760673`

**Response Fields:**
- `organisasjonsnummer` - 9-digit organization number
- `navn` - Company name
- `organisasjonsform` - Organization form (AS, ASA, ENK, etc.)
- `registreringsdatoEnhetsregisteret` - Registration date
- `registrertIMvaregisteret` - VAT registered (boolean)
- `naeringskode1` - Primary industry code
- `antallAnsatte` - Number of employees
- `forretningsadresse` - Business address
- `postadresse` - Postal address
- `institusjonellSektorkode` - Institutional sector code
- `registrertIForetaksregisteret` - Registered in business registry (boolean)
- `registrertIStiftelsesregisteret` - Registered in foundation registry (boolean)
- `registrertIFrivillighetsregisteret` - Registered in voluntary registry (boolean)
- `konkurs` - Bankruptcy status (boolean)
- `underAvvikling` - Under liquidation (boolean)
- `underTvangsavviklingEllerTvangsopplosning` - Under forced liquidation (boolean)
- `maalform` - Language form (Bokmål/Nynorsk)

### 3. Get Company Roles (Roller)
```
GET /api/enheter/{orgnr}/roller
```

**Response:** List of roles (board members, CEO, auditor, etc.)

**Role Types:**
- `INNH` - Owner (innehaver)
- `DAGL` - CEO (daglig leder)
- `LEDE` - Board chairman (styrets leder)
- `NEST` - Deputy chairman (nestleder)
- `MEDL` - Board member (styremedlem)
- `VARA` - Deputy member (varamedlem)
- `REVI` - Auditor (revisor)
- `REGN` - Accountant (regnskapsfører)

### 4. Get Subunits (Underenheter)
```
GET /api/underenheter?overordnetEnhet={orgnr}
```

**Response:** List of subunits for a parent company

### 5. Download Complete Dataset
```
GET /api/enheter/lastned          # JSON format
GET /api/enheter/lastned/csv      # CSV format
GET /api/enheter/lastned/regneark # Excel format
```

---

## Integration Plan for TRUE ASI System

### Phase 1: Organization Number Input & Company Lookup

**Frontend Component:** `client/src/pages/CompanyLookup.tsx`
- Input field for 9-digit organization number
- Search button
- Display company information card

**Backend tRPC Procedure:** `server/routers/brreg.ts`
```typescript
getCompanyByOrgnr: publicProcedure
  .input(z.object({ orgnr: z.string().length(9) }))
  .query(async ({ input }) => {
    const response = await fetch(
      `https://data.brreg.no/enhetsregisteret/api/enheter/${input.orgnr}`
    );
    if (!response.ok) throw new TRPCError({ code: 'NOT_FOUND' });
    return await response.json();
  }),
```

### Phase 2: Company Data Storage

**Database Schema:** `drizzle/schema.ts`
```typescript
export const companies = mysqlTable("companies", {
  id: varchar("id", { length: 128 }).primaryKey(),
  orgnr: varchar("orgnr", { length: 9 }).notNull().unique(),
  name: text("name").notNull(),
  organizationForm: varchar("organizationForm", { length: 50 }),
  registrationDate: timestamp("registrationDate"),
  industryCode: varchar("industryCode", { length: 10 }),
  employees: int("employees"),
  businessAddress: text("businessAddress"),
  postalAddress: text("postalAddress"),
  vatRegistered: int("vatRegistered"), // boolean as int
  bankrupt: int("bankrupt"), // boolean as int
  underLiquidation: int("underLiquidation"), // boolean as int
  rawData: text("rawData"), // Store full JSON response
  createdAt: timestamp("createdAt").notNull().defaultNow(),
  updatedAt: timestamp("updatedAt").notNull().defaultNow().onUpdateNow(),
});
```

### Phase 3: Company Roles Integration

**Database Schema:**
```typescript
export const companyRoles = mysqlTable("company_roles", {
  id: varchar("id", { length: 128 }).primaryKey(),
  companyId: varchar("companyId", { length: 128 }).notNull(),
  roleType: varchar("roleType", { length: 10 }).notNull(), // DAGL, LEDE, etc.
  personName: text("personName"),
  personBirthDate: varchar("personBirthDate", { length: 10 }),
  organizationNumber: varchar("organizationNumber", { length: 9 }),
  createdAt: timestamp("createdAt").notNull().defaultNow(),
});
```

### Phase 4: Automated Data Enrichment Pipeline

**Workflow:**
1. User inputs organization number
2. Fetch company data from Brreg.no
3. Fetch company roles from Brreg.no
4. Store in database
5. Trigger Proff.no financial data fetch
6. Trigger LinkedIn company enrichment
7. Trigger multi-model AI analysis
8. Generate recommendations with deeplinks

---

## Example API Calls

### Get Company by Organization Number
```bash
curl -X GET "https://data.brreg.no/enhetsregisteret/api/enheter/974760673" \
  -H "Accept: application/json"
```

**Response:**
```json
{
  "organisasjonsnummer": "974760673",
  "navn": "BRØNNØYSUNDREGISTRENE",
  "organisasjonsform": {
    "kode": "ORGL",
    "beskrivelse": "Ordinært organ for stat eller kommune"
  },
  "registreringsdatoEnhetsregisteret": "1995-08-09",
  "registrertIMvaregisteret": false,
  "naeringskode1": {
    "beskrivelse": "Generell offentlig administrasjon",
    "kode": "84.110"
  },
  "antallAnsatte": 321,
  "forretningsadresse": {
    "land": "Norge",
    "landkode": "NO",
    "postnummer": "8910",
    "poststed": "BRØNNØYSUND",
    "adresse": ["Havnegata 48"],
    "kommune": "BRØNNØY",
    "kommunenummer": "1813"
  },
  "institusjonellSektorkode": {
    "kode": "6100",
    "beskrivelse": "Statsforvaltningen"
  },
  "registrertIForetaksregisteret": false,
  "registrertIStiftelsesregisteret": false,
  "registrertIFrivillighetsregisteret": false,
  "konkurs": false,
  "underAvvikling": false,
  "underTvangsavviklingEllerTvangsopplosning": false,
  "maalform": "Bokmål"
}
```

### Get Company Roles
```bash
curl -X GET "https://data.brreg.no/enhetsregisteret/api/enheter/974760673/roller" \
  -H "Accept: application/json"
```

---

## Error Handling

### 404 Not Found
- Organization number does not exist
- Return user-friendly error message

### 400 Bad Request
- Invalid organization number format (must be 9 digits)
- Validate input before API call

### Rate Limiting
- No explicit rate limits documented
- Implement exponential backoff for retries

---

## Next Steps

1. ✅ Create `server/routers/brreg.ts` with tRPC procedures
2. ✅ Create `companies` and `companyRoles` database tables
3. ✅ Create `client/src/pages/CompanyLookup.tsx` component
4. ✅ Integrate with existing business analysis workflow
5. ✅ Connect to Proff.no API for financial data
6. ✅ Connect to LinkedIn API for social data
7. ✅ Build automated pipeline orchestration
