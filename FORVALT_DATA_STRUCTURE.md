# Forvalt.no Data Structure

## Example: EQUINOR ASA (923609016)

### Credit Rating & Risk
- **Proff Premium Rating:** A+ (Meget lav risiko)
- **Score:** 95/100
- **Bankruptcy Probability:** 0.11%
- **Credit Limit:** NOK 493,080,000,000

### Rating Components
1. **Leadership & Ownership:** 5/5
2. **Economy:** 5/5
3. **Payment History:** 2/5
4. **General:** 5/5

### Financial Metrics (2024, USD thousands)
- **Revenue:** 72,543,000
- **EBITDA:** 11,036,000
- **Operating Result:** 10,347,000
- **Total Assets:** 109,150,000
- **Profitability:** 9.2% (Satisfactory)
- **Liquidity:** 1.07 (Satisfactory)
- **Solidity:** 37.6% (Good)
- **EBITDA Margin:** 15.2%

### Payment Remarks
- **Voluntary Liens:** 0
- **Factoring Agreements:** 0
- **Forced Liens:** 0
- **Payment Remarks:** Yes (See details)

### Company Information
- **Org Number:** 923609016 MVA
- **Company Name:** EQUINOR ASA
- **Organization Form:** Public Limited Company (ASA)
- **Share Capital:** 6,392,018,780
- **Founded:** 18.09.1972
- **Registered:** 12.03.1995
- **Employees:** 21,496 (AA-register), 21,000 (annual report)
- **Website:** www.equinor.com
- **Phone:** 51 99 00 00

### Address
- **Business Address:** Forusbeen 50, 4035 STAVANGER
- **Postal Address:** Postboks 8500, 4035 STAVANGER

### Industry Classification
- **Sector:** 1120 - State-owned joint-stock companies
- **NACE Primary:** 06.100 - Extraction of crude oil
- **NACE Secondary:** 06.200 - Extraction of natural gas
- **NACE Tertiary:** 19.200 - Production of refined petroleum products

### Leadership (Updated 05.12.2025)
- **CEO:** Anders Opedal
- **Board Chairman:** Jon Erik Reinhardsen
- **Deputy Chairman:** Anne Drinkwater
- **Auditor:** ERNST & YOUNG AS

### Shareholders (Top 20, as of 31.12.2024)
1. NÆRINGS- OG FISKERIDEPARTEMENTET - 67.00%
2. Jpmorgan Chase Bank, N.A., London - 5.09%
3. State Street Bank And Trust Comp - 4.88%
4. FOLKETRYGDFONDET - 3.99%
5. EQUINOR ASA - 2.30%

### Available Premium Reports
- **Valuation Report Premium:** NOK 5,990
- **Credit Report Premium:** NOK 200
- **Company Report Premium:** NOK 495
- **Annual Report 2024:** NOK 15
- **Register Extract:** NOK 125
- **Company Information:** NOK 45

## Data Integration Strategy

### API Endpoints (Estimated)
Since Forvalt.no doesn't have a public API, we'll use web scraping:

1. **Company Search:** `https://forvalt.no/ForetaksIndex/Firma/FirmaSide/{orgnr}`
2. **Credit Rating:** Extract from "Proff Premium rating" section
3. **Financial Data:** Extract from "Nøkkeltall" table
4. **Payment Remarks:** Extract from "Betalingsanmerkninger" section
5. **Leadership:** Extract from "Roller" section

### Data Points to Extract
```typescript
interface ForvaltCompanyData {
  // Credit Rating
  creditRating: string; // "A+", "A", "B", etc.
  creditScore: number; // 0-100
  bankruptcyProbability: number; // 0-100%
  creditLimit: number; // NOK
  riskLevel: "very_low" | "low" | "moderate" | "high" | "very_high";
  
  // Rating Components
  leadershipScore: number; // 1-5
  economyScore: number; // 1-5
  paymentHistoryScore: number; // 1-5
  generalScore: number; // 1-5
  
  // Financial Metrics
  revenue: number;
  ebitda: number;
  operatingResult: number;
  totalAssets: number;
  profitability: number; // %
  liquidity: number;
  solidity: number; // %
  ebitdaMargin: number; // %
  
  // Payment Remarks
  voluntaryLiens: number;
  factoringAgreements: number;
  forcedLiens: number;
  hasPaymentRemarks: boolean;
  
  // Company Info
  companyName: string;
  orgNumber: string;
  organizationForm: string;
  shareCapital: number;
  founded: string;
  employees: number;
  website: string;
  phone: string;
  
  // Leadership
  ceo: string;
  boardChairman: string;
  auditor: string;
}
```

## Implementation Plan

1. **Create Forvalt Scraper Module** (`server/helpers/forvalt_scraper.ts`)
   - Use Puppeteer or Playwright for authenticated scraping
   - Store session cookies for persistent login
   - Extract all data points from company page

2. **Integrate into businessOrchestrator**
   - Call Forvalt scraper after Brreg fetch
   - Combine Brreg + Forvalt data for comprehensive analysis
   - Use credit rating in AI recommendations

3. **Display in CompanyLookup Page**
   - Show credit rating badge (A+, A, B, etc.)
   - Display bankruptcy probability
   - Show financial health indicators
   - Add "View Full Forvalt Report" button

4. **Store in Database**
   - Save to `company_financials` table
   - Update on each analysis run
   - Track historical changes

## Next Steps

1. Install Puppeteer: `pnpm add puppeteer`
2. Create `forvalt_scraper.ts` with login + scraping logic
3. Update `businessOrchestrator` to call Forvalt scraper
4. Add Forvalt data display to CompanyLookup page
5. Test with real Norwegian companies
