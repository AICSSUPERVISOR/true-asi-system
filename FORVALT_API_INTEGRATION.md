# Forvalt.no API Integration Documentation

## Overview
Proff Forvalt provides premium Norwegian company data including credit ratings, financial statements, and risk scores through their REST API.

## API Offerings

### 1. Proff Premium API
**Base Features:**
- Company basic information (orgnr, name, addresses, industry, status)
- Updated phone numbers
- Financial statements (as soon as available)
- Historical financial data
- Role information (board, auditor, accountant, owner, CEO)
- Shareholders, subsidiaries, and ownership interests
- Group relations
- Beneficial owners

### 2. Proff Credit API
**Credit & Risk Features:**
- **Rating/Score**: Bankruptcy rating and credit score
- **Credit Limit**: Recommended credit limit for the company
- **Payment Remarks**: Payment history and defaults
- **Liens**: Registered liens and encumbrances
- **Automated Credit Check**: Real-time credit assessment

### 3. Compliance API
**AML/KYC Features:**
- **KYC (Know Your Customer)**: Identity verification
- **AML (Anti-Money Laundering)**: Risk assessment
- **PEP (Politically Exposed Persons)**: PEP and PEP RCA checks
- **Sanctions**: EU, UN, GB, OFAC (USA) sanctions lists
- **SOE (State-Owned Enterprises)**: International state-owned companies
- **Compliance Report**: Full compliance documentation

## Technical Details
- **API Type**: REST API
- **Data Format**: JSON (recommended) and XML
- **Coverage**: Norway, Sweden, Denmark, Finland
- **Update Frequency**: Daily
- **Documentation**: Available at Forvalt.no (requires account)

## Integration Strategy for TRUE ASI System

### Phase 1: Basic Company Data (IMPLEMENTED)
✅ Brreg.no integration for free basic company data
- Organization number validation
- Company name, address, industry
- Board members and CEO

### Phase 2: Premium Financial Data (TO IMPLEMENT)
🔄 Forvalt.no Premium API integration
- Financial statements (revenue, profit, assets, liabilities)
- Historical financial data (3-5 years)
- Key financial ratios
- Shareholders and ownership structure

### Phase 3: Credit & Risk Assessment (TO IMPLEMENT)
🔄 Forvalt.no Credit API integration
- Credit score (0-100 scale)
- Credit rating (AAA, AA, A, BBB, BB, B, CCC, CC, C, D)
- Bankruptcy risk probability
- Recommended credit limit
- Payment remarks and defaults

### Phase 4: Compliance Checks (TO IMPLEMENT)
🔄 Forvalt.no Compliance API integration
- PEP screening
- Sanctions list checking
- AML risk assessment
- KYC compliance report

## API Access
**Contact**: Proff Forvalt
**Phone**: +47 21 51 66 02
**Website**: https://forvalt.no/ProffAPI/PremiumAPI
**Note**: Requires paid subscription and API credentials

## Implementation Plan

### Step 1: API Credentials Setup
1. Contact Forvalt.no for API access
2. Receive API key and endpoint URLs
3. Store credentials in environment variables:
   - `FORVALT_API_KEY`
   - `FORVALT_API_URL`

### Step 2: Backend Integration
1. Create `server/routers/forvalt.ts` tRPC router
2. Implement procedures:
   - `forvalt.getFinancialData` - Fetch financial statements
   - `forvalt.getCreditRating` - Get credit score and rating
   - `forvalt.getComplianceCheck` - Run PEP/sanctions check
3. Save data to `company_financials` table

### Step 3: Frontend Display
1. Add financial metrics to CompanyLookup page
2. Display credit rating badge (AAA, AA, A, etc.)
3. Show bankruptcy risk indicator
4. Add compliance status badges

### Step 4: AI Analysis Integration
1. Include financial data in businessOrchestrator
2. Use credit rating in risk assessment
3. Factor compliance status into recommendations

## Data Mapping

### Credit Rating Scale
| Rating | Description | Risk Level |
|--------|-------------|------------|
| AAA | Excellent | Very Low |
| AA | Very Good | Low |
| A | Good | Low-Medium |
| BBB | Satisfactory | Medium |
| BB | Adequate | Medium-High |
| B | Weak | High |
| CCC | Very Weak | Very High |
| CC | Extremely Weak | Extremely High |
| C | Default Imminent | Critical |
| D | Default | Failed |

### Bankruptcy Risk Probability
- 0-1%: Very Low Risk (Green)
- 1-5%: Low Risk (Light Green)
- 5-10%: Medium Risk (Yellow)
- 10-20%: High Risk (Orange)
- 20%+: Very High Risk (Red)

## Cost Considerations
- Forvalt.no API requires paid subscription
- Pricing based on API call volume
- Consider caching strategy to minimize API calls
- Cache financial data for 24 hours
- Cache credit ratings for 7 days

## Alternative: Free Data Sources
If Forvalt.no API is not available, use these free alternatives:
1. **Brreg.no** - Basic company data (already integrated)
2. **Proff.no** - Public financial statements (web scraping)
3. **Regnskapstall.no** - Historical financial data (web scraping)

## Status
- **Brreg.no Integration**: ✅ COMPLETE
- **Forvalt.no API Access**: ⏳ PENDING (requires subscription)
- **Financial Data Integration**: 📋 PLANNED
- **Credit Rating Display**: 📋 PLANNED
- **Compliance Checks**: 📋 PLANNED
