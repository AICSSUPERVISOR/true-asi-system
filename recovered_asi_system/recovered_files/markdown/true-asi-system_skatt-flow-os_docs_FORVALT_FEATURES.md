# Forvalt.no Full Feature Integration Map

## Logged In Successfully
- **User**: Lucas (Kundenr: 90347)
- **Account**: LL2020365@gmail.com

## Available Features from Forvalt.no Dashboard

### Main Navigation
1. **Startside** - Main dashboard with market statistics
2. **Segmentering** - Market segmentation tools
3. **Analyse** - Company analysis tools
4. **PEP & Sanksjoner** - Politically Exposed Persons & Sanctions screening
5. **Overvåking** - Company monitoring/alerts
6. **Aksjonærregisteret** - Shareholder registry
7. **Eiendomsregisteret** - Property registry
8. **Utenlandske foretak** - Foreign companies

### Quick Access Tools
- **Nyetableringer** - New company registrations
- **Nettverksøk** - Network search (ownership structures)
- **Oslo Børs** - Stock exchange data
- **Konkursåpninger** - Bankruptcy filings
- **Personopplysninger** - Personal information lookup
- **PEP & Sanksjoner** - PEP/Sanctions screening

### Real-time Statistics (from dashboard)
- **Nyetableringer siste døgn**: 318 (3,079 last 30 days, +48.9% last week)
- **Konkurser siste døgn**: 5 (413 last 30 days, +15.5% last week)
- **Betalingsanmerkninger**: 47,632 active companies (4.3 mrd NOK total)
- **Tvungen pant**: 5,116 companies (2.6 mrd NOK total)
- **Aktive roller**: 3.4 million
- **Historiske roller**: 8.2 million
- **Aktive selskaper**: 1.1 million
- **Inaktive selskaper**: 1.6 million

### Risk Classification System
- **A-klasse**: Most solid companies, very low bankruptcy risk
- **B-klasse**: Moderate risk, stable but some vulnerability
- **C-klasse**: High risk - 20x higher bankruptcy rate than A-class

### Board Composition Analytics
- Gender distribution across age groups
- Compliance with gender balance requirements
- Current requirements: 10,904 companies, 85.31% compliant
- Future requirements: 19,993 companies, 78.19% compliant

## API Endpoints to Integrate

### Company Search & Data
- `/ForetaksIndex` - Main search
- `/ForetaksIndex/Segmentering` - Segmentation
- `/ForetaksIndex/Analyse` - Analysis
- `/ForetaksIndex/Sanctions` - PEP & Sanctions
- `/ForetaksIndex/Overvaaking/MineFirmaer` - Monitoring
- `/ForetaksIndex/Shareholders` - Shareholder registry
- `/ForetaksIndex/Eiendom` - Property registry
- `/ForetaksIndex/UtenlandskeForetak` - Foreign companies

### Data Points per Company
1. **Basic Info**: Name, Org.nr, Address, Industry code
2. **Financial Data**: Revenue, Profit, Assets, Liabilities
3. **Credit Score**: Rating (A/B/C), Credit limit, Risk assessment
4. **Payment Remarks**: Betalingsanmerkninger, amounts, dates
5. **Forced Liens**: Tvungen pant details
6. **Ownership**: Shareholders, ownership percentages
7. **Board/Management**: Roles, persons, historical changes
8. **Property**: Real estate holdings
9. **Legal**: Court cases, bankruptcy history
10. **PEP/Sanctions**: Screening results

## Integration Requirements for Skatt-Flow OS

### Must Implement
1. Company lookup by org.nr or name
2. Credit check with full rating details
3. Financial statement retrieval
4. Role/board member lookup
5. Payment remarks check
6. Monitoring/alerts setup
7. PEP & Sanctions screening
8. Ownership structure visualization

### API Authentication
- Session-based authentication via login
- Customer number: 90347
- Need to check for API key access in "Min Forvalt"


## Subscription Details (Forvalt Premium)

### Account Information
- **Kundenr**: 90347
- **Brukernavn**: LL2020365@gmail.com
- **Firma**: INNOVATECH KAPITAL AS
- **Postadresse**: c/o Lucas Bjelland Armauer-Hansen Markalleen 44, 1368 STABEKK
- **Telefon**: 944 67 815
- **Utløpsdato**: 05.12.2026

### Included Features (Forvalt Premium)
1. **Overvåking** - Up to 5,000 companies
2. **Basisinformasjon** - Basic company info
3. **Regnskapsinformasjon** - Financial statements
4. **Rating/Scoringsmodell** - Credit rating/scoring
5. **Eksporter DM- og TM-lister til Excel** - Export marketing lists
6. **Eksporter regnskapstall til Excel** - Export financials
7. **Konkurrentanalyser** - Competitor analysis
8. **Ekstra felt - utvidet eksport** - Extended export fields
9. **Firmarapporter** - Company reports
10. **Saker i domstolene** - Court cases
11. **PropCloud data** - Property data
12. **Betalingsanmerkninger** - Payment remarks (not for sole proprietors/individuals)

### Available Add-on Modules
- Utenlandske foretak (Foreign companies) - 0 clips
- Klippekort PEP (PEP screening) - 0 clips  
- Klippekort betalingsanmerkninger (Payment remarks clips) - 0 clips

## Integration Strategy for Skatt-Flow OS

Since this is a Premium subscription with web access (not API), we need to:
1. Use web scraping with authenticated session for data retrieval
2. Implement caching to minimize requests
3. Store Forvalt data in our database for quick access
4. Sync data periodically rather than real-time

### Key Data Points to Extract per Company
- Basic info (name, org.nr, address, industry)
- Financial statements (revenue, profit, assets, liabilities)
- Credit rating (A/B/C score, credit limit)
- Payment remarks (betalingsanmerkninger)
- Court cases (saker i domstolene)
- Property data (PropCloud)
- Ownership structure
- Board/management roles


## Complete Company Data Fields (from Equinor ASA example)

### Basic Information
- **Organisasjonsnr**: 923609016 MVA
- **Selskapsnavn**: EQUINOR ASA
- **Organisasjonsform**: Allmennaksjeselskap (ASA)
- **Aksjekapital**: 6,392,018,780
- **Stiftelsedato**: 18.09.1972
- **Registreringsdato**: 12.03.1995
- **Status**: Aktivt
- **Registrert i**: Foretaksregisteret, Merverdiavgiftsmanntallet, NAV AA-registeret, Oslo Børs
- **Kontaktperson**: DAGL Anders Opedal
- **Internett**: www.equinor.com
- **Telefon**: 51 99 00 00
- **Antall ansatte**: 21,496 (Aa-registeret) / 21,000 (årsregnskap)
- **EHF-faktura**: Ja
- **Forretningsadresse**: Forusbeen 50, 4035 STAVANGER
- **Postadresse**: Postboks 8500, 4035 STAVANGER
- **Underavdelinger**: 35 enheter
- **Klienter (regnskap)**: 58 foretak

### Industry Classification
- **Sektor**: 1120 - Statlig eide aksjeselskaper mv.
- **NACE-bransje**: 06.100 - Utvinning av råolje
- **Sekundær NACE**: 06.200 - Utvinning av naturgass
- **Tertiær NACE**: 19.200 - Produksjon av raffinerte petroleumsprodukter
- **Proff-bransje**: Olje og gass, Utvinning av råolje og naturgass
- **Vedtektsfestet formål**: Full text available

### Credit Rating (Proff Premium)
- **Rating**: A+ Meget lav risiko
- **Score**: 95/100
- **Konkursrisiko**: 0.11%
- **Kredittramme**: 493,080,000,000 NOK
- **Vurderinger**:
  - Ledelse og eierskap: 5
  - Økonomi: 5
  - Betalingshistorikk: 2
  - Generelt: 5

### Financial Analysis
- **Lønnsomhet**: 9.2% (Tilfredsstillende)
- **Likviditetsgrad 1**: 1.07 (Tilfredsstillende)
- **Soliditet**: 37.6% (God)
- **EBITDA**: 15.2%

### Financial Statements (Regnskapstall)
| Year | Valuta | Sum driftsinnt. | Driftsresultat | Ord. res. f. skatt | Sum eiend. |
|------|--------|-----------------|----------------|--------------------| -----------|
| 2024 | USD | 72,543,000 | 10,347,000 | 8,168,000 | 109,150,000 |
| 2023 | USD | 72,442,000 | 10,658,000 | 12,299,000 | 127,220,000 |
| 2022 | USD | 96,784,000 | 28,365,000 | 27,478,000 | 159,342,000 |

### Payment Remarks & Liens
- **Frivillig pant**: 0
- **Factoringavtaler**: 0
- **Tvungen pant**: 0
- **Betalingsanmerkninger**: Ja (details available)

### Court Cases
- Ingen aktive saker funnet
- Historical cases available

### Property Information (PropCloud)
- **Adresse**: Forusbeen 50, 4035 Stavanger
- **Tomt**: 63,069.9 m²
- **Type**: Kontor- og adm.bygning rådhus
- **Sist omsatt**: 04.06.2015
- **Eiere**: FORUSBEEN 50 AS

### Management & Board (Roller)
- **Daglig leder**: Anders Opedal
- **Styrets leder**: Jon Erik Reinhardsen
- **Nestleder**: Anne Drinkwater
- **Styremedlemmer**: Multiple listed
- **Varamedlemmer**: Listed
- **Revisor**: ERNST & YOUNG AS
- **Signatur**: Rules defined
- **Prokura**: Multiple persons listed

### Shareholders (from Skatteetaten)
- Top 20 shareholders with percentages
- Total 127,204 shareholders
- Largest: NÆRINGS- OG FISKERIDEPARTEMENTET (67.00%)

### Subsidiaries (Datterselskaper)
- Full list with org.nr and ownership percentage
- 100% owned subsidiaries listed

### Other Ownership Interests
- Minority stakes in other companies

### Announcements (Kunngjøringer)
- Last 10 announcements with dates and types
- Full history available

### Available Premium Products
- Verdirapport Premium: NOK 5,990
- Verdirapport Premium Konsern: NOK 8,990
- Årsregnskap: NOK 15 per year
- Kredittrapport Premium: NOK 200
- Firmarapport Premium: NOK 495
- Registerutskrift: NOK 125
- Foretaksopplysninger: NOK 45
- Rolleopplysninger: NOK 15
- Konkursregister bekreftelse: NOK 15
- Firmaattest: NOK 20
- Pantattest: NOK 115

## Tabs Available per Company
1. **Firmainformasjon** - Basic company info
2. **Regnskapstall** - Financial statements
3. **Regnskapsanalyse** - Financial analysis
4. **AI-Analyse ✨** - AI-powered analysis (NEW!)
5. **Betalingsanmerkninger og pant** - Payment remarks and liens
6. **Nettverk** - Network/ownership structure
