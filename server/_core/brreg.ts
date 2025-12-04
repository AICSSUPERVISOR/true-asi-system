/**
 * Brønnøysund Register Centre (BRREG) API Integration
 * 
 * Official Norwegian Business Registry API
 * Documentation: https://data.brreg.no/enhetsregisteret/api/dokumentasjon/en/index.html
 * License: Norwegian Licence for Open Government Data (NLOD)
 * Authentication: None required (open data)
 */

const BRREG_BASE_URL = "https://data.brreg.no/enhetsregisteret/api";

export interface BrregCompany {
  organisasjonsnummer: string;
  navn: string;
  organisasjonsform: {
    kode: string;
    beskrivelse: string;
  };
  hjemmeside?: string;
  postadresse?: {
    land: string;
    landkode: string;
    postnummer: string;
    poststed: string;
    adresse: string[];
    kommune: string;
    kommunenummer: string;
  };
  forretningsadresse?: {
    land: string;
    landkode: string;
    postnummer: string;
    poststed: string;
    adresse: string[];
    kommune: string;
    kommunenummer: string;
  };
  registreringsdatoEnhetsregisteret: string;
  registrertIMvaregisteret: boolean;
  naeringskode1?: {
    kode: string;
    beskrivelse: string;
  };
  naeringskode2?: {
    kode: string;
    beskrivelse: string;
  };
  naeringskode3?: {
    kode: string;
    beskrivelse: string;
  };
  antallAnsatte?: number;
  harRegistrertAntallAnsatte: boolean;
  overordnetEnhet?: string;
  epostadresse?: string;
  telefon?: string;
  institusjonellSektorkode?: {
    kode: string;
    beskrivelse: string;
  };
  sisteInnsendteAarsregnskap?: string;
  konkurs?: boolean;
  underAvvikling?: boolean;
  underTvangsavviklingEllerTvangsopplosning?: boolean;
  maalform?: string;
  _links?: {
    self: {
      href: string;
    };
    overordnetEnhet?: {
      href: string;
    };
  };
}

export interface BrregSearchResult {
  _embedded: {
    enheter: BrregCompany[];
  };
  page: {
    size: number;
    totalElements: number;
    totalPages: number;
    number: number;
  };
  _links: {
    self: {
      href: string;
    };
    first?: {
      href: string;
    };
    next?: {
      href: string;
    };
    last?: {
      href: string;
    };
  };
}

/**
 * Get company information by organization number
 * @param orgNumber - Norwegian organization number (9 digits)
 * @returns Company information or null if not found
 */
export async function getCompanyByOrgNumber(
  orgNumber: string
): Promise<BrregCompany | null> {
  try {
    // Remove spaces and validate format
    const cleanOrgNumber = orgNumber.replace(/\s/g, "");
    
    if (!/^\d{9}$/.test(cleanOrgNumber)) {
      throw new Error("Invalid organization number format. Must be 9 digits.");
    }

    const response = await fetch(
      `${BRREG_BASE_URL}/enheter/${cleanOrgNumber}`
    );

    if (!response.ok) {
      if (response.status === 404) {
        return null;
      }
      throw new Error(`BRREG API error: ${response.status} ${response.statusText}`);
    }

    const data: BrregCompany = await response.json();
    return data;
  } catch (error) {
    console.error("Error fetching company from BRREG:", error);
    throw error;
  }
}

/**
 * Search for companies by name
 * @param name - Company name to search for
 * @param page - Page number (0-indexed)
 * @param size - Number of results per page (max 100)
 * @returns Search results
 */
export async function searchCompaniesByName(
  name: string,
  page: number = 0,
  size: number = 20
): Promise<BrregSearchResult> {
  try {
    const params = new URLSearchParams({
      navn: name,
      page: page.toString(),
      size: Math.min(size, 100).toString(),
    });

    const response = await fetch(
      `${BRREG_BASE_URL}/enheter?${params.toString()}`
    );

    if (!response.ok) {
      throw new Error(`BRREG API error: ${response.status} ${response.statusText}`);
    }

    const data: BrregSearchResult = await response.json();
    return data;
  } catch (error) {
    console.error("Error searching companies in BRREG:", error);
    throw error;
  }
}

/**
 * Get parent company information
 * @param orgNumber - Organization number of the parent company
 * @returns Parent company information or null if not found
 */
export async function getParentCompany(
  orgNumber: string
): Promise<BrregCompany | null> {
  return getCompanyByOrgNumber(orgNumber);
}

/**
 * Categorize company industry based on NACE codes
 * @param company - Company data from BRREG
 * @returns Industry category
 */
export function categorizeIndustry(company: BrregCompany): string {
  if (!company.naeringskode1) {
    return "Unknown";
  }

  const code = company.naeringskode1.kode;
  const firstTwo = code.substring(0, 2);

  // NACE Rev. 2 industry classification
  const industryMap: Record<string, string> = {
    "01": "Agriculture",
    "02": "Forestry",
    "03": "Fishing",
    "05": "Mining - Coal",
    "06": "Mining - Oil & Gas",
    "07": "Mining - Metal Ores",
    "08": "Mining - Other",
    "09": "Mining Support Services",
    "10": "Food Products",
    "11": "Beverages",
    "12": "Tobacco Products",
    "13": "Textiles",
    "14": "Wearing Apparel",
    "15": "Leather Products",
    "16": "Wood Products",
    "17": "Paper Products",
    "18": "Printing",
    "19": "Petroleum Products",
    "20": "Chemicals",
    "21": "Pharmaceuticals",
    "22": "Rubber & Plastics",
    "23": "Non-Metallic Minerals",
    "24": "Basic Metals",
    "25": "Fabricated Metal Products",
    "26": "Computer & Electronics",
    "27": "Electrical Equipment",
    "28": "Machinery",
    "29": "Motor Vehicles",
    "30": "Other Transport Equipment",
    "31": "Furniture",
    "32": "Other Manufacturing",
    "33": "Repair & Installation",
    "35": "Electricity & Gas",
    "36": "Water Supply",
    "37": "Sewerage",
    "38": "Waste Management",
    "39": "Remediation",
    "41": "Construction - Buildings",
    "42": "Construction - Civil Engineering",
    "43": "Construction - Specialized",
    "45": "Wholesale & Retail - Motor Vehicles",
    "46": "Wholesale Trade",
    "47": "Retail Trade",
    "49": "Land Transport",
    "50": "Water Transport",
    "51": "Air Transport",
    "52": "Warehousing",
    "53": "Postal Services",
    "55": "Accommodation",
    "56": "Food Services",
    "58": "Publishing",
    "59": "Film & Video",
    "60": "Broadcasting",
    "61": "Telecommunications",
    "62": "IT Services",
    "63": "Information Services",
    "64": "Financial Services",
    "65": "Insurance",
    "66": "Financial Support Activities",
    "68": "Real Estate",
    "69": "Legal & Accounting",
    "70": "Management Consulting",
    "71": "Architecture & Engineering",
    "72": "Scientific Research",
    "73": "Advertising & Market Research",
    "74": "Professional Services",
    "75": "Veterinary",
    "77": "Rental & Leasing",
    "78": "Employment Services",
    "79": "Travel & Tourism",
    "80": "Security Services",
    "81": "Facility Services",
    "82": "Office Support Services",
    "84": "Public Administration",
    "85": "Education",
    "86": "Healthcare",
    "87": "Residential Care",
    "88": "Social Work",
    "90": "Creative Arts",
    "91": "Libraries & Museums",
    "92": "Gambling",
    "93": "Sports & Recreation",
    "94": "Membership Organizations",
    "95": "Repair Services",
    "96": "Personal Services",
    "97": "Household Services",
    "98": "Household Production",
    "99": "International Organizations",
  };

  return industryMap[firstTwo] || "Other";
}

/**
 * Extract key business information for AI analysis
 * @param company - Company data from BRREG
 * @returns Structured business information
 */
export function extractBusinessInfo(company: BrregCompany) {
  return {
    organizationNumber: company.organisasjonsnummer,
    name: company.navn,
    industry: categorizeIndustry(company),
    industryCode: company.naeringskode1?.kode,
    industryDescription: company.naeringskode1?.beskrivelse,
    employees: company.antallAnsatte || 0,
    website: company.hjemmeside,
    email: company.epostadresse,
    phone: company.telefon,
    address: company.forretningsadresse || company.postadresse,
    registrationDate: company.registreringsdatoEnhetsregisteret,
    vatRegistered: company.registrertIMvaregisteret,
    parentCompany: company.overordnetEnhet,
    organizationType: company.organisasjonsform.beskrivelse,
    isBankrupt: company.konkurs || false,
    isUnderLiquidation: company.underAvvikling || false,
  };
}
