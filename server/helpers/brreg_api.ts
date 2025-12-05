/**
 * Brønnøysund Register API Integration
 * Official Norwegian business registry API
 * Documentation: https://data.brreg.no/enhetsregisteret/api/docs/index.html
 */

import { cacheGet, cacheSet } from './cache';

const BRREG_API_BASE = 'https://data.brreg.no/enhetsregisteret/api';
const CACHE_TTL = 24 * 60 * 60; // 24 hours

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
    postnummer?: string;
    poststed?: string;
    adresse?: string[];
    kommune?: string;
    kommunenummer?: string;
  };
  forretningsadresse?: {
    land: string;
    landkode: string;
    postnummer?: string;
    poststed?: string;
    adresse?: string[];
    kommune?: string;
    kommunenummer?: string;
  };
  stiftelsesdato?: string;
  registreringsdatoEnhetsregisteret?: string;
  registrertIMvaregisteret?: boolean;
  naeringskode1?: {
    beskrivelse: string;
    kode: string;
  };
  antallAnsatte?: number;
  overordnetEnhet?: string;
  oppstartsdato?: string;
  datoEierskifte?: string;
  vedtektsdato?: string;
  vedtektsfestetFormaal?: string;
  aktivitet?: string;
}

export interface BrregSearchResult {
  _embedded?: {
    enheter: BrregCompany[];
  };
  page: {
    size: number;
    totalElements: number;
    totalPages: number;
    number: number;
  };
}

/**
 * Search for companies by organization number
 */
export async function searchCompanyByOrgNumber(orgNumber: string): Promise<BrregCompany | null> {
  const cacheKey = `brreg:company:${orgNumber}`;
  
  // Check cache first
  const cached = await cacheGet<BrregCompany>(cacheKey);
  if (cached) {
    console.log(`[Brønnøysund] Cache hit for ${orgNumber}`);
    return cached;
  }

  try {
    console.log(`[Brønnøysund] Fetching company data for ${orgNumber}`);
    
    const response = await fetch(`${BRREG_API_BASE}/enheter/${orgNumber}`, {
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      if (response.status === 404) {
        console.log(`[Brønnøysund] Company not found: ${orgNumber}`);
        return null;
      }
      throw new Error(`Brønnøysund API error: ${response.status} ${response.statusText}`);
    }

    const data: BrregCompany = await response.json();
    
    // Cache the result
    await cacheSet(cacheKey, data, CACHE_TTL);
    
    console.log(`[Brønnøysund] Successfully fetched data for ${data.navn}`);
    return data;
  } catch (error) {
    console.error(`[Brønnøysund] Error fetching company:`, error);
    throw error;
  }
}

/**
 * Search for companies by name
 */
export async function searchCompaniesByName(name: string, limit: number = 10): Promise<BrregCompany[]> {
  const cacheKey = `brreg:search:${name}:${limit}`;
  
  // Check cache first
  const cached = await cacheGet<BrregCompany[]>(cacheKey);
  if (cached) {
    console.log(`[Brønnøysund] Cache hit for search: ${name}`);
    return cached;
  }

  try {
    console.log(`[Brønnøysund] Searching companies by name: ${name}`);
    
    const response = await fetch(
      `${BRREG_API_BASE}/enheter?navn=${encodeURIComponent(name)}&size=${limit}`,
      {
        headers: {
          'Accept': 'application/json',
        },
      }
    );

    if (!response.ok) {
      throw new Error(`Brønnøysund API error: ${response.status} ${response.statusText}`);
    }

    const data: BrregSearchResult = await response.json();
    const companies = data._embedded?.enheter || [];
    
    // Cache the result
    await cacheSet(cacheKey, companies, CACHE_TTL);
    
    console.log(`[Brønnøysund] Found ${companies.length} companies`);
    return companies;
  } catch (error) {
    console.error(`[Brønnøysund] Error searching companies:`, error);
    throw error;
  }
}

/**
 * Get company's parent organization
 */
export async function getParentOrganization(orgNumber: string): Promise<BrregCompany | null> {
  const company = await searchCompanyByOrgNumber(orgNumber);
  
  if (!company || !company.overordnetEnhet) {
    return null;
  }

  return searchCompanyByOrgNumber(company.overordnetEnhet);
}

/**
 * Get company's subsidiaries
 */
export async function getSubsidiaries(orgNumber: string): Promise<BrregCompany[]> {
  const cacheKey = `brreg:subsidiaries:${orgNumber}`;
  
  // Check cache first
  const cached = await cacheGet<BrregCompany[]>(cacheKey);
  if (cached) {
    console.log(`[Brønnøysund] Cache hit for subsidiaries: ${orgNumber}`);
    return cached;
  }

  try {
    console.log(`[Brønnøysund] Fetching subsidiaries for ${orgNumber}`);
    
    const response = await fetch(
      `${BRREG_API_BASE}/enheter?overordnetEnhet=${orgNumber}&size=100`,
      {
        headers: {
          'Accept': 'application/json',
        },
      }
    );

    if (!response.ok) {
      throw new Error(`Brønnøysund API error: ${response.status} ${response.statusText}`);
    }

    const data: BrregSearchResult = await response.json();
    const subsidiaries = data._embedded?.enheter || [];
    
    // Cache the result
    await cacheSet(cacheKey, subsidiaries, CACHE_TTL);
    
    console.log(`[Brønnøysund] Found ${subsidiaries.length} subsidiaries`);
    return subsidiaries;
  } catch (error) {
    console.error(`[Brønnøysund] Error fetching subsidiaries:`, error);
    throw error;
  }
}

/**
 * Get industry information
 */
export function getIndustryInfo(company: BrregCompany): {
  code: string;
  name: string;
  category: string;
} {
  const naeringskode = company.naeringskode1;
  
  if (!naeringskode) {
    return {
      code: 'UNKNOWN',
      name: 'Unknown Industry',
      category: 'other'
    };
  }

  // Map NACE codes to our industry categories
  const code = naeringskode.kode;
  const name = naeringskode.beskrivelse;
  
  let category = 'other';
  
  // Healthcare: 86-88
  if (code.startsWith('86') || code.startsWith('87') || code.startsWith('88')) {
    category = 'healthcare';
  }
  // Finance: 64-66
  else if (code.startsWith('64') || code.startsWith('65') || code.startsWith('66')) {
    category = 'finance';
  }
  // Retail/E-commerce: 47
  else if (code.startsWith('47')) {
    category = 'ecommerce';
  }
  // Manufacturing: 10-33
  else if (parseInt(code.substring(0, 2)) >= 10 && parseInt(code.substring(0, 2)) <= 33) {
    category = 'manufacturing';
  }
  // Construction: 41-43
  else if (code.startsWith('41') || code.startsWith('42') || code.startsWith('43')) {
    category = 'construction';
  }
  // Real Estate: 68
  else if (code.startsWith('68')) {
    category = 'real_estate';
  }
  // IT/Technology: 62-63
  else if (code.startsWith('62') || code.startsWith('63')) {
    category = 'technology';
  }
  // Education: 85
  else if (code.startsWith('85')) {
    category = 'education';
  }
  // Hospitality: 55-56
  else if (code.startsWith('55') || code.startsWith('56')) {
    category = 'hospitality';
  }
  // Transportation: 49-53
  else if (parseInt(code.substring(0, 2)) >= 49 && parseInt(code.substring(0, 2)) <= 53) {
    category = 'transportation';
  }
  
  return { code, name, category };
}

/**
 * Format company data for our system
 */
export function formatCompanyData(company: BrregCompany) {
  const industry = getIndustryInfo(company);
  
  return {
    organizationNumber: company.organisasjonsnummer,
    name: company.navn,
    legalForm: company.organisasjonsform.beskrivelse,
    website: company.hjemmeside,
    address: {
      street: company.forretningsadresse?.adresse?.join(', '),
      postalCode: company.forretningsadresse?.postnummer,
      city: company.forretningsadresse?.poststed,
      municipality: company.forretningsadresse?.kommune,
      country: company.forretningsadresse?.land || 'Norway'
    },
    foundedDate: company.stiftelsesdato,
    registeredDate: company.registreringsdatoEnhetsregisteret,
    employees: company.antallAnsatte,
    industry: {
      code: industry.code,
      name: industry.name,
      category: industry.category
    },
    parentOrganization: company.overordnetEnhet,
    registeredInVAT: company.registrertIMvaregisteret || false,
    purpose: company.vedtektsfestetFormaal,
    activity: company.aktivitet
  };
}
