/**
 * Database helpers for business profiles
 */

import mysql from "mysql2/promise";
import { ENV } from "./_core/env";

async function getConnection() {
  return await mysql.createConnection(ENV.databaseUrl);
}
import type { BrregCompany } from "./_core/brreg";

export interface BusinessProfile {
  id: number;
  organization_number: string;
  name: string;
  industry: string | null;
  industry_code: string | null;
  industry_description: string | null;
  employees: number;
  website: string | null;
  email: string | null;
  phone: string | null;
  address_street: string | null;
  address_postal_code: string | null;
  address_city: string | null;
  address_country: string | null;
  registration_date: Date | null;
  vat_registered: boolean;
  parent_company: string | null;
  organization_type: string | null;
  is_bankrupt: boolean;
  is_under_liquidation: boolean;
  raw_data: any;
  created_at: Date;
  updated_at: Date;
}

/**
 * Save or update business profile from BRREG data
 */
export async function saveBusinessProfile(
  brregData: BrregCompany
): Promise<BusinessProfile> {
  const connection = await getConnection();

  const address = brregData.forretningsadresse || brregData.postadresse;

  const profileData = {
    organization_number: brregData.organisasjonsnummer,
    name: brregData.navn,
    industry: brregData.naeringskode1?.beskrivelse || null,
    industry_code: brregData.naeringskode1?.kode || null,
    industry_description: brregData.naeringskode1?.beskrivelse || null,
    employees: brregData.antallAnsatte || 0,
    website: brregData.hjemmeside || null,
    email: brregData.epostadresse || null,
    phone: brregData.telefon || null,
    address_street: address?.adresse?.join(", ") || null,
    address_postal_code: address?.postnummer || null,
    address_city: address?.poststed || null,
    address_country: address?.land || null,
    registration_date: brregData.registreringsdatoEnhetsregisteret
      ? new Date(brregData.registreringsdatoEnhetsregisteret)
      : null,
    vat_registered: brregData.registrertIMvaregisteret || false,
    parent_company: brregData.overordnetEnhet || null,
    organization_type: brregData.organisasjonsform?.beskrivelse || null,
    is_bankrupt: brregData.konkurs || false,
    is_under_liquidation: brregData.underAvvikling || false,
    raw_data: JSON.stringify(brregData),
  };

  // Upsert (insert or update if exists)
  try {
    await connection.execute(
    `INSERT INTO business_profiles 
      (organization_number, name, industry, industry_code, industry_description, 
       employees, website, email, phone, address_street, address_postal_code, 
       address_city, address_country, registration_date, vat_registered, 
       parent_company, organization_type, is_bankrupt, is_under_liquidation, raw_data)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
     ON DUPLICATE KEY UPDATE
       name = VALUES(name),
       industry = VALUES(industry),
       industry_code = VALUES(industry_code),
       industry_description = VALUES(industry_description),
       employees = VALUES(employees),
       website = VALUES(website),
       email = VALUES(email),
       phone = VALUES(phone),
       address_street = VALUES(address_street),
       address_postal_code = VALUES(address_postal_code),
       address_city = VALUES(address_city),
       address_country = VALUES(address_country),
       registration_date = VALUES(registration_date),
       vat_registered = VALUES(vat_registered),
       parent_company = VALUES(parent_company),
       organization_type = VALUES(organization_type),
       is_bankrupt = VALUES(is_bankrupt),
       is_under_liquidation = VALUES(is_under_liquidation),
       raw_data = VALUES(raw_data),
       updated_at = CURRENT_TIMESTAMP`,
    [
      profileData.organization_number,
      profileData.name,
      profileData.industry,
      profileData.industry_code,
      profileData.industry_description,
      profileData.employees,
      profileData.website,
      profileData.email,
      profileData.phone,
      profileData.address_street,
      profileData.address_postal_code,
      profileData.address_city,
      profileData.address_country,
      profileData.registration_date,
      profileData.vat_registered,
      profileData.parent_company,
      profileData.organization_type,
      profileData.is_bankrupt,
      profileData.is_under_liquidation,
      profileData.raw_data,
    ]
  );
  } finally {
    await connection.end();
  }

  // Fetch the saved profile
  return getBusinessProfile(brregData.organisasjonsnummer);
}

/**
 * Get business profile by organization number
 */
export async function getBusinessProfile(
  orgNumber: string
): Promise<BusinessProfile> {
  const connection = await getConnection();

  try {
    const [rows] = await connection.execute(
      "SELECT * FROM business_profiles WHERE organization_number = ?",
      [orgNumber]
    );

    const profiles = rows as BusinessProfile[];
    if (profiles.length === 0) {
      throw new Error("Business profile not found");
    }

    return profiles[0];
  } finally {
    await connection.end();
  }
}

/**
 * Get all business profiles
 */
export async function getAllBusinessProfiles(): Promise<BusinessProfile[]> {
  const connection = await getConnection();

  try {
    const [rows] = await connection.execute(
      "SELECT * FROM business_profiles ORDER BY created_at DESC"
    );

    return rows as BusinessProfile[];
  } finally {
    await connection.end();
  }
}

/**
 * Search business profiles by name
 */
export async function searchBusinessProfiles(
  query: string
): Promise<BusinessProfile[]> {
  const connection = await getConnection();

  try {
    const [rows] = await connection.execute(
      "SELECT * FROM business_profiles WHERE name LIKE ? ORDER BY name ASC LIMIT 50",
      [`%${query}%`]
    );

    return rows as BusinessProfile[];
  } finally {
    await connection.end();
  }
}

/**
 * Get business profiles by industry
 */
export async function getBusinessProfilesByIndustry(
  industry: string
): Promise<BusinessProfile[]> {
  const connection = await getConnection();

  try {
    const [rows] = await connection.execute(
      "SELECT * FROM business_profiles WHERE industry = ? ORDER BY employees DESC",
      [industry]
    );

    return rows as BusinessProfile[];
  } finally {
    await connection.end();
  }
}

/**
 * Delete business profile
 */
export async function deleteBusinessProfile(orgNumber: string): Promise<void> {
  const connection = await getConnection();

  try {
    await connection.execute(
      "DELETE FROM business_profiles WHERE organization_number = ?",
      [orgNumber]
    );
  } finally {
    await connection.end();
  }
}
