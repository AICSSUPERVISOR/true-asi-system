/**
 * Database helpers for website analyses
 */

import mysql from "mysql2/promise";
import type { WebsiteAnalysis, Recommendation } from "./_core/website_analysis";

const databaseUrl = process.env.DATABASE_URL;
if (!databaseUrl) {
  throw new Error("DATABASE_URL environment variable is not set");
}

let connection: mysql.Connection | null = null;

async function getConnection() {
  if (!connection) {
    connection = await mysql.createConnection(databaseUrl!);
  }
  return connection;
}

export interface WebsiteAnalysisRecord {
  id: number;
  business_id: number;
  url: string;
  overall_score: number;
  seo_score: number;
  performance_score: number;
  accessibility_score: number;
  content_score: number;
  ux_score: number;
  analysis_data: string; // JSON string
  recommendations: string; // JSON string
  created_at: Date;
  updated_at: Date;
}

/**
 * Save website analysis to database
 */
export async function saveWebsiteAnalysis(
  businessId: number,
  analysis: WebsiteAnalysis
): Promise<number> {
  const conn = await getConnection();

  const [result] = await conn.execute(
    `INSERT INTO website_analyses (
      business_id, url, overall_score, seo_score, performance_score,
      accessibility_score, content_score, ux_score, analysis_data, recommendations
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    [
      businessId,
      analysis.url,
      analysis.overallScore,
      analysis.seo.score,
      analysis.performance.score,
      analysis.accessibility.score,
      analysis.contentQuality.score,
      analysis.ux.score,
      JSON.stringify({
        seo: analysis.seo,
        performance: analysis.performance,
        accessibility: analysis.accessibility,
        contentQuality: analysis.contentQuality,
        ux: analysis.ux,
      }),
      JSON.stringify(analysis.recommendations),
    ]
  );

  return (result as any).insertId;
}

/**
 * Get website analysis by ID
 */
export async function getWebsiteAnalysisById(
  id: number
): Promise<WebsiteAnalysis | null> {
  const conn = await getConnection();

  const [rows] = await conn.execute(
    `SELECT * FROM website_analyses WHERE id = ? LIMIT 1`,
    [id]
  );

  const records = rows as WebsiteAnalysisRecord[];
  if (records.length === 0) return null;

  return parseWebsiteAnalysisRecord(records[0]);
}

/**
 * Get website analyses by business ID
 */
export async function getWebsiteAnalysesByBusinessId(
  businessId: number
): Promise<WebsiteAnalysis[]> {
  const conn = await getConnection();

  const [rows] = await conn.execute(
    `SELECT * FROM website_analyses WHERE business_id = ? ORDER BY created_at DESC`,
    [businessId]
  );

  const records = rows as WebsiteAnalysisRecord[];
  return records.map(parseWebsiteAnalysisRecord);
}

/**
 * Get latest website analysis for a URL
 */
export async function getLatestWebsiteAnalysis(
  url: string
): Promise<WebsiteAnalysis | null> {
  const conn = await getConnection();

  const [rows] = await conn.execute(
    `SELECT * FROM website_analyses WHERE url = ? ORDER BY created_at DESC LIMIT 1`,
    [url]
  );

  const records = rows as WebsiteAnalysisRecord[];
  if (records.length === 0) return null;

  return parseWebsiteAnalysisRecord(records[0]);
}

/**
 * Parse database record to WebsiteAnalysis
 */
function parseWebsiteAnalysisRecord(record: WebsiteAnalysisRecord): WebsiteAnalysis {
  const analysisData = JSON.parse(record.analysis_data);
  const recommendations = JSON.parse(record.recommendations);

  return {
    url: record.url,
    seo: analysisData.seo,
    performance: analysisData.performance,
    accessibility: analysisData.accessibility,
    contentQuality: analysisData.contentQuality,
    ux: analysisData.ux,
    overallScore: record.overall_score,
    recommendations: recommendations as Recommendation[],
  };
}

/**
 * Get all website analyses (for admin/analytics)
 */
export async function getAllWebsiteAnalyses(
  limit: number = 100
): Promise<WebsiteAnalysis[]> {
  const conn = await getConnection();

  const [rows] = await conn.execute(
    `SELECT * FROM website_analyses ORDER BY created_at DESC LIMIT ?`,
    [limit]
  );

  const records = rows as WebsiteAnalysisRecord[];
  return records.map(parseWebsiteAnalysisRecord);
}

/**
 * Delete website analysis
 */
export async function deleteWebsiteAnalysis(id: number): Promise<boolean> {
  const conn = await getConnection();

  const [result] = await conn.execute(`DELETE FROM website_analyses WHERE id = ?`, [id]);

  return (result as any).affectedRows > 0;
}
