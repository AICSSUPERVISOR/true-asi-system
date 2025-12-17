import { int, mysqlTable, text, timestamp, varchar, json, boolean, decimal } from "drizzle-orm/mysql-core";

// ============================================================================
// VELOCITY COMPETITOR PRICING TABLE
// Stores scraped competitor pricing data for analytics
// ============================================================================
export const competitorPrices = mysqlTable("competitorPrices", {
  id: int("id").autoincrement().primaryKey(),
  competitor: varchar("competitor", { length: 50 }).notNull(), // Uber, Bolt, Lyft
  route: varchar("route", { length: 255 }).notNull(),
  cityFrom: varchar("cityFrom", { length: 100 }).notNull(),
  cityTo: varchar("cityTo", { length: 100 }).notNull(),
  currency: varchar("currency", { length: 3 }).notNull(),
  standardPrice: decimal("standardPrice", { precision: 10, scale: 2 }).notNull(),
  premiumPrice: decimal("premiumPrice", { precision: 10, scale: 2 }),
  xlPrice: decimal("xlPrice", { precision: 10, scale: 2 }),
  sourceUrl: text("sourceUrl"),
  scrapedAt: timestamp("scrapedAt").defaultNow().notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

export type CompetitorPrice = typeof competitorPrices.$inferSelect;
export type InsertCompetitorPrice = typeof competitorPrices.$inferInsert;

// ============================================================================
// VELOCITY PRICING TABLE
// Stores calculated VELOCITY competitive pricing
// ============================================================================
export const velocityPrices = mysqlTable("velocityPrices", {
  id: int("id").autoincrement().primaryKey(),
  route: varchar("route", { length: 255 }).notNull(),
  cityFrom: varchar("cityFrom", { length: 100 }).notNull(),
  cityTo: varchar("cityTo", { length: 100 }).notNull(),
  currency: varchar("currency", { length: 3 }).notNull(),
  // VELOCITY pricing tiers
  velocityStandard: decimal("velocityStandard", { precision: 10, scale: 2 }).notNull(),
  velocityPremium: decimal("velocityPremium", { precision: 10, scale: 2 }),
  velocityXl: decimal("velocityXl", { precision: 10, scale: 2 }),
  velocityAutonomous: decimal("velocityAutonomous", { precision: 10, scale: 2 }).notNull(),
  // Competitor reference prices
  uberPrice: decimal("uberPrice", { precision: 10, scale: 2 }),
  boltPrice: decimal("boltPrice", { precision: 10, scale: 2 }),
  lyftPrice: decimal("lyftPrice", { precision: 10, scale: 2 }),
  // Discount percentages
  uberDiscount: decimal("uberDiscount", { precision: 5, scale: 2 }).default("10.00"),
  boltDiscount: decimal("boltDiscount", { precision: 5, scale: 2 }).default("8.00"),
  lyftDiscount: decimal("lyftDiscount", { precision: 5, scale: 2 }).default("10.00"),
  autonomousDiscount: decimal("autonomousDiscount", { precision: 5, scale: 2 }).default("40.00"),
  // Price change tracking
  priceChangePercent: decimal("priceChangePercent", { precision: 5, scale: 2 }).default("0.00"),
  significantChange: boolean("significantChange").default(false),
  // Metadata
  calculatedAt: timestamp("calculatedAt").defaultNow().notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

export type VelocityPrice = typeof velocityPrices.$inferSelect;
export type InsertVelocityPrice = typeof velocityPrices.$inferInsert;

// ============================================================================
// VELOCITY PRICE MONITORING LOG TABLE
// Audit trail for price monitoring runs
// ============================================================================
export const priceMonitoringLogs = mysqlTable("priceMonitoringLogs", {
  id: int("id").autoincrement().primaryKey(),
  runId: varchar("runId", { length: 64 }).notNull().unique(),
  status: varchar("status", { length: 20 }).notNull(), // RUNNING, COMPLETED, FAILED
  routesProcessed: int("routesProcessed").default(0),
  competitorPricesFound: int("competitorPricesFound").default(0),
  velocityPricesCalculated: int("velocityPricesCalculated").default(0),
  slackNotificationSent: boolean("slackNotificationSent").default(false),
  slackMessageTimestamp: varchar("slackMessageTimestamp", { length: 50 }),
  errorMessage: text("errorMessage"),
  rawResultsJson: json("rawResultsJson"),
  startedAt: timestamp("startedAt").defaultNow().notNull(),
  completedAt: timestamp("completedAt"),
});

export type PriceMonitoringLog = typeof priceMonitoringLogs.$inferSelect;
export type InsertPriceMonitoringLog = typeof priceMonitoringLogs.$inferInsert;
