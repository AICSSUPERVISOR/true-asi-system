import { mysqlTable, varchar, text, int, timestamp, mysqlEnum } from "drizzle-orm/mysql-core";

// Existing users table - DO NOT MODIFY to avoid migration issues
export const users = mysqlTable("users", {
  id: int("id").primaryKey().autoincrement(),
  openId: varchar("openId", { length: 64 }).notNull().unique(),
  name: text("name"),
  email: varchar("email", { length: 320 }),
  loginMethod: varchar("loginMethod", { length: 64 }),
  role: mysqlEnum("role", ["user", "admin"]).notNull().default("user"),
  createdAt: timestamp("createdAt").notNull().defaultNow(),
  updatedAt: timestamp("updatedAt").notNull().defaultNow().onUpdateNow(),
  lastSignedIn: timestamp("lastSignedIn").notNull().defaultNow(),
});

export type User = typeof users.$inferSelect;
export type InsertUser = typeof users.$inferInsert;

/**
 * S-7 Test Submissions
 * Stores all user submissions for S-7 test questions with automated scoring
 */
export const s7Submissions = mysqlTable("s7_submissions", {
  id: varchar("id", { length: 128 }).primaryKey(),
  userId: int("userId").notNull(), // References users.id
  questionNumber: int("questionNumber").notNull(), // 1-40
  answer: text("answer").notNull(), // User's full answer
  
  // Automated 6-category rubric scores (0-10 each, stored as int * 10 for precision)
  scoreNovelty: int("scoreNovelty"), // Novelty & originality
  scoreCoherence: int("scoreCoherence"), // Logical coherence
  scoreRigor: int("scoreRigor"), // Mathematical rigor
  scoreSynthesis: int("scoreSynthesis"), // Cross-domain synthesis
  scoreFormalization: int("scoreFormalization"), // Formalization quality
  scoreDepth: int("scoreDepth"), // Depth of insight
  
  // Overall metrics
  totalScore: int("totalScore"), // Sum of all 6 categories (0-600, stored as int * 10)
  averageScore: int("averageScore"), // Average across categories (0-100, stored as int * 10)
  meetsThreshold: int("meetsThreshold"), // Boolean: ≥8.8 all, ≥9.6 in 2 (0 or 1)
  
  // Metadata
  evaluationModel: varchar("evaluationModel", { length: 100 }), // Which AI model evaluated
  evaluationTime: int("evaluationTime"), // Time taken to evaluate (ms)
  submittedAt: timestamp("submittedAt").notNull().defaultNow(),
  evaluatedAt: timestamp("evaluatedAt"),
});

export type S7Submission = typeof s7Submissions.$inferSelect;
export type InsertS7Submission = typeof s7Submissions.$inferInsert;

/**
 * S-7 Leaderboard Rankings
 * Aggregated user performance across all questions
 */
/**
 * S-7 Answer Comparisons
 * Stores AI-powered gap analysis between user answers and top performers
 */
export const answerComparisons = mysqlTable("answer_comparisons", {
  id: varchar("id", { length: 128 }).primaryKey(),
  userId: int("userId").notNull(),
  questionNumber: int("questionNumber").notNull(),
  userSubmissionId: varchar("userSubmissionId", { length: 128 }).notNull(),
  comparedWithSubmissionId: varchar("comparedWithSubmissionId", { length: 128 }),
  
  // Gap analysis results (stored as decimal * 10 for precision)
  noveltyGap: int("noveltyGap"),
  coherenceGap: int("coherenceGap"),
  rigorGap: int("rigorGap"),
  synthesisGap: int("synthesisGap"),
  formalizationGap: int("formalizationGap"),
  depthGap: int("depthGap"),
  
  // AI-generated insights
  overallAnalysis: text("overallAnalysis"),
  noveltyRecommendations: text("noveltyRecommendations"),
  coherenceRecommendations: text("coherenceRecommendations"),
  rigorRecommendations: text("rigorRecommendations"),
  synthesisRecommendations: text("synthesisRecommendations"),
  formalizationRecommendations: text("formalizationRecommendations"),
  depthRecommendations: text("depthRecommendations"),
  
  // Metadata
  comparisonModel: varchar("comparisonModel", { length: 50 }),
  comparisonTime: int("comparisonTime"),
  createdAt: timestamp("createdAt").defaultNow(),
});

export const s7Rankings = mysqlTable("s7_rankings", {
  id: varchar("id", { length: 128 }).primaryKey(),
  userId: int("userId").notNull().unique(), // References users.id
  
  // Aggregate scores
  totalSubmissions: int("totalSubmissions").notNull().default(0),
  questionsCompleted: int("questionsCompleted").notNull().default(0), // Unique questions answered
  averageScore: int("averageScore"), // Average across all submissions (stored as int * 10)
  bestScore: int("bestScore"), // Highest single submission score (stored as int * 10)
  
  // Category averages (stored as int * 10 for precision)
  avgNovelty: int("avgNovelty"),
  avgCoherence: int("avgCoherence"),
  avgRigor: int("avgRigor"),
  avgSynthesis: int("avgSynthesis"),
  avgFormalization: int("avgFormalization"),
  avgDepth: int("avgDepth"),
  
  // S-7 threshold tracking
  questionsAboveThreshold: int("questionsAboveThreshold").notNull().default(0),
  s7Certified: int("s7Certified").notNull().default(0), // Boolean: Passed all 40 questions (0 or 1)
  
  // Ranking metadata
  globalRank: int("globalRank"), // Position in global leaderboard
  lastUpdated: timestamp("lastUpdated").notNull().defaultNow(),
});

export type S7Ranking = typeof s7Rankings.$inferSelect;
export type InsertS7Ranking = typeof s7Rankings.$inferInsert;
