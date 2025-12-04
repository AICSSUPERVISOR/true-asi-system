import { eq, desc, sql } from "drizzle-orm";
import { getDb } from "./db";
import { s7Submissions, s7Rankings } from "../drizzle/schema";
import { createId } from "@paralleldrive/cuid2";

/**
 * Submit an S-7 answer for evaluation
 */
export async function submitS7Answer(params: {
  userId: number;
  questionNumber: number;
  answer: string;
}) {
  const db = await getDb();
  if (!db) throw new Error("Database not available");

  const submissionId = createId();
  
  await db.insert(s7Submissions).values({
    id: submissionId,
    userId: params.userId,
    questionNumber: params.questionNumber,
    answer: params.answer,
    submittedAt: new Date(),
  });

  return submissionId;
}

/**
 * Update submission with evaluation scores
 */
export async function updateS7Scores(params: {
  submissionId: string;
  scores: {
    novelty: number;
    coherence: number;
    rigor: number;
    synthesis: number;
    formalization: number;
    depth: number;
  };
  evaluationModel: string;
  evaluationTime: number;
}) {
  const db = await getDb();
  if (!db) throw new Error("Database not available");

  const { scores } = params;
  
  // Calculate metrics (scores stored as int * 10 for precision)
  const totalScore = (scores.novelty + scores.coherence + scores.rigor + 
                      scores.synthesis + scores.formalization + scores.depth) * 10;
  const averageScore = Math.round((totalScore / 6));
  
  // Check S-7 threshold: ≥8.8 all categories, ≥9.6 in at least 2
  const allAbove88 = Object.values(scores).every(s => s >= 8.8);
  const countAbove96 = Object.values(scores).filter(s => s >= 9.6).length;
  const meetsThreshold = allAbove88 && countAbove96 >= 2 ? 1 : 0;

  await db.update(s7Submissions)
    .set({
      scoreNovelty: Math.round(scores.novelty * 10),
      scoreCoherence: Math.round(scores.coherence * 10),
      scoreRigor: Math.round(scores.rigor * 10),
      scoreSynthesis: Math.round(scores.synthesis * 10),
      scoreFormalization: Math.round(scores.formalization * 10),
      scoreDepth: Math.round(scores.depth * 10),
      totalScore,
      averageScore,
      meetsThreshold,
      evaluationModel: params.evaluationModel,
      evaluationTime: params.evaluationTime,
      evaluatedAt: new Date(),
    })
    .where(eq(s7Submissions.id, params.submissionId));

  return { totalScore, averageScore, meetsThreshold };
}

/**
 * Get user's submission history
 */
export async function getUserSubmissions(userId: number) {
  const db = await getDb();
  if (!db) return [];

  return await db.select()
    .from(s7Submissions)
    .where(eq(s7Submissions.userId, userId))
    .orderBy(desc(s7Submissions.submittedAt));
}

/**
 * Get global leaderboard
 */
export async function getLeaderboard(limit: number = 100) {
  const db = await getDb();
  if (!db) return [];

  return await db.select()
    .from(s7Rankings)
    .orderBy(desc(s7Rankings.averageScore))
    .limit(limit);
}

/**
 * Get user ranking
 */
export async function getUserRanking(userId: number) {
  const db = await getDb();
  if (!db) return null;

  const result = await db.select()
    .from(s7Rankings)
    .where(eq(s7Rankings.userId, userId))
    .limit(1);

  return result.length > 0 ? result[0] : null;
}

/**
 * Update user ranking after new submission
 */
export async function updateUserRanking(userId: number) {
  const db = await getDb();
  if (!db) throw new Error("Database not available");

  // Get all user submissions
  const submissions = await db.select()
    .from(s7Submissions)
    .where(eq(s7Submissions.userId, userId));

  if (submissions.length === 0) return;

  // Calculate aggregates
  const totalSubmissions = submissions.length;
  const questionsCompleted = new Set(submissions.map(s => s.questionNumber)).size;
  
  const evaluatedSubmissions = submissions.filter(s => s.averageScore !== null);
  if (evaluatedSubmissions.length === 0) return;

  const avgScore = Math.round(
    evaluatedSubmissions.reduce((sum, s) => sum + (s.averageScore || 0), 0) / evaluatedSubmissions.length
  );
  const bestScore = Math.max(...evaluatedSubmissions.map(s => s.averageScore || 0));

  // Calculate category averages
  const avgNovelty = Math.round(
    evaluatedSubmissions.reduce((sum, s) => sum + (s.scoreNovelty || 0), 0) / evaluatedSubmissions.length
  );
  const avgCoherence = Math.round(
    evaluatedSubmissions.reduce((sum, s) => sum + (s.scoreCoherence || 0), 0) / evaluatedSubmissions.length
  );
  const avgRigor = Math.round(
    evaluatedSubmissions.reduce((sum, s) => sum + (s.scoreRigor || 0), 0) / evaluatedSubmissions.length
  );
  const avgSynthesis = Math.round(
    evaluatedSubmissions.reduce((sum, s) => sum + (s.scoreSynthesis || 0), 0) / evaluatedSubmissions.length
  );
  const avgFormalization = Math.round(
    evaluatedSubmissions.reduce((sum, s) => sum + (s.scoreFormalization || 0), 0) / evaluatedSubmissions.length
  );
  const avgDepth = Math.round(
    evaluatedSubmissions.reduce((sum, s) => sum + (s.scoreDepth || 0), 0) / evaluatedSubmissions.length
  );

  const questionsAboveThreshold = evaluatedSubmissions.filter(s => s.meetsThreshold === 1).length;
  const s7Certified = questionsCompleted >= 40 && questionsAboveThreshold >= 40 ? 1 : 0;

  // Upsert ranking
  const rankingId = createId();
  await db.insert(s7Rankings).values({
    id: rankingId,
    userId,
    totalSubmissions,
    questionsCompleted,
    averageScore: avgScore,
    bestScore,
    avgNovelty,
    avgCoherence,
    avgRigor,
    avgSynthesis,
    avgFormalization,
    avgDepth,
    questionsAboveThreshold,
    s7Certified,
    lastUpdated: new Date(),
  }).onDuplicateKeyUpdate({
    set: {
      totalSubmissions,
      questionsCompleted,
      averageScore: avgScore,
      bestScore,
      avgNovelty,
      avgCoherence,
      avgRigor,
      avgSynthesis,
      avgFormalization,
      avgDepth,
      questionsAboveThreshold,
      s7Certified,
      lastUpdated: new Date(),
    },
  });

  // Update global ranks
  await updateGlobalRanks();
}

/**
 * Update global ranks for all users
 */
async function updateGlobalRanks() {
  const db = await getDb();
  if (!db) return;

  const rankings = await db.select()
    .from(s7Rankings)
    .orderBy(desc(s7Rankings.averageScore));

  for (let i = 0; i < rankings.length; i++) {
    await db.update(s7Rankings)
      .set({ globalRank: i + 1 })
      .where(eq(s7Rankings.userId, rankings[i].userId));
  }
}

/**
 * Get leaderboard by category
 */
export async function getLeaderboardByCategory(category: string, limit: number = 100) {
  const db = await getDb();
  if (!db) return [];

  const categoryMap: Record<string, any> = {
    novelty: s7Rankings.avgNovelty,
    coherence: s7Rankings.avgCoherence,
    rigor: s7Rankings.avgRigor,
    synthesis: s7Rankings.avgSynthesis,
    formalization: s7Rankings.avgFormalization,
    depth: s7Rankings.avgDepth,
  };

  const orderByColumn = categoryMap[category] || s7Rankings.averageScore;

  return await db.select()
    .from(s7Rankings)
    .orderBy(desc(orderByColumn))
    .limit(limit);
}
