import { eq, and, desc } from "drizzle-orm";
import { getDb } from "./db";
import { answerComparisons, s7Submissions } from "../drizzle/schema";
import { createId } from "@paralleldrive/cuid2";

/**
 * Create a new answer comparison with AI gap analysis
 */
export async function createAnswerComparison(params: {
  userId: number;
  questionNumber: number;
  userSubmissionId: string;
  comparedWithSubmissionId?: string;
  gaps: {
    novelty: number;
    coherence: number;
    rigor: number;
    synthesis: number;
    formalization: number;
    depth: number;
  };
  recommendations: {
    overall: string;
    novelty: string;
    coherence: string;
    rigor: string;
    synthesis: string;
    formalization: string;
    depth: string;
  };
  comparisonModel: string;
  comparisonTime: number;
}) {
  const db = await getDb();
  if (!db) throw new Error("Database not available");

  const comparisonId = createId();
  
  await db.insert(answerComparisons).values({
    id: comparisonId,
    userId: params.userId,
    questionNumber: params.questionNumber,
    userSubmissionId: params.userSubmissionId,
    comparedWithSubmissionId: params.comparedWithSubmissionId,
    
    // Store gaps as integers (multiply by 10 for precision)
    noveltyGap: Math.round(params.gaps.novelty * 10),
    coherenceGap: Math.round(params.gaps.coherence * 10),
    rigorGap: Math.round(params.gaps.rigor * 10),
    synthesisGap: Math.round(params.gaps.synthesis * 10),
    formalizationGap: Math.round(params.gaps.formalization * 10),
    depthGap: Math.round(params.gaps.depth * 10),
    
    // AI recommendations
    overallAnalysis: params.recommendations.overall,
    noveltyRecommendations: params.recommendations.novelty,
    coherenceRecommendations: params.recommendations.coherence,
    rigorRecommendations: params.recommendations.rigor,
    synthesisRecommendations: params.recommendations.synthesis,
    formalizationRecommendations: params.recommendations.formalization,
    depthRecommendations: params.recommendations.depth,
    
    // Metadata
    comparisonModel: params.comparisonModel,
    comparisonTime: params.comparisonTime,
  });

  return comparisonId;
}

/**
 * Get user's comparison history for a question
 */
export async function getUserComparisons(userId: number, questionNumber?: number) {
  const db = await getDb();
  if (!db) return [];

  if (questionNumber) {
    return await db.select()
      .from(answerComparisons)
      .where(and(
        eq(answerComparisons.userId, userId),
        eq(answerComparisons.questionNumber, questionNumber)
      ))
      .orderBy(desc(answerComparisons.createdAt));
  }

  return await db.select()
    .from(answerComparisons)
    .where(eq(answerComparisons.userId, userId))
    .orderBy(desc(answerComparisons.createdAt));
}

/**
 * Get a specific comparison by ID
 */
export async function getComparisonById(comparisonId: string) {
  const db = await getDb();
  if (!db) return null;

  const result = await db.select()
    .from(answerComparisons)
    .where(eq(answerComparisons.id, comparisonId))
    .limit(1);

  return result.length > 0 ? result[0] : null;
}

/**
 * Get top-ranked answer for a question (for comparison)
 */
export async function getTopRankedAnswer(questionNumber: number) {
  const db = await getDb();
  if (!db) return null;

  const result = await db.select()
    .from(s7Submissions)
    .where(eq(s7Submissions.questionNumber, questionNumber))
    .orderBy(desc(s7Submissions.averageScore))
    .limit(1);

  return result.length > 0 ? result[0] : null;
}

/**
 * Get user's submission for a question
 */
export async function getUserSubmission(userId: number, questionNumber: number) {
  const db = await getDb();
  if (!db) return null;

  const result = await db.select()
    .from(s7Submissions)
    .where(and(
      eq(s7Submissions.userId, userId),
      eq(s7Submissions.questionNumber, questionNumber)
    ))
    .orderBy(desc(s7Submissions.submittedAt))
    .limit(1);

  return result.length > 0 ? result[0] : null;
}
