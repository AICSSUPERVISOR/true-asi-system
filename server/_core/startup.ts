/**
 * Server Startup Tasks
 * Runs once when the server starts to initialize caches and connections
 */

import { warmUpCache, getCacheStats } from "./cache";
import { enhancedS7Answers } from "../enhanced_s7_answers";

export async function runStartupTasks() {
  console.log("[Startup] Running initialization tasks...");

  // Warm up Redis cache with all enhanced S-7 answers
  try {
    await warmUpCache(enhancedS7Answers);
    const stats = await getCacheStats();
    console.log(`[Startup] Redis cache ready: ${stats.keys} keys, ${stats.memory} memory`);
  } catch (error) {
    console.error("[Startup] Failed to warm up cache:", error);
  }

  // Additional startup tasks can be added here
  // - Pre-compute frequently accessed data
  // - Initialize WebSocket connections
  // - Load configuration from AWS S3
  // - Verify database connections
  // - Start background jobs

  console.log("[Startup] Initialization complete");
}

// Export a promise that resolves when startup is complete
let startupPromise: Promise<void> | null = null;

export function ensureStartupComplete(): Promise<void> {
  if (!startupPromise) {
    startupPromise = runStartupTasks();
  }
  return startupPromise;
}
