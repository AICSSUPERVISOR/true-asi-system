import * as db from "../db";
import { notifyOwner } from "../_core/notification";

// ============================================================================
// TASK SCHEDULER
// Cron-like scheduling for automated accounting tasks
// ============================================================================

export interface ScheduledTask {
  id: string;
  name: string;
  cronExpression: string;
  handler: () => Promise<void>;
  lastRun?: Date;
  nextRun?: Date;
  enabled: boolean;
}

export interface TaskResult {
  taskId: string;
  success: boolean;
  message: string;
  executedAt: Date;
  duration: number;
}

// Norwegian MVA deadlines (10th of the month following the period)
const MVA_DEADLINES = {
  1: { month: 3, day: 10 },  // Termin 1 (Jan-Feb) -> 10. mars
  2: { month: 5, day: 10 },  // Termin 2 (Mar-Apr) -> 10. mai
  3: { month: 7, day: 10 },  // Termin 3 (May-Jun) -> 10. juli
  4: { month: 9, day: 10 },  // Termin 4 (Jul-Aug) -> 10. september
  5: { month: 11, day: 10 }, // Termin 5 (Sep-Oct) -> 10. november
  6: { month: 1, day: 10 },  // Termin 6 (Nov-Dec) -> 10. januar (neste år)
};

// A-melding deadline (5th of the following month)
const A_MELDING_DEADLINE_DAY = 5;

class TaskScheduler {
  private tasks: Map<string, ScheduledTask> = new Map();
  private intervalId: NodeJS.Timeout | null = null;
  private checkInterval: number = 60000; // Check every minute

  constructor() {
    this.registerDefaultTasks();
  }

  /**
   * Register default accounting tasks
   */
  private registerDefaultTasks(): void {
    // MVA deadline reminder (daily at 8:00)
    this.registerTask({
      id: "mva-deadline-reminder",
      name: "MVA Deadline Reminder",
      cronExpression: "0 8 * * *",
      handler: this.checkMVADeadlines.bind(this),
      enabled: true,
    });

    // A-melding deadline reminder (daily at 8:00)
    this.registerTask({
      id: "a-melding-deadline-reminder",
      name: "A-melding Deadline Reminder",
      cronExpression: "0 8 * * *",
      handler: this.checkAMeldingDeadlines.bind(this),
      enabled: true,
    });

    // Forvalt data refresh (weekly on Monday at 6:00)
    this.registerTask({
      id: "forvalt-refresh",
      name: "Forvalt Data Refresh",
      cronExpression: "0 6 * * 1",
      handler: this.refreshForvaltData.bind(this),
      enabled: true,
    });

    // SAF-T backup generation (monthly on 1st at 2:00)
    this.registerTask({
      id: "saft-backup",
      name: "SAF-T Backup Generation",
      cronExpression: "0 2 1 * *",
      handler: this.generateSAFTBackups.bind(this),
      enabled: true,
    });

    // Document processing check (every 15 minutes)
    this.registerTask({
      id: "document-processing",
      name: "Document Processing Check",
      cronExpression: "*/15 * * * *",
      handler: this.processQueuedDocuments.bind(this),
      enabled: true,
    });
  }

  /**
   * Register a new task
   */
  registerTask(task: Omit<ScheduledTask, "lastRun" | "nextRun">): void {
    const fullTask: ScheduledTask = {
      ...task,
      nextRun: this.calculateNextRun(task.cronExpression),
    };
    this.tasks.set(task.id, fullTask);
    console.log(`[Scheduler] Registered task: ${task.name}`);
  }

  /**
   * Start the scheduler
   */
  start(): void {
    if (this.intervalId) {
      console.log("[Scheduler] Already running");
      return;
    }

    console.log("[Scheduler] Starting task scheduler...");
    this.intervalId = setInterval(() => this.tick(), this.checkInterval);
    
    // Run initial check
    this.tick();
  }

  /**
   * Stop the scheduler
   */
  stop(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
      console.log("[Scheduler] Stopped");
    }
  }

  /**
   * Check and run due tasks
   */
  private async tick(): Promise<void> {
    const now = new Date();

    for (const [id, task] of Array.from(this.tasks.entries())) {
      if (!task.enabled) continue;
      if (!task.nextRun || task.nextRun > now) continue;

      console.log(`[Scheduler] Running task: ${task.name}`);
      const startTime = Date.now();

      try {
        await task.handler();
        
        const result: TaskResult = {
          taskId: id,
          success: true,
          message: "Task completed successfully",
          executedAt: now,
          duration: Date.now() - startTime,
        };

        console.log(`[Scheduler] Task ${task.name} completed in ${result.duration}ms`);
      } catch (error) {
        console.error(`[Scheduler] Task ${task.name} failed:`, error);
        
        // Notify owner of task failure
        await notifyOwner({
          title: `Scheduled Task Failed: ${task.name}`,
          content: `Task ${task.name} failed with error: ${error instanceof Error ? error.message : "Unknown error"}`,
        });
      }

      // Update task timing
      task.lastRun = now;
      task.nextRun = this.calculateNextRun(task.cronExpression);
    }
  }

  /**
   * Calculate next run time from cron expression
   * Simplified cron parser (minute hour day month weekday)
   */
  private calculateNextRun(cronExpression: string): Date {
    const now = new Date();
    const parts = cronExpression.split(" ");
    
    if (parts.length !== 5) {
      // Default to 1 hour from now
      return new Date(now.getTime() + 3600000);
    }

    const [minute, hour, day, month, weekday] = parts;
    const next = new Date(now);
    next.setSeconds(0);
    next.setMilliseconds(0);

    // Simple parsing - just handle common cases
    if (minute !== "*") {
      const mins = minute.startsWith("*/") 
        ? parseInt(minute.slice(2)) 
        : parseInt(minute);
      
      if (minute.startsWith("*/")) {
        // Every X minutes
        const currentMin = now.getMinutes();
        const nextMin = Math.ceil((currentMin + 1) / mins) * mins;
        if (nextMin >= 60) {
          next.setHours(next.getHours() + 1);
          next.setMinutes(nextMin - 60);
        } else {
          next.setMinutes(nextMin);
        }
      } else {
        next.setMinutes(mins);
        if (next <= now) {
          next.setHours(next.getHours() + 1);
        }
      }
    }

    if (hour !== "*") {
      const hrs = parseInt(hour);
      next.setHours(hrs);
      if (next <= now) {
        next.setDate(next.getDate() + 1);
      }
    }

    if (day !== "*") {
      const d = parseInt(day);
      next.setDate(d);
      if (next <= now) {
        next.setMonth(next.getMonth() + 1);
      }
    }

    return next;
  }

  // ============================================================================
  // TASK HANDLERS
  // ============================================================================

  /**
   * Check for upcoming MVA deadlines
   */
  private async checkMVADeadlines(): Promise<void> {
    const today = new Date();
    const companies = await db.listCompanies();

    for (const company of companies) {
      // Determine current MVA term
      const month = today.getMonth() + 1;
      const currentTerm = Math.ceil(month / 2);
      const deadline = MVA_DEADLINES[currentTerm as keyof typeof MVA_DEADLINES];
      
      if (!deadline) continue;

      const deadlineDate = new Date(
        today.getFullYear() + (deadline.month < month ? 1 : 0),
        deadline.month - 1,
        deadline.day
      );

      const daysUntilDeadline = Math.ceil(
        (deadlineDate.getTime() - today.getTime()) / (1000 * 60 * 60 * 24)
      );

      // Check if we need to send reminder (7 days, 3 days, 1 day before)
      if ([7, 3, 1].includes(daysUntilDeadline)) {
        // Check if filing exists for this period
        const existingFilings = await db.listFilings(company.id);
        const hasFilingForTerm = existingFilings.some(
          (f) => f.filingType === "MVA_MELDING" && 
                 f.periodStart && 
                 new Date(f.periodStart).getMonth() === (currentTerm - 1) * 2
        );

        if (!hasFilingForTerm) {
          await notifyOwner({
            title: `MVA-frist for ${company.name}`,
            content: `MVA-melding for termin ${currentTerm} må leveres innen ${daysUntilDeadline} dag(er) (${deadlineDate.toLocaleDateString("nb-NO")}).`,
          });
        }
      }
    }
  }

  /**
   * Check for upcoming A-melding deadlines
   */
  private async checkAMeldingDeadlines(): Promise<void> {
    const today = new Date();
    const deadlineDate = new Date(
      today.getFullYear(),
      today.getMonth(),
      A_MELDING_DEADLINE_DAY
    );

    // If deadline has passed this month, check next month
    if (deadlineDate < today) {
      deadlineDate.setMonth(deadlineDate.getMonth() + 1);
    }

    const daysUntilDeadline = Math.ceil(
      (deadlineDate.getTime() - today.getTime()) / (1000 * 60 * 60 * 24)
    );

    if ([3, 1].includes(daysUntilDeadline)) {
      const companies = await db.listCompanies();
      
      for (const company of companies) {
        await notifyOwner({
          title: `A-melding frist for ${company.name}`,
          content: `A-melding for forrige måned må leveres innen ${daysUntilDeadline} dag(er) (${deadlineDate.toLocaleDateString("nb-NO")}).`,
        });
      }
    }
  }

  /**
   * Refresh Forvalt data for all companies
   */
  private async refreshForvaltData(): Promise<void> {
    const companies = await db.listCompanies();
    let refreshed = 0;
    let failed = 0;

    for (const company of companies) {
      if (!company.orgNumber) continue;

      try {
        // In production, call Forvalt API here
        console.log(`[Scheduler] Refreshing Forvalt data for ${company.name}`);
        refreshed++;
      } catch (error) {
        console.error(`[Scheduler] Failed to refresh Forvalt for ${company.name}:`, error);
        failed++;
      }
    }

    console.log(`[Scheduler] Forvalt refresh complete: ${refreshed} refreshed, ${failed} failed`);
  }

  /**
   * Generate SAF-T backups for all companies
   */
  private async generateSAFTBackups(): Promise<void> {
    const companies = await db.listCompanies();
    const lastMonth = new Date();
    lastMonth.setMonth(lastMonth.getMonth() - 1);
    
    const periodStart = new Date(lastMonth.getFullYear(), lastMonth.getMonth(), 1);
    const periodEnd = new Date(lastMonth.getFullYear(), lastMonth.getMonth() + 1, 0);

    for (const company of companies) {
      try {
        console.log(`[Scheduler] Generating SAF-T backup for ${company.name}`);
        
        // Create a filing record for the backup
        await db.createFiling({
          companyId: company.id,
          filingType: "SAF_T",
          periodStart,
          periodEnd,
          status: "DRAFT",
          createdById: 1, // System user
        });
      } catch (error) {
        console.error(`[Scheduler] Failed to generate SAF-T for ${company.name}:`, error);
      }
    }
  }

  /**
   * Process queued documents
   */
  private async processQueuedDocuments(): Promise<void> {
    const companies = await db.listCompanies();

    for (const company of companies) {
      const documents = await db.listAccountingDocuments(company.id);
      const pendingDocs = documents.filter((d) => d.status === "NEW");

      if (pendingDocs.length > 0) {
        console.log(`[Scheduler] Found ${pendingDocs.length} pending documents for ${company.name}`);
        
        for (const doc of pendingDocs) {
          try {
            // Update status to processing
            await db.updateAccountingDocument(doc.id, { status: "PROCESSED" });
            
            // In production, trigger AI processing here
            console.log(`[Scheduler] Processing document ${doc.id}: ${doc.originalFileName}`);
          } catch (error) {
            console.error(`[Scheduler] Failed to process document ${doc.id}:`, error);
          }
        }
      }
    }
  }

  /**
   * Get task status
   */
  getTaskStatus(): Array<{ id: string; name: string; enabled: boolean; lastRun?: Date; nextRun?: Date }> {
    return Array.from(this.tasks.values()).map((task) => ({
      id: task.id,
      name: task.name,
      enabled: task.enabled,
      lastRun: task.lastRun,
      nextRun: task.nextRun,
    }));
  }

  /**
   * Enable/disable a task
   */
  setTaskEnabled(taskId: string, enabled: boolean): boolean {
    const task = this.tasks.get(taskId);
    if (task) {
      task.enabled = enabled;
      return true;
    }
    return false;
  }

  /**
   * Run a task manually
   */
  async runTask(taskId: string): Promise<TaskResult> {
    const task = this.tasks.get(taskId);
    if (!task) {
      return {
        taskId,
        success: false,
        message: "Task not found",
        executedAt: new Date(),
        duration: 0,
      };
    }

    const startTime = Date.now();
    try {
      await task.handler();
      return {
        taskId,
        success: true,
        message: "Task completed successfully",
        executedAt: new Date(),
        duration: Date.now() - startTime,
      };
    } catch (error) {
      return {
        taskId,
        success: false,
        message: error instanceof Error ? error.message : "Unknown error",
        executedAt: new Date(),
        duration: Date.now() - startTime,
      };
    }
  }
}

// Singleton instance
let schedulerInstance: TaskScheduler | null = null;

export function getScheduler(): TaskScheduler {
  if (!schedulerInstance) {
    schedulerInstance = new TaskScheduler();
  }
  return schedulerInstance;
}

export function startScheduler(): void {
  getScheduler().start();
}

export function stopScheduler(): void {
  if (schedulerInstance) {
    schedulerInstance.stop();
  }
}
