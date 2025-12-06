/**
 * QStash Workflow Automation
 * 
 * Provides scheduled workflow automation for company analysis,
 * recommendation updates, and async task processing using Upstash QStash.
 */

import { Client } from '@upstash/qstash';

// Initialize QStash client
const qstashClient = new Client({
  token: 'eyJVc2VySUQiOiJiMGQ2YmZmNi1jOTRiLTRhYmEtYTc0My00ZDEzZDc5ZGYxMzYiLCJQYXNzd29yZCI6IjdkZmIzMWI4NDMwNTQ4NGJiNDRiNWFiY2U3ZmI5ODM4In0=',
});

export interface ScheduledAnalysis {
  companyId: string;
  orgNumber: string;
  frequency: 'daily' | 'weekly' | 'monthly';
  nextRun: Date;
}

export interface WorkflowResult {
  messageId: string;
  status: 'scheduled' | 'delivered' | 'failed';
  scheduledFor?: Date;
}

/**
 * Schedule automated company analysis
 */
export async function scheduleCompanyAnalysis(
  companyId: string,
  orgNumber: string,
  frequency: 'daily' | 'weekly' | 'monthly' = 'weekly',
  callbackUrl: string
): Promise<WorkflowResult> {
  try {
    // Calculate cron schedule based on frequency
    const cronSchedule = {
      daily: '0 0 * * *',      // Every day at midnight
      weekly: '0 0 * * 0',     // Every Sunday at midnight
      monthly: '0 0 1 * *',    // First day of month at midnight
    }[frequency];
    
    const result = await qstashClient.publishJSON({
      url: callbackUrl,
      body: {
        type: 'scheduled_analysis',
        companyId,
        orgNumber,
        frequency,
        timestamp: new Date().toISOString(),
      },
      cron: cronSchedule,
    });
    
    console.log(`[QStash] Scheduled ${frequency} analysis for company ${companyId}`);
    
    return {
      messageId: result.messageId,
      status: 'scheduled',
      scheduledFor: new Date(),
    };
  } catch (error) {
    console.error('[QStash] Error scheduling analysis:', error);
    throw error;
  }
}

/**
 * Schedule one-time company analysis (delayed execution)
 */
export async function scheduleOneTimeAnalysis(
  companyId: string,
  orgNumber: string,
  delaySeconds: number,
  callbackUrl: string
): Promise<WorkflowResult> {
  try {
    const result = await qstashClient.publishJSON({
      url: callbackUrl,
      body: {
        type: 'one_time_analysis',
        companyId,
        orgNumber,
        timestamp: new Date().toISOString(),
      },
      delay: delaySeconds,
    });
    
    const scheduledFor = new Date(Date.now() + delaySeconds * 1000);
    console.log(`[QStash] Scheduled one-time analysis for ${scheduledFor.toISOString()}`);
    
    return {
      messageId: result.messageId,
      status: 'scheduled',
      scheduledFor,
    };
  } catch (error) {
    console.error('[QStash] Error scheduling one-time analysis:', error);
    throw error;
  }
}

/**
 * Schedule recommendation updates
 */
export async function scheduleRecommendationUpdate(
  companyId: string,
  recommendationId: string,
  callbackUrl: string
): Promise<WorkflowResult> {
  try {
    const result = await qstashClient.publishJSON({
      url: callbackUrl,
      body: {
        type: 'recommendation_update',
        companyId,
        recommendationId,
        timestamp: new Date().toISOString(),
      },
      delay: 3600, // 1 hour delay
    });
    
    console.log(`[QStash] Scheduled recommendation update for ${recommendationId}`);
    
    return {
      messageId: result.messageId,
      status: 'scheduled',
      scheduledFor: new Date(Date.now() + 3600 * 1000),
    };
  } catch (error) {
    console.error('[QStash] Error scheduling recommendation update:', error);
    throw error;
  }
}

/**
 * Schedule batch company analysis (multiple companies)
 */
export async function scheduleBatchAnalysis(
  companies: Array<{ companyId: string; orgNumber: string }>,
  callbackUrl: string
): Promise<WorkflowResult[]> {
  try {
    const results = await Promise.all(
      companies.map(async (company, index) => {
        // Stagger execution by 10 seconds per company to avoid overload
        const delay = index * 10;
        
        const result = await qstashClient.publishJSON({
          url: callbackUrl,
          body: {
            type: 'batch_analysis',
            companyId: company.companyId,
            orgNumber: company.orgNumber,
            batchIndex: index,
            timestamp: new Date().toISOString(),
          },
          delay,
        });
        
        return {
          messageId: result.messageId,
          status: 'scheduled' as const,
          scheduledFor: new Date(Date.now() + delay * 1000),
        };
      })
    );
    
    console.log(`[QStash] Scheduled batch analysis for ${companies.length} companies`);
    return results;
  } catch (error) {
    console.error('[QStash] Error scheduling batch analysis:', error);
    throw error;
  }
}

/**
 * Cancel scheduled workflow
 */
export async function cancelScheduledWorkflow(messageId: string): Promise<void> {
  try {
    await qstashClient.messages.delete(messageId);
    console.log(`[QStash] Cancelled workflow ${messageId}`);
  } catch (error) {
    console.error('[QStash] Error cancelling workflow:', error);
    throw error;
  }
}

/**
 * Get workflow status
 */
export async function getWorkflowStatus(messageId: string): Promise<{
  messageId: string;
  status: string;
  createdAt: number;
  url: string;
}> {
  try {
    const message = await qstashClient.messages.get(messageId);
    
    return {
      messageId: message.messageId,
      status: (message as any).state || 'unknown',
      createdAt: message.createdAt,
      url: message.url,
    };
  } catch (error) {
    console.error('[QStash] Error getting workflow status:', error);
    throw error;
  }
}

/**
 * List all scheduled workflows
 * Note: QStash API doesn't support listing messages, return empty array
 */
export async function listScheduledWorkflows(): Promise<Array<{
  messageId: string;
  status: string;
  createdAt: number;
  url: string;
}>> {
  try {
    // QStash doesn't support listing messages via SDK
    console.log('[QStash] List workflows not supported by SDK');
    return [];
  } catch (error) {
    console.error('[QStash] Error listing workflows:', error);
    return [];
  }
}

/**
 * Verify QStash webhook signature
 */
export async function verifyWebhookSignature(
  signature: string,
  body: string,
  currentSigningKey: string = 'sig_5ZyfsAyuAGWZXQVbYo2eHCG9eeGs'
): Promise<boolean> {
  try {
    const { Receiver } = await import('@upstash/qstash');
    const receiver = new Receiver({
      currentSigningKey,
      nextSigningKey: 'sig_5Mz3FbfTd7tZgviPef9erz3B84na',
    });
    
    await receiver.verify({
      signature,
      body,
    });
    
    return true;
  } catch (error) {
    console.error('[QStash] Webhook signature verification failed:', error);
    return false;
  }
}

/**
 * Schedule periodic credit rating refresh
 */
export async function scheduleCreditRatingRefresh(
  companyId: string,
  orgNumber: string,
  callbackUrl: string
): Promise<WorkflowResult> {
  try {
    // Refresh credit ratings every Monday at 6 AM
    const result = await qstashClient.publishJSON({
      url: callbackUrl,
      body: {
        type: 'credit_rating_refresh',
        companyId,
        orgNumber,
        timestamp: new Date().toISOString(),
      },
      cron: '0 6 * * 1', // Every Monday at 6 AM
    });
    
    console.log(`[QStash] Scheduled weekly credit rating refresh for ${companyId}`);
    
    return {
      messageId: result.messageId,
      status: 'scheduled',
      scheduledFor: new Date(),
    };
  } catch (error) {
    console.error('[QStash] Error scheduling credit rating refresh:', error);
    throw error;
  }
}

/**
 * Get QStash statistics
 * Note: QStash API doesn't support listing messages, return placeholder stats
 */
export async function getQStashStats(): Promise<{
  totalMessages: number;
  scheduledMessages: number;
  status: string;
}> {
  try {
    // QStash doesn't support listing messages via SDK
    return {
      totalMessages: 0,
      scheduledMessages: 0,
      status: 'operational',
    };
  } catch (error) {
    console.error('[QStash] Error getting stats:', error);
    return {
      totalMessages: 0,
      scheduledMessages: 0,
      status: 'error',
    };
  }
}
