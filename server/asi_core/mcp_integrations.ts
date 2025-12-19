/**
 * TRUE ASI - MCP SERVER INTEGRATIONS
 * 
 * Complete integration with all available MCP servers:
 * - Stripe: Payments, subscriptions, customers
 * - Zapier: Workflow automation
 * - Supabase: Database management
 * - Vercel: Deployment management
 * - Firecrawl: Web scraping
 * - HuggingFace: AI models
 * - Canva: Design generation
 * - Gmail: Email automation
 */

import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

// =============================================================================
// TYPES
// =============================================================================

export interface MCPToolResult {
  success: boolean;
  data?: any;
  error?: string;
  executionTime: number;
}

export interface MCPTool {
  name: string;
  description: string;
  server: string;
  parameters: Record<string, any>;
}

// =============================================================================
// MCP CLI WRAPPER
// =============================================================================

class MCPClient {
  private async execute(server: string, tool: string, input: Record<string, any>): Promise<MCPToolResult> {
    const startTime = Date.now();
    
    try {
      const inputJson = JSON.stringify(input).replace(/"/g, '\\"');
      const command = `manus-mcp-cli tool call ${tool} --server ${server} --input "${inputJson}"`;
      
      const { stdout, stderr } = await execAsync(command, { timeout: 60000 });
      
      if (stderr && !stdout) {
        return {
          success: false,
          error: stderr,
          executionTime: Date.now() - startTime
        };
      }
      
      try {
        const data = JSON.parse(stdout);
        return {
          success: true,
          data,
          executionTime: Date.now() - startTime
        };
      } catch {
        return {
          success: true,
          data: stdout.trim(),
          executionTime: Date.now() - startTime
        };
      }
    } catch (error: any) {
      return {
        success: false,
        error: error.message || String(error),
        executionTime: Date.now() - startTime
      };
    }
  }
  
  async listTools(server: string): Promise<MCPTool[]> {
    try {
      const { stdout } = await execAsync(`manus-mcp-cli tool list --server ${server}`);
      // Parse tool list from output
      const lines = stdout.split('\n').filter(l => l.trim());
      return lines.map(line => ({
        name: line.trim(),
        description: '',
        server,
        parameters: {}
      }));
    } catch {
      return [];
    }
  }
  
  // Stripe MCP
  stripe = {
    createCustomer: (email: string, name?: string, metadata?: Record<string, string>) =>
      this.execute('stripe', 'create_customer', { email, name, metadata }),
    
    listCustomers: (limit?: number) =>
      this.execute('stripe', 'list_customers', { limit: limit || 10 }),
    
    createSubscription: (customerId: string, priceId: string) =>
      this.execute('stripe', 'create_subscription', { customer_id: customerId, price_id: priceId }),
    
    listSubscriptions: (customerId?: string) =>
      this.execute('stripe', 'list_subscriptions', { customer_id: customerId }),
    
    cancelSubscription: (subscriptionId: string) =>
      this.execute('stripe', 'cancel_subscription', { subscription_id: subscriptionId }),
    
    createPaymentIntent: (amount: number, currency: string, customerId?: string) =>
      this.execute('stripe', 'create_payment_intent', { amount, currency, customer_id: customerId }),
    
    createPaymentLink: (priceId: string, quantity?: number) =>
      this.execute('stripe', 'create_payment_link', { price_id: priceId, quantity: quantity || 1 }),
    
    createProduct: (name: string, description?: string) =>
      this.execute('stripe', 'create_product', { name, description }),
    
    createPrice: (productId: string, unitAmount: number, currency: string, recurring?: { interval: string }) =>
      this.execute('stripe', 'create_price', { product_id: productId, unit_amount: unitAmount, currency, recurring }),
    
    listProducts: (limit?: number) =>
      this.execute('stripe', 'list_products', { limit: limit || 10 }),
    
    listPrices: (productId?: string) =>
      this.execute('stripe', 'list_prices', { product_id: productId }),
    
    createInvoice: (customerId: string) =>
      this.execute('stripe', 'create_invoice', { customer_id: customerId }),
    
    listInvoices: (customerId?: string) =>
      this.execute('stripe', 'list_invoices', { customer_id: customerId }),
    
    createCoupon: (percentOff?: number, amountOff?: number, currency?: string, duration?: string) =>
      this.execute('stripe', 'create_coupon', { percent_off: percentOff, amount_off: amountOff, currency, duration }),
    
    listCoupons: () =>
      this.execute('stripe', 'list_coupons', {}),
    
    createRefund: (paymentIntentId: string, amount?: number) =>
      this.execute('stripe', 'create_refund', { payment_intent_id: paymentIntentId, amount }),
    
    getBalance: () =>
      this.execute('stripe', 'get_balance', {}),
    
    listDisputes: () =>
      this.execute('stripe', 'list_disputes', {})
  };
  
  // Zapier MCP
  zapier = {
    search: (query: string, app?: string) =>
      this.execute('zapier', 'search', { query, app }),
    
    executeAction: (actionId: string, data: Record<string, any>) =>
      this.execute('zapier', 'execute_action', { action_id: actionId, data }),
    
    listZaps: () =>
      this.execute('zapier', 'list_zaps', {}),
    
    triggerZap: (zapId: string, data: Record<string, any>) =>
      this.execute('zapier', 'trigger_zap', { zap_id: zapId, data })
  };
  
  // Supabase MCP
  supabase = {
    listProjects: () =>
      this.execute('supabase', 'list_projects', {}),
    
    getProject: (projectId: string) =>
      this.execute('supabase', 'get_project', { project_id: projectId }),
    
    listTables: (projectId: string) =>
      this.execute('supabase', 'list_tables', { project_id: projectId }),
    
    executeQuery: (projectId: string, query: string) =>
      this.execute('supabase', 'execute_query', { project_id: projectId, query }),
    
    createTable: (projectId: string, tableName: string, columns: Array<{ name: string; type: string }>) =>
      this.execute('supabase', 'create_table', { project_id: projectId, table_name: tableName, columns }),
    
    insertRow: (projectId: string, tableName: string, data: Record<string, any>) =>
      this.execute('supabase', 'insert_row', { project_id: projectId, table_name: tableName, data }),
    
    selectRows: (projectId: string, tableName: string, filters?: Record<string, any>) =>
      this.execute('supabase', 'select_rows', { project_id: projectId, table_name: tableName, filters }),
    
    updateRow: (projectId: string, tableName: string, id: string, data: Record<string, any>) =>
      this.execute('supabase', 'update_row', { project_id: projectId, table_name: tableName, id, data }),
    
    deleteRow: (projectId: string, tableName: string, id: string) =>
      this.execute('supabase', 'delete_row', { project_id: projectId, table_name: tableName, id }),
    
    listFunctions: (projectId: string) =>
      this.execute('supabase', 'list_functions', { project_id: projectId }),
    
    invokeFunction: (projectId: string, functionName: string, body: any) =>
      this.execute('supabase', 'invoke_function', { project_id: projectId, function_name: functionName, body })
  };
  
  // Vercel MCP
  vercel = {
    listProjects: (teamId?: string) =>
      this.execute('vercel', 'list_projects', { team_id: teamId }),
    
    getProject: (projectId: string) =>
      this.execute('vercel', 'get_project', { project_id: projectId }),
    
    listDeployments: (projectId?: string) =>
      this.execute('vercel', 'list_deployments', { project_id: projectId }),
    
    getDeployment: (deploymentId: string) =>
      this.execute('vercel', 'get_deployment', { deployment_id: deploymentId }),
    
    getDeploymentLogs: (deploymentId: string) =>
      this.execute('vercel', 'get_deployment_logs', { deployment_id: deploymentId }),
    
    listDomains: (projectId?: string) =>
      this.execute('vercel', 'list_domains', { project_id: projectId }),
    
    addDomain: (projectId: string, domain: string) =>
      this.execute('vercel', 'add_domain', { project_id: projectId, domain }),
    
    removeDomain: (projectId: string, domain: string) =>
      this.execute('vercel', 'remove_domain', { project_id: projectId, domain }),
    
    listTeams: () =>
      this.execute('vercel', 'list_teams', {}),
    
    searchDocs: (query: string) =>
      this.execute('vercel', 'search_docs', { query })
  };
  
  // Firecrawl MCP
  firecrawl = {
    scrape: (url: string, options?: { formats?: string[]; onlyMainContent?: boolean }) =>
      this.execute('firecrawl', 'firecrawl_scrape', { url, ...options }),
    
    crawl: (url: string, options?: { maxDepth?: number; limit?: number }) =>
      this.execute('firecrawl', 'firecrawl_crawl', { url, ...options }),
    
    map: (url: string) =>
      this.execute('firecrawl', 'firecrawl_map', { url }),
    
    search: (query: string, options?: { limit?: number }) =>
      this.execute('firecrawl', 'firecrawl_search', { query, ...options }),
    
    extract: (url: string, schema: Record<string, any>) =>
      this.execute('firecrawl', 'firecrawl_extract', { url, schema })
  };
  
  // HuggingFace MCP
  huggingface = {
    searchModels: (query: string, options?: { limit?: number; filter?: string }) =>
      this.execute('hugging-face', 'search_models', { query, ...options }),
    
    getModel: (modelId: string) =>
      this.execute('hugging-face', 'get_model', { model_id: modelId }),
    
    searchDatasets: (query: string, options?: { limit?: number }) =>
      this.execute('hugging-face', 'search_datasets', { query, ...options }),
    
    getDataset: (datasetId: string) =>
      this.execute('hugging-face', 'get_dataset', { dataset_id: datasetId }),
    
    searchPapers: (query: string, options?: { limit?: number }) =>
      this.execute('hugging-face', 'search_papers', { query, ...options }),
    
    getPaper: (paperId: string) =>
      this.execute('hugging-face', 'get_paper', { paper_id: paperId }),
    
    listSpaces: (options?: { limit?: number; sort?: string }) =>
      this.execute('hugging-face', 'list_spaces', options || {}),
    
    getSpace: (spaceId: string) =>
      this.execute('hugging-face', 'get_space', { space_id: spaceId })
  };
  
  // Canva MCP
  canva = {
    listDesigns: (options?: { limit?: number }) =>
      this.execute('canva', 'list_designs', options || {}),
    
    getDesign: (designId: string) =>
      this.execute('canva', 'get_design', { design_id: designId }),
    
    generateDesign: (query: string) =>
      this.execute('canva', 'generate_design', { query }),
    
    createDesignFromCandidate: (candidateId: string) =>
      this.execute('canva', 'create_design_from_candidate', { candidate_id: candidateId }),
    
    getExportFormats: (designId: string) =>
      this.execute('canva', 'get_export_formats', { design_id: designId }),
    
    exportDesign: (designId: string, format: string) =>
      this.execute('canva', 'export_design', { design_id: designId, format }),
    
    resizeDesign: (designId: string, width: number, height: number) =>
      this.execute('canva', 'resize_design', { design_id: designId, width, height }),
    
    listTemplates: (options?: { category?: string; limit?: number }) =>
      this.execute('canva', 'list_templates', options || {}),
    
    createFromTemplate: (templateId: string) =>
      this.execute('canva', 'create_from_template', { template_id: templateId })
  };
  
  // Gmail MCP
  gmail = {
    listMessages: (options?: { maxResults?: number; query?: string }) =>
      this.execute('gmail', 'list_messages', options || {}),
    
    getMessage: (messageId: string) =>
      this.execute('gmail', 'get_message', { message_id: messageId }),
    
    sendMessage: (to: string, subject: string, body: string, options?: { cc?: string; bcc?: string }) =>
      this.execute('gmail', 'send_message', { to, subject, body, ...options }),
    
    replyToMessage: (messageId: string, body: string) =>
      this.execute('gmail', 'reply_to_message', { message_id: messageId, body }),
    
    createDraft: (to: string, subject: string, body: string) =>
      this.execute('gmail', 'create_draft', { to, subject, body }),
    
    listDrafts: () =>
      this.execute('gmail', 'list_drafts', {}),
    
    sendDraft: (draftId: string) =>
      this.execute('gmail', 'send_draft', { draft_id: draftId }),
    
    deleteDraft: (draftId: string) =>
      this.execute('gmail', 'delete_draft', { draft_id: draftId }),
    
    listLabels: () =>
      this.execute('gmail', 'list_labels', {}),
    
    addLabel: (messageId: string, labelId: string) =>
      this.execute('gmail', 'add_label', { message_id: messageId, label_id: labelId }),
    
    removeLabel: (messageId: string, labelId: string) =>
      this.execute('gmail', 'remove_label', { message_id: messageId, label_id: labelId }),
    
    markAsRead: (messageId: string) =>
      this.execute('gmail', 'mark_as_read', { message_id: messageId }),
    
    markAsUnread: (messageId: string) =>
      this.execute('gmail', 'mark_as_unread', { message_id: messageId }),
    
    deleteMessage: (messageId: string) =>
      this.execute('gmail', 'delete_message', { message_id: messageId }),
    
    searchMessages: (query: string, options?: { maxResults?: number }) =>
      this.execute('gmail', 'search_messages', { query, ...options })
  };
}

// =============================================================================
// UNIFIED MCP MANAGER
// =============================================================================

export class UnifiedMCPManager {
  private client: MCPClient;
  private executionHistory: Array<{
    server: string;
    tool: string;
    result: MCPToolResult;
    timestamp: Date;
  }> = [];
  
  constructor() {
    this.client = new MCPClient();
  }
  
  // Get all MCP clients
  get stripe() { return this.client.stripe; }
  get zapier() { return this.client.zapier; }
  get supabase() { return this.client.supabase; }
  get vercel() { return this.client.vercel; }
  get firecrawl() { return this.client.firecrawl; }
  get huggingface() { return this.client.huggingface; }
  get canva() { return this.client.canva; }
  get gmail() { return this.client.gmail; }
  
  // List all available tools across all servers
  async listAllTools(): Promise<Record<string, MCPTool[]>> {
    const servers = ['stripe', 'zapier', 'supabase', 'vercel', 'firecrawl', 'hugging-face', 'canva', 'gmail'];
    const tools: Record<string, MCPTool[]> = {};
    
    for (const server of servers) {
      tools[server] = await this.client.listTools(server);
    }
    
    return tools;
  }
  
  // Get execution statistics
  getStats(): {
    totalExecutions: number;
    successRate: number;
    averageExecutionTime: number;
    executionsByServer: Record<string, number>;
  } {
    const successCount = this.executionHistory.filter(e => e.result.success).length;
    const totalTime = this.executionHistory.reduce((sum, e) => sum + e.result.executionTime, 0);
    
    const executionsByServer: Record<string, number> = {};
    for (const execution of this.executionHistory) {
      executionsByServer[execution.server] = (executionsByServer[execution.server] || 0) + 1;
    }
    
    return {
      totalExecutions: this.executionHistory.length,
      successRate: this.executionHistory.length > 0 ? successCount / this.executionHistory.length : 0,
      averageExecutionTime: this.executionHistory.length > 0 ? totalTime / this.executionHistory.length : 0,
      executionsByServer
    };
  }
  
  // Get available servers
  getAvailableServers(): string[] {
    return ['stripe', 'zapier', 'supabase', 'vercel', 'firecrawl', 'hugging-face', 'canva', 'gmail'];
  }
  
  // Get server capabilities
  getServerCapabilities(): Record<string, string[]> {
    return {
      stripe: ['payments', 'subscriptions', 'customers', 'products', 'invoices', 'refunds', 'coupons'],
      zapier: ['workflow_automation', 'app_integration', 'triggers', 'actions'],
      supabase: ['database', 'auth', 'storage', 'functions', 'realtime'],
      vercel: ['deployments', 'domains', 'projects', 'teams', 'logs'],
      firecrawl: ['web_scraping', 'crawling', 'search', 'extraction'],
      'hugging-face': ['models', 'datasets', 'papers', 'spaces'],
      canva: ['design_generation', 'templates', 'export', 'resize'],
      gmail: ['email', 'drafts', 'labels', 'search']
    };
  }
}

// Export singleton instance
export const mcpManager = new UnifiedMCPManager();
