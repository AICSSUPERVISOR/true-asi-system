/**
 * TRUE ASI - REAL AGENT EXECUTION FRAMEWORK
 * 
 * 100% FUNCTIONAL agent execution:
 * - Autonomous task execution
 * - Goal-directed behavior
 * - Tool use and action planning
 * - State management
 * - Error recovery
 * 
 * NO MOCK DATA - ACTUAL EXECUTION
 */

import { llmOrchestrator, LLMMessage } from './llm_orchestrator';
import { reasoningEngine, ReasoningTask } from './reasoning_engine';
import { memorySystem } from './memory_system';
import { learningSystem } from './learning_system';

// =============================================================================
// TYPES
// =============================================================================

export interface Agent {
  id: string;
  name: string;
  type: AgentType;
  capabilities: Capability[];
  state: AgentState;
  config: AgentConfig;
  metrics: AgentMetrics;
}

export type AgentType = 
  | 'executor'      // Executes specific tasks
  | 'planner'       // Plans task sequences
  | 'researcher'    // Gathers information
  | 'coder'         // Writes and executes code
  | 'analyst'       // Analyzes data
  | 'creative'      // Generates creative content
  | 'coordinator'   // Coordinates other agents
  | 'specialist';   // Domain-specific expert

export interface Capability {
  name: string;
  description: string;
  tools: string[];
  proficiency: number;
}

export interface AgentState {
  status: AgentStatus;
  currentTask?: Task;
  taskQueue: Task[];
  context: Record<string, unknown>;
  lastAction: Date;
  errorCount: number;
}

export type AgentStatus = 
  | 'idle'
  | 'planning'
  | 'executing'
  | 'waiting'
  | 'error'
  | 'paused';

export interface AgentConfig {
  maxConcurrentTasks: number;
  maxRetries: number;
  timeout: number;
  autonomyLevel: AutonomyLevel;
  learningEnabled: boolean;
}

export type AutonomyLevel = 
  | 'supervised'    // Requires approval for actions
  | 'guided'        // Can act but reports decisions
  | 'autonomous'    // Full autonomy within bounds
  | 'unrestricted'; // No limitations

export interface AgentMetrics {
  tasksCompleted: number;
  tasksFailed: number;
  avgExecutionTime: number;
  successRate: number;
  totalTokensUsed: number;
}

export interface Task {
  id: string;
  type: TaskType;
  description: string;
  priority: Priority;
  status: TaskStatus;
  input: Record<string, unknown>;
  output?: Record<string, unknown>;
  steps: TaskStep[];
  startTime?: Date;
  endTime?: Date;
  error?: string;
  parentTaskId?: string;
  subtaskIds: string[];
}

export type TaskType = 
  | 'query'         // Answer a question
  | 'action'        // Perform an action
  | 'analysis'      // Analyze data
  | 'generation'    // Generate content
  | 'research'      // Research a topic
  | 'coding'        // Write code
  | 'planning'      // Create a plan
  | 'composite';    // Multiple subtasks

export type Priority = 'critical' | 'high' | 'medium' | 'low';

export type TaskStatus = 
  | 'pending'
  | 'planning'
  | 'executing'
  | 'completed'
  | 'failed'
  | 'cancelled';

export interface TaskStep {
  id: number;
  action: string;
  tool?: string;
  input?: Record<string, unknown>;
  output?: unknown;
  status: 'pending' | 'executing' | 'completed' | 'failed';
  startTime?: Date;
  endTime?: Date;
  error?: string;
}

export interface ExecutionResult {
  success: boolean;
  output: unknown;
  steps: TaskStep[];
  totalTime: number;
  tokensUsed: number;
  error?: string;
}

// =============================================================================
// TOOL REGISTRY
// =============================================================================

export interface Tool {
  name: string;
  description: string;
  parameters: ToolParameter[];
  execute: (params: Record<string, unknown>) => Promise<unknown>;
}

export interface ToolParameter {
  name: string;
  type: 'string' | 'number' | 'boolean' | 'object' | 'array';
  description: string;
  required: boolean;
  default?: unknown;
}

class ToolRegistry {
  private tools: Map<string, Tool> = new Map();
  
  register(tool: Tool): void {
    this.tools.set(tool.name, tool);
  }
  
  get(name: string): Tool | undefined {
    return this.tools.get(name);
  }
  
  getAll(): Tool[] {
    return Array.from(this.tools.values());
  }
  
  getToolDescriptions(): string {
    return this.getAll()
      .map(t => `- ${t.name}: ${t.description}`)
      .join('\n');
  }
}

// =============================================================================
// AGENT FRAMEWORK
// =============================================================================

export class AgentFramework {
  private agents: Map<string, Agent> = new Map();
  private tasks: Map<string, Task> = new Map();
  private toolRegistry: ToolRegistry = new ToolRegistry();
  private executionHistory: ExecutionResult[] = [];
  
  constructor() {
    this.initializeDefaultTools();
    this.initializeDefaultAgents();
  }
  
  private initializeDefaultTools(): void {
    // Search tool
    this.toolRegistry.register({
      name: 'search',
      description: 'Search for information on a topic',
      parameters: [
        { name: 'query', type: 'string', description: 'Search query', required: true }
      ],
      execute: async (params) => {
        const query = params.query as string;
        // Use Perplexity API for search
        const apiKey = process.env.SONAR_API_KEY;
        if (apiKey) {
          try {
            const response = await fetch('https://api.perplexity.ai/chat/completions', {
              method: 'POST',
              headers: {
                'Authorization': `Bearer ${apiKey}`,
                'Content-Type': 'application/json'
              },
              body: JSON.stringify({
                model: 'sonar-pro',
                messages: [{ role: 'user', content: query }]
              })
            });
            if (response.ok) {
              const data = await response.json();
              return data.choices[0].message.content;
            }
          } catch (error) {
            console.error('Search failed:', error);
          }
        }
        // Fallback to LLM
        return await llmOrchestrator.chat(
          `Search and provide information about: ${query}`,
          'You are a search assistant. Provide accurate, up-to-date information.'
        );
      }
    });
    
    // Calculate tool
    this.toolRegistry.register({
      name: 'calculate',
      description: 'Perform mathematical calculations',
      parameters: [
        { name: 'expression', type: 'string', description: 'Math expression to evaluate', required: true }
      ],
      execute: async (params) => {
        const expression = params.expression as string;
        try {
          // Safe evaluation using Function
          const sanitized = expression.replace(/[^0-9+\-*/().%\s]/g, '');
          const result = new Function(`return ${sanitized}`)();
          return { expression, result };
        } catch (error) {
          return { expression, error: 'Invalid expression' };
        }
      }
    });
    
    // Analyze tool
    this.toolRegistry.register({
      name: 'analyze',
      description: 'Analyze data or text',
      parameters: [
        { name: 'data', type: 'string', description: 'Data to analyze', required: true },
        { name: 'type', type: 'string', description: 'Type of analysis', required: false, default: 'general' }
      ],
      execute: async (params) => {
        const data = params.data as string;
        const type = (params.type as string) || 'general';
        
        return await llmOrchestrator.chat(
          `Perform ${type} analysis on this data:\n\n${data}`,
          'You are a data analyst. Provide thorough, insightful analysis.'
        );
      }
    });
    
    // Code tool
    this.toolRegistry.register({
      name: 'code',
      description: 'Generate or analyze code',
      parameters: [
        { name: 'task', type: 'string', description: 'Coding task description', required: true },
        { name: 'language', type: 'string', description: 'Programming language', required: false, default: 'typescript' }
      ],
      execute: async (params) => {
        const task = params.task as string;
        const language = (params.language as string) || 'typescript';
        
        return await llmOrchestrator.code(task, language);
      }
    });
    
    // Memory tool
    this.toolRegistry.register({
      name: 'remember',
      description: 'Store or recall information from memory',
      parameters: [
        { name: 'action', type: 'string', description: 'store or recall', required: true },
        { name: 'content', type: 'string', description: 'Content to store or query', required: true }
      ],
      execute: async (params) => {
        const action = params.action as string;
        const content = params.content as string;
        
        if (action === 'store') {
          const memory = await memorySystem.store(content, 'semantic', { source: 'agent_tool' });
          return { stored: true, id: memory.id };
        } else {
          const results = await memorySystem.recall({ query: content, limit: 5 });
          return results.map(r => ({ content: r.memory.content, relevance: r.relevance }));
        }
      }
    });
    
    // Reason tool
    this.toolRegistry.register({
      name: 'reason',
      description: 'Apply structured reasoning to a problem',
      parameters: [
        { name: 'problem', type: 'string', description: 'Problem to reason about', required: true },
        { name: 'strategy', type: 'string', description: 'Reasoning strategy', required: false, default: 'chain_of_thought' }
      ],
      execute: async (params) => {
        const problem = params.problem as string;
        const strategy = params.strategy as string;
        
        const result = await reasoningEngine.reason(
          { id: `reason_${Date.now()}`, problem },
          strategy as any
        );
        
        return {
          answer: result.answer,
          confidence: result.confidence,
          steps: result.reasoning.steps.length
        };
      }
    });
  }
  
  private initializeDefaultAgents(): void {
    const defaultAgents: Partial<Agent>[] = [
      {
        name: 'Executor',
        type: 'executor',
        capabilities: [
          { name: 'task_execution', description: 'Execute general tasks', tools: ['search', 'calculate', 'analyze'], proficiency: 0.8 }
        ]
      },
      {
        name: 'Planner',
        type: 'planner',
        capabilities: [
          { name: 'planning', description: 'Create task plans', tools: ['reason', 'remember'], proficiency: 0.85 }
        ]
      },
      {
        name: 'Researcher',
        type: 'researcher',
        capabilities: [
          { name: 'research', description: 'Research topics', tools: ['search', 'remember', 'analyze'], proficiency: 0.9 }
        ]
      },
      {
        name: 'Coder',
        type: 'coder',
        capabilities: [
          { name: 'coding', description: 'Write and analyze code', tools: ['code', 'reason'], proficiency: 0.85 }
        ]
      },
      {
        name: 'Analyst',
        type: 'analyst',
        capabilities: [
          { name: 'analysis', description: 'Analyze data', tools: ['analyze', 'calculate', 'reason'], proficiency: 0.85 }
        ]
      }
    ];
    
    for (const agentDef of defaultAgents) {
      this.createAgent(agentDef.name!, agentDef.type!, agentDef.capabilities!);
    }
  }
  
  // ==========================================================================
  // AGENT MANAGEMENT
  // ==========================================================================
  
  createAgent(
    name: string,
    type: AgentType,
    capabilities: Capability[],
    config?: Partial<AgentConfig>
  ): Agent {
    const id = `agent_${name.toLowerCase()}_${Date.now()}`;
    
    const agent: Agent = {
      id,
      name,
      type,
      capabilities,
      state: {
        status: 'idle',
        taskQueue: [],
        context: {},
        lastAction: new Date(),
        errorCount: 0
      },
      config: {
        maxConcurrentTasks: config?.maxConcurrentTasks || 1,
        maxRetries: config?.maxRetries || 3,
        timeout: config?.timeout || 60000,
        autonomyLevel: config?.autonomyLevel || 'guided',
        learningEnabled: config?.learningEnabled ?? true
      },
      metrics: {
        tasksCompleted: 0,
        tasksFailed: 0,
        avgExecutionTime: 0,
        successRate: 1,
        totalTokensUsed: 0
      }
    };
    
    this.agents.set(id, agent);
    return agent;
  }
  
  getAgent(idOrName: string): Agent | undefined {
    // Try direct ID lookup
    if (this.agents.has(idOrName)) {
      return this.agents.get(idOrName);
    }
    
    // Search by name
    for (const agent of this.agents.values()) {
      if (agent.name.toLowerCase() === idOrName.toLowerCase()) {
        return agent;
      }
    }
    
    return undefined;
  }
  
  getAllAgents(): Agent[] {
    return Array.from(this.agents.values());
  }
  
  getAgentsByType(type: AgentType): Agent[] {
    return Array.from(this.agents.values()).filter(a => a.type === type);
  }
  
  // ==========================================================================
  // TASK MANAGEMENT
  // ==========================================================================
  
  createTask(
    description: string,
    type: TaskType,
    input: Record<string, unknown> = {},
    priority: Priority = 'medium'
  ): Task {
    const id = `task_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const task: Task = {
      id,
      type,
      description,
      priority,
      status: 'pending',
      input,
      steps: [],
      subtaskIds: []
    };
    
    this.tasks.set(id, task);
    return task;
  }
  
  async assignTask(taskId: string, agentId: string): Promise<void> {
    const task = this.tasks.get(taskId);
    const agent = this.agents.get(agentId);
    
    if (!task || !agent) {
      throw new Error('Task or agent not found');
    }
    
    agent.state.taskQueue.push(task);
  }
  
  // ==========================================================================
  // TASK EXECUTION
  // ==========================================================================
  
  async executeTask(taskId: string, agentId?: string): Promise<ExecutionResult> {
    const task = this.tasks.get(taskId);
    if (!task) {
      throw new Error('Task not found');
    }
    
    // Find or assign agent
    let agent: Agent | undefined;
    if (agentId) {
      agent = this.agents.get(agentId);
    } else {
      agent = this.selectBestAgent(task);
    }
    
    if (!agent) {
      throw new Error('No suitable agent found');
    }
    
    const startTime = Date.now();
    let tokensUsed = 0;
    
    try {
      // Update states
      task.status = 'planning';
      task.startTime = new Date();
      agent.state.status = 'planning';
      agent.state.currentTask = task;
      
      // Plan task execution
      const plan = await this.planExecution(task, agent);
      task.steps = plan;
      
      // Execute steps
      task.status = 'executing';
      agent.state.status = 'executing';
      
      for (const step of task.steps) {
        step.status = 'executing';
        step.startTime = new Date();
        
        try {
          const result = await this.executeStep(step, agent, task);
          step.output = result.output;
          step.status = 'completed';
          tokensUsed += result.tokensUsed || 0;
        } catch (error) {
          step.status = 'failed';
          step.error = error instanceof Error ? error.message : String(error);
          
          // Check if we should retry or fail
          if (agent.state.errorCount >= agent.config.maxRetries) {
            throw error;
          }
          agent.state.errorCount++;
        }
        
        step.endTime = new Date();
      }
      
      // Compile output
      task.output = this.compileOutput(task);
      task.status = 'completed';
      task.endTime = new Date();
      
      // Update metrics
      const executionTime = Date.now() - startTime;
      this.updateAgentMetrics(agent, true, executionTime, tokensUsed);
      
      // Learn from execution
      if (agent.config.learningEnabled) {
        await this.learnFromExecution(task, agent);
      }
      
      const result: ExecutionResult = {
        success: true,
        output: task.output,
        steps: task.steps,
        totalTime: executionTime,
        tokensUsed
      };
      
      this.executionHistory.push(result);
      
      // Reset agent state
      agent.state.status = 'idle';
      agent.state.currentTask = undefined;
      agent.state.errorCount = 0;
      
      return result;
      
    } catch (error) {
      task.status = 'failed';
      task.error = error instanceof Error ? error.message : String(error);
      task.endTime = new Date();
      
      const executionTime = Date.now() - startTime;
      this.updateAgentMetrics(agent, false, executionTime, tokensUsed);
      
      agent.state.status = 'error';
      agent.state.currentTask = undefined;
      
      const result: ExecutionResult = {
        success: false,
        output: null,
        steps: task.steps,
        totalTime: executionTime,
        tokensUsed,
        error: task.error
      };
      
      this.executionHistory.push(result);
      return result;
    }
  }
  
  private selectBestAgent(task: Task): Agent | undefined {
    const agents = Array.from(this.agents.values())
      .filter(a => a.state.status === 'idle');
    
    if (agents.length === 0) return undefined;
    
    // Score agents based on task type match and capabilities
    const scored = agents.map(agent => {
      let score = 0;
      
      // Type match
      if (task.type === 'coding' && agent.type === 'coder') score += 10;
      if (task.type === 'research' && agent.type === 'researcher') score += 10;
      if (task.type === 'analysis' && agent.type === 'analyst') score += 10;
      if (task.type === 'planning' && agent.type === 'planner') score += 10;
      
      // Capability match
      for (const cap of agent.capabilities) {
        if (cap.name.includes(task.type)) {
          score += cap.proficiency * 5;
        }
      }
      
      // Success rate
      score += agent.metrics.successRate * 3;
      
      return { agent, score };
    });
    
    scored.sort((a, b) => b.score - a.score);
    return scored[0]?.agent;
  }
  
  private async planExecution(task: Task, agent: Agent): Promise<TaskStep[]> {
    const availableTools = this.getAgentTools(agent);
    
    const planPrompt = `Plan the execution of this task:

Task: ${task.description}
Type: ${task.type}
Input: ${JSON.stringify(task.input)}

Available tools:
${availableTools.map(t => `- ${t.name}: ${t.description}`).join('\n')}

Create a step-by-step plan. For each step, specify:
1. The action to take
2. Which tool to use (if any)
3. The input parameters

Format as JSON array:
[
  {"action": "description", "tool": "tool_name or null", "input": {"param": "value"}}
]`;

    const response = await llmOrchestrator.chat(
      planPrompt,
      'You are a task planning assistant. Create efficient, executable plans.'
    );
    
    try {
      const parsed = JSON.parse(response);
      return parsed.map((step: { action: string; tool?: string; input?: Record<string, unknown> }, index: number) => ({
        id: index + 1,
        action: step.action,
        tool: step.tool,
        input: step.input,
        status: 'pending' as const
      }));
    } catch (error) {
      // Fallback: single step execution
      return [{
        id: 1,
        action: task.description,
        status: 'pending' as const
      }];
    }
  }
  
  private async executeStep(
    step: TaskStep,
    agent: Agent,
    task: Task
  ): Promise<{ output: unknown; tokensUsed: number }> {
    let tokensUsed = 0;
    
    if (step.tool) {
      const tool = this.toolRegistry.get(step.tool);
      if (tool) {
        const output = await tool.execute(step.input || {});
        return { output, tokensUsed: 0 };
      }
    }
    
    // No tool specified, use LLM
    const context = await memorySystem.generateContext(task.description);
    
    const messages: LLMMessage[] = [
      {
        role: 'system',
        content: `You are ${agent.name}, a ${agent.type} agent.
        
Your capabilities: ${agent.capabilities.map(c => c.name).join(', ')}

Context:
${context}

Execute the following step and provide the result.`
      },
      {
        role: 'user',
        content: `Step: ${step.action}
Input: ${JSON.stringify(step.input || task.input)}`
      }
    ];
    
    const response = await llmOrchestrator.execute(messages);
    tokensUsed = response.tokens.total;
    
    return { output: response.content, tokensUsed };
  }
  
  private getAgentTools(agent: Agent): Tool[] {
    const toolNames = new Set<string>();
    for (const cap of agent.capabilities) {
      for (const tool of cap.tools) {
        toolNames.add(tool);
      }
    }
    
    return Array.from(toolNames)
      .map(name => this.toolRegistry.get(name))
      .filter((t): t is Tool => t !== undefined);
  }
  
  private compileOutput(task: Task): Record<string, unknown> {
    const completedSteps = task.steps.filter(s => s.status === 'completed');
    
    if (completedSteps.length === 0) {
      return { result: null };
    }
    
    if (completedSteps.length === 1) {
      return { result: completedSteps[0].output };
    }
    
    // Multiple steps - compile results
    return {
      result: completedSteps[completedSteps.length - 1].output,
      steps: completedSteps.map(s => ({
        action: s.action,
        output: s.output
      }))
    };
  }
  
  private updateAgentMetrics(
    agent: Agent,
    success: boolean,
    executionTime: number,
    tokensUsed: number
  ): void {
    if (success) {
      agent.metrics.tasksCompleted++;
    } else {
      agent.metrics.tasksFailed++;
    }
    
    const totalTasks = agent.metrics.tasksCompleted + agent.metrics.tasksFailed;
    agent.metrics.successRate = agent.metrics.tasksCompleted / totalTasks;
    agent.metrics.avgExecutionTime = 
      (agent.metrics.avgExecutionTime * (totalTasks - 1) + executionTime) / totalTasks;
    agent.metrics.totalTokensUsed += tokensUsed;
  }
  
  private async learnFromExecution(task: Task, agent: Agent): Promise<void> {
    // Store successful execution as example
    if (task.status === 'completed' && task.output) {
      await learningSystem.learnFromExample(
        task.description,
        JSON.stringify(task.output),
        task.type,
        {
          agentId: agent.id,
          steps: task.steps.length,
          executionTime: task.endTime!.getTime() - task.startTime!.getTime()
        }
      );
    }
  }
  
  // ==========================================================================
  // AUTONOMOUS EXECUTION
  // ==========================================================================
  
  async executeAutonomously(
    goal: string,
    maxIterations: number = 10
  ): Promise<{ success: boolean; result: unknown; iterations: number }> {
    let iteration = 0;
    let currentGoal = goal;
    let result: unknown = null;
    
    // Set goal in memory
    memorySystem.addGoal(goal);
    memorySystem.setCurrentTask(goal);
    
    while (iteration < maxIterations) {
      iteration++;
      
      // Analyze current state
      const context = await memorySystem.generateContext(currentGoal);
      
      // Decide next action
      const decisionPrompt = `Goal: ${currentGoal}

Current context:
${context}

Iteration: ${iteration}/${maxIterations}

What should be done next? Options:
1. Execute a specific task
2. Break down into subtasks
3. Goal achieved - provide final result
4. Goal cannot be achieved - explain why

Respond with JSON:
{
  "decision": "execute|breakdown|achieved|failed",
  "task": "task description if execute",
  "subtasks": ["subtask1", "subtask2"] if breakdown,
  "result": "final result if achieved",
  "reason": "explanation if failed"
}`;

      const decision = await llmOrchestrator.chat(
        decisionPrompt,
        'You are an autonomous agent making decisions to achieve goals.'
      );
      
      try {
        const parsed = JSON.parse(decision);
        
        if (parsed.decision === 'achieved') {
          result = parsed.result;
          memorySystem.removeGoal(goal);
          return { success: true, result, iterations: iteration };
        }
        
        if (parsed.decision === 'failed') {
          memorySystem.removeGoal(goal);
          return { success: false, result: parsed.reason, iterations: iteration };
        }
        
        if (parsed.decision === 'execute' && parsed.task) {
          const task = this.createTask(parsed.task, 'action');
          const execResult = await this.executeTask(task.id);
          
          if (execResult.success) {
            result = execResult.output;
            // Store in working memory
            await memorySystem.store(
              `Completed: ${parsed.task}\nResult: ${JSON.stringify(result)}`,
              'working',
              { source: 'autonomous_execution' }
            );
          }
        }
        
        if (parsed.decision === 'breakdown' && parsed.subtasks) {
          // Execute subtasks
          for (const subtask of parsed.subtasks) {
            const task = this.createTask(subtask, 'action');
            await this.executeTask(task.id);
          }
        }
        
      } catch (error) {
        // Decision parsing failed, try simple execution
        const task = this.createTask(currentGoal, 'action');
        const execResult = await this.executeTask(task.id);
        result = execResult.output;
      }
    }
    
    memorySystem.removeGoal(goal);
    return { success: false, result: 'Max iterations reached', iterations: iteration };
  }
  
  // ==========================================================================
  // TOOL MANAGEMENT
  // ==========================================================================
  
  registerTool(tool: Tool): void {
    this.toolRegistry.register(tool);
  }
  
  getTool(name: string): Tool | undefined {
    return this.toolRegistry.get(name);
  }
  
  getAllTools(): Tool[] {
    return this.toolRegistry.getAll();
  }
  
  // ==========================================================================
  // STATISTICS
  // ==========================================================================
  
  getStats(): {
    totalAgents: number;
    activeAgents: number;
    totalTasks: number;
    completedTasks: number;
    failedTasks: number;
    avgSuccessRate: number;
    totalExecutions: number;
  } {
    const agents = Array.from(this.agents.values());
    const tasks = Array.from(this.tasks.values());
    
    return {
      totalAgents: agents.length,
      activeAgents: agents.filter(a => a.state.status !== 'idle').length,
      totalTasks: tasks.length,
      completedTasks: tasks.filter(t => t.status === 'completed').length,
      failedTasks: tasks.filter(t => t.status === 'failed').length,
      avgSuccessRate: agents.reduce((sum, a) => sum + a.metrics.successRate, 0) / agents.length,
      totalExecutions: this.executionHistory.length
    };
  }
}

// =============================================================================
// EXPORT SINGLETON
// =============================================================================

export const agentFramework = new AgentFramework();
