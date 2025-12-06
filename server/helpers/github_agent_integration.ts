/**
 * GitHub Agent Integration
 * 
 * Integrates 251 specialized agents from AICSSUPERVISOR/true-asi-system repository
 * for enhanced AI-powered business analysis and recommendations.
 * 
 * Repository: https://github.com/AICSSUPERVISOR/true-asi-system
 * Total Agents: 251
 * Total Files: 567
 */

import { readFileSync, readdirSync } from 'fs';
import { join } from 'path';

const REPO_PATH = '/home/ubuntu/true-asi-system';
const AGENTS_PATH = join(REPO_PATH, 'agents');

export interface Agent {
  id: number;
  name: string;
  specialty: string;
  description: string;
  capabilities: string[];
  status: 'operational' | 'idle' | 'busy';
  tasksCompleted: number;
  successRate: number;
}

export interface AgentTask {
  type: string;
  data: any;
  parameters?: Record<string, any>;
}

export interface AgentResult {
  agentId: number;
  specialty: string;
  result: any;
  executionTime: number;
  success: boolean;
}

/**
 * Parse agent file to extract metadata
 */
function parseAgentFile(filePath: string): Agent | null {
  try {
    const content = readFileSync(filePath, 'utf-8');
    
    // Extract agent ID from filename (agent_000.py -> 0)
    const filename = filePath.split('/').pop() || '';
    const idMatch = filename.match(/agent_(\d+)\.py/);
    const id = idMatch ? parseInt(idMatch[1]) : 0;
    
    // Extract specialty from docstring
    const specialtyMatch = content.match(/Specialty:\s*(\w+)/i);
    const specialty = specialtyMatch ? specialtyMatch[1] : 'general';
    
    // Extract description
    const descMatch = content.match(/Agent \d+ - (\w+)/);
    const name = descMatch ? descMatch[1] : `Agent ${id}`;
    
    // Extract capabilities from docstring
    const capabilitiesSection = content.match(/Capabilities:([\s\S]*?)(?=Agent ID:|Status:|$)/);
    const capabilities: string[] = [];
    
    if (capabilitiesSection) {
      const lines = capabilitiesSection[1].split('\n');
      lines.forEach(line => {
        const capMatch = line.match(/- (.+)/);
        if (capMatch) {
          capabilities.push(capMatch[1].trim());
        }
      });
    }
    
    return {
      id,
      name,
      specialty,
      description: `Specialized autonomous agent for ${specialty} tasks`,
      capabilities: capabilities.length > 0 ? capabilities : [
        'Autonomous task execution',
        'Hivemind communication',
        'Continuous learning',
        'Self-optimization',
      ],
      status: 'operational',
      tasksCompleted: 0,
      successRate: 1.0,
    };
  } catch (error) {
    console.error(`[GitHub] Error parsing agent file ${filePath}:`, error);
    return null;
  }
}

/**
 * Load all agents from repository
 */
export function loadAllAgents(): Agent[] {
  try {
    const agentFiles = readdirSync(AGENTS_PATH)
      .filter(file => file.startsWith('agent_') && file.endsWith('.py'))
      .sort();
    
    const agents = agentFiles
      .map(file => parseAgentFile(join(AGENTS_PATH, file)))
      .filter((agent): agent is Agent => agent !== null);
    
    console.log(`[GitHub] Loaded ${agents.length} agents from repository`);
    return agents;
  } catch (error) {
    console.error('[GitHub] Error loading agents:', error);
    return [];
  }
}

/**
 * Get agent by ID
 */
export function getAgentById(id: number): Agent | null {
  try {
    const filePath = join(AGENTS_PATH, `agent_${id.toString().padStart(3, '0')}.py`);
    return parseAgentFile(filePath);
  } catch (error) {
    console.error(`[GitHub] Error getting agent ${id}:`, error);
    return null;
  }
}

/**
 * Get agents by specialty
 */
export function getAgentsBySpecialty(specialty: string): Agent[] {
  const allAgents = loadAllAgents();
  return allAgents.filter(agent => 
    agent.specialty.toLowerCase() === specialty.toLowerCase()
  );
}

/**
 * Get recommended agents for a task type
 */
export function getRecommendedAgents(taskType: string, count: number = 5): Agent[] {
  const allAgents = loadAllAgents();
  
  // Map task types to agent specialties
  const specialtyMap: Record<string, string[]> = {
    'business_analysis': ['reasoning', 'analysis', 'strategy', 'financial', 'market'],
    'financial_analysis': ['financial', 'accounting', 'analysis', 'data', 'statistics'],
    'marketing': ['marketing', 'content', 'social', 'seo', 'advertising'],
    'operations': ['operations', 'logistics', 'optimization', 'automation', 'efficiency'],
    'technology': ['technology', 'coding', 'development', 'infrastructure', 'security'],
    'sales': ['sales', 'crm', 'negotiation', 'communication', 'relationship'],
    'customer_service': ['support', 'communication', 'problem_solving', 'empathy', 'service'],
    'leadership': ['leadership', 'management', 'strategy', 'decision_making', 'vision'],
  };
  
  const relevantSpecialties = specialtyMap[taskType] || ['reasoning', 'analysis'];
  
  // Filter agents by relevant specialties
  const relevantAgents = allAgents.filter(agent =>
    relevantSpecialties.some(specialty => 
      agent.specialty.toLowerCase().includes(specialty.toLowerCase())
    )
  );
  
  // If not enough relevant agents, add general agents
  if (relevantAgents.length < count) {
    const generalAgents = allAgents.filter(agent => 
      !relevantAgents.includes(agent)
    );
    relevantAgents.push(...generalAgents.slice(0, count - relevantAgents.length));
  }
  
  return relevantAgents.slice(0, count);
}

/**
 * Simulate agent task execution (placeholder for actual Python integration)
 */
export async function executeAgentTask(
  agentId: number,
  task: AgentTask
): Promise<AgentResult> {
  const startTime = Date.now();
  
  try {
    const agent = getAgentById(agentId);
    
    if (!agent) {
      throw new Error(`Agent ${agentId} not found`);
    }
    
    // Simulate task execution (in production, this would call Python agent)
    await new Promise(resolve => setTimeout(resolve, 100));
    
    const executionTime = Date.now() - startTime;
    
    console.log(`[GitHub] Agent ${agentId} (${agent.specialty}) executed task in ${executionTime}ms`);
    
    return {
      agentId,
      specialty: agent.specialty,
      result: {
        status: 'completed',
        message: `Task executed successfully by ${agent.name}`,
        data: task.data,
      },
      executionTime,
      success: true,
    };
  } catch (error) {
    const executionTime = Date.now() - startTime;
    
    console.error(`[GitHub] Error executing agent ${agentId} task:`, error);
    
    return {
      agentId,
      specialty: 'unknown',
      result: {
        status: 'failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      },
      executionTime,
      success: false,
    };
  }
}

/**
 * Execute task with multiple agents in parallel
 */
export async function executeMultiAgentTask(
  agentIds: number[],
  task: AgentTask
): Promise<AgentResult[]> {
  console.log(`[GitHub] Executing task with ${agentIds.length} agents in parallel`);
  
  const results = await Promise.all(
    agentIds.map(id => executeAgentTask(id, task))
  );
  
  const successCount = results.filter(r => r.success).length;
  console.log(`[GitHub] Multi-agent task completed: ${successCount}/${results.length} successful`);
  
  return results;
}

/**
 * Get agent statistics
 */
export function getAgentStats(): {
  totalAgents: number;
  specialties: Record<string, number>;
  topSpecialties: Array<{ specialty: string; count: number }>;
} {
  const allAgents = loadAllAgents();
  
  const specialties: Record<string, number> = {};
  allAgents.forEach(agent => {
    specialties[agent.specialty] = (specialties[agent.specialty] || 0) + 1;
  });
  
  const topSpecialties = Object.entries(specialties)
    .map(([specialty, count]) => ({ specialty, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 10);
  
  return {
    totalAgents: allAgents.length,
    specialties,
    topSpecialties,
  };
}

/**
 * Search agents by capability
 */
export function searchAgentsByCapability(capability: string): Agent[] {
  const allAgents = loadAllAgents();
  
  return allAgents.filter(agent =>
    agent.capabilities.some(cap =>
      cap.toLowerCase().includes(capability.toLowerCase())
    )
  );
}

/**
 * Get agent recommendations for company analysis
 */
export function getCompanyAnalysisAgents(): Agent[] {
  return getRecommendedAgents('business_analysis', 10);
}

/**
 * Get agent recommendations for financial analysis
 */
export function getFinancialAnalysisAgents(): Agent[] {
  return getRecommendedAgents('financial_analysis', 10);
}

/**
 * Get agent recommendations for marketing strategy
 */
export function getMarketingStrategyAgents(): Agent[] {
  return getRecommendedAgents('marketing', 10);
}
