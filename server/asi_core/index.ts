/**
 * TRUE ASI - Core Module Exports
 * 
 * This is the main entry point for the TRUE ASI system.
 * All core functionality is exported from here.
 */

// Main ASI System
export { trueASI, initializeASI, askASI, executeASITask, chatWithASI } from './true_asi';
export type { ASIConfig, ASIState, ASIRequest, ASIResponse, ASICapabilities } from './true_asi';

// LLM Orchestrator
export { llmOrchestrator } from './llm_orchestrator';
export type { LLMMessage, LLMResponse } from './llm_orchestrator';

// Reasoning Engine
export { reasoningEngine } from './reasoning_engine';
export type { ReasoningTask, ReasoningResult, ReasoningStrategy } from './reasoning_engine';

// Memory System
export { memorySystem } from './memory_system';
export type { Memory, MemoryType, MemoryQuery } from './memory_system';

// Learning System
export { learningSystem } from './learning_system';
export type { LearningExample, Feedback, Skill } from './learning_system';

// Agent Framework
export { agentFramework } from './agent_framework';
export type { Agent, Task, ExecutionResult } from './agent_framework';

// Knowledge Graph
export { knowledgeGraph } from './knowledge_graph';
export type { Entity, Relationship, KnowledgeTriple } from './knowledge_graph';

// Tool Executor
export { toolExecutor } from './tool_executor';
export type { ToolExecutionResult, CodeExecutionResult } from './tool_executor';

// Multi-Agent Coordinator
export { multiAgentCoordinator } from './multi_agent_coordinator';
export type { CollectiveDecision, SwarmTask } from './multi_agent_coordinator';

// Self-Improvement Engine
export { selfImprovementEngine } from './self_improvement';
export type { SelfReflection, EvolutionGeneration } from './self_improvement';

// Benchmark System
export { benchmarkSystem } from './benchmark_system';
export type { BenchmarkResult, ASIScorecard } from './benchmark_system';

// Router
export { asiRouter } from './asi_router';

// Unified ASI Hub
export { asiHub, UnifiedASIHub } from './unified_asi_hub';
export type { ASICapabilities as HubCapabilities, ASIStatus, ASIRequest as HubRequest, ASIResponse as HubResponse } from './unified_asi_hub';

// Full-Weight LLM Providers
export { unifiedLLM, UnifiedLLMManager } from './llm_providers';

// Manus Connectors
export { manusConnector, UnifiedManusConnector } from './manus_connectors';

// MCP Integrations
export { mcpManager, UnifiedMCPManager } from './mcp_integrations';

// Business APIs
export { businessAPIs, UnifiedBusinessAPIManager } from './business_apis';

// Knowledge Infrastructure (10+ TB)
export { knowledgeInfrastructure, KnowledgeInfrastructure } from './knowledge_infrastructure';
export type { KnowledgeItem, KnowledgeStats, KnowledgeSource } from './knowledge_infrastructure';

// Repository Mining
export { repositoryMiner, RepositoryMiner } from './repository_miner';
export type { Repository, MiningResult, CodePattern } from './repository_miner';

// Knowledge Synthesis
export { knowledgeSynthesis, KnowledgeSynthesisEngine } from './knowledge_synthesis';
export type { SynthesisRequest, SynthesisResult, Insight } from './knowledge_synthesis';
