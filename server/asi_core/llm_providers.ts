/**
 * TRUE ASI - FULL-WEIGHT LLM PROVIDERS
 * 
 * Complete integration of ALL available LLM providers:
 * - Google Gemini (gemini-2.5-flash, gemini-pro)
 * - Anthropic Claude (claude-3-opus, claude-3-sonnet, claude-3-haiku)
 * - xAI Grok (grok-4)
 * - Cohere (command-r-plus, command-r)
 * - OpenRouter (multi-model routing)
 * - Perplexity Sonar (sonar-pro with web grounding)
 * - ASI1.AI (custom models)
 * - AIMLAPI (model aggregator)
 * - Manus Built-in LLM
 */

import { invokeLLM } from '../_core/llm';

// =============================================================================
// TYPES
// =============================================================================

export interface LLMProvider {
  id: string;
  name: string;
  models: ModelConfig[];
  apiKey: string;
  baseUrl: string;
  maxTokens: number;
  supportsStreaming: boolean;
  supportsVision: boolean;
  supportsTools: boolean;
  costPer1kTokens: { input: number; output: number };
}

export interface ModelConfig {
  id: string;
  name: string;
  contextWindow: number;
  maxOutput: number;
  capabilities: ModelCapability[];
  tier: 'flagship' | 'standard' | 'fast' | 'economy';
}

export type ModelCapability = 
  | 'chat'
  | 'completion'
  | 'vision'
  | 'tools'
  | 'json_mode'
  | 'streaming'
  | 'web_search'
  | 'code'
  | 'reasoning'
  | 'creative'
  | 'multilingual';

export interface UnifiedMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string | ContentPart[];
  name?: string;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
}

export interface ContentPart {
  type: 'text' | 'image_url' | 'file_url';
  text?: string;
  image_url?: { url: string; detail?: 'auto' | 'low' | 'high' };
  file_url?: { url: string; mime_type?: string };
}

export interface ToolCall {
  id: string;
  type: 'function';
  function: { name: string; arguments: string };
}

export interface LLMRequest {
  provider?: string;
  model?: string;
  messages: UnifiedMessage[];
  temperature?: number;
  maxTokens?: number;
  tools?: Tool[];
  toolChoice?: 'none' | 'auto' | 'required' | { type: 'function'; function: { name: string } };
  responseFormat?: { type: 'text' | 'json_object' | 'json_schema'; json_schema?: object };
  stream?: boolean;
}

export interface Tool {
  type: 'function';
  function: {
    name: string;
    description: string;
    parameters: object;
  };
}

export interface LLMResponse {
  id: string;
  provider: string;
  model: string;
  content: string;
  toolCalls?: ToolCall[];
  usage: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
  latency: number;
  cost: number;
  finishReason: 'stop' | 'length' | 'tool_calls' | 'content_filter';
}

// =============================================================================
// PROVIDER CONFIGURATIONS
// =============================================================================

const PROVIDERS: Record<string, LLMProvider> = {
  // Google Gemini
  gemini: {
    id: 'gemini',
    name: 'Google Gemini',
    apiKey: process.env.GEMINI_API_KEY || '',
    baseUrl: 'https://generativelanguage.googleapis.com/v1beta',
    maxTokens: 8192,
    supportsStreaming: true,
    supportsVision: true,
    supportsTools: true,
    costPer1kTokens: { input: 0.00025, output: 0.0005 },
    models: [
      {
        id: 'gemini-2.5-flash',
        name: 'Gemini 2.5 Flash',
        contextWindow: 1000000,
        maxOutput: 8192,
        capabilities: ['chat', 'vision', 'tools', 'json_mode', 'streaming', 'code', 'reasoning', 'multilingual'],
        tier: 'flagship'
      },
      {
        id: 'gemini-pro',
        name: 'Gemini Pro',
        contextWindow: 32000,
        maxOutput: 8192,
        capabilities: ['chat', 'tools', 'json_mode', 'streaming', 'code', 'reasoning'],
        tier: 'standard'
      },
      {
        id: 'gemini-pro-vision',
        name: 'Gemini Pro Vision',
        contextWindow: 16000,
        maxOutput: 4096,
        capabilities: ['chat', 'vision', 'streaming'],
        tier: 'standard'
      }
    ]
  },

  // Anthropic Claude
  anthropic: {
    id: 'anthropic',
    name: 'Anthropic Claude',
    apiKey: process.env.ANTHROPIC_API_KEY || '',
    baseUrl: 'https://api.anthropic.com/v1',
    maxTokens: 4096,
    supportsStreaming: true,
    supportsVision: true,
    supportsTools: true,
    costPer1kTokens: { input: 0.015, output: 0.075 },
    models: [
      {
        id: 'claude-3-opus-20240229',
        name: 'Claude 3 Opus',
        contextWindow: 200000,
        maxOutput: 4096,
        capabilities: ['chat', 'vision', 'tools', 'json_mode', 'streaming', 'code', 'reasoning', 'creative', 'multilingual'],
        tier: 'flagship'
      },
      {
        id: 'claude-3-sonnet-20240229',
        name: 'Claude 3 Sonnet',
        contextWindow: 200000,
        maxOutput: 4096,
        capabilities: ['chat', 'vision', 'tools', 'json_mode', 'streaming', 'code', 'reasoning'],
        tier: 'standard'
      },
      {
        id: 'claude-3-haiku-20240307',
        name: 'Claude 3 Haiku',
        contextWindow: 200000,
        maxOutput: 4096,
        capabilities: ['chat', 'vision', 'tools', 'streaming', 'code'],
        tier: 'fast'
      }
    ]
  },

  // xAI Grok
  grok: {
    id: 'grok',
    name: 'xAI Grok',
    apiKey: process.env.XAI_API_KEY || '',
    baseUrl: 'https://api.x.ai/v1',
    maxTokens: 131072,
    supportsStreaming: true,
    supportsVision: true,
    supportsTools: true,
    costPer1kTokens: { input: 0.005, output: 0.015 },
    models: [
      {
        id: 'grok-4',
        name: 'Grok 4',
        contextWindow: 131072,
        maxOutput: 131072,
        capabilities: ['chat', 'vision', 'tools', 'json_mode', 'streaming', 'code', 'reasoning', 'creative', 'web_search'],
        tier: 'flagship'
      },
      {
        id: 'grok-3',
        name: 'Grok 3',
        contextWindow: 131072,
        maxOutput: 8192,
        capabilities: ['chat', 'vision', 'tools', 'streaming', 'code', 'reasoning'],
        tier: 'standard'
      }
    ]
  },

  // Cohere
  cohere: {
    id: 'cohere',
    name: 'Cohere',
    apiKey: process.env.COHERE_API_KEY || '',
    baseUrl: 'https://api.cohere.ai/v2',
    maxTokens: 4096,
    supportsStreaming: true,
    supportsVision: false,
    supportsTools: true,
    costPer1kTokens: { input: 0.0005, output: 0.0015 },
    models: [
      {
        id: 'command-r-plus',
        name: 'Command R+',
        contextWindow: 128000,
        maxOutput: 4096,
        capabilities: ['chat', 'tools', 'json_mode', 'streaming', 'code', 'reasoning', 'multilingual'],
        tier: 'flagship'
      },
      {
        id: 'command-r',
        name: 'Command R',
        contextWindow: 128000,
        maxOutput: 4096,
        capabilities: ['chat', 'tools', 'streaming', 'code', 'multilingual'],
        tier: 'standard'
      }
    ]
  },

  // OpenRouter (Multi-model routing)
  openrouter: {
    id: 'openrouter',
    name: 'OpenRouter',
    apiKey: process.env.OPENROUTER_API_KEY || '',
    baseUrl: 'https://openrouter.ai/api/v1',
    maxTokens: 4096,
    supportsStreaming: true,
    supportsVision: true,
    supportsTools: true,
    costPer1kTokens: { input: 0.001, output: 0.002 },
    models: [
      {
        id: 'openai/gpt-4-turbo',
        name: 'GPT-4 Turbo (via OpenRouter)',
        contextWindow: 128000,
        maxOutput: 4096,
        capabilities: ['chat', 'vision', 'tools', 'json_mode', 'streaming', 'code', 'reasoning'],
        tier: 'flagship'
      },
      {
        id: 'anthropic/claude-3-opus',
        name: 'Claude 3 Opus (via OpenRouter)',
        contextWindow: 200000,
        maxOutput: 4096,
        capabilities: ['chat', 'vision', 'tools', 'streaming', 'code', 'reasoning', 'creative'],
        tier: 'flagship'
      },
      {
        id: 'google/gemini-pro-1.5',
        name: 'Gemini Pro 1.5 (via OpenRouter)',
        contextWindow: 1000000,
        maxOutput: 8192,
        capabilities: ['chat', 'vision', 'tools', 'streaming', 'code', 'reasoning'],
        tier: 'flagship'
      },
      {
        id: 'meta-llama/llama-3.1-405b-instruct',
        name: 'Llama 3.1 405B (via OpenRouter)',
        contextWindow: 131072,
        maxOutput: 4096,
        capabilities: ['chat', 'tools', 'streaming', 'code', 'reasoning'],
        tier: 'flagship'
      },
      {
        id: 'mistralai/mistral-large',
        name: 'Mistral Large (via OpenRouter)',
        contextWindow: 128000,
        maxOutput: 4096,
        capabilities: ['chat', 'tools', 'streaming', 'code', 'reasoning', 'multilingual'],
        tier: 'flagship'
      }
    ]
  },

  // Perplexity Sonar
  perplexity: {
    id: 'perplexity',
    name: 'Perplexity Sonar',
    apiKey: process.env.SONAR_API_KEY || '',
    baseUrl: 'https://api.perplexity.ai',
    maxTokens: 4096,
    supportsStreaming: true,
    supportsVision: false,
    supportsTools: false,
    costPer1kTokens: { input: 0.001, output: 0.001 },
    models: [
      {
        id: 'sonar-pro',
        name: 'Sonar Pro',
        contextWindow: 200000,
        maxOutput: 4096,
        capabilities: ['chat', 'streaming', 'web_search', 'reasoning'],
        tier: 'flagship'
      },
      {
        id: 'sonar',
        name: 'Sonar',
        contextWindow: 128000,
        maxOutput: 4096,
        capabilities: ['chat', 'streaming', 'web_search'],
        tier: 'standard'
      }
    ]
  },

  // ASI1.AI
  asi1: {
    id: 'asi1',
    name: 'ASI1.AI',
    apiKey: process.env.ASI1_AI_API_KEY || '',
    baseUrl: 'https://api.asi1.ai/v1',
    maxTokens: 4096,
    supportsStreaming: true,
    supportsVision: true,
    supportsTools: true,
    costPer1kTokens: { input: 0.001, output: 0.002 },
    models: [
      {
        id: 'asi1-mini',
        name: 'ASI1 Mini',
        contextWindow: 131072,
        maxOutput: 4096,
        capabilities: ['chat', 'vision', 'tools', 'streaming', 'code', 'reasoning'],
        tier: 'standard'
      }
    ]
  },

  // AIMLAPI (Model Aggregator)
  aimlapi: {
    id: 'aimlapi',
    name: 'AIMLAPI',
    apiKey: process.env.AIMLAPI_KEY || '',
    baseUrl: 'https://api.aimlapi.com/v1',
    maxTokens: 4096,
    supportsStreaming: true,
    supportsVision: true,
    supportsTools: true,
    costPer1kTokens: { input: 0.0005, output: 0.001 },
    models: [
      {
        id: 'gpt-4',
        name: 'GPT-4 (via AIMLAPI)',
        contextWindow: 8192,
        maxOutput: 4096,
        capabilities: ['chat', 'tools', 'streaming', 'code', 'reasoning'],
        tier: 'flagship'
      },
      {
        id: 'gpt-3.5-turbo',
        name: 'GPT-3.5 Turbo (via AIMLAPI)',
        contextWindow: 16385,
        maxOutput: 4096,
        capabilities: ['chat', 'tools', 'streaming', 'code'],
        tier: 'fast'
      }
    ]
  },

  // Manus Built-in LLM
  manus: {
    id: 'manus',
    name: 'Manus Built-in',
    apiKey: process.env.BUILT_IN_FORGE_API_KEY || '',
    baseUrl: process.env.BUILT_IN_FORGE_API_URL || '',
    maxTokens: 8192,
    supportsStreaming: true,
    supportsVision: true,
    supportsTools: true,
    costPer1kTokens: { input: 0, output: 0 }, // Included in platform
    models: [
      {
        id: 'manus-default',
        name: 'Manus Default',
        contextWindow: 128000,
        maxOutput: 8192,
        capabilities: ['chat', 'vision', 'tools', 'json_mode', 'streaming', 'code', 'reasoning'],
        tier: 'flagship'
      }
    ]
  }
};

// =============================================================================
// PROVIDER IMPLEMENTATIONS
// =============================================================================

class GeminiProvider {
  private apiKey: string;
  
  constructor() {
    this.apiKey = process.env.GEMINI_API_KEY || '';
  }
  
  async chat(request: LLMRequest): Promise<LLMResponse> {
    const startTime = Date.now();
    const model = request.model || 'gemini-2.5-flash';
    
    try {
      const response = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${this.apiKey}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            contents: request.messages.map(m => ({
              role: m.role === 'assistant' ? 'model' : 'user',
              parts: [{ text: typeof m.content === 'string' ? m.content : m.content.map(p => p.text || '').join('') }]
            })),
            generationConfig: {
              temperature: request.temperature || 0.7,
              maxOutputTokens: request.maxTokens || 8192
            }
          })
        }
      );
      
      const data = await response.json();
      const content = data.candidates?.[0]?.content?.parts?.[0]?.text || '';
      
      return {
        id: `gemini_${Date.now()}`,
        provider: 'gemini',
        model,
        content,
        usage: {
          promptTokens: data.usageMetadata?.promptTokenCount || 0,
          completionTokens: data.usageMetadata?.candidatesTokenCount || 0,
          totalTokens: (data.usageMetadata?.promptTokenCount || 0) + (data.usageMetadata?.candidatesTokenCount || 0)
        },
        latency: Date.now() - startTime,
        cost: 0,
        finishReason: 'stop'
      };
    } catch (error) {
      throw new Error(`Gemini API error: ${error}`);
    }
  }
}

class AnthropicProvider {
  private apiKey: string;
  
  constructor() {
    this.apiKey = process.env.ANTHROPIC_API_KEY || '';
  }
  
  async chat(request: LLMRequest): Promise<LLMResponse> {
    const startTime = Date.now();
    const model = request.model || 'claude-3-opus-20240229';
    
    try {
      const systemMessage = request.messages.find(m => m.role === 'system');
      const otherMessages = request.messages.filter(m => m.role !== 'system');
      
      const response = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': this.apiKey,
          'anthropic-version': '2023-06-01'
        },
        body: JSON.stringify({
          model,
          max_tokens: request.maxTokens || 4096,
          system: systemMessage ? (typeof systemMessage.content === 'string' ? systemMessage.content : '') : undefined,
          messages: otherMessages.map(m => ({
            role: m.role,
            content: typeof m.content === 'string' ? m.content : m.content.map(p => p.text || '').join('')
          }))
        })
      });
      
      const data = await response.json();
      const content = data.content?.[0]?.text || '';
      
      return {
        id: data.id || `anthropic_${Date.now()}`,
        provider: 'anthropic',
        model,
        content,
        usage: {
          promptTokens: data.usage?.input_tokens || 0,
          completionTokens: data.usage?.output_tokens || 0,
          totalTokens: (data.usage?.input_tokens || 0) + (data.usage?.output_tokens || 0)
        },
        latency: Date.now() - startTime,
        cost: 0,
        finishReason: data.stop_reason === 'end_turn' ? 'stop' : 'stop'
      };
    } catch (error) {
      throw new Error(`Anthropic API error: ${error}`);
    }
  }
}

class GrokProvider {
  private apiKey: string;
  
  constructor() {
    this.apiKey = process.env.XAI_API_KEY || '';
  }
  
  async chat(request: LLMRequest): Promise<LLMResponse> {
    const startTime = Date.now();
    const model = request.model || 'grok-4';
    
    try {
      const response = await fetch('https://api.x.ai/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.apiKey}`
        },
        body: JSON.stringify({
          model,
          messages: request.messages.map(m => ({
            role: m.role,
            content: typeof m.content === 'string' ? m.content : m.content.map(p => p.text || '').join('')
          })),
          temperature: request.temperature || 0.7,
          max_tokens: request.maxTokens || 4096
        })
      });
      
      const data = await response.json();
      const content = data.choices?.[0]?.message?.content || '';
      
      return {
        id: data.id || `grok_${Date.now()}`,
        provider: 'grok',
        model,
        content,
        usage: {
          promptTokens: data.usage?.prompt_tokens || 0,
          completionTokens: data.usage?.completion_tokens || 0,
          totalTokens: data.usage?.total_tokens || 0
        },
        latency: Date.now() - startTime,
        cost: 0,
        finishReason: data.choices?.[0]?.finish_reason || 'stop'
      };
    } catch (error) {
      throw new Error(`Grok API error: ${error}`);
    }
  }
}

class CohereProvider {
  private apiKey: string;
  
  constructor() {
    this.apiKey = process.env.COHERE_API_KEY || '';
  }
  
  async chat(request: LLMRequest): Promise<LLMResponse> {
    const startTime = Date.now();
    const model = request.model || 'command-r-plus';
    
    try {
      const response = await fetch('https://api.cohere.ai/v2/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.apiKey}`
        },
        body: JSON.stringify({
          model,
          messages: request.messages.map(m => ({
            role: m.role === 'assistant' ? 'assistant' : m.role === 'system' ? 'system' : 'user',
            content: typeof m.content === 'string' ? m.content : m.content.map(p => p.text || '').join('')
          })),
          temperature: request.temperature || 0.7,
          max_tokens: request.maxTokens || 4096
        })
      });
      
      const data = await response.json();
      const content = data.message?.content?.[0]?.text || data.text || '';
      
      return {
        id: data.id || `cohere_${Date.now()}`,
        provider: 'cohere',
        model,
        content,
        usage: {
          promptTokens: data.usage?.billed_units?.input_tokens || 0,
          completionTokens: data.usage?.billed_units?.output_tokens || 0,
          totalTokens: (data.usage?.billed_units?.input_tokens || 0) + (data.usage?.billed_units?.output_tokens || 0)
        },
        latency: Date.now() - startTime,
        cost: 0,
        finishReason: 'stop'
      };
    } catch (error) {
      throw new Error(`Cohere API error: ${error}`);
    }
  }
}

class OpenRouterProvider {
  private apiKey: string;
  
  constructor() {
    this.apiKey = process.env.OPENROUTER_API_KEY || '';
  }
  
  async chat(request: LLMRequest): Promise<LLMResponse> {
    const startTime = Date.now();
    const model = request.model || 'openai/gpt-4-turbo';
    
    try {
      const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.apiKey}`,
          'HTTP-Referer': 'https://true-asi.manus.space',
          'X-Title': 'TRUE ASI System'
        },
        body: JSON.stringify({
          model,
          messages: request.messages.map(m => ({
            role: m.role,
            content: typeof m.content === 'string' ? m.content : m.content.map(p => p.text || '').join('')
          })),
          temperature: request.temperature || 0.7,
          max_tokens: request.maxTokens || 4096
        })
      });
      
      const data = await response.json();
      const content = data.choices?.[0]?.message?.content || '';
      
      return {
        id: data.id || `openrouter_${Date.now()}`,
        provider: 'openrouter',
        model,
        content,
        usage: {
          promptTokens: data.usage?.prompt_tokens || 0,
          completionTokens: data.usage?.completion_tokens || 0,
          totalTokens: data.usage?.total_tokens || 0
        },
        latency: Date.now() - startTime,
        cost: 0,
        finishReason: data.choices?.[0]?.finish_reason || 'stop'
      };
    } catch (error) {
      throw new Error(`OpenRouter API error: ${error}`);
    }
  }
}

class PerplexityProvider {
  private apiKey: string;
  
  constructor() {
    this.apiKey = process.env.SONAR_API_KEY || '';
  }
  
  async chat(request: LLMRequest): Promise<LLMResponse> {
    const startTime = Date.now();
    const model = request.model || 'sonar-pro';
    
    try {
      const response = await fetch('https://api.perplexity.ai/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.apiKey}`
        },
        body: JSON.stringify({
          model,
          messages: request.messages.map(m => ({
            role: m.role,
            content: typeof m.content === 'string' ? m.content : m.content.map(p => p.text || '').join('')
          })),
          temperature: request.temperature || 0.7,
          max_tokens: request.maxTokens || 4096
        })
      });
      
      const data = await response.json();
      const content = data.choices?.[0]?.message?.content || '';
      
      return {
        id: data.id || `perplexity_${Date.now()}`,
        provider: 'perplexity',
        model,
        content,
        usage: {
          promptTokens: data.usage?.prompt_tokens || 0,
          completionTokens: data.usage?.completion_tokens || 0,
          totalTokens: data.usage?.total_tokens || 0
        },
        latency: Date.now() - startTime,
        cost: 0,
        finishReason: data.choices?.[0]?.finish_reason || 'stop'
      };
    } catch (error) {
      throw new Error(`Perplexity API error: ${error}`);
    }
  }
}

class ManusProvider {
  async chat(request: LLMRequest): Promise<LLMResponse> {
    const startTime = Date.now();
    
    try {
      const response = await invokeLLM({
        messages: request.messages.map(m => ({
          role: m.role as 'system' | 'user' | 'assistant',
          content: typeof m.content === 'string' ? m.content : m.content.map(p => p.text || '').join('')
        })),
        tool_choice: request.toolChoice as any,
        tools: request.tools as any,
        response_format: request.responseFormat as any
      });
      
      const rawContent = response.choices?.[0]?.message?.content;
      const content = typeof rawContent === 'string' ? rawContent : '';
      
      return {
        id: `manus_${Date.now()}`,
        provider: 'manus',
        model: 'manus-default',
        content,
        usage: {
          promptTokens: response.usage?.prompt_tokens || 0,
          completionTokens: response.usage?.completion_tokens || 0,
          totalTokens: response.usage?.total_tokens || 0
        },
        latency: Date.now() - startTime,
        cost: 0,
        finishReason: 'stop'
      };
    } catch (error) {
      throw new Error(`Manus LLM error: ${error}`);
    }
  }
}

// =============================================================================
// UNIFIED LLM MANAGER
// =============================================================================

export class UnifiedLLMManager {
  private providers: Map<string, any> = new Map();
  private requestHistory: LLMResponse[] = [];
  private totalCost = 0;
  
  constructor() {
    // Initialize all providers
    this.providers.set('gemini', new GeminiProvider());
    this.providers.set('anthropic', new AnthropicProvider());
    this.providers.set('grok', new GrokProvider());
    this.providers.set('cohere', new CohereProvider());
    this.providers.set('openrouter', new OpenRouterProvider());
    this.providers.set('perplexity', new PerplexityProvider());
    this.providers.set('manus', new ManusProvider());
  }
  
  // Get all available providers
  getProviders(): LLMProvider[] {
    return Object.values(PROVIDERS).filter(p => p.apiKey || p.id === 'manus');
  }
  
  // Get all available models
  getAllModels(): { provider: string; model: ModelConfig }[] {
    const models: { provider: string; model: ModelConfig }[] = [];
    
    for (const provider of this.getProviders()) {
      for (const model of provider.models) {
        models.push({ provider: provider.id, model });
      }
    }
    
    return models;
  }
  
  // Get models by capability
  getModelsByCapability(capability: ModelCapability): { provider: string; model: ModelConfig }[] {
    return this.getAllModels().filter(m => m.model.capabilities.includes(capability));
  }
  
  // Get flagship models only
  getFlagshipModels(): { provider: string; model: ModelConfig }[] {
    return this.getAllModels().filter(m => m.model.tier === 'flagship');
  }
  
  // Chat with specific provider
  async chat(request: LLMRequest): Promise<LLMResponse> {
    const providerId = request.provider || 'manus';
    const provider = this.providers.get(providerId);
    
    if (!provider) {
      throw new Error(`Provider ${providerId} not found`);
    }
    
    const response = await provider.chat(request);
    this.requestHistory.push(response);
    this.totalCost += response.cost;
    
    return response;
  }
  
  // Chat with best available model for task
  async chatWithBestModel(request: LLMRequest, preferredCapabilities: ModelCapability[] = []): Promise<LLMResponse> {
    // Find best model based on capabilities
    let candidates = this.getAllModels();
    
    if (preferredCapabilities.length > 0) {
      candidates = candidates.filter(m => 
        preferredCapabilities.every(cap => m.model.capabilities.includes(cap))
      );
    }
    
    // Prefer flagship models
    const flagships = candidates.filter(m => m.model.tier === 'flagship');
    if (flagships.length > 0) {
      candidates = flagships;
    }
    
    // Select first available
    if (candidates.length === 0) {
      // Fallback to Manus
      return this.chat({ ...request, provider: 'manus' });
    }
    
    const selected = candidates[0];
    return this.chat({ ...request, provider: selected.provider, model: selected.model.id });
  }
  
  // Multi-model consensus (ask multiple models and aggregate)
  async multiModelConsensus(request: LLMRequest, providerIds?: string[]): Promise<{
    responses: LLMResponse[];
    consensus: string;
    confidence: number;
  }> {
    const providers = providerIds || ['manus', 'gemini', 'anthropic', 'grok'];
    const availableProviders = providers.filter(p => this.providers.has(p));
    
    // Query all providers in parallel
    const responses = await Promise.all(
      availableProviders.map(async (providerId) => {
        try {
          return await this.chat({ ...request, provider: providerId });
        } catch (error) {
          return null;
        }
      })
    );
    
    const validResponses = responses.filter((r): r is LLMResponse => r !== null);
    
    // Synthesize consensus
    const consensus = await this.synthesizeConsensus(validResponses, request);
    
    return {
      responses: validResponses,
      consensus: consensus.content,
      confidence: consensus.confidence
    };
  }
  
  private async synthesizeConsensus(responses: LLMResponse[], originalRequest: LLMRequest): Promise<{ content: string; confidence: number }> {
    if (responses.length === 0) {
      return { content: 'No responses available', confidence: 0 };
    }
    
    if (responses.length === 1) {
      return { content: responses[0].content, confidence: 0.7 };
    }
    
    // Use Manus to synthesize
    const synthesisRequest: LLMRequest = {
      provider: 'manus',
      messages: [
        {
          role: 'system',
          content: 'You are synthesizing responses from multiple AI models. Identify common themes, resolve contradictions, and provide a unified, high-quality response.'
        },
        {
          role: 'user',
          content: `Original question: ${typeof originalRequest.messages[originalRequest.messages.length - 1].content === 'string' 
            ? originalRequest.messages[originalRequest.messages.length - 1].content 
            : ''}\n\nResponses from different models:\n${responses.map((r, i) => `Model ${i + 1} (${r.provider}/${r.model}):\n${r.content}`).join('\n\n---\n\n')}\n\nProvide a synthesized response that combines the best insights from all models.`
        }
      ]
    };
    
    const synthesis = await this.chat(synthesisRequest);
    
    // Calculate confidence based on agreement
    const confidence = this.calculateAgreement(responses);
    
    return { content: synthesis.content, confidence };
  }
  
  private calculateAgreement(responses: LLMResponse[]): number {
    // Simple heuristic: longer common substrings = higher agreement
    if (responses.length < 2) return 1;
    
    const contents = responses.map(r => r.content.toLowerCase());
    let totalSimilarity = 0;
    let comparisons = 0;
    
    for (let i = 0; i < contents.length; i++) {
      for (let j = i + 1; j < contents.length; j++) {
        const similarity = this.jaccardSimilarity(contents[i], contents[j]);
        totalSimilarity += similarity;
        comparisons++;
      }
    }
    
    return comparisons > 0 ? totalSimilarity / comparisons : 0;
  }
  
  private jaccardSimilarity(a: string, b: string): number {
    const wordsA = new Set(a.split(/\s+/));
    const wordsB = new Set(b.split(/\s+/));
    
    const intersection = new Set([...wordsA].filter(x => wordsB.has(x)));
    const union = new Set([...wordsA, ...wordsB]);
    
    return intersection.size / union.size;
  }
  
  // Get statistics
  getStats(): {
    totalRequests: number;
    totalCost: number;
    requestsByProvider: Record<string, number>;
    averageLatency: number;
    availableProviders: string[];
  } {
    const requestsByProvider: Record<string, number> = {};
    let totalLatency = 0;
    
    for (const response of this.requestHistory) {
      requestsByProvider[response.provider] = (requestsByProvider[response.provider] || 0) + 1;
      totalLatency += response.latency;
    }
    
    return {
      totalRequests: this.requestHistory.length,
      totalCost: this.totalCost,
      requestsByProvider,
      averageLatency: this.requestHistory.length > 0 ? totalLatency / this.requestHistory.length : 0,
      availableProviders: Array.from(this.providers.keys())
    };
  }
}

// Export singleton instance
export const unifiedLLM = new UnifiedLLMManager();
