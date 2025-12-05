/**
 * COMPLETE AI MODELS INTEGRATION
 * 
 * Integrates ALL AI model providers with intelligent routing, automatic failover,
 * cost tracking, and performance monitoring.
 * 
 * Providers (in priority order):
 * 1. ASI1.AI (Primary - 99.5% success rate)
 * 2. AIMLAPI (200+ models)
 * 3. OpenAI (GPT-4o, GPT-4-turbo)
 * 4. Claude (Claude 3.5 Sonnet)
 * 5. Gemini (Gemini 2.0)
 * 6. Grok (Grok Beta)
 * 7. Cohere (Command R+)
 * 8. Perplexity (Sonar Pro - web-grounded)
 */

import axios from 'axios';

// ============================================================================
// TYPES
// ============================================================================

export interface AIMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface AIModelRequest {
  messages: AIMessage[];
  maxTokens?: number;
  temperature?: number;
  model?: string;
  provider?: AIProvider;
}

export interface AIModelResponse {
  content: string;
  provider: AIProvider;
  model: string;
  tokensUsed: number;
  cost: number;
  responseTime: number;
  success: boolean;
}

export type AIProvider = 'asi1' | 'aimlapi' | 'openai' | 'claude' | 'gemini' | 'grok' | 'cohere' | 'perplexity';

interface ProviderConfig {
  name: string;
  apiKey: string;
  endpoint: string;
  models: string[];
  costPerToken: number;
  priority: number;
  enabled: boolean;
}

// ============================================================================
// PROVIDER CONFIGURATIONS
// ============================================================================

const PROVIDERS: Record<AIProvider, ProviderConfig> = {
  asi1: {
    name: 'ASI1.AI',
    apiKey: process.env.ASI1_AI_API_KEY || 'sk_26ec4938b6274ae089bfa915d02bf10036bde0326b5845c5b87c50b5dbc2c9ad',
    endpoint: 'https://api.asi1.ai/v1/chat/completions',
    models: ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo'],
    costPerToken: 0.00001,
    priority: 1,
    enabled: true
  },
  aimlapi: {
    name: 'AIMLAPI',
    apiKey: process.env.AIMLAPI_KEY || '147620aa16e04b96bb2f12b79527593f',
    endpoint: 'https://api.aimlapi.com/chat/completions',
    models: ['gpt-4', 'claude-3-opus', 'gemini-pro'],
    costPerToken: 0.000015,
    priority: 2,
    enabled: true
  },
  openai: {
    name: 'OpenAI',
    apiKey: process.env.OPENAI_API_KEY || '',
    endpoint: 'https://api.openai.com/v1/chat/completions',
    models: ['gpt-4o', 'gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo'],
    costPerToken: 0.00003,
    priority: 3,
    enabled: !!process.env.OPENAI_API_KEY
  },
  claude: {
    name: 'Anthropic Claude',
    apiKey: process.env.ANTHROPIC_API_KEY || '',
    endpoint: 'https://api.anthropic.com/v1/messages',
    models: ['claude-3-5-sonnet-20241022', 'claude-3-opus-20240229'],
    costPerToken: 0.000015,
    priority: 4,
    enabled: !!process.env.ANTHROPIC_API_KEY
  },
  gemini: {
    name: 'Google Gemini',
    apiKey: process.env.GEMINI_API_KEY || '',
    endpoint: 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent',
    models: ['gemini-2.0-flash-exp', 'gemini-1.5-pro'],
    costPerToken: 0.0000075,
    priority: 5,
    enabled: !!process.env.GEMINI_API_KEY
  },
  grok: {
    name: 'xAI Grok',
    apiKey: process.env.XAI_API_KEY || '',
    endpoint: 'https://api.x.ai/v1/chat/completions',
    models: ['grok-beta'],
    costPerToken: 0.00002,
    priority: 6,
    enabled: !!process.env.XAI_API_KEY
  },
  cohere: {
    name: 'Cohere',
    apiKey: process.env.COHERE_API_KEY || '',
    endpoint: 'https://api.cohere.ai/v2/chat',
    models: ['command-r-plus', 'command-r'],
    costPerToken: 0.000015,
    priority: 7,
    enabled: !!process.env.COHERE_API_KEY
  },
  perplexity: {
    name: 'Perplexity',
    apiKey: process.env.SONAR_API_KEY || '',
    endpoint: 'https://api.perplexity.ai/chat/completions',
    models: ['sonar-pro', 'sonar'],
    costPerToken: 0.00001,
    priority: 8,
    enabled: !!process.env.SONAR_API_KEY
  }
};

// ============================================================================
// INTELLIGENT AI ROUTING
// ============================================================================

/**
 * Get available providers sorted by priority
 */
function getAvailableProviders(): ProviderConfig[] {
  return Object.values(PROVIDERS)
    .filter(p => p.enabled)
    .sort((a, b) => a.priority - b.priority);
}

/**
 * Invoke AI with automatic provider selection and failover
 */
export async function invokeAI(request: AIModelRequest): Promise<AIModelResponse> {
  const providers = request.provider 
    ? [PROVIDERS[request.provider]]
    : getAvailableProviders();

  let lastError: Error | null = null;

  for (const provider of providers) {
    try {
      console.log(`[AI] Trying provider: ${provider.name}`);
      const response = await callProvider(provider, request);
      console.log(`[AI] Success with ${provider.name}`);
      return response;
    } catch (error) {
      console.error(`[AI] Failed with ${provider.name}:`, error);
      lastError = error as Error;
      // Continue to next provider
    }
  }

  throw new Error(`All AI providers failed. Last error: ${lastError?.message}`);
}

/**
 * Call specific AI provider
 */
async function callProvider(
  provider: ProviderConfig,
  request: AIModelRequest
): Promise<AIModelResponse> {
  const startTime = Date.now();
  const model = request.model || provider.models[0];

  // Build request based on provider
  let apiRequest: any;
  let headers: any;

  switch (provider.name) {
    case 'Anthropic Claude':
      apiRequest = {
        model,
        messages: request.messages.filter(m => m.role !== 'system'),
        system: request.messages.find(m => m.role === 'system')?.content,
        max_tokens: request.maxTokens || 4096,
        temperature: request.temperature || 0.7
      };
      headers = {
        'Content-Type': 'application/json',
        'x-api-key': provider.apiKey,
        'anthropic-version': '2023-06-01'
      };
      break;

    case 'Google Gemini':
      apiRequest = {
        contents: request.messages.map(m => ({
          role: m.role === 'assistant' ? 'model' : 'user',
          parts: [{ text: m.content }]
        }))
      };
      headers = {
        'Content-Type': 'application/json'
      };
      // Add API key to URL
      provider.endpoint += `?key=${provider.apiKey}`;
      break;

    case 'Cohere':
      apiRequest = {
        model,
        messages: request.messages,
        max_tokens: request.maxTokens || 4096,
        temperature: request.temperature || 0.7
      };
      headers = {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${provider.apiKey}`
      };
      break;

    default:
      // OpenAI-compatible (ASI1.AI, AIMLAPI, OpenAI, Grok, Perplexity)
      apiRequest = {
        model,
        messages: request.messages,
        max_tokens: request.maxTokens || 4096,
        temperature: request.temperature || 0.7
      };
      headers = {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${provider.apiKey}`
      };
  }

  const response = await axios.post(provider.endpoint, apiRequest, {
    headers,
    timeout: 60000 // 60 second timeout
  });

  const responseTime = Date.now() - startTime;

  // Parse response based on provider
  let content: string;
  let tokensUsed: number;

  switch (provider.name) {
    case 'Anthropic Claude':
      content = response.data.content[0].text;
      tokensUsed = response.data.usage.input_tokens + response.data.usage.output_tokens;
      break;

    case 'Google Gemini':
      content = response.data.candidates[0].content.parts[0].text;
      tokensUsed = response.data.usageMetadata?.totalTokenCount || 0;
      break;

    case 'Cohere':
      content = response.data.message.content[0].text;
      tokensUsed = response.data.usage?.tokens?.total || 0;
      break;

    default:
      // OpenAI-compatible
      content = response.data.choices[0].message.content;
      tokensUsed = response.data.usage?.total_tokens || 0;
  }

  const cost = tokensUsed * provider.costPerToken;

  return {
    content,
    provider: Object.keys(PROVIDERS).find(k => PROVIDERS[k as AIProvider] === provider) as AIProvider,
    model,
    tokensUsed,
    cost,
    responseTime,
    success: true
  };
}

/**
 * Invoke multiple AI providers in parallel and compare results
 */
export async function invokeAIParallel(
  request: AIModelRequest,
  providers: AIProvider[]
): Promise<AIModelResponse[]> {
  const promises = providers.map(provider =>
    invokeAI({ ...request, provider }).catch(error => ({
      content: '',
      provider,
      model: '',
      tokensUsed: 0,
      cost: 0,
      responseTime: 0,
      success: false,
      error: error.message
    }))
  );

  return Promise.all(promises);
}

/**
 * Get AI provider health status
 */
export async function getAIProvidersHealth(): Promise<Record<AIProvider, boolean>> {
  const health: Record<string, boolean> = {};

  for (const [key, provider] of Object.entries(PROVIDERS)) {
    if (!provider.enabled) {
      health[key] = false;
      continue;
    }

    try {
      await callProvider(provider, {
        messages: [{ role: 'user', content: 'ping' }],
        maxTokens: 10
      });
      health[key] = true;
    } catch (error) {
      health[key] = false;
    }
  }

  return health as Record<AIProvider, boolean>;
}

/**
 * Get cost estimate for request
 */
export function estimateCost(
  messages: AIMessage[],
  provider?: AIProvider
): { provider: string; estimatedCost: number } {
  const selectedProvider = provider ? PROVIDERS[provider] : getAvailableProviders()[0];
  
  // Rough estimate: 1 token ≈ 4 characters
  const totalChars = messages.reduce((sum, m) => sum + m.content.length, 0);
  const estimatedTokens = Math.ceil(totalChars / 4);
  const estimatedCost = estimatedTokens * selectedProvider.costPerToken;

  return {
    provider: selectedProvider.name,
    estimatedCost
  };
}

// ============================================================================
// EXPORTS
// ============================================================================

export { PROVIDERS, getAvailableProviders };
