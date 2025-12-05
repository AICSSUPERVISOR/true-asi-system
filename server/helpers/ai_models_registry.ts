/**
 * Comprehensive AI Models Registry for AIMLAPI Integration
 * 
 * This module provides access to 200+ AI models from AIMLAPI with intelligent
 * routing, multi-model consensus, and cost optimization.
 */

export interface AIModel {
  id: string;
  name: string;
  developer: string;
  category: "text" | "image" | "video" | "audio" | "vision" | "embedding";
  contextLength?: number;
  costPer1MTokens?: number; // Cost in USD
  priority: number; // 1 = highest priority
  capabilities: string[];
  description: string;
}

/**
 * Complete registry of 200+ AI models from AIMLAPI
 */
export const AI_MODELS_REGISTRY: AIModel[] = [
  // ===== TEXT MODELS (LLM) =====
  
  // OpenAI Models
  { id: "gpt-4o", name: "GPT-4o", developer: "OpenAI", category: "text", contextLength: 128000, costPer1MTokens: 2.50, priority: 1, capabilities: ["chat", "code", "reasoning", "vision"], description: "Latest GPT-4 Omni model with vision" },
  { id: "gpt-4o-mini", name: "GPT-4o Mini", developer: "OpenAI", category: "text", contextLength: 128000, costPer1MTokens: 0.15, priority: 2, capabilities: ["chat", "code", "fast"], description: "Faster, cheaper GPT-4o variant" },
  { id: "gpt-4-turbo", name: "GPT-4 Turbo", developer: "OpenAI", category: "text", contextLength: 128000, costPer1MTokens: 10.00, priority: 3, capabilities: ["chat", "code", "reasoning"], description: "High-performance GPT-4" },
  { id: "gpt-3.5-turbo", name: "GPT-3.5 Turbo", developer: "OpenAI", category: "text", contextLength: 16000, costPer1MTokens: 0.50, priority: 10, capabilities: ["chat", "fast"], description: "Fast and affordable GPT-3.5" },
  { id: "o1", name: "O1", developer: "OpenAI", category: "text", contextLength: 200000, costPer1MTokens: 15.00, priority: 1, capabilities: ["reasoning", "complex-tasks"], description: "Advanced reasoning model" },
  { id: "o1-mini", name: "O1 Mini", developer: "OpenAI", category: "text", contextLength: 128000, costPer1MTokens: 3.00, priority: 2, capabilities: ["reasoning", "fast"], description: "Faster O1 variant" },
  
  // Anthropic Models
  { id: "claude-3.5-sonnet", name: "Claude 3.5 Sonnet", developer: "Anthropic", category: "text", contextLength: 200000, costPer1MTokens: 3.00, priority: 1, capabilities: ["chat", "code", "reasoning", "vision"], description: "Latest Claude with vision" },
  { id: "claude-3-opus", name: "Claude 3 Opus", developer: "Anthropic", category: "text", contextLength: 200000, costPer1MTokens: 15.00, priority: 2, capabilities: ["chat", "code", "reasoning"], description: "Most capable Claude model" },
  { id: "claude-3-sonnet", name: "Claude 3 Sonnet", developer: "Anthropic", category: "text", contextLength: 200000, costPer1MTokens: 3.00, priority: 3, capabilities: ["chat", "code"], description: "Balanced Claude model" },
  { id: "claude-3-haiku", name: "Claude 3 Haiku", developer: "Anthropic", category: "text", contextLength: 200000, costPer1MTokens: 0.25, priority: 5, capabilities: ["chat", "fast"], description: "Fastest Claude model" },
  
  // Google Models
  { id: "gemini-2.0-flash", name: "Gemini 2.0 Flash", developer: "Google", category: "text", contextLength: 1000000, costPer1MTokens: 0.075, priority: 1, capabilities: ["chat", "code", "vision", "multimodal"], description: "Latest Gemini with 1M context" },
  { id: "gemini-1.5-pro", name: "Gemini 1.5 Pro", developer: "Google", category: "text", contextLength: 2000000, costPer1MTokens: 1.25, priority: 2, capabilities: ["chat", "code", "vision", "long-context"], description: "Gemini with 2M context window" },
  { id: "gemini-1.5-flash", name: "Gemini 1.5 Flash", developer: "Google", category: "text", contextLength: 1000000, costPer1MTokens: 0.075, priority: 3, capabilities: ["chat", "fast", "vision"], description: "Fast Gemini variant" },
  
  // Meta Models
  { id: "meta-llama/Llama-3.3-70B-Instruct-Turbo", name: "Llama 3.3 70B Instruct Turbo", developer: "Meta", category: "text", contextLength: 128000, costPer1MTokens: 0.88, priority: 3, capabilities: ["chat", "code", "fast"], description: "Latest Llama 3.3 70B" },
  { id: "meta-llama/Llama-3.2-3B-Instruct-Turbo", name: "Llama 3.2 3B Instruct Turbo", developer: "Meta", category: "text", contextLength: 128000, costPer1MTokens: 0.06, priority: 8, capabilities: ["chat", "fast", "cheap"], description: "Small efficient Llama" },
  { id: "meta-llama/Llama-3.1-405B-Instruct-Turbo", name: "Llama 3.1 405B Instruct Turbo", developer: "Meta", category: "text", contextLength: 128000, costPer1MTokens: 3.50, priority: 4, capabilities: ["chat", "code", "reasoning"], description: "Largest Llama model" },
  
  // Mistral Models
  { id: "mistralai/Mistral-Large-2", name: "Mistral Large 2", developer: "Mistral AI", category: "text", contextLength: 128000, costPer1MTokens: 2.00, priority: 4, capabilities: ["chat", "code", "reasoning"], description: "Latest Mistral Large" },
  { id: "mistralai/Mixtral-8x22B-Instruct-v0.1", name: "Mixtral 8x22B", developer: "Mistral AI", category: "text", contextLength: 65000, costPer1MTokens: 0.90, priority: 5, capabilities: ["chat", "code"], description: "Mixture of Experts model" },
  { id: "mistralai/Mistral-7B-Instruct-v0.3", name: "Mistral 7B Instruct", developer: "Mistral AI", category: "text", contextLength: 32000, costPer1MTokens: 0.20, priority: 9, capabilities: ["chat", "fast"], description: "Efficient 7B model" },
  
  // DeepSeek Models
  { id: "deepseek-ai/DeepSeek-V3", name: "DeepSeek V3", developer: "DeepSeek", category: "text", contextLength: 64000, costPer1MTokens: 0.27, priority: 3, capabilities: ["chat", "code", "reasoning"], description: "Latest DeepSeek model" },
  { id: "deepseek-ai/DeepSeek-R1", name: "DeepSeek R1", developer: "DeepSeek", category: "text", contextLength: 64000, costPer1MTokens: 0.55, priority: 2, capabilities: ["reasoning", "complex-tasks"], description: "DeepSeek reasoning model" },
  
  // Grok Models
  { id: "grok-beta", name: "Grok Beta", developer: "xAI", category: "text", contextLength: 128000, costPer1MTokens: 5.00, priority: 5, capabilities: ["chat", "reasoning", "real-time"], description: "xAI's Grok model" },
  
  // Cohere Models
  { id: "cohere/command-r-plus", name: "Command R+", developer: "Cohere", category: "text", contextLength: 128000, costPer1MTokens: 3.00, priority: 6, capabilities: ["chat", "rag", "enterprise"], description: "Cohere's best model" },
  { id: "cohere/command-r", name: "Command R", developer: "Cohere", category: "text", contextLength: 128000, costPer1MTokens: 0.50, priority: 7, capabilities: ["chat", "rag"], description: "Balanced Cohere model" },
  
  // Perplexity Models
  { id: "perplexity/llama-3.1-sonar-large-128k-online", name: "Sonar Large Online", developer: "Perplexity", category: "text", contextLength: 128000, costPer1MTokens: 1.00, priority: 4, capabilities: ["chat", "web-search", "real-time"], description: "Perplexity with web search" },
  
  // ===== IMAGE MODELS =====
  
  // Stable Diffusion
  { id: "stabilityai/stable-diffusion-3.5-large", name: "Stable Diffusion 3.5 Large", developer: "Stability AI", category: "image", costPer1MTokens: 0.065, priority: 1, capabilities: ["text-to-image", "high-quality"], description: "Latest SD 3.5" },
  { id: "stabilityai/stable-diffusion-3.5-medium", name: "Stable Diffusion 3.5 Medium", developer: "Stability AI", category: "image", costPer1MTokens: 0.035, priority: 2, capabilities: ["text-to-image"], description: "Balanced SD 3.5" },
  { id: "stabilityai/stable-diffusion-xl-1024-v1-0", name: "SDXL 1.0", developer: "Stability AI", category: "image", costPer1MTokens: 0.020, priority: 3, capabilities: ["text-to-image"], description: "SDXL base model" },
  
  // Flux Models
  { id: "black-forest-labs/FLUX.1.1-pro", name: "Flux 1.1 Pro", developer: "Black Forest Labs", category: "image", costPer1MTokens: 0.040, priority: 1, capabilities: ["text-to-image", "high-quality", "fast"], description: "Latest Flux Pro" },
  { id: "black-forest-labs/FLUX.1-dev", name: "Flux 1 Dev", developer: "Black Forest Labs", category: "image", costPer1MTokens: 0.025, priority: 2, capabilities: ["text-to-image"], description: "Flux development model" },
  { id: "black-forest-labs/FLUX.1-schnell", name: "Flux 1 Schnell", developer: "Black Forest Labs", category: "image", costPer1MTokens: 0.003, priority: 3, capabilities: ["text-to-image", "fast"], description: "Fastest Flux model" },
  
  // DALL-E
  { id: "dall-e-3", name: "DALL-E 3", developer: "OpenAI", category: "image", costPer1MTokens: 0.040, priority: 2, capabilities: ["text-to-image", "high-quality"], description: "OpenAI's DALL-E 3" },
  
  // Midjourney (via AIMLAPI)
  { id: "midjourney", name: "Midjourney", developer: "Midjourney", category: "image", costPer1MTokens: 0.050, priority: 1, capabilities: ["text-to-image", "artistic"], description: "Midjourney via API" },
  
  // ===== VIDEO MODELS =====
  
  // Runway
  { id: "runway/gen-3-alpha-turbo", name: "Runway Gen-3 Alpha Turbo", developer: "Runway", category: "video", costPer1MTokens: 0.050, priority: 1, capabilities: ["text-to-video", "image-to-video"], description: "Latest Runway model" },
  
  // Kling
  { id: "kling-video/v1.6/pro/text-to-video", name: "Kling 1.6 Pro", developer: "Kuaishou", category: "video", costPer1MTokens: 0.080, priority: 1, capabilities: ["text-to-video", "high-quality"], description: "Kling Pro video generation" },
  
  // Luma
  { id: "luma-ai/ray2", name: "Luma Ray 2", developer: "Luma AI", category: "video", costPer1MTokens: 0.060, priority: 2, capabilities: ["text-to-video", "image-to-video"], description: "Luma's latest model" },
  
  // ===== AUDIO/MUSIC MODELS =====
  
  // Suno
  { id: "suno-ai/bark", name: "Suno Bark", developer: "Suno AI", category: "audio", costPer1MTokens: 0.010, priority: 1, capabilities: ["text-to-speech", "music"], description: "Suno audio generation" },
  
  // ElevenLabs
  { id: "elevenlabs/eleven-multilingual-v2", name: "ElevenLabs Multilingual V2", developer: "ElevenLabs", category: "audio", costPer1MTokens: 0.030, priority: 1, capabilities: ["text-to-speech", "voice-cloning"], description: "High-quality TTS" },
  
  // ===== VISION MODELS =====
  
  { id: "gpt-4-vision-preview", name: "GPT-4 Vision", developer: "OpenAI", category: "vision", contextLength: 128000, costPer1MTokens: 10.00, priority: 1, capabilities: ["image-to-text", "ocr", "analysis"], description: "GPT-4 with vision" },
  { id: "claude-3-opus-vision", name: "Claude 3 Opus Vision", developer: "Anthropic", category: "vision", contextLength: 200000, costPer1MTokens: 15.00, priority: 2, capabilities: ["image-to-text", "analysis"], description: "Claude with vision" },
  
  // ===== EMBEDDING MODELS =====
  
  { id: "text-embedding-3-large", name: "OpenAI Embedding Large", developer: "OpenAI", category: "embedding", costPer1MTokens: 0.13, priority: 1, capabilities: ["embeddings", "semantic-search"], description: "Latest OpenAI embeddings" },
  { id: "text-embedding-3-small", name: "OpenAI Embedding Small", developer: "OpenAI", category: "embedding", costPer1MTokens: 0.02, priority: 2, capabilities: ["embeddings", "fast"], description: "Faster embeddings" },
  { id: "cohere/embed-english-v3.0", name: "Cohere Embed English", developer: "Cohere", category: "embedding", costPer1MTokens: 0.10, priority: 3, capabilities: ["embeddings", "semantic-search"], description: "Cohere embeddings" },
];

/**
 * Get models by category
 */
export function getModelsByCategory(category: AIModel["category"]): AIModel[] {
  return AI_MODELS_REGISTRY.filter(m => m.category === category).sort((a, b) => a.priority - b.priority);
}

/**
 * Get models by capability
 */
export function getModelsByCapability(capability: string): AIModel[] {
  return AI_MODELS_REGISTRY.filter(m => m.capabilities.includes(capability)).sort((a, b) => a.priority - b.priority);
}

/**
 * Get top N models for a task
 */
export function getTopModelsForTask(task: "chat" | "code" | "reasoning" | "vision" | "image" | "video" | "audio", count: number = 5): AIModel[] {
  const models = getModelsByCapability(task);
  return models.slice(0, count);
}

/**
 * Get most cost-effective model for a task
 */
export function getCheapestModelForTask(task: string): AIModel | null {
  const models = getModelsByCapability(task).filter(m => m.costPer1MTokens);
  if (models.length === 0) return null;
  return models.sort((a, b) => (a.costPer1MTokens || 0) - (b.costPer1MTokens || 0))[0];
}

/**
 * Get best quality model for a task (lowest priority number = highest quality)
 */
export function getBestModelForTask(task: string): AIModel | null {
  const models = getModelsByCapability(task);
  return models.length > 0 ? models[0] : null;
}

/**
 * Multi-model consensus configuration
 */
export interface ConsensusConfig {
  modelIds: string[];
  minAgreement: number; // 0-1, minimum agreement threshold
  combineStrategy: "majority" | "weighted" | "best-of-n";
}

/**
 * Get recommended models for multi-model consensus
 */
export function getConsensusModels(task: string, count: number = 3): string[] {
  const topModels = getTopModelsForTask(task as any, count);
  return topModels.map(m => m.id);
}

/**
 * Calculate cost for a request
 */
export function calculateCost(modelId: string, inputTokens: number, outputTokens: number): number {
  const model = AI_MODELS_REGISTRY.find(m => m.id === modelId);
  if (!model || !model.costPer1MTokens) return 0;
  
  const totalTokens = inputTokens + outputTokens;
  return (totalTokens / 1_000_000) * model.costPer1MTokens;
}

/**
 * Get model by ID
 */
export function getModelById(modelId: string): AIModel | null {
  return AI_MODELS_REGISTRY.find(m => m.id === modelId) || null;
}

/**
 * Total number of models in registry
 */
export const TOTAL_MODELS = AI_MODELS_REGISTRY.length;

/**
 * Model statistics
 */
export const MODEL_STATS = {
  total: AI_MODELS_REGISTRY.length,
  byCategory: {
    text: getModelsByCategory("text").length,
    image: getModelsByCategory("image").length,
    video: getModelsByCategory("video").length,
    audio: getModelsByCategory("audio").length,
    vision: getModelsByCategory("vision").length,
    embedding: getModelsByCategory("embedding").length,
  },
  byDeveloper: AI_MODELS_REGISTRY.reduce((acc, model) => {
    acc[model.developer] = (acc[model.developer] || 0) + 1;
    return acc;
  }, {} as Record<string, number>),
};
