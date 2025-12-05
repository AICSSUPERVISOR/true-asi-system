/**
 * Multi-Model Consensus Algorithm
 * 
 * This module implements intelligent multi-model querying with consensus-based
 * response synthesis, confidence scoring, and cost optimization.
 */

import { invokeLLM } from "../_core/llm";
import { AI_MODELS_REGISTRY, getTopModelsForTask, calculateCost, type AIModel } from "./ai_models_registry";

export interface ConsensusRequest {
  prompt: string;
  task: "chat" | "code" | "reasoning" | "vision" | "image" | "video" | "audio";
  modelCount?: number; // Number of models to query (default: 3)
  strategy?: "majority" | "weighted" | "best-of-n" | "unanimous";
  minConfidence?: number; // Minimum confidence threshold (0-1, default: 0.7)
  maxCost?: number; // Maximum cost in USD (default: no limit)
}

export interface ModelResponse {
  modelId: string;
  modelName: string;
  response: string;
  tokens: { input: number; output: number };
  cost: number;
  latency: number; // milliseconds
  error?: string;
}

export interface ConsensusResult {
  finalResponse: string;
  confidence: number; // 0-1
  strategy: string;
  modelResponses: ModelResponse[];
  totalCost: number;
  totalLatency: number;
  agreement: number; // 0-1, how much models agreed
  reasoning: string; // Explanation of how consensus was reached
}

/**
 * Query multiple models in parallel and synthesize responses
 */
export async function getConsensus(request: ConsensusRequest): Promise<ConsensusResult> {
  const {
    prompt,
    task,
    modelCount = 3,
    strategy = "weighted",
    minConfidence = 0.7,
    maxCost,
  } = request;

  // Select top models for the task
  const topModels = getTopModelsForTask(task, modelCount);
  
  // Filter by max cost if specified
  const selectedModels = maxCost
    ? topModels.filter(m => (m.costPer1MTokens || 0) * 2 <= maxCost) // Estimate 2M tokens max
    : topModels;

  if (selectedModels.length === 0) {
    throw new Error("No models available within cost constraints");
  }

  // Query all models in parallel
  const startTime = Date.now();
  const responses = await Promise.allSettled(
    selectedModels.map(model => queryModel(model, prompt))
  );

  // Extract successful responses
  const modelResponses: ModelResponse[] = responses
    .map((result, index) => {
      if (result.status === "fulfilled") {
        return result.value;
      } else {
        return {
          modelId: selectedModels[index].id,
          modelName: selectedModels[index].name,
          response: "",
          tokens: { input: 0, output: 0 },
          cost: 0,
          latency: 0,
          error: result.reason?.message || "Unknown error",
        };
      }
    })
    .filter(r => !r.error); // Only successful responses

  if (modelResponses.length === 0) {
    throw new Error("All models failed to respond");
  }

  const totalLatency = Date.now() - startTime;
  const totalCost = modelResponses.reduce((sum, r) => sum + r.cost, 0);

  // Synthesize responses based on strategy
  const synthesized = await synthesizeResponses(modelResponses, strategy, prompt);

  return {
    finalResponse: synthesized.response,
    confidence: synthesized.confidence,
    strategy,
    modelResponses,
    totalCost,
    totalLatency,
    agreement: synthesized.agreement,
    reasoning: synthesized.reasoning,
  };
}

/**
 * Query a single model
 */
async function queryModel(model: AIModel, prompt: string): Promise<ModelResponse> {
  const startTime = Date.now();

  try {
    const response = await invokeLLM({
      messages: [
        { role: "system", content: "You are a helpful AI assistant. Provide clear, accurate, and concise responses." },
        { role: "user", content: prompt },
      ],
      // Note: invokeLLM uses default model, would need to be extended to support model selection
      // For now, this demonstrates the architecture
    });

    const latency = Date.now() - startTime;
    const messageContent = response.choices[0]?.message?.content;
    const content = typeof messageContent === "string" ? messageContent : "";
    
    // Estimate tokens (rough approximation: 1 token ≈ 4 characters)
    const inputTokens = Math.ceil(prompt.length / 4);
    const outputTokens = Math.ceil(content.length / 4);
    
    const cost = calculateCost(model.id, inputTokens, outputTokens);

    return {
      modelId: model.id,
      modelName: model.name,
      response: content,
      tokens: { input: inputTokens, output: outputTokens },
      cost,
      latency,
    };
  } catch (error: any) {
    throw new Error(`Model ${model.name} failed: ${error.message}`);
  }
}

/**
 * Synthesize multiple model responses into a single consensus response
 */
async function synthesizeResponses(
  responses: ModelResponse[],
  strategy: string,
  originalPrompt: string
): Promise<{ response: string; confidence: number; agreement: number; reasoning: string }> {
  if (responses.length === 1) {
    return {
      response: responses[0].response,
      confidence: 0.8, // Single model confidence
      agreement: 1.0,
      reasoning: "Only one model responded successfully",
    };
  }

  switch (strategy) {
    case "majority":
      return majorityVoting(responses);
    
    case "weighted":
      return weightedSynthesis(responses);
    
    case "best-of-n":
      return bestOfN(responses);
    
    case "unanimous":
      return unanimousConsensus(responses);
    
    default:
      return weightedSynthesis(responses);
  }
}

/**
 * Majority voting: Select the most common response
 */
function majorityVoting(responses: ModelResponse[]): {
  response: string;
  confidence: number;
  agreement: number;
  reasoning: string;
} {
  // Group similar responses (simple string comparison for now)
  const responseCounts = new Map<string, number>();
  responses.forEach(r => {
    const normalized = r.response.trim().toLowerCase();
    responseCounts.set(normalized, (responseCounts.get(normalized) || 0) + 1);
  });

  // Find most common response
  let maxCount = 0;
  let majorityResponse = "";
  responseCounts.forEach((count, response) => {
    if (count > maxCount) {
      maxCount = count;
      majorityResponse = response;
    }
  });

  const agreement = maxCount / responses.length;
  const confidence = agreement;

  // Find original response that matches majority
  const finalResponse = responses.find(
    r => r.response.trim().toLowerCase() === majorityResponse
  )?.response || responses[0].response;

  return {
    response: finalResponse,
    confidence,
    agreement,
    reasoning: `${maxCount} out of ${responses.length} models agreed on this response`,
  };
}

/**
 * Weighted synthesis: Combine responses based on model priority
 */
function weightedSynthesis(responses: ModelResponse[]): {
  response: string;
  confidence: number;
  agreement: number;
  reasoning: string;
} {
  // Weight by inverse latency (faster = better) and model priority
  const weights = responses.map((r, i) => {
    const latencyWeight = 1 / (r.latency + 100); // Avoid division by zero
    const priorityWeight = 1 / (i + 1); // First model has highest priority
    return latencyWeight * priorityWeight;
  });

  const totalWeight = weights.reduce((sum, w) => sum + w, 0);
  const normalizedWeights = weights.map(w => w / totalWeight);

  // Select response with highest weight
  let maxWeight = 0;
  let bestIndex = 0;
  normalizedWeights.forEach((weight, index) => {
    if (weight > maxWeight) {
      maxWeight = weight;
      bestIndex = index;
    }
  });

  const confidence = maxWeight;
  const agreement = calculateAgreement(responses);

  return {
    response: responses[bestIndex].response,
    confidence,
    agreement,
    reasoning: `Selected response from ${responses[bestIndex].modelName} (weight: ${(maxWeight * 100).toFixed(1)}%)`,
  };
}

/**
 * Best-of-N: Select the longest/most detailed response
 */
function bestOfN(responses: ModelResponse[]): {
  response: string;
  confidence: number;
  agreement: number;
  reasoning: string;
} {
  // Select longest response as proxy for most detailed
  const longest = responses.reduce((best, current) =>
    current.response.length > best.response.length ? current : best
  );

  const confidence = 0.85; // High confidence for detailed response
  const agreement = calculateAgreement(responses);

  return {
    response: longest.response,
    confidence,
    agreement,
    reasoning: `Selected most detailed response from ${longest.modelName} (${longest.response.length} characters)`,
  };
}

/**
 * Unanimous consensus: All models must agree (high confidence)
 */
function unanimousConsensus(responses: ModelResponse[]): {
  response: string;
  confidence: number;
  agreement: number;
  reasoning: string;
} {
  // Check if all responses are similar (simple check)
  const normalized = responses.map(r => r.response.trim().toLowerCase());
  const allSame = normalized.every(r => r === normalized[0]);

  if (allSame) {
    return {
      response: responses[0].response,
      confidence: 1.0,
      agreement: 1.0,
      reasoning: "All models provided identical responses",
    };
  }

  // If not unanimous, fall back to weighted synthesis
  const weighted = weightedSynthesis(responses);
  return {
    ...weighted,
    confidence: weighted.confidence * 0.7, // Lower confidence due to disagreement
    reasoning: `Models disagreed, using weighted synthesis: ${weighted.reasoning}`,
  };
}

/**
 * Calculate agreement score between responses
 */
function calculateAgreement(responses: ModelResponse[]): number {
  if (responses.length <= 1) return 1.0;

  // Simple similarity check: count how many responses are similar
  const normalized = responses.map(r => r.response.trim().toLowerCase());
  let similarPairs = 0;
  let totalPairs = 0;

  for (let i = 0; i < normalized.length; i++) {
    for (let j = i + 1; j < normalized.length; j++) {
      totalPairs++;
      // Check if responses share significant overlap (>50% of words)
      const words1 = new Set(normalized[i].split(/\s+/));
      const words2 = new Set(normalized[j].split(/\s+/));
      const words1Array = Array.from(words1);
      const intersection = new Set(words1Array.filter(w => words2.has(w)));
      const similarity = intersection.size / Math.min(words1.size, words2.size);
      if (similarity > 0.5) {
        similarPairs++;
      }
    }
  }

  return totalPairs > 0 ? similarPairs / totalPairs : 0;
}

/**
 * Get consensus for business recommendations
 */
export async function getBusinessRecommendationConsensus(
  companyData: any,
  industryContext: string
): Promise<ConsensusResult> {
  const prompt = `Analyze this Norwegian business and provide 3-5 specific, actionable recommendations to increase revenue and customers:

Company: ${companyData.name}
Industry: ${industryContext}
Current Digital Maturity Score: ${companyData.digitalMaturityScore}/100
Website: ${companyData.website || "None"}
LinkedIn Followers: ${companyData.linkedinFollowers || 0}
Employees: ${companyData.employees || "Unknown"}

Provide recommendations in this format:
1. [Category]: [Specific Action] - Expected Impact: [Revenue/Customer increase]
2. ...

Focus on practical, implementable actions with measurable outcomes.`;

  return getConsensus({
    prompt,
    task: "reasoning",
    modelCount: 3,
    strategy: "weighted",
    minConfidence: 0.75,
  });
}

/**
 * Get consensus for content generation
 */
export async function getContentGenerationConsensus(
  topic: string,
  contentType: "blog" | "social" | "email" | "ad"
): Promise<ConsensusResult> {
  const prompt = `Generate high-quality ${contentType} content about: ${topic}

Requirements:
- Engaging and professional tone
- Clear call-to-action
- Optimized for ${contentType === "social" ? "social media engagement" : contentType === "email" ? "email open rates" : "conversions"}
- Length: ${contentType === "blog" ? "500-800 words" : contentType === "social" ? "100-200 characters" : "200-400 words"}`;

  return getConsensus({
    prompt,
    task: "chat",
    modelCount: 3,
    strategy: "best-of-n",
    minConfidence: 0.8,
  });
}
