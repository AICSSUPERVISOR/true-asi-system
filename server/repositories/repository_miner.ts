/**
 * TRUE ASI - COMPLETE REPOSITORY MINING SYSTEM
 * 
 * Mines and integrates ALL major AI/ML repositories:
 * 1. Foundation model repos (Llama, Mistral, Qwen, etc.)
 * 2. Framework repos (PyTorch, TensorFlow, JAX)
 * 3. Agent repos (LangChain, AutoGPT, CrewAI)
 * 4. Benchmark repos (ARC, MMLU, HumanEval)
 * 5. Research repos (Papers with Code, arXiv)
 * 
 * NO MOCK DATA - 100% FUNCTIONAL
 */

import { invokeLLM } from '../_core/llm';

// ============================================================================
// COMPLETE REPOSITORY REGISTRY - ALL MAJOR AI/ML REPOS
// ============================================================================

export const REPOSITORY_REGISTRY = {
  // Foundation Models
  foundation_models: [
    { name: 'meta-llama/llama', url: 'https://github.com/meta-llama/llama', category: 'llm', priority: 1 },
    { name: 'meta-llama/llama3', url: 'https://github.com/meta-llama/llama3', category: 'llm', priority: 1 },
    { name: 'mistralai/mistral-src', url: 'https://github.com/mistralai/mistral-src', category: 'llm', priority: 1 },
    { name: 'QwenLM/Qwen', url: 'https://github.com/QwenLM/Qwen', category: 'llm', priority: 1 },
    { name: 'QwenLM/Qwen2', url: 'https://github.com/QwenLM/Qwen2', category: 'llm', priority: 1 },
    { name: 'google/gemma', url: 'https://github.com/google/gemma_pytorch', category: 'llm', priority: 1 },
    { name: 'deepseek-ai/DeepSeek-V2', url: 'https://github.com/deepseek-ai/DeepSeek-V2', category: 'llm', priority: 1 },
    { name: 'deepseek-ai/DeepSeek-Coder', url: 'https://github.com/deepseek-ai/DeepSeek-Coder', category: 'code', priority: 1 },
    { name: 'THUDM/ChatGLM-6B', url: 'https://github.com/THUDM/ChatGLM-6B', category: 'llm', priority: 2 },
    { name: 'THUDM/GLM-4', url: 'https://github.com/THUDM/GLM-4', category: 'llm', priority: 1 },
    { name: 'bigcode/starcoder', url: 'https://github.com/bigcode-project/starcoder', category: 'code', priority: 1 },
    { name: 'codellama/CodeLlama', url: 'https://github.com/facebookresearch/codellama', category: 'code', priority: 1 },
    { name: 'WizardLM/WizardCoder', url: 'https://github.com/nlpxucan/WizardLM', category: 'code', priority: 2 },
    { name: 'microsoft/phi-2', url: 'https://github.com/microsoft/phi-2', category: 'llm', priority: 2 },
    { name: 'stabilityai/stablelm', url: 'https://github.com/Stability-AI/StableLM', category: 'llm', priority: 2 },
    { name: 'EleutherAI/gpt-neox', url: 'https://github.com/EleutherAI/gpt-neox', category: 'llm', priority: 2 },
    { name: 'mosaicml/llm-foundry', url: 'https://github.com/mosaicml/llm-foundry', category: 'llm', priority: 2 },
    { name: 'databricks/dolly', url: 'https://github.com/databrickslabs/dolly', category: 'llm', priority: 3 },
    { name: 'togethercomputer/RedPajama', url: 'https://github.com/togethercomputer/RedPajama-Data', category: 'data', priority: 2 },
    { name: 'allenai/OLMo', url: 'https://github.com/allenai/OLMo', category: 'llm', priority: 2 },
  ],

  // Multimodal Models
  multimodal: [
    { name: 'openai/CLIP', url: 'https://github.com/openai/CLIP', category: 'vision', priority: 1 },
    { name: 'salesforce/BLIP', url: 'https://github.com/salesforce/BLIP', category: 'vision', priority: 1 },
    { name: 'haotian-liu/LLaVA', url: 'https://github.com/haotian-liu/LLaVA', category: 'vision', priority: 1 },
    { name: 'THUDM/CogVLM', url: 'https://github.com/THUDM/CogVLM', category: 'vision', priority: 1 },
    { name: 'microsoft/Florence-2', url: 'https://github.com/microsoft/Florence-2', category: 'vision', priority: 1 },
    { name: 'OpenGVLab/InternVL', url: 'https://github.com/OpenGVLab/InternVL', category: 'vision', priority: 1 },
    { name: 'QwenLM/Qwen-VL', url: 'https://github.com/QwenLM/Qwen-VL', category: 'vision', priority: 1 },
    { name: 'google/paligemma', url: 'https://github.com/google-research/big_vision', category: 'vision', priority: 2 },
    { name: 'openai/whisper', url: 'https://github.com/openai/whisper', category: 'audio', priority: 1 },
    { name: 'suno-ai/bark', url: 'https://github.com/suno-ai/bark', category: 'audio', priority: 2 },
    { name: 'coqui-ai/TTS', url: 'https://github.com/coqui-ai/TTS', category: 'audio', priority: 2 },
    { name: 'CompVis/stable-diffusion', url: 'https://github.com/CompVis/stable-diffusion', category: 'image_gen', priority: 1 },
    { name: 'Stability-AI/stablediffusion', url: 'https://github.com/Stability-AI/stablediffusion', category: 'image_gen', priority: 1 },
    { name: 'black-forest-labs/flux', url: 'https://github.com/black-forest-labs/flux', category: 'image_gen', priority: 1 },
  ],

  // Agent Frameworks
  agents: [
    { name: 'langchain-ai/langchain', url: 'https://github.com/langchain-ai/langchain', category: 'agent', priority: 1 },
    { name: 'langchain-ai/langgraph', url: 'https://github.com/langchain-ai/langgraph', category: 'agent', priority: 1 },
    { name: 'Significant-Gravitas/AutoGPT', url: 'https://github.com/Significant-Gravitas/AutoGPT', category: 'agent', priority: 1 },
    { name: 'joaomdmoura/crewAI', url: 'https://github.com/joaomdmoura/crewAI', category: 'agent', priority: 1 },
    { name: 'microsoft/autogen', url: 'https://github.com/microsoft/autogen', category: 'agent', priority: 1 },
    { name: 'OpenBMB/AgentVerse', url: 'https://github.com/OpenBMB/AgentVerse', category: 'agent', priority: 2 },
    { name: 'geekan/MetaGPT', url: 'https://github.com/geekan/MetaGPT', category: 'agent', priority: 1 },
    { name: 'OpenBMB/ChatDev', url: 'https://github.com/OpenBMB/ChatDev', category: 'agent', priority: 2 },
    { name: 'yoheinakajima/babyagi', url: 'https://github.com/yoheinakajima/babyagi', category: 'agent', priority: 2 },
    { name: 'SamurAIGPT/Camel-AutoGPT', url: 'https://github.com/camel-ai/camel', category: 'agent', priority: 2 },
    { name: 'run-llama/llama_index', url: 'https://github.com/run-llama/llama_index', category: 'rag', priority: 1 },
    { name: 'chroma-core/chroma', url: 'https://github.com/chroma-core/chroma', category: 'vector_db', priority: 1 },
    { name: 'qdrant/qdrant', url: 'https://github.com/qdrant/qdrant', category: 'vector_db', priority: 1 },
    { name: 'weaviate/weaviate', url: 'https://github.com/weaviate/weaviate', category: 'vector_db', priority: 2 },
    { name: 'milvus-io/milvus', url: 'https://github.com/milvus-io/milvus', category: 'vector_db', priority: 2 },
  ],

  // ML Frameworks
  frameworks: [
    { name: 'pytorch/pytorch', url: 'https://github.com/pytorch/pytorch', category: 'framework', priority: 1 },
    { name: 'tensorflow/tensorflow', url: 'https://github.com/tensorflow/tensorflow', category: 'framework', priority: 1 },
    { name: 'google/jax', url: 'https://github.com/google/jax', category: 'framework', priority: 1 },
    { name: 'huggingface/transformers', url: 'https://github.com/huggingface/transformers', category: 'framework', priority: 1 },
    { name: 'huggingface/diffusers', url: 'https://github.com/huggingface/diffusers', category: 'framework', priority: 1 },
    { name: 'huggingface/peft', url: 'https://github.com/huggingface/peft', category: 'training', priority: 1 },
    { name: 'huggingface/trl', url: 'https://github.com/huggingface/trl', category: 'training', priority: 1 },
    { name: 'vllm-project/vllm', url: 'https://github.com/vllm-project/vllm', category: 'inference', priority: 1 },
    { name: 'ggerganov/llama.cpp', url: 'https://github.com/ggerganov/llama.cpp', category: 'inference', priority: 1 },
    { name: 'mlc-ai/mlc-llm', url: 'https://github.com/mlc-ai/mlc-llm', category: 'inference', priority: 1 },
    { name: 'NVIDIA/TensorRT-LLM', url: 'https://github.com/NVIDIA/TensorRT-LLM', category: 'inference', priority: 1 },
    { name: 'Lightning-AI/pytorch-lightning', url: 'https://github.com/Lightning-AI/pytorch-lightning', category: 'training', priority: 2 },
    { name: 'microsoft/DeepSpeed', url: 'https://github.com/microsoft/DeepSpeed', category: 'training', priority: 1 },
    { name: 'facebookresearch/fairscale', url: 'https://github.com/facebookresearch/fairscale', category: 'training', priority: 2 },
    { name: 'ray-project/ray', url: 'https://github.com/ray-project/ray', category: 'distributed', priority: 1 },
  ],

  // Benchmarks & Evaluation
  benchmarks: [
    { name: 'fchollet/ARC-AGI', url: 'https://github.com/fchollet/ARC-AGI', category: 'benchmark', priority: 1 },
    { name: 'openai/evals', url: 'https://github.com/openai/evals', category: 'benchmark', priority: 1 },
    { name: 'EleutherAI/lm-evaluation-harness', url: 'https://github.com/EleutherAI/lm-evaluation-harness', category: 'benchmark', priority: 1 },
    { name: 'openai/human-eval', url: 'https://github.com/openai/human-eval', category: 'benchmark', priority: 1 },
    { name: 'google/BIG-bench', url: 'https://github.com/google/BIG-bench', category: 'benchmark', priority: 1 },
    { name: 'hendrycks/test', url: 'https://github.com/hendrycks/test', category: 'benchmark', priority: 1 },
    { name: 'openai/grade-school-math', url: 'https://github.com/openai/grade-school-math', category: 'benchmark', priority: 2 },
    { name: 'allenai/ai2-arc', url: 'https://github.com/allenai/arc-solvers', category: 'benchmark', priority: 2 },
    { name: 'google-research/MATH', url: 'https://github.com/hendrycks/math', category: 'benchmark', priority: 2 },
    { name: 'SWE-bench/SWE-bench', url: 'https://github.com/princeton-nlp/SWE-bench', category: 'benchmark', priority: 1 },
  ],

  // Safety & Alignment
  safety: [
    { name: 'anthropics/hh-rlhf', url: 'https://github.com/anthropics/hh-rlhf', category: 'safety', priority: 1 },
    { name: 'openai/moderation-api', url: 'https://github.com/openai/moderation-api-release', category: 'safety', priority: 2 },
    { name: 'allenai/reward-bench', url: 'https://github.com/allenai/reward-bench', category: 'safety', priority: 2 },
    { name: 'lm-sys/FastChat', url: 'https://github.com/lm-sys/FastChat', category: 'safety', priority: 1 },
    { name: 'NVIDIA/NeMo-Guardrails', url: 'https://github.com/NVIDIA/NeMo-Guardrails', category: 'safety', priority: 1 },
    { name: 'llm-attacks/llm-attacks', url: 'https://github.com/llm-attacks/llm-attacks', category: 'safety', priority: 2 },
  ],

  // Research & Papers
  research: [
    { name: 'paperswithcode/paperswithcode', url: 'https://github.com/paperswithcode/paperswithcode-data', category: 'research', priority: 1 },
    { name: 'karpathy/nanoGPT', url: 'https://github.com/karpathy/nanoGPT', category: 'research', priority: 1 },
    { name: 'karpathy/minGPT', url: 'https://github.com/karpathy/minGPT', category: 'research', priority: 2 },
    { name: 'lucidrains/vit-pytorch', url: 'https://github.com/lucidrains/vit-pytorch', category: 'research', priority: 2 },
    { name: 'lucidrains/DALLE2-pytorch', url: 'https://github.com/lucidrains/DALLE2-pytorch', category: 'research', priority: 2 },
    { name: 'facebookresearch/llama-recipes', url: 'https://github.com/facebookresearch/llama-recipes', category: 'research', priority: 1 },
    { name: 'tatsu-lab/stanford_alpaca', url: 'https://github.com/tatsu-lab/stanford_alpaca', category: 'research', priority: 2 },
    { name: 'lm-sys/arena-hard', url: 'https://github.com/lm-sys/arena-hard-auto', category: 'research', priority: 2 },
  ],

  // Knowledge & RAG
  knowledge: [
    { name: 'getzep/graphiti', url: 'https://github.com/getzep/graphiti', category: 'knowledge', priority: 1 },
    { name: 'neo4j/neo4j', url: 'https://github.com/neo4j/neo4j', category: 'knowledge', priority: 1 },
    { name: 'microsoft/graphrag', url: 'https://github.com/microsoft/graphrag', category: 'knowledge', priority: 1 },
    { name: 'stanfordnlp/dspy', url: 'https://github.com/stanfordnlp/dspy', category: 'knowledge', priority: 1 },
    { name: 'Unstructured-IO/unstructured', url: 'https://github.com/Unstructured-IO/unstructured', category: 'knowledge', priority: 2 },
  ],
};

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

export interface RepositoryInfo {
  name: string;
  url: string;
  category: string;
  priority: number;
  description?: string;
  stars?: number;
  language?: string;
  topics?: string[];
  last_updated?: Date;
  readme_content?: string;
  key_files?: string[];
  dependencies?: string[];
  model_weights?: ModelWeight[];
  api_endpoints?: APIEndpoint[];
  integration_status: 'pending' | 'mined' | 'integrated' | 'failed';
}

export interface ModelWeight {
  name: string;
  huggingface_id: string;
  size_gb: number;
  parameters: string;
  quantizations: string[];
  download_url: string;
}

export interface APIEndpoint {
  name: string;
  url: string;
  method: string;
  auth_type: string;
  description: string;
}

export interface MiningResult {
  repository: RepositoryInfo;
  extracted_knowledge: string;
  model_weights: ModelWeight[];
  api_endpoints: APIEndpoint[];
  integration_code: string;
  deep_links: DeepLink[];
}

export interface DeepLink {
  type: 'api' | 'model' | 'dataset' | 'documentation' | 'code';
  name: string;
  url: string;
  description: string;
}

// ============================================================================
// REPOSITORY MINER CLASS
// ============================================================================

export class RepositoryMiner {
  private repositories: Map<string, RepositoryInfo> = new Map();
  private miningResults: Map<string, MiningResult> = new Map();
  private allDeepLinks: DeepLink[] = [];

  constructor() {
    this.initializeRegistry();
  }

  private initializeRegistry(): void {
    // Initialize all repositories from registry
    for (const category of Object.values(REPOSITORY_REGISTRY)) {
      for (const repo of category) {
        const info: RepositoryInfo = {
          ...repo,
          integration_status: 'pending'
        };
        this.repositories.set(repo.name, info);
      }
    }
  }

  // ============================================================================
  // MINING OPERATIONS
  // ============================================================================

  async mineRepository(repoName: string): Promise<MiningResult | null> {
    const repo = this.repositories.get(repoName);
    if (!repo) return null;

    try {
      // Extract owner and repo from URL
      const match = repo.url.match(/github\.com\/([^\/]+)\/([^\/]+)/);
      if (!match) {
        repo.integration_status = 'failed';
        return null;
      }

      const [, owner, repoId] = match;

      // Fetch repository metadata
      const apiUrl = `https://api.github.com/repos/${owner}/${repoId}`;
      const response = await fetch(apiUrl, {
        headers: {
          'Accept': 'application/vnd.github.v3+json',
          'User-Agent': 'TRUE-ASI-Repository-Miner/1.0'
        }
      });

      if (response.ok) {
        const data = await response.json();
        repo.description = data.description;
        repo.stars = data.stargazers_count;
        repo.language = data.language;
        repo.topics = data.topics;
        repo.last_updated = new Date(data.updated_at);
      }

      // Fetch README
      const readmeUrl = `https://api.github.com/repos/${owner}/${repoId}/readme`;
      const readmeResponse = await fetch(readmeUrl, {
        headers: {
          'Accept': 'application/vnd.github.v3+json',
          'User-Agent': 'TRUE-ASI-Repository-Miner/1.0'
        }
      });

      if (readmeResponse.ok) {
        const readmeData = await readmeResponse.json();
        repo.readme_content = Buffer.from(readmeData.content, 'base64').toString('utf-8');
      }

      // Extract knowledge using LLM
      const extractedKnowledge = await this.extractKnowledge(repo);

      // Extract model weights
      const modelWeights = await this.extractModelWeights(repo);

      // Extract API endpoints
      const apiEndpoints = await this.extractAPIEndpoints(repo);

      // Generate integration code
      const integrationCode = await this.generateIntegrationCode(repo, modelWeights, apiEndpoints);

      // Extract deep links
      const deepLinks = await this.extractDeepLinks(repo);

      const result: MiningResult = {
        repository: repo,
        extracted_knowledge: extractedKnowledge,
        model_weights: modelWeights,
        api_endpoints: apiEndpoints,
        integration_code: integrationCode,
        deep_links: deepLinks
      };

      repo.integration_status = 'mined';
      this.miningResults.set(repoName, result);
      this.allDeepLinks.push(...deepLinks);

      return result;

    } catch (error) {
      repo.integration_status = 'failed';
      console.error(`Failed to mine ${repoName}:`, error);
      return null;
    }
  }

  async mineAllRepositories(): Promise<{
    total: number;
    mined: number;
    failed: number;
    results: MiningResult[];
  }> {
    const results: MiningResult[] = [];
    let mined = 0;
    let failed = 0;

    // Sort by priority
    const sortedRepos = Array.from(this.repositories.values())
      .sort((a, b) => a.priority - b.priority);

    for (const repo of sortedRepos) {
      const result = await this.mineRepository(repo.name);
      if (result) {
        results.push(result);
        mined++;
      } else {
        failed++;
      }

      // Rate limiting
      await new Promise(resolve => setTimeout(resolve, 500));
    }

    return {
      total: this.repositories.size,
      mined,
      failed,
      results
    };
  }

  // ============================================================================
  // KNOWLEDGE EXTRACTION
  // ============================================================================

  private async extractKnowledge(repo: RepositoryInfo): Promise<string> {
    const systemPrompt = `You are an AI/ML repository analyst.
Extract key knowledge from this repository:
- Main purpose and capabilities
- Key algorithms and techniques
- Model architectures
- Training methods
- Inference optimizations
- Integration patterns
Provide a comprehensive technical summary.`;

    const userPrompt = `Repository: ${repo.name}
URL: ${repo.url}
Category: ${repo.category}
Description: ${repo.description || 'N/A'}
Language: ${repo.language || 'N/A'}
Topics: ${repo.topics?.join(', ') || 'N/A'}
Stars: ${repo.stars || 'N/A'}

README (excerpt):
${(repo.readme_content || '').slice(0, 5000)}

Extract key technical knowledge.`;

    const response = await invokeLLM({
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt }
      ]
    });

    const content = response.choices[0]?.message?.content;
    return typeof content === 'string' ? content : '';
  }

  // ============================================================================
  // MODEL WEIGHT EXTRACTION
  // ============================================================================

  private async extractModelWeights(repo: RepositoryInfo): Promise<ModelWeight[]> {
    const systemPrompt = `You are an AI model expert.
Extract model weight information from this repository.
Identify HuggingFace model IDs, sizes, and quantizations.
Output valid JSON with: models (array of {name, huggingface_id, size_gb, parameters, quantizations, download_url}).`;

    const userPrompt = `Repository: ${repo.name}
Category: ${repo.category}
README:
${(repo.readme_content || '').slice(0, 5000)}

Extract all model weights available.`;

    const response = await invokeLLM({
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt }
      ],
      response_format: {
        type: 'json_schema',
        json_schema: {
          name: 'model_extraction',
          strict: true,
          schema: {
            type: 'object',
            properties: {
              models: {
                type: 'array',
                items: {
                  type: 'object',
                  properties: {
                    name: { type: 'string' },
                    huggingface_id: { type: 'string' },
                    size_gb: { type: 'number' },
                    parameters: { type: 'string' },
                    quantizations: { type: 'array', items: { type: 'string' } },
                    download_url: { type: 'string' }
                  },
                  required: ['name', 'huggingface_id', 'size_gb', 'parameters', 'quantizations', 'download_url'],
                  additionalProperties: false
                }
              }
            },
            required: ['models'],
            additionalProperties: false
          }
        }
      }
    });

    const content = response.choices[0]?.message?.content;
    const parsed = JSON.parse(typeof content === 'string' ? content : '{"models":[]}');
    return parsed.models || [];
  }

  // ============================================================================
  // API ENDPOINT EXTRACTION
  // ============================================================================

  private async extractAPIEndpoints(repo: RepositoryInfo): Promise<APIEndpoint[]> {
    const systemPrompt = `You are an API documentation expert.
Extract API endpoints from this repository.
Identify REST APIs, gRPC endpoints, and inference APIs.
Output valid JSON with: endpoints (array of {name, url, method, auth_type, description}).`;

    const userPrompt = `Repository: ${repo.name}
Category: ${repo.category}
README:
${(repo.readme_content || '').slice(0, 5000)}

Extract all API endpoints.`;

    const response = await invokeLLM({
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt }
      ],
      response_format: {
        type: 'json_schema',
        json_schema: {
          name: 'api_extraction',
          strict: true,
          schema: {
            type: 'object',
            properties: {
              endpoints: {
                type: 'array',
                items: {
                  type: 'object',
                  properties: {
                    name: { type: 'string' },
                    url: { type: 'string' },
                    method: { type: 'string' },
                    auth_type: { type: 'string' },
                    description: { type: 'string' }
                  },
                  required: ['name', 'url', 'method', 'auth_type', 'description'],
                  additionalProperties: false
                }
              }
            },
            required: ['endpoints'],
            additionalProperties: false
          }
        }
      }
    });

    const content = response.choices[0]?.message?.content;
    const parsed = JSON.parse(typeof content === 'string' ? content : '{"endpoints":[]}');
    return parsed.endpoints || [];
  }

  // ============================================================================
  // INTEGRATION CODE GENERATION
  // ============================================================================

  private async generateIntegrationCode(
    repo: RepositoryInfo,
    modelWeights: ModelWeight[],
    apiEndpoints: APIEndpoint[]
  ): Promise<string> {
    const systemPrompt = `You are a software integration expert.
Generate TypeScript integration code for this AI/ML repository.
Include:
- Model loading and inference
- API client code
- Error handling
- Type definitions
Output production-ready TypeScript code.`;

    const userPrompt = `Repository: ${repo.name}
Category: ${repo.category}
Language: ${repo.language}

Model Weights:
${modelWeights.map(m => `- ${m.name}: ${m.huggingface_id}`).join('\n')}

API Endpoints:
${apiEndpoints.map(e => `- ${e.name}: ${e.method} ${e.url}`).join('\n')}

Generate integration code.`;

    const response = await invokeLLM({
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt }
      ]
    });

    const content = response.choices[0]?.message?.content;
    return typeof content === 'string' ? content : '';
  }

  // ============================================================================
  // DEEP LINK EXTRACTION
  // ============================================================================

  private async extractDeepLinks(repo: RepositoryInfo): Promise<DeepLink[]> {
    const systemPrompt = `You are a resource discovery expert.
Extract all deep links from this repository:
- API documentation URLs
- Model download URLs
- Dataset URLs
- Code examples
- Related resources
Output valid JSON with: links (array of {type, name, url, description}).`;

    const userPrompt = `Repository: ${repo.name}
URL: ${repo.url}
README:
${(repo.readme_content || '').slice(0, 5000)}

Extract all deep links.`;

    const response = await invokeLLM({
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt }
      ],
      response_format: {
        type: 'json_schema',
        json_schema: {
          name: 'deeplink_extraction',
          strict: true,
          schema: {
            type: 'object',
            properties: {
              links: {
                type: 'array',
                items: {
                  type: 'object',
                  properties: {
                    type: { type: 'string' },
                    name: { type: 'string' },
                    url: { type: 'string' },
                    description: { type: 'string' }
                  },
                  required: ['type', 'name', 'url', 'description'],
                  additionalProperties: false
                }
              }
            },
            required: ['links'],
            additionalProperties: false
          }
        }
      }
    });

    const content = response.choices[0]?.message?.content;
    const parsed = JSON.parse(typeof content === 'string' ? content : '{"links":[]}');
    return (parsed.links || []).map((l: { type: string; name: string; url: string; description: string }) => ({
      type: l.type as DeepLink['type'],
      name: l.name,
      url: l.url,
      description: l.description
    }));
  }

  // ============================================================================
  // GETTERS
  // ============================================================================

  getRepository(name: string): RepositoryInfo | undefined {
    return this.repositories.get(name);
  }

  getAllRepositories(): RepositoryInfo[] {
    return Array.from(this.repositories.values());
  }

  getRepositoriesByCategory(category: string): RepositoryInfo[] {
    return Array.from(this.repositories.values())
      .filter(r => r.category === category);
  }

  getMiningResult(name: string): MiningResult | undefined {
    return this.miningResults.get(name);
  }

  getAllMiningResults(): MiningResult[] {
    return Array.from(this.miningResults.values());
  }

  getAllDeepLinks(): DeepLink[] {
    return this.allDeepLinks;
  }

  getStats(): {
    total_repositories: number;
    mined: number;
    integrated: number;
    failed: number;
    total_model_weights: number;
    total_api_endpoints: number;
    total_deep_links: number;
  } {
    const repos = Array.from(this.repositories.values());
    const results = Array.from(this.miningResults.values());

    return {
      total_repositories: repos.length,
      mined: repos.filter(r => r.integration_status === 'mined').length,
      integrated: repos.filter(r => r.integration_status === 'integrated').length,
      failed: repos.filter(r => r.integration_status === 'failed').length,
      total_model_weights: results.reduce((sum, r) => sum + r.model_weights.length, 0),
      total_api_endpoints: results.reduce((sum, r) => sum + r.api_endpoints.length, 0),
      total_deep_links: this.allDeepLinks.length
    };
  }
}

// Export singleton instance
export const repositoryMiner = new RepositoryMiner();
