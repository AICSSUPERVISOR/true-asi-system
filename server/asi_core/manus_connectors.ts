/**
 * TRUE ASI - COMPLETE MANUS CONNECTOR SYSTEM
 * 
 * Full integration with Manus platform capabilities:
 * - LLM invocation (invokeLLM)
 * - Image generation (generateImage)
 * - Voice transcription (transcribeAudio)
 * - Storage (storagePut, storageGet)
 * - Notifications (notifyOwner)
 * - Data API access
 * - OAuth integration
 */

import { invokeLLM } from '../_core/llm';
import { generateImage } from '../_core/imageGeneration';
import { transcribeAudio } from '../_core/voiceTranscription';
import { storagePut, storageGet } from '../storage';
import { notifyOwner } from '../_core/notification';

// =============================================================================
// TYPES
// =============================================================================

export interface ManusLLMRequest {
  messages: Array<{
    role: 'system' | 'user' | 'assistant' | 'tool';
    content: string | ContentPart[];
  }>;
  tools?: ManusToolDefinition[];
  toolChoice?: 'none' | 'auto' | 'required' | { type: 'function'; function: { name: string } };
  responseFormat?: {
    type: 'text' | 'json_object' | 'json_schema';
    json_schema?: {
      name: string;
      strict?: boolean;
      schema: object;
    };
  };
  temperature?: number;
  maxTokens?: number;
}

export interface ContentPart {
  type: 'text' | 'image_url' | 'file_url';
  text?: string;
  image_url?: { url: string; detail?: 'auto' | 'low' | 'high' };
  file_url?: { url: string; mime_type?: string };
}

export interface ManusToolDefinition {
  type: 'function';
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

export interface ManusLLMResponse {
  id: string;
  content: string;
  toolCalls?: Array<{
    id: string;
    type: 'function';
    function: { name: string; arguments: string };
  }>;
  usage: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
  finishReason: string;
}

export interface ImageGenerationRequest {
  prompt: string;
  originalImages?: Array<{
    url: string;
    mimeType: string;
  }>;
  style?: 'realistic' | 'artistic' | 'cartoon' | 'abstract';
  size?: '1024x1024' | '1792x1024' | '1024x1792';
}

export interface ImageGenerationResponse {
  url: string;
  prompt: string;
  style?: string;
}

export interface TranscriptionRequest {
  audioUrl: string;
  language?: string;
  prompt?: string;
}

export interface TranscriptionResponse {
  text: string;
  language: string;
  segments?: Array<{
    start: number;
    end: number;
    text: string;
  }>;
}

export interface StorageRequest {
  key: string;
  data?: Buffer | Uint8Array | string;
  contentType?: string;
  expiresIn?: number;
}

export interface StorageResponse {
  key: string;
  url: string;
}

export interface NotificationRequest {
  title: string;
  content: string;
  priority?: 'low' | 'normal' | 'high' | 'urgent';
}

// =============================================================================
// MANUS LLM CONNECTOR
// =============================================================================

export class ManusLLMConnector {
  private requestCount = 0;
  private totalTokens = 0;
  
  async chat(request: ManusLLMRequest): Promise<ManusLLMResponse> {
    try {
      const response = await invokeLLM({
        messages: request.messages.map(m => ({
          role: m.role as 'system' | 'user' | 'assistant',
          content: typeof m.content === 'string' ? m.content : this.formatContentParts(m.content)
        })),
        tools: request.tools,
        tool_choice: request.toolChoice as any,
        response_format: request.responseFormat as any
      });
      
      this.requestCount++;
      this.totalTokens += response.usage?.total_tokens || 0;
      
      const rawContent = response.choices?.[0]?.message?.content;
      
      return {
        id: `manus_${Date.now()}`,
        content: typeof rawContent === 'string' ? rawContent : '',
        toolCalls: response.choices?.[0]?.message?.tool_calls,
        usage: {
          promptTokens: response.usage?.prompt_tokens || 0,
          completionTokens: response.usage?.completion_tokens || 0,
          totalTokens: response.usage?.total_tokens || 0
        },
        finishReason: response.choices?.[0]?.finish_reason || 'stop'
      };
    } catch (error) {
      throw new Error(`Manus LLM error: ${error}`);
    }
  }
  
  private formatContentParts(parts: ContentPart[]): any {
    return parts.map(part => {
      if (part.type === 'text') {
        return { type: 'text' as const, text: part.text };
      } else if (part.type === 'image_url') {
        return { type: 'image_url' as const, image_url: part.image_url };
      } else if (part.type === 'file_url') {
        return { type: 'file_url' as const, file_url: part.file_url };
      }
      return { type: 'text' as const, text: '' };
    });
  }
  
  async chatWithStructuredOutput<T>(
    request: ManusLLMRequest,
    schema: { name: string; schema: object }
  ): Promise<T> {
    const response = await this.chat({
      ...request,
      responseFormat: {
        type: 'json_schema',
        json_schema: {
          name: schema.name,
          strict: true,
          schema: schema.schema
        }
      }
    });
    
    return JSON.parse(response.content) as T;
  }
  
  async chatWithTools(
    request: ManusLLMRequest,
    toolHandlers: Record<string, (args: any) => Promise<any>>
  ): Promise<ManusLLMResponse> {
    let response = await this.chat(request);
    
    // Handle tool calls
    while (response.toolCalls && response.toolCalls.length > 0) {
      const toolResults: Array<{ role: 'tool'; content: string; tool_call_id: string }> = [];
      
      for (const toolCall of response.toolCalls) {
        const handler = toolHandlers[toolCall.function.name];
        if (handler) {
          try {
            const args = JSON.parse(toolCall.function.arguments);
            const result = await handler(args);
            toolResults.push({
              role: 'tool',
              content: JSON.stringify(result),
              tool_call_id: toolCall.id
            });
          } catch (error) {
            toolResults.push({
              role: 'tool',
              content: JSON.stringify({ error: String(error) }),
              tool_call_id: toolCall.id
            });
          }
        }
      }
      
      // Continue conversation with tool results
      response = await this.chat({
        ...request,
        messages: [
          ...request.messages,
          { role: 'assistant', content: response.content },
          ...toolResults
        ] as any
      });
    }
    
    return response;
  }
  
  getStats(): { requestCount: number; totalTokens: number } {
    return { requestCount: this.requestCount, totalTokens: this.totalTokens };
  }
}

// =============================================================================
// MANUS IMAGE CONNECTOR
// =============================================================================

export class ManusImageConnector {
  private generationCount = 0;
  
  async generate(request: ImageGenerationRequest): Promise<ImageGenerationResponse> {
    try {
      const result = await generateImage({
        prompt: request.prompt,
        originalImages: request.originalImages
      });
      
      this.generationCount++;
      
      return {
        url: result.url || '',
        prompt: request.prompt,
        style: request.style
      };
    } catch (error) {
      throw new Error(`Manus image generation error: ${error}`);
    }
  }
  
  async edit(
    originalImageUrl: string,
    editPrompt: string,
    mimeType: string = 'image/png'
  ): Promise<ImageGenerationResponse> {
    return this.generate({
      prompt: editPrompt,
      originalImages: [{ url: originalImageUrl, mimeType }]
    });
  }
  
  async generateMultiple(prompts: string[]): Promise<ImageGenerationResponse[]> {
    return Promise.all(prompts.map(prompt => this.generate({ prompt })));
  }
  
  getStats(): { generationCount: number } {
    return { generationCount: this.generationCount };
  }
}

// =============================================================================
// MANUS VOICE CONNECTOR
// =============================================================================

export class ManusVoiceConnector {
  private transcriptionCount = 0;
  
  async transcribe(request: TranscriptionRequest): Promise<TranscriptionResponse> {
    try {
      const result = await transcribeAudio({
        audioUrl: request.audioUrl,
        language: request.language,
        prompt: request.prompt
      });
      
      this.transcriptionCount++;
      
      const transcriptionResult = result as any;
      return {
        text: transcriptionResult.text || '',
        language: transcriptionResult.language || 'unknown',
        segments: transcriptionResult.segments?.map((s: any) => ({
          start: s.start,
          end: s.end,
          text: s.text
        }))
      };
    } catch (error) {
      throw new Error(`Manus transcription error: ${error}`);
    }
  }
  
  async transcribeMultiple(audioUrls: string[]): Promise<TranscriptionResponse[]> {
    return Promise.all(audioUrls.map(url => this.transcribe({ audioUrl: url })));
  }
  
  getStats(): { transcriptionCount: number } {
    return { transcriptionCount: this.transcriptionCount };
  }
}

// =============================================================================
// MANUS STORAGE CONNECTOR
// =============================================================================

export class ManusStorageConnector {
  private uploadCount = 0;
  private downloadCount = 0;
  
  async upload(request: StorageRequest): Promise<StorageResponse> {
    if (!request.data) {
      throw new Error('No data provided for upload');
    }
    
    try {
      const result = await storagePut(
        request.key,
        request.data,
        request.contentType
      );
      
      this.uploadCount++;
      
      return {
        key: result.key,
        url: result.url
      };
    } catch (error) {
      throw new Error(`Manus storage upload error: ${error}`);
    }
  }
  
  async getUrl(key: string, expiresIn?: number): Promise<StorageResponse> {
    try {
      const result = await storageGet(key);
      
      this.downloadCount++;
      
      return {
        key: result.key,
        url: result.url
      };
    } catch (error) {
      throw new Error(`Manus storage get error: ${error}`);
    }
  }
  
  async uploadJson(key: string, data: object): Promise<StorageResponse> {
    return this.upload({
      key: `${key}.json`,
      data: JSON.stringify(data, null, 2),
      contentType: 'application/json'
    });
  }
  
  async uploadText(key: string, text: string): Promise<StorageResponse> {
    return this.upload({
      key: `${key}.txt`,
      data: text,
      contentType: 'text/plain'
    });
  }
  
  getStats(): { uploadCount: number; downloadCount: number } {
    return { uploadCount: this.uploadCount, downloadCount: this.downloadCount };
  }
}

// =============================================================================
// MANUS NOTIFICATION CONNECTOR
// =============================================================================

export class ManusNotificationConnector {
  private notificationCount = 0;
  
  async notify(request: NotificationRequest): Promise<boolean> {
    try {
      const result = await notifyOwner({
        title: request.title,
        content: request.content
      });
      
      this.notificationCount++;
      
      return result;
    } catch (error) {
      console.error('Manus notification error:', error);
      return false;
    }
  }
  
  async notifySuccess(title: string, message: string): Promise<boolean> {
    return this.notify({
      title: `✅ ${title}`,
      content: message,
      priority: 'normal'
    });
  }
  
  async notifyError(title: string, error: string): Promise<boolean> {
    return this.notify({
      title: `❌ ${title}`,
      content: error,
      priority: 'high'
    });
  }
  
  async notifyWarning(title: string, warning: string): Promise<boolean> {
    return this.notify({
      title: `⚠️ ${title}`,
      content: warning,
      priority: 'normal'
    });
  }
  
  async notifyInfo(title: string, info: string): Promise<boolean> {
    return this.notify({
      title: `ℹ️ ${title}`,
      content: info,
      priority: 'low'
    });
  }
  
  getStats(): { notificationCount: number } {
    return { notificationCount: this.notificationCount };
  }
}

// =============================================================================
// MANUS DATA API CONNECTOR
// =============================================================================

export class ManusDataAPIConnector {
  private baseUrl: string;
  private apiKey: string;
  private requestCount = 0;
  
  constructor() {
    this.baseUrl = process.env.BUILT_IN_FORGE_API_URL || '';
    this.apiKey = process.env.BUILT_IN_FORGE_API_KEY || '';
  }
  
  async request(endpoint: string, options: {
    method?: 'GET' | 'POST' | 'PUT' | 'DELETE';
    body?: object;
    headers?: Record<string, string>;
  } = {}): Promise<any> {
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        method: options.method || 'GET',
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json',
          ...options.headers
        },
        body: options.body ? JSON.stringify(options.body) : undefined
      });
      
      this.requestCount++;
      
      if (!response.ok) {
        throw new Error(`API error: ${response.status} ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error) {
      throw new Error(`Manus Data API error: ${error}`);
    }
  }
  
  async search(query: string, type: 'info' | 'news' | 'api' | 'data' = 'info'): Promise<any> {
    return this.request('/search', {
      method: 'POST',
      body: { query, type }
    });
  }
  
  getStats(): { requestCount: number } {
    return { requestCount: this.requestCount };
  }
}

// =============================================================================
// UNIFIED MANUS CONNECTOR
// =============================================================================

export class UnifiedManusConnector {
  public llm: ManusLLMConnector;
  public image: ManusImageConnector;
  public voice: ManusVoiceConnector;
  public storage: ManusStorageConnector;
  public notification: ManusNotificationConnector;
  public dataApi: ManusDataAPIConnector;
  
  constructor() {
    this.llm = new ManusLLMConnector();
    this.image = new ManusImageConnector();
    this.voice = new ManusVoiceConnector();
    this.storage = new ManusStorageConnector();
    this.notification = new ManusNotificationConnector();
    this.dataApi = new ManusDataAPIConnector();
  }
  
  // Quick access methods
  async chat(message: string, systemPrompt?: string): Promise<string> {
    const messages: ManusLLMRequest['messages'] = [];
    
    if (systemPrompt) {
      messages.push({ role: 'system', content: systemPrompt });
    }
    messages.push({ role: 'user', content: message });
    
    const response = await this.llm.chat({ messages });
    return response.content;
  }
  
  async generateImage(prompt: string): Promise<string> {
    const response = await this.image.generate({ prompt });
    return response.url;
  }
  
  async transcribeAudio(audioUrl: string): Promise<string> {
    const response = await this.voice.transcribe({ audioUrl });
    return response.text;
  }
  
  async uploadFile(key: string, data: Buffer | string, contentType?: string): Promise<string> {
    const response = await this.storage.upload({ key, data, contentType });
    return response.url;
  }
  
  async notify(title: string, content: string): Promise<boolean> {
    return this.notification.notify({ title, content });
  }
  
  // Get all stats
  getStats(): {
    llm: { requestCount: number; totalTokens: number };
    image: { generationCount: number };
    voice: { transcriptionCount: number };
    storage: { uploadCount: number; downloadCount: number };
    notification: { notificationCount: number };
    dataApi: { requestCount: number };
  } {
    return {
      llm: this.llm.getStats(),
      image: this.image.getStats(),
      voice: this.voice.getStats(),
      storage: this.storage.getStats(),
      notification: this.notification.getStats(),
      dataApi: this.dataApi.getStats()
    };
  }
  
  // Check available capabilities
  getCapabilities(): {
    llm: boolean;
    image: boolean;
    voice: boolean;
    storage: boolean;
    notification: boolean;
    dataApi: boolean;
  } {
    return {
      llm: true, // Always available via Manus
      image: true,
      voice: true,
      storage: !!process.env.AWS_S3_BUCKET,
      notification: true,
      dataApi: !!process.env.BUILT_IN_FORGE_API_URL
    };
  }
}

// Export singleton instance
export const manusConnector = new UnifiedManusConnector();
