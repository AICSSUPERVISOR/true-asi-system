/**
 * WhatsApp Business API Integration
 * 
 * Inspired by Daniel Gross's whatsapp-gpt repository
 * Enables automated customer communication and business workflow automation
 * 
 * API Documentation: https://developers.facebook.com/docs/whatsapp/cloud-api
 */

/**
 * WhatsApp message types
 */
export type WhatsAppMessageType = 'text' | 'template' | 'image' | 'document' | 'video' | 'audio';

/**
 * WhatsApp message status
 */
export type WhatsAppMessageStatus = 'sent' | 'delivered' | 'read' | 'failed';

/**
 * WhatsApp template message
 */
export interface WhatsAppTemplate {
  name: string;
  language: string;
  components?: Array<{
    type: 'header' | 'body' | 'footer' | 'button';
    parameters?: Array<{
      type: 'text' | 'image' | 'document';
      text?: string;
      image?: { link: string };
      document?: { link: string; filename: string };
    }>;
  }>;
}

/**
 * WhatsApp text message
 */
export interface WhatsAppTextMessage {
  to: string; // Phone number in international format (e.g., "4791234567")
  type: 'text';
  text: {
    body: string;
    preview_url?: boolean;
  };
}

/**
 * WhatsApp template message
 */
export interface WhatsAppTemplateMessage {
  to: string;
  type: 'template';
  template: WhatsAppTemplate;
}

/**
 * WhatsApp media message
 */
export interface WhatsAppMediaMessage {
  to: string;
  type: 'image' | 'document' | 'video' | 'audio';
  image?: {
    link: string;
    caption?: string;
  };
  document?: {
    link: string;
    caption?: string;
    filename?: string;
  };
  video?: {
    link: string;
    caption?: string;
  };
  audio?: {
    link: string;
  };
}

/**
 * WhatsApp message (union type)
 */
export type WhatsAppMessage = WhatsAppTextMessage | WhatsAppTemplateMessage | WhatsAppMediaMessage;

/**
 * WhatsApp API response
 */
export interface WhatsAppAPIResponse {
  messaging_product: 'whatsapp';
  contacts: Array<{
    input: string;
    wa_id: string;
  }>;
  messages: Array<{
    id: string;
  }>;
}

/**
 * WhatsApp webhook event
 */
export interface WhatsAppWebhookEvent {
  object: 'whatsapp_business_account';
  entry: Array<{
    id: string;
    changes: Array<{
      value: {
        messaging_product: 'whatsapp';
        metadata: {
          display_phone_number: string;
          phone_number_id: string;
        };
        contacts?: Array<{
          profile: {
            name: string;
          };
          wa_id: string;
        }>;
        messages?: Array<{
          from: string;
          id: string;
          timestamp: string;
          type: 'text' | 'image' | 'document' | 'audio' | 'video';
          text?: {
            body: string;
          };
          image?: {
            mime_type: string;
            sha256: string;
            id: string;
          };
          document?: {
            mime_type: string;
            sha256: string;
            id: string;
            filename: string;
          };
        }>;
        statuses?: Array<{
          id: string;
          status: WhatsAppMessageStatus;
          timestamp: string;
          recipient_id: string;
        }>;
      };
      field: 'messages';
    }>;
  }>;
}

/**
 * WhatsApp Business API Client
 */
export class WhatsAppAPI {
  private accessToken: string;
  private phoneNumberId: string;
  private baseUrl: string = 'https://graph.facebook.com/v18.0';

  constructor(accessToken?: string, phoneNumberId?: string) {
    this.accessToken = accessToken || process.env.WHATSAPP_ACCESS_TOKEN || '';
    this.phoneNumberId = phoneNumberId || process.env.WHATSAPP_PHONE_NUMBER_ID || '';

    if (!this.accessToken) {
      console.warn('[WhatsApp API] No access token provided. Set WHATSAPP_ACCESS_TOKEN environment variable.');
    }
    if (!this.phoneNumberId) {
      console.warn('[WhatsApp API] No phone number ID provided. Set WHATSAPP_PHONE_NUMBER_ID environment variable.');
    }
  }

  /**
   * Check if WhatsApp API is configured
   */
  isConfigured(): boolean {
    return !!(this.accessToken && this.phoneNumberId);
  }

  /**
   * Send a text message
   * 
   * @param to - Recipient phone number (international format, e.g., "4791234567")
   * @param text - Message text
   * @param previewUrl - Enable URL preview (default: false)
   * @returns API response
   */
  async sendTextMessage(
    to: string,
    text: string,
    previewUrl: boolean = false
  ): Promise<WhatsAppAPIResponse> {
    const message: WhatsAppTextMessage = {
      to,
      type: 'text',
      text: {
        body: text,
        preview_url: previewUrl,
      },
    };

    return this.sendMessage(message);
  }

  /**
   * Send a template message
   * 
   * @param to - Recipient phone number
   * @param template - Template configuration
   * @returns API response
   */
  async sendTemplateMessage(
    to: string,
    template: WhatsAppTemplate
  ): Promise<WhatsAppAPIResponse> {
    const message: WhatsAppTemplateMessage = {
      to,
      type: 'template',
      template,
    };

    return this.sendMessage(message);
  }

  /**
   * Send an image message
   * 
   * @param to - Recipient phone number
   * @param imageUrl - Image URL (must be publicly accessible)
   * @param caption - Optional caption
   * @returns API response
   */
  async sendImageMessage(
    to: string,
    imageUrl: string,
    caption?: string
  ): Promise<WhatsAppAPIResponse> {
    const message: WhatsAppMediaMessage = {
      to,
      type: 'image',
      image: {
        link: imageUrl,
        caption,
      },
    };

    return this.sendMessage(message);
  }

  /**
   * Send a document message
   * 
   * @param to - Recipient phone number
   * @param documentUrl - Document URL (must be publicly accessible)
   * @param filename - Document filename
   * @param caption - Optional caption
   * @returns API response
   */
  async sendDocumentMessage(
    to: string,
    documentUrl: string,
    filename: string,
    caption?: string
  ): Promise<WhatsAppAPIResponse> {
    const message: WhatsAppMediaMessage = {
      to,
      type: 'document',
      document: {
        link: documentUrl,
        filename,
        caption,
      },
    };

    return this.sendMessage(message);
  }

  /**
   * Send a message (generic)
   * 
   * @param message - WhatsApp message object
   * @returns API response
   */
  private async sendMessage(message: WhatsAppMessage): Promise<WhatsAppAPIResponse> {
    if (!this.isConfigured()) {
      throw new Error('WhatsApp API is not configured. Set WHATSAPP_ACCESS_TOKEN and WHATSAPP_PHONE_NUMBER_ID environment variables.');
    }

    const url = `${this.baseUrl}/${this.phoneNumberId}/messages`;

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.accessToken}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        messaging_product: 'whatsapp',
        ...message,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(`WhatsApp API error: ${JSON.stringify(error)}`);
    }

    return response.json();
  }

  /**
   * Mark a message as read
   * 
   * @param messageId - Message ID
   */
  async markAsRead(messageId: string): Promise<void> {
    if (!this.isConfigured()) {
      throw new Error('WhatsApp API is not configured.');
    }

    const url = `${this.baseUrl}/${this.phoneNumberId}/messages`;

    await fetch(url, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.accessToken}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        messaging_product: 'whatsapp',
        status: 'read',
        message_id: messageId,
      }),
    });
  }

  /**
   * Download media from WhatsApp
   * 
   * @param mediaId - Media ID from webhook
   * @returns Media URL
   */
  async getMediaUrl(mediaId: string): Promise<string> {
    if (!this.isConfigured()) {
      throw new Error('WhatsApp API is not configured.');
    }

    const url = `${this.baseUrl}/${mediaId}`;

    const response = await fetch(url, {
      headers: {
        'Authorization': `Bearer ${this.accessToken}`,
      },
    });

    if (!response.ok) {
      throw new Error('Failed to get media URL');
    }

    const data = await response.json();
    return data.url;
  }

  /**
   * Download media file
   * 
   * @param mediaUrl - Media URL from getMediaUrl()
   * @returns Media file buffer
   */
  async downloadMedia(mediaUrl: string): Promise<Buffer> {
    const response = await fetch(mediaUrl, {
      headers: {
        'Authorization': `Bearer ${this.accessToken}`,
      },
    });

    if (!response.ok) {
      throw new Error('Failed to download media');
    }

    const arrayBuffer = await response.arrayBuffer();
    return Buffer.from(arrayBuffer);
  }
}

/**
 * Singleton WhatsApp API instance
 */
let whatsappAPI: WhatsAppAPI | null = null;

/**
 * Get WhatsApp API instance
 * 
 * @returns WhatsApp API instance
 */
export function getWhatsAppAPI(): WhatsAppAPI {
  if (!whatsappAPI) {
    whatsappAPI = new WhatsAppAPI();
  }
  return whatsappAPI;
}

/**
 * Send WhatsApp notification (convenience function)
 * 
 * @param to - Recipient phone number
 * @param message - Message text
 * @returns Success boolean
 */
export async function sendWhatsAppNotification(
  to: string,
  message: string
): Promise<boolean> {
  try {
    const api = getWhatsAppAPI();
    
    if (!api.isConfigured()) {
      console.log('[WhatsApp] API not configured, skipping notification');
      return false;
    }

    await api.sendTextMessage(to, message);
    console.log(`[WhatsApp] Notification sent to ${to}`);
    return true;
  } catch (error) {
    console.error('[WhatsApp] Failed to send notification:', error);
    return false;
  }
}

/**
 * Verify WhatsApp webhook signature
 * 
 * @param signature - X-Hub-Signature-256 header
 * @param body - Request body
 * @returns Valid boolean
 */
export function verifyWebhookSignature(signature: string, body: string): boolean {
  const crypto = require('crypto');
  const appSecret = process.env.WHATSAPP_APP_SECRET || '';
  
  if (!appSecret) {
    console.warn('[WhatsApp] No app secret configured');
    return false;
  }

  const expectedSignature = crypto
    .createHmac('sha256', appSecret)
    .update(body)
    .digest('hex');

  return signature === `sha256=${expectedSignature}`;
}

/**
 * Handle WhatsApp webhook verification (GET request)
 * 
 * @param mode - hub.mode query parameter
 * @param token - hub.verify_token query parameter
 * @param challenge - hub.challenge query parameter
 * @returns Challenge string if valid, null otherwise
 */
export function handleWebhookVerification(
  mode: string,
  token: string,
  challenge: string
): string | null {
  const verifyToken = process.env.WHATSAPP_VERIFY_TOKEN || '';

  if (mode === 'subscribe' && token === verifyToken) {
    console.log('[WhatsApp] Webhook verified');
    return challenge;
  }

  console.warn('[WhatsApp] Webhook verification failed');
  return null;
}
