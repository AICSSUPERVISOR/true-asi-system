// ============================================================================
// INPUT SANITIZATION MIDDLEWARE
// Protects against XSS, SQL injection, and other injection attacks
// ============================================================================

/**
 * Sanitize a string by removing potentially dangerous characters
 */
export function sanitizeString(input: string): string {
  if (typeof input !== "string") return input;

  return input
    // Remove null bytes
    .replace(/\0/g, "")
    // Escape HTML entities
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#x27;")
    // Remove potential script injections
    .replace(/javascript:/gi, "")
    .replace(/data:/gi, "")
    .replace(/vbscript:/gi, "")
    // Remove event handlers
    .replace(/on\w+\s*=/gi, "");
}

/**
 * Sanitize HTML content (for rich text fields)
 */
export function sanitizeHtml(input: string): string {
  if (typeof input !== "string") return input;

  // Allowed tags
  const allowedTags = [
    "p", "br", "b", "i", "u", "strong", "em", "ul", "ol", "li",
    "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "code", "pre"
  ];

  // Remove all tags except allowed ones
  let result = input;

  // Remove script tags and their content
  result = result.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, "");

  // Remove style tags and their content
  result = result.replace(/<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>/gi, "");

  // Remove event handlers from all tags
  result = result.replace(/\s*on\w+\s*=\s*["'][^"']*["']/gi, "");
  result = result.replace(/\s*on\w+\s*=\s*[^\s>]*/gi, "");

  // Remove javascript: and data: URLs
  result = result.replace(/href\s*=\s*["']?\s*javascript:[^"'>]*/gi, 'href="#"');
  result = result.replace(/src\s*=\s*["']?\s*javascript:[^"'>]*/gi, 'src=""');
  result = result.replace(/href\s*=\s*["']?\s*data:[^"'>]*/gi, 'href="#"');
  result = result.replace(/src\s*=\s*["']?\s*data:[^"'>]*/gi, 'src=""');

  // Remove tags not in allowed list
  const tagPattern = /<\/?([a-z][a-z0-9]*)\b[^>]*>/gi;
  result = result.replace(tagPattern, (match, tagName) => {
    if (allowedTags.includes(tagName.toLowerCase())) {
      // Keep allowed tags but remove dangerous attributes
      return match
        .replace(/\s+style\s*=\s*["'][^"']*["']/gi, "")
        .replace(/\s+class\s*=\s*["'][^"']*["']/gi, "");
    }
    return "";
  });

  return result;
}

/**
 * Sanitize a filename
 */
export function sanitizeFilename(filename: string): string {
  if (typeof filename !== "string") return "file";

  return filename
    // Remove path traversal attempts
    .replace(/\.\./g, "")
    .replace(/[\/\\]/g, "")
    // Remove null bytes
    .replace(/\0/g, "")
    // Keep only safe characters
    .replace(/[^a-zA-Z0-9._-]/g, "_")
    // Limit length
    .slice(0, 255);
}

/**
 * Sanitize an organization number (Norwegian format)
 */
export function sanitizeOrgNumber(orgNumber: string): string {
  if (typeof orgNumber !== "string") return "";

  // Remove all non-digits
  return orgNumber.replace(/\D/g, "").slice(0, 9);
}

/**
 * Sanitize an email address
 */
export function sanitizeEmail(email: string): string {
  if (typeof email !== "string") return "";

  // Basic email sanitization
  return email
    .toLowerCase()
    .trim()
    .replace(/[<>]/g, "")
    .slice(0, 320);
}

/**
 * Sanitize a URL
 */
export function sanitizeUrl(url: string): string {
  if (typeof url !== "string") return "";

  try {
    const parsed = new URL(url);
    // Only allow http and https protocols
    if (!["http:", "https:"].includes(parsed.protocol)) {
      return "";
    }
    return parsed.href;
  } catch {
    return "";
  }
}

/**
 * Sanitize a number input
 */
export function sanitizeNumber(input: unknown, min?: number, max?: number): number | null {
  const num = Number(input);
  if (isNaN(num) || !isFinite(num)) return null;

  let result = num;
  if (min !== undefined) result = Math.max(min, result);
  if (max !== undefined) result = Math.min(max, result);

  return result;
}

/**
 * Sanitize a date input
 */
export function sanitizeDate(input: unknown): Date | null {
  if (input instanceof Date) {
    return isNaN(input.getTime()) ? null : input;
  }

  if (typeof input === "string" || typeof input === "number") {
    const date = new Date(input);
    return isNaN(date.getTime()) ? null : date;
  }

  return null;
}

/**
 * Deep sanitize an object
 */
export function sanitizeObject<T extends Record<string, unknown>>(obj: T): T {
  const result: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(obj)) {
    if (typeof value === "string") {
      result[key] = sanitizeString(value);
    } else if (Array.isArray(value)) {
      result[key] = value.map((item) =>
        typeof item === "string"
          ? sanitizeString(item)
          : typeof item === "object" && item !== null
          ? sanitizeObject(item as Record<string, unknown>)
          : item
      );
    } else if (typeof value === "object" && value !== null) {
      result[key] = sanitizeObject(value as Record<string, unknown>);
    } else {
      result[key] = value;
    }
  }

  return result as T;
}

// ============================================================================
// VALIDATION HELPERS
// ============================================================================

/**
 * Validate Norwegian organization number (Modulus 11 check)
 */
export function isValidOrgNumber(orgNumber: string): boolean {
  const cleaned = sanitizeOrgNumber(orgNumber);
  if (cleaned.length !== 9) return false;

  const weights = [3, 2, 7, 6, 5, 4, 3, 2];
  const digits = cleaned.split("").map(Number);

  let sum = 0;
  for (let i = 0; i < 8; i++) {
    sum += digits[i] * weights[i];
  }

  const remainder = sum % 11;
  const checkDigit = remainder === 0 ? 0 : 11 - remainder;

  // Check digit of 10 is invalid
  if (checkDigit === 10) return false;

  return digits[8] === checkDigit;
}

/**
 * Validate Norwegian bank account number (Modulus 11 check)
 */
export function isValidBankAccount(accountNumber: string): boolean {
  const cleaned = accountNumber.replace(/\D/g, "");
  if (cleaned.length !== 11) return false;

  const weights = [5, 4, 3, 2, 7, 6, 5, 4, 3, 2];
  const digits = cleaned.split("").map(Number);

  let sum = 0;
  for (let i = 0; i < 10; i++) {
    sum += digits[i] * weights[i];
  }

  const remainder = sum % 11;
  const checkDigit = remainder === 0 ? 0 : 11 - remainder;

  // Check digit of 10 is invalid
  if (checkDigit === 10) return false;

  return digits[10] === checkDigit;
}

/**
 * Validate email format
 */
export function isValidEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

/**
 * Validate URL format
 */
export function isValidUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    return ["http:", "https:"].includes(parsed.protocol);
  } catch {
    return false;
  }
}
