import rateLimit from "express-rate-limit";
import helmet from "helmet";
import * as Sentry from "@sentry/node";
import { httpIntegration, expressIntegration, expressErrorHandler, setupExpressErrorHandler } from "@sentry/node";
import type { Express, Request, Response, NextFunction } from "express";

/**
 * Initialize Sentry error monitoring
 */
export function initSentry(app: Express) {
  // Only initialize in production or if SENTRY_DSN is provided
  const sentryDsn = process.env.SENTRY_DSN;
  
  if (sentryDsn) {
    Sentry.init({
      dsn: sentryDsn,
      environment: process.env.NODE_ENV || "development",
      tracesSampleRate: 1.0,
      integrations: [
        httpIntegration(),
        expressIntegration(),
      ],
    });

    // Setup Express error handler
    setupExpressErrorHandler(app);
    
    console.log("[Security] Sentry error monitoring initialized");
  } else {
    console.log("[Security] Sentry DSN not configured, skipping error monitoring");
  }
}

/**
 * Add Sentry error handler (must be added after routes)
 */
export function addSentryErrorHandler(app: Express) {
  const sentryDsn = process.env.SENTRY_DSN;
  
  if (sentryDsn) {
    // The error handler must be before any other error middleware
    app.use(expressErrorHandler());
  }
}

/**
 * Configure security headers with helmet
 */
export function configureSecurityHeaders(app: Express) {
  app.use(
    helmet({
      contentSecurityPolicy: {
        directives: {
          defaultSrc: ["'self'"],
          scriptSrc: ["'self'", "'unsafe-inline'", "'unsafe-eval'"], // Needed for Vite HMR in dev
          styleSrc: ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com"],
          fontSrc: ["'self'", "https://fonts.gstatic.com"],
          imgSrc: ["'self'", "data:", "https:", "blob:"],
          connectSrc: ["'self'", "https://api.asi1.ai", "https://api.aimlapi.com", "wss:", "ws:"],
          frameSrc: ["'none'"],
          objectSrc: ["'none'"],
          upgradeInsecureRequests: process.env.NODE_ENV === "production" ? [] : null,
        },
      },
      hsts: {
        maxAge: 31536000, // 1 year
        includeSubDomains: true,
        preload: true,
      },
      frameguard: {
        action: "deny",
      },
      noSniff: true,
      xssFilter: true,
      referrerPolicy: {
        policy: "strict-origin-when-cross-origin",
      },
    })
  );
  
  console.log("[Security] Security headers configured with helmet");
}

/**
 * Configure rate limiting
 */
export function configureRateLimiting(app: Express) {
  // General API rate limiter
  const apiLimiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100, // Limit each IP to 100 requests per windowMs
    message: "Too many requests from this IP, please try again later.",
    standardHeaders: true,
    legacyHeaders: false,
    // Skip rate limiting for localhost in development
    skip: (req) => {
      return process.env.NODE_ENV === "development" && 
             (req.ip === "127.0.0.1" || req.ip === "::1");
    },
  });

  // Stricter rate limiter for authentication endpoints
  const authLimiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 5, // Limit each IP to 5 requests per windowMs
    message: "Too many authentication attempts, please try again later.",
    standardHeaders: true,
    legacyHeaders: false,
    skip: (req) => {
      return process.env.NODE_ENV === "development" && 
             (req.ip === "127.0.0.1" || req.ip === "::1");
    },
  });

  // Apply rate limiters
  app.use("/api/trpc", apiLimiter);
  app.use("/api/oauth", authLimiter);
  
  console.log("[Security] Rate limiting configured (100 req/15min general, 5 req/15min auth)");
}

/**
 * Request logging middleware
 */
export function configureRequestLogging(app: Express) {
  app.use((req: Request, res: Response, next: NextFunction) => {
    const start = Date.now();
    
    res.on("finish", () => {
      const duration = Date.now() - start;
      const logLevel = res.statusCode >= 400 ? "error" : "info";
      
      console.log(
        `[${logLevel.toUpperCase()}] ${req.method} ${req.path} - ${res.statusCode} - ${duration}ms`
      );
    });
    
    next();
  });
  
  console.log("[Security] Request logging configured");
}

/**
 * Body size limits
 */
export function configureBodyLimits() {
  return {
    json: { limit: "10mb" },
    urlencoded: { limit: "10mb", extended: true },
  };
}

/**
 * Input sanitization middleware
 * Sanitizes all string inputs to prevent XSS attacks
 */
export function sanitizeInput(req: Request, res: Response, next: NextFunction) {
  // Sanitize query parameters
  if (req.query) {
    for (const key in req.query) {
      if (typeof req.query[key] === 'string') {
        req.query[key] = sanitizeString(req.query[key] as string);
      }
    }
  }

  // Sanitize body parameters
  if (req.body) {
    req.body = sanitizeObject(req.body);
  }

  next();
}

/**
 * Sanitize a string to prevent XSS attacks
 */
function sanitizeString(str: string): string {
  return str
    .replace(/[<>]/g, '') // Remove < and >
    .replace(/javascript:/gi, '') // Remove javascript: protocol
    .replace(/on\w+=/gi, '') // Remove event handlers (onclick=, onload=, etc.)
    .trim();
}

/**
 * Recursively sanitize an object
 */
function sanitizeObject(obj: any): any {
  if (typeof obj === 'string') {
    return sanitizeString(obj);
  }
  
  if (Array.isArray(obj)) {
    return obj.map(sanitizeObject);
  }
  
  if (obj !== null && typeof obj === 'object') {
    const sanitized: any = {};
    for (const key in obj) {
      sanitized[key] = sanitizeObject(obj[key]);
    }
    return sanitized;
  }
  
  return obj;
}

/**
 * Initialize all security middleware
 */
export function initializeSecurity(app: Express) {
  console.log("[Security] Initializing security middleware...");
  
  // 1. Sentry (must be first)
  initSentry(app);
  
  // 2. Security headers
  configureSecurityHeaders(app);
  
  // 3. Rate limiting
  configureRateLimiting(app);
  
  // 4. Input sanitization
  app.use(sanitizeInput);
  
  // 5. Request logging
  configureRequestLogging(app);
  
  console.log("[Security] All security middleware initialized successfully");
}
