/**
 * Website analysis and automated enhancement system
 */

import { invokeLLM } from "./llm";

export interface WebsiteAnalysis {
  url: string;
  seo: SEOAnalysis;
  performance: PerformanceAnalysis;
  accessibility: AccessibilityAnalysis;
  contentQuality: ContentQualityAnalysis;
  ux: UXAnalysis;
  overallScore: number;
  recommendations: Recommendation[];
}

export interface SEOAnalysis {
  score: number;
  title: string;
  titleLength: number;
  metaDescription: string;
  metaDescriptionLength: number;
  h1Count: number;
  h1Text: string[];
  imageAltMissing: number;
  internalLinks: number;
  externalLinks: number;
  canonicalUrl: string;
  robotsMeta: string;
  structuredData: boolean;
  issues: string[];
  improvements: string[];
}

export interface PerformanceAnalysis {
  score: number;
  loadTime: number;
  pageSize: number;
  requests: number;
  images: number;
  scripts: number;
  stylesheets: number;
  fonts: number;
  issues: string[];
  improvements: string[];
}

export interface AccessibilityAnalysis {
  score: number;
  missingAltText: number;
  lowContrast: number;
  missingLabels: number;
  keyboardNavigation: boolean;
  ariaLabels: number;
  semanticHTML: boolean;
  issues: string[];
  improvements: string[];
}

export interface ContentQualityAnalysis {
  score: number;
  wordCount: number;
  readabilityScore: number;
  headingStructure: boolean;
  paragraphLength: number;
  callToActions: number;
  contactInfo: boolean;
  socialLinks: number;
  issues: string[];
  improvements: string[];
}

export interface UXAnalysis {
  score: number;
  mobileResponsive: boolean;
  navigationClarity: number;
  formUsability: number;
  visualHierarchy: number;
  whitespace: number;
  consistency: number;
  issues: string[];
  improvements: string[];
}

export interface Recommendation {
  category: "seo" | "performance" | "accessibility" | "content" | "ux";
  priority: "critical" | "high" | "medium" | "low";
  title: string;
  description: string;
  impact: string;
  effort: "low" | "medium" | "high";
  implementation: string[];
}

/**
 * Fetch and analyze website
 */
export async function analyzeWebsite(url: string): Promise<WebsiteAnalysis> {
  console.log(`[Website Analysis] Starting analysis for ${url}`);

  // Fetch website HTML
  const html = await fetchWebsiteHTML(url);
  
  // Parallel analysis
  const [seo, performance, accessibility, contentQuality, ux] = await Promise.all([
    analyzeSEO(url, html),
    analyzePerformance(url, html),
    analyzeAccessibility(url, html),
    analyzeContentQuality(url, html),
    analyzeUX(url, html),
  ]);

  // Calculate overall score
  const overallScore = Math.round(
    (seo.score + performance.score + accessibility.score + contentQuality.score + ux.score) / 5
  );

  // Generate prioritized recommendations
  const recommendations = await generateRecommendations({
    url,
    seo,
    performance,
    accessibility,
    contentQuality,
    ux,
  });

  console.log(`[Website Analysis] Completed with overall score: ${overallScore}/100`);

  return {
    url,
    seo,
    performance,
    accessibility,
    contentQuality,
    ux,
    overallScore,
    recommendations,
  };
}

/**
 * Fetch website HTML
 */
async function fetchWebsiteHTML(url: string): Promise<string> {
  try {
    // Ensure URL has protocol
    if (!url.startsWith("http")) {
      url = "https://" + url;
    }

    const response = await fetch(url, {
      headers: {
        "User-Agent":
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return await response.text();
  } catch (error) {
    console.error(`[Website Analysis] Failed to fetch ${url}:`, error);
    throw new Error(`Failed to fetch website: ${error}`);
  }
}

/**
 * Analyze SEO
 */
async function analyzeSEO(url: string, html: string): Promise<SEOAnalysis> {
  const issues: string[] = [];
  const improvements: string[] = [];
  let score = 100;

  // Extract title
  const titleMatch = html.match(/<title[^>]*>(.*?)<\/title>/i);
  const title = titleMatch ? titleMatch[1].trim() : "";
  const titleLength = title.length;

  if (!title) {
    issues.push("Missing page title");
    score -= 15;
  } else if (titleLength < 30 || titleLength > 60) {
    issues.push(`Title length ${titleLength} chars (optimal: 30-60)`);
    score -= 5;
  }

  // Extract meta description
  const metaDescMatch = html.match(
    /<meta\s+name=["']description["']\s+content=["'](.*?)["']/i
  );
  const metaDescription = metaDescMatch ? metaDescMatch[1].trim() : "";
  const metaDescriptionLength = metaDescription.length;

  if (!metaDescription) {
    issues.push("Missing meta description");
    score -= 10;
  } else if (metaDescriptionLength < 120 || metaDescriptionLength > 160) {
    issues.push(`Meta description length ${metaDescriptionLength} chars (optimal: 120-160)`);
    score -= 5;
  }

  // Count H1 tags
  const h1Matches = html.match(/<h1[^>]*>(.*?)<\/h1>/gi) || [];
  const h1Count = h1Matches.length;
  const h1Text = h1Matches.map((h1) => h1.replace(/<[^>]+>/g, "").trim());

  if (h1Count === 0) {
    issues.push("Missing H1 heading");
    score -= 10;
  } else if (h1Count > 1) {
    issues.push(`Multiple H1 headings (${h1Count}) - should have only one`);
    score -= 5;
  }

  // Count images without alt text
  const imgMatches = html.match(/<img[^>]*>/gi) || [];
  const imageAltMissing = imgMatches.filter((img) => !img.includes("alt=")).length;

  if (imageAltMissing > 0) {
    issues.push(`${imageAltMissing} images missing alt text`);
    score -= Math.min(10, imageAltMissing * 2);
  }

  // Count links
  const internalLinks = (html.match(/<a\s+[^>]*href=["']\//gi) || []).length;
  const externalLinks = (html.match(/<a\s+[^>]*href=["']https?:\/\//gi) || []).length;

  // Check canonical URL
  const canonicalMatch = html.match(/<link\s+rel=["']canonical["']\s+href=["'](.*?)["']/i);
  const canonicalUrl = canonicalMatch ? canonicalMatch[1] : "";

  if (!canonicalUrl) {
    issues.push("Missing canonical URL");
    score -= 5;
  }

  // Check robots meta
  const robotsMatch = html.match(/<meta\s+name=["']robots["']\s+content=["'](.*?)["']/i);
  const robotsMeta = robotsMatch ? robotsMatch[1] : "";

  // Check structured data
  const structuredData =
    html.includes('type="application/ld+json"') || html.includes("schema.org");

  if (!structuredData) {
    improvements.push("Add structured data (JSON-LD) for better search visibility");
  }

  // Generate improvements
  if (titleLength < 30) {
    improvements.push("Expand title to 50-60 characters with target keywords");
  }
  if (metaDescriptionLength < 120) {
    improvements.push("Expand meta description to 150-160 characters with call-to-action");
  }
  if (imageAltMissing > 0) {
    improvements.push(`Add descriptive alt text to all ${imageAltMissing} images`);
  }
  if (internalLinks < 5) {
    improvements.push("Add more internal links to improve site structure");
  }

  return {
    score: Math.max(0, score),
    title,
    titleLength,
    metaDescription,
    metaDescriptionLength,
    h1Count,
    h1Text,
    imageAltMissing,
    internalLinks,
    externalLinks,
    canonicalUrl,
    robotsMeta,
    structuredData,
    issues,
    improvements,
  };
}

/**
 * Analyze performance
 */
async function analyzePerformance(url: string, html: string): Promise<PerformanceAnalysis> {
  const issues: string[] = [];
  const improvements: string[] = [];
  let score = 100;

  const pageSize = Buffer.byteLength(html, "utf8");
  const images = (html.match(/<img[^>]*>/gi) || []).length;
  const scripts = (html.match(/<script[^>]*>/gi) || []).length;
  const stylesheets = (html.match(/<link[^>]*rel=["']stylesheet["']/gi) || []).length;
  const fonts = (html.match(/@font-face/gi) || []).length;
  const requests = images + scripts + stylesheets + fonts;

  // Estimate load time based on page size
  const loadTime = pageSize / 100000; // Rough estimate: 100KB/s

  if (pageSize > 1000000) {
    issues.push(`Large page size: ${(pageSize / 1024).toFixed(0)}KB (target: <500KB)`);
    score -= 15;
  }

  if (scripts > 10) {
    issues.push(`Too many scripts: ${scripts} (target: <10)`);
    score -= 10;
  }

  if (images > 20) {
    issues.push(`Too many images: ${images} (consider lazy loading)`);
    score -= 5;
  }

  if (loadTime > 3) {
    issues.push(`Slow estimated load time: ${loadTime.toFixed(1)}s (target: <3s)`);
    score -= 10;
  }

  // Generate improvements
  improvements.push("Compress and optimize all images (WebP format)");
  improvements.push("Minify CSS and JavaScript files");
  improvements.push("Enable browser caching with proper headers");
  improvements.push("Use CDN for static assets");
  improvements.push("Implement lazy loading for images");

  return {
    score: Math.max(0, score),
    loadTime,
    pageSize,
    requests,
    images,
    scripts,
    stylesheets,
    fonts,
    issues,
    improvements,
  };
}

/**
 * Analyze accessibility
 */
async function analyzeAccessibility(url: string, html: string): Promise<AccessibilityAnalysis> {
  const issues: string[] = [];
  const improvements: string[] = [];
  let score = 100;

  const missingAltText = (html.match(/<img[^>]*>/gi) || []).filter(
    (img) => !img.includes("alt=")
  ).length;

  const missingLabels = (html.match(/<input[^>]*>/gi) || []).filter(
    (input) => !input.includes("aria-label") && !input.includes("id=")
  ).length;

  const ariaLabels = (html.match(/aria-label=/gi) || []).length;
  const semanticHTML =
    html.includes("<header") &&
    html.includes("<nav") &&
    html.includes("<main") &&
    html.includes("<footer");

  if (missingAltText > 0) {
    issues.push(`${missingAltText} images missing alt text`);
    score -= Math.min(20, missingAltText * 3);
  }

  if (missingLabels > 0) {
    issues.push(`${missingLabels} form inputs missing labels`);
    score -= Math.min(15, missingLabels * 5);
  }

  if (!semanticHTML) {
    issues.push("Missing semantic HTML5 elements");
    score -= 10;
  }

  if (ariaLabels < 3) {
    improvements.push("Add ARIA labels for better screen reader support");
  }

  improvements.push("Ensure keyboard navigation works for all interactive elements");
  improvements.push("Test with screen readers (NVDA, JAWS)");
  improvements.push("Add skip-to-content link");

  return {
    score: Math.max(0, score),
    missingAltText,
    lowContrast: 0, // Would require CSS analysis
    missingLabels,
    keyboardNavigation: true, // Assume true by default
    ariaLabels,
    semanticHTML,
    issues,
    improvements,
  };
}

/**
 * Analyze content quality
 */
async function analyzeContentQuality(url: string, html: string): Promise<ContentQualityAnalysis> {
  const issues: string[] = [];
  const improvements: string[] = [];
  let score = 100;

  // Extract text content (remove HTML tags)
  const textContent = html.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
  const wordCount = textContent.split(/\s+/).length;

  const headingStructure =
    html.includes("<h1") && html.includes("<h2") && html.includes("<h3");
  const callToActions = (
    html.match(/<button|<a[^>]*class=["'][^"']*btn[^"']*["']/gi) || []
  ).length;
  const contactInfo =
    html.includes("@") || html.includes("tel:") || html.includes("contact");
  const socialLinks = (html.match(/facebook|twitter|linkedin|instagram/gi) || []).length;

  if (wordCount < 300) {
    issues.push(`Low word count: ${wordCount} (target: >500)`);
    score -= 15;
  }

  if (!headingStructure) {
    issues.push("Poor heading structure (missing H2/H3)");
    score -= 10;
  }

  if (callToActions === 0) {
    issues.push("No clear call-to-action buttons");
    score -= 10;
  }

  if (!contactInfo) {
    issues.push("Missing contact information");
    score -= 10;
  }

  if (socialLinks === 0) {
    improvements.push("Add social media links");
  }

  improvements.push("Expand content to 800-1500 words with valuable information");
  improvements.push("Add more headings (H2, H3) to break up content");
  improvements.push("Include clear calls-to-action throughout the page");

  return {
    score: Math.max(0, score),
    wordCount,
    readabilityScore: 70, // Simplified - would require proper readability calculation
    headingStructure,
    paragraphLength: Math.round(wordCount / 10),
    callToActions,
    contactInfo,
    socialLinks,
    issues,
    improvements,
  };
}

/**
 * Analyze UX
 */
async function analyzeUX(url: string, html: string): Promise<UXAnalysis> {
  const issues: string[] = [];
  const improvements: string[] = [];
  let score = 100;

  const mobileResponsive =
    html.includes('name="viewport"') || html.includes("@media");
  const navigationClarity = html.includes("<nav") ? 80 : 50;
  const formUsability = (html.match(/<form[^>]*>/gi) || []).length > 0 ? 70 : 100;
  const visualHierarchy = html.includes("<h1") && html.includes("<h2") ? 80 : 60;
  const whitespace = 75; // Simplified - would require CSS analysis
  const consistency = 80; // Simplified - would require design analysis

  if (!mobileResponsive) {
    issues.push("Not mobile responsive");
    score -= 20;
  }

  if (navigationClarity < 70) {
    issues.push("Navigation unclear or missing");
    score -= 10;
  }

  improvements.push("Add mobile-first responsive design");
  improvements.push("Improve navigation clarity with clear labels");
  improvements.push("Enhance visual hierarchy with consistent typography");
  improvements.push("Add more whitespace for better readability");

  return {
    score: Math.max(0, score),
    mobileResponsive,
    navigationClarity,
    formUsability,
    visualHierarchy,
    whitespace,
    consistency,
    issues,
    improvements,
  };
}

/**
 * Generate prioritized recommendations using AI
 */
async function generateRecommendations(analysis: {
  url: string;
  seo: SEOAnalysis;
  performance: PerformanceAnalysis;
  accessibility: AccessibilityAnalysis;
  contentQuality: ContentQualityAnalysis;
  ux: UXAnalysis;
}): Promise<Recommendation[]> {
  const prompt = `Generate prioritized website improvement recommendations based on this analysis:

URL: ${analysis.url}

SEO Score: ${analysis.seo.score}/100
Issues: ${analysis.seo.issues.join(", ")}

Performance Score: ${analysis.performance.score}/100
Issues: ${analysis.performance.issues.join(", ")}

Accessibility Score: ${analysis.accessibility.score}/100
Issues: ${analysis.accessibility.issues.join(", ")}

Content Quality Score: ${analysis.contentQuality.score}/100
Issues: ${analysis.contentQuality.issues.join(", ")}

UX Score: ${analysis.ux.score}/100
Issues: ${analysis.ux.issues.join(", ")}

Generate 8-12 specific, actionable recommendations with:
- Category (seo/performance/accessibility/content/ux)
- Priority (critical/high/medium/low)
- Title (concise)
- Description (detailed explanation)
- Impact (expected improvement)
- Effort (low/medium/high)
- Implementation steps (array of specific actions)

Focus on highest-impact, most critical issues first.`;

  const response = await invokeLLM({
    messages: [
      {
        role: "system",
        content:
          "You are a website optimization expert. Provide specific, actionable recommendations.",
      },
      { role: "user", content: prompt },
    ],
    response_format: {
      type: "json_schema",
      json_schema: {
        name: "recommendations",
        strict: true,
        schema: {
          type: "object",
          properties: {
            recommendations: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  category: {
                    type: "string",
                    enum: ["seo", "performance", "accessibility", "content", "ux"],
                  },
                  priority: {
                    type: "string",
                    enum: ["critical", "high", "medium", "low"],
                  },
                  title: { type: "string" },
                  description: { type: "string" },
                  impact: { type: "string" },
                  effort: { type: "string", enum: ["low", "medium", "high"] },
                  implementation: { type: "array", items: { type: "string" } },
                },
                required: [
                  "category",
                  "priority",
                  "title",
                  "description",
                  "impact",
                  "effort",
                  "implementation",
                ],
                additionalProperties: false,
              },
            },
          },
          required: ["recommendations"],
          additionalProperties: false,
        },
      },
    },
  });

  const content = response.choices[0].message.content as string;
  const result = JSON.parse(content);
  return result.recommendations as Recommendation[];
}
