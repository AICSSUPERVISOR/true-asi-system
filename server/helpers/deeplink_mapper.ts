/**
 * Deeplink Mapper
 * Maps AI recommendations to specific platform deeplinks for one-click execution
 * Uses the comprehensive industry_deeplinks.ts registry (300+ platforms)
 */

// Import industry deeplinks (will be implemented)
// import { getIndustryDeeplinks } from "./industry_deeplinks";

export interface Recommendation {
  category: "revenue" | "marketing" | "leadership" | "operations" | "technology";
  action: string;
  impact: "high" | "medium" | "low";
  difficulty: "easy" | "medium" | "hard";
  roi: string;
  estimatedCost?: string;
  timeframe?: string;
}

export interface ExecutableRecommendation extends Recommendation {
  deeplinks: DeeplinkAction[];
  priority: number; // 1-10, calculated from impact + difficulty
}

export interface DeeplinkAction {
  platform: string;
  category: string;
  url: string;
  description: string;
  setupTime?: string;
  cost?: string;
}

/**
 * Map recommendations to deeplinks based on keywords and industry
 */
export function mapRecommendationsToDeeplinks(
  recommendations: Recommendation[],
  industryCode: string
): ExecutableRecommendation[] {
  // const industryDeeplinks = getIndustryDeeplinks(industryCode);
  const industryDeeplinks = {}; // Placeholder

  return recommendations.map((rec) => {
    const deeplinks = findMatchingDeeplinks(rec, industryDeeplinks);
    const priority = calculatePriority(rec);

    return {
      ...rec,
      deeplinks,
      priority,
    };
  });
}

/**
 * Find matching deeplinks for a recommendation
 */
function findMatchingDeeplinks(
  recommendation: Recommendation,
  industryDeeplinks: any
): DeeplinkAction[] {
  const deeplinks: DeeplinkAction[] = [];
  const actionLower = recommendation.action.toLowerCase();

  // Marketing recommendations
  if (recommendation.category === "marketing") {
    if (actionLower.includes("social media") || actionLower.includes("linkedin")) {
      deeplinks.push({
        platform: "LinkedIn Ads",
        category: "Marketing",
        url: "https://www.linkedin.com/campaignmanager/",
        description: "Create targeted LinkedIn advertising campaigns",
        setupTime: "30 minutes",
        cost: "From $10/day",
      });
      deeplinks.push({
        platform: "Facebook Ads",
        category: "Marketing",
        url: "https://www.facebook.com/adsmanager/",
        description: "Launch Facebook and Instagram ad campaigns",
        setupTime: "20 minutes",
        cost: "From $5/day",
      });
    }

    if (actionLower.includes("email") || actionLower.includes("newsletter")) {
      deeplinks.push({
        platform: "Mailchimp",
        category: "Email Marketing",
        url: "https://mailchimp.com/",
        description: "Set up email marketing campaigns",
        setupTime: "1 hour",
        cost: "Free for up to 500 contacts",
      });
      deeplinks.push({
        platform: "HubSpot",
        category: "Marketing Automation",
        url: "https://www.hubspot.com/products/marketing",
        description: "Complete marketing automation platform",
        setupTime: "2 hours",
        cost: "From $45/month",
      });
    }

    if (actionLower.includes("seo") || actionLower.includes("search")) {
      deeplinks.push({
        platform: "Google Search Console",
        category: "SEO",
        url: "https://search.google.com/search-console/",
        description: "Monitor and optimize Google search presence",
        setupTime: "15 minutes",
        cost: "Free",
      });
      deeplinks.push({
        platform: "SEMrush",
        category: "SEO Tools",
        url: "https://www.semrush.com/",
        description: "Comprehensive SEO and competitor analysis",
        setupTime: "30 minutes",
        cost: "From $119.95/month",
      });
    }

    if (actionLower.includes("content") || actionLower.includes("blog")) {
      deeplinks.push({
        platform: "WordPress",
        category: "Content Management",
        url: "https://wordpress.com/",
        description: "Create and manage blog content",
        setupTime: "1 hour",
        cost: "Free or from $4/month",
      });
    }

    if (actionLower.includes("analytics") || actionLower.includes("tracking")) {
      deeplinks.push({
        platform: "Google Analytics",
        category: "Analytics",
        url: "https://analytics.google.com/",
        description: "Track website traffic and user behavior",
        setupTime: "30 minutes",
        cost: "Free",
      });
    }
  }

  // Revenue recommendations
  if (recommendation.category === "revenue") {
    if (actionLower.includes("crm") || actionLower.includes("customer")) {
      deeplinks.push({
        platform: "Salesforce",
        category: "CRM",
        url: "https://www.salesforce.com/form/signup/freetrial-sales/",
        description: "Enterprise CRM for sales and customer management",
        setupTime: "2 hours",
        cost: "From $25/user/month",
      });
      deeplinks.push({
        platform: "HubSpot CRM",
        category: "CRM",
        url: "https://www.hubspot.com/products/crm",
        description: "Free CRM with sales pipeline management",
        setupTime: "1 hour",
        cost: "Free",
      });
      deeplinks.push({
        platform: "Pipedrive",
        category: "CRM",
        url: "https://www.pipedrive.com/",
        description: "Sales-focused CRM for small businesses",
        setupTime: "1 hour",
        cost: "From $14.90/user/month",
      });
    }

    if (actionLower.includes("e-commerce") || actionLower.includes("online store")) {
      deeplinks.push({
        platform: "Shopify",
        category: "E-commerce",
        url: "https://www.shopify.com/",
        description: "Complete e-commerce platform",
        setupTime: "4 hours",
        cost: "From $29/month",
      });
      deeplinks.push({
        platform: "WooCommerce",
        category: "E-commerce",
        url: "https://woocommerce.com/",
        description: "WordPress e-commerce plugin",
        setupTime: "3 hours",
        cost: "Free (hosting required)",
      });
    }

    if (actionLower.includes("payment") || actionLower.includes("checkout")) {
      deeplinks.push({
        platform: "Stripe",
        category: "Payments",
        url: "https://dashboard.stripe.com/register",
        description: "Online payment processing",
        setupTime: "1 hour",
        cost: "2.9% + $0.30 per transaction",
      });
      deeplinks.push({
        platform: "PayPal",
        category: "Payments",
        url: "https://www.paypal.com/bizsignup/",
        description: "Payment gateway for online transactions",
        setupTime: "30 minutes",
        cost: "2.9% + fixed fee per transaction",
      });
    }

    if (actionLower.includes("pricing") || actionLower.includes("subscription")) {
      deeplinks.push({
        platform: "Chargebee",
        category: "Subscription Management",
        url: "https://www.chargebee.com/",
        description: "Subscription billing and revenue management",
        setupTime: "2 hours",
        cost: "From $249/month",
      });
    }
  }

  // Operations recommendations
  if (recommendation.category === "operations") {
    if (actionLower.includes("project") || actionLower.includes("task")) {
      deeplinks.push({
        platform: "Asana",
        category: "Project Management",
        url: "https://asana.com/",
        description: "Team project and task management",
        setupTime: "1 hour",
        cost: "Free for up to 15 users",
      });
      deeplinks.push({
        platform: "Monday.com",
        category: "Project Management",
        url: "https://monday.com/",
        description: "Visual project management platform",
        setupTime: "1 hour",
        cost: "From $8/user/month",
      });
      deeplinks.push({
        platform: "Trello",
        category: "Project Management",
        url: "https://trello.com/",
        description: "Kanban-style project boards",
        setupTime: "30 minutes",
        cost: "Free or from $5/user/month",
      });
    }

    if (actionLower.includes("communication") || actionLower.includes("team")) {
      deeplinks.push({
        platform: "Slack",
        category: "Team Communication",
        url: "https://slack.com/",
        description: "Team messaging and collaboration",
        setupTime: "30 minutes",
        cost: "Free or from $7.25/user/month",
      });
      deeplinks.push({
        platform: "Microsoft Teams",
        category: "Team Communication",
        url: "https://www.microsoft.com/microsoft-teams/",
        description: "Enterprise team collaboration",
        setupTime: "1 hour",
        cost: "From $4/user/month",
      });
    }

    if (actionLower.includes("automation") || actionLower.includes("workflow")) {
      deeplinks.push({
        platform: "Zapier",
        category: "Automation",
        url: "https://zapier.com/",
        description: "Automate workflows between apps",
        setupTime: "1 hour",
        cost: "Free for 100 tasks/month",
      });
      deeplinks.push({
        platform: "Make (Integromat)",
        category: "Automation",
        url: "https://www.make.com/",
        description: "Visual automation platform",
        setupTime: "1 hour",
        cost: "Free for 1,000 operations/month",
      });
      deeplinks.push({
        platform: "n8n",
        category: "Automation",
        url: "https://n8n.io/",
        description: "Open-source workflow automation",
        setupTime: "2 hours",
        cost: "Free (self-hosted) or from $20/month",
      });
    }

    if (actionLower.includes("accounting") || actionLower.includes("finance")) {
      deeplinks.push({
        platform: "QuickBooks",
        category: "Accounting",
        url: "https://quickbooks.intuit.com/",
        description: "Small business accounting software",
        setupTime: "2 hours",
        cost: "From $30/month",
      });
      deeplinks.push({
        platform: "Xero",
        category: "Accounting",
        url: "https://www.xero.com/",
        description: "Cloud accounting platform",
        setupTime: "2 hours",
        cost: "From $13/month",
      });
    }

    if (actionLower.includes("inventory") || actionLower.includes("stock")) {
      deeplinks.push({
        platform: "Cin7",
        category: "Inventory Management",
        url: "https://www.cin7.com/",
        description: "Inventory and order management",
        setupTime: "3 hours",
        cost: "From $299/month",
      });
    }
  }

  // Technology recommendations
  if (recommendation.category === "technology") {
    if (actionLower.includes("website") || actionLower.includes("web")) {
      deeplinks.push({
        platform: "Webflow",
        category: "Website Builder",
        url: "https://webflow.com/",
        description: "Professional website builder",
        setupTime: "4 hours",
        cost: "From $14/month",
      });
      deeplinks.push({
        platform: "Wix",
        category: "Website Builder",
        url: "https://www.wix.com/",
        description: "Easy website builder",
        setupTime: "2 hours",
        cost: "From $16/month",
      });
    }

    if (actionLower.includes("cloud") || actionLower.includes("hosting")) {
      deeplinks.push({
        platform: "AWS",
        category: "Cloud Hosting",
        url: "https://aws.amazon.com/",
        description: "Enterprise cloud infrastructure",
        setupTime: "4 hours",
        cost: "Pay-as-you-go",
      });
      deeplinks.push({
        platform: "DigitalOcean",
        category: "Cloud Hosting",
        url: "https://www.digitalocean.com/",
        description: "Simple cloud hosting",
        setupTime: "2 hours",
        cost: "From $4/month",
      });
    }

    if (actionLower.includes("ai") || actionLower.includes("chatbot")) {
      deeplinks.push({
        platform: "OpenAI",
        category: "AI Platform",
        url: "https://platform.openai.com/",
        description: "GPT-4 API for AI integration",
        setupTime: "2 hours",
        cost: "Pay-per-use",
      });
      deeplinks.push({
        platform: "Intercom",
        category: "Customer Support",
        url: "https://www.intercom.com/",
        description: "AI-powered customer messaging",
        setupTime: "2 hours",
        cost: "From $74/month",
      });
    }

    if (actionLower.includes("security") || actionLower.includes("backup")) {
      deeplinks.push({
        platform: "1Password",
        category: "Security",
        url: "https://1password.com/",
        description: "Password management for teams",
        setupTime: "1 hour",
        cost: "From $7.99/user/month",
      });
      deeplinks.push({
        platform: "Backblaze",
        category: "Backup",
        url: "https://www.backblaze.com/",
        description: "Cloud backup solution",
        setupTime: "1 hour",
        cost: "From $7/month",
      });
    }
  }

  // Leadership recommendations
  if (recommendation.category === "leadership") {
    if (actionLower.includes("hiring") || actionLower.includes("recruitment")) {
      deeplinks.push({
        platform: "LinkedIn Recruiter",
        category: "Recruitment",
        url: "https://business.linkedin.com/talent-solutions/recruiter",
        description: "Professional recruitment platform",
        setupTime: "1 hour",
        cost: "From $170/month",
      });
      deeplinks.push({
        platform: "Indeed",
        category: "Job Posting",
        url: "https://employers.indeed.com/",
        description: "Post jobs and find candidates",
        setupTime: "30 minutes",
        cost: "Pay-per-click",
      });
    }

    if (actionLower.includes("training") || actionLower.includes("learning")) {
      deeplinks.push({
        platform: "Udemy for Business",
        category: "Training",
        url: "https://business.udemy.com/",
        description: "Employee training platform",
        setupTime: "1 hour",
        cost: "From $360/user/year",
      });
      deeplinks.push({
        platform: "LinkedIn Learning",
        category: "Training",
        url: "https://learning.linkedin.com/",
        description: "Professional development courses",
        setupTime: "30 minutes",
        cost: "From $29.99/month",
      });
    }

    if (actionLower.includes("performance") || actionLower.includes("review")) {
      deeplinks.push({
        platform: "Lattice",
        category: "Performance Management",
        url: "https://lattice.com/",
        description: "Employee performance and engagement",
        setupTime: "2 hours",
        cost: "From $11/user/month",
      });
    }
  }

  // If no specific deeplinks found, add general recommendations
  if (deeplinks.length === 0) {
    deeplinks.push({
      platform: "Google Search",
      category: "Research",
      url: `https://www.google.com/search?q=${encodeURIComponent(recommendation.action)}`,
      description: "Research this recommendation online",
      setupTime: "Variable",
      cost: "Free",
    });
  }

  return deeplinks;
}

/**
 * Calculate priority score (1-10) based on impact and difficulty
 */
function calculatePriority(recommendation: Recommendation): number {
  const impactScores = { high: 10, medium: 6, low: 3 };
  const difficultyScores = { easy: 3, medium: 2, hard: 1 };

  const impactScore = impactScores[recommendation.impact];
  const difficultyScore = difficultyScores[recommendation.difficulty];

  // Priority = (Impact * 0.7) + (Difficulty * 0.3)
  // Higher impact = higher priority, easier difficulty = higher priority
  return Math.round((impactScore * 0.7) + (difficultyScore * 0.3));
}
