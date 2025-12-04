/**
 * LinkedIn integration using Manus Data API Hub
 */

import { callDataApi } from "./dataApi";

export interface LinkedInCompanyProfile {
  name: string;
  universalName: string;
  linkedinUrl: string;
  tagline: string;
  description: string;
  website: string;
  phone: string;
  staffCount: number;
  staffCountRange: string;
  followerCount: number;
  industries: string[];
  specialities: string[];
  crunchbaseUrl?: string;
  type: string;
}

export interface LinkedInPersonProfile {
  id: string;
  username: string;
  firstName: string;
  lastName: string;
  headline: string;
  summary: string;
  location: string;
  profilePicture: string;
  isTopVoice: boolean;
  isCreator: boolean;
  isPremium: boolean;
  position: Array<{
    title: string;
    companyName: string;
    start: { year: number; month: number };
    end: { year: number; month: number };
    description: string;
  }>;
  educations: Array<{
    schoolName: string;
    degree: string;
    fieldOfStudy: string;
    start: { year: number };
    end: { year: number };
  }>;
  skills: Array<{
    name: string;
    endorsementsCount: number;
  }>;
}

export interface LinkedInSearchResult {
  fullName: string;
  headline: string;
  location: string;
  profileURL: string;
  username: string;
  summary?: string;
  profilePicture?: string;
}

/**
 * Get LinkedIn company profile by username
 */
export async function getLinkedInCompany(
  username: string
): Promise<LinkedInCompanyProfile | null> {
  try {
    const response = await callDataApi("LinkedIn/get_company_details", {
      query: { username },
    }) as any;

    if (!response.success || !response.data) {
      console.error("LinkedIn company lookup failed:", response);
      return null;
    }

    return response.data as LinkedInCompanyProfile;
  } catch (error) {
    console.error("Error fetching LinkedIn company:", error);
    return null;
  }
}

/**
 * Get LinkedIn person profile by username
 */
export async function getLinkedInPerson(
  username: string
): Promise<LinkedInPersonProfile | null> {
  try {
    const response = await callDataApi("LinkedIn/get_user_profile_by_username", {
      query: { username },
    }) as any;

    // LinkedIn API returns profile data directly (not wrapped in success/data)
    if (response.id && response.username) {
      return response as LinkedInPersonProfile;
    }

    console.error("LinkedIn person lookup failed:", response);
    return null;
  } catch (error) {
    console.error("Error fetching LinkedIn person:", error);
    return null;
  }
}

/**
 * Search for people on LinkedIn
 */
export async function searchLinkedInPeople(params: {
  keywords?: string;
  firstName?: string;
  lastName?: string;
  company?: string;
  keywordTitle?: string;
  keywordSchool?: string;
  start?: string;
}): Promise<LinkedInSearchResult[]> {
  try {
    const response = await callDataApi("LinkedIn/search_people", {
      query: params,
    }) as any;

    if (!response.success || !response.data) {
      console.error("LinkedIn people search failed:", response);
      return [];
    }

    return (response.data.items || []) as LinkedInSearchResult[];
  } catch (error) {
    console.error("Error searching LinkedIn people:", error);
    return [];
  }
}

/**
 * Get all employees of a company from LinkedIn
 */
export async function getCompanyEmployees(
  companyName: string,
  maxResults: number = 50
): Promise<LinkedInSearchResult[]> {
  const allEmployees: LinkedInSearchResult[] = [];
  let start = 0;

  while (allEmployees.length < maxResults) {
    const employees = await searchLinkedInPeople({
      company: companyName,
      start: start.toString(),
    });

    if (employees.length === 0) break;

    allEmployees.push(...employees);
    start += 10; // LinkedIn returns 10 results per page

    // Avoid rate limiting
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }

  return allEmployees.slice(0, maxResults);
}

/**
 * Extract LinkedIn username from company website or search
 */
export async function findCompanyLinkedInUsername(
  companyName: string
): Promise<string | null> {
  // Try searching for the company
  const searchResults = await searchLinkedInPeople({
    company: companyName,
    start: "0",
  });

  if (searchResults.length > 0) {
    // Extract company username from first employee's profile URL
    const profileURL = searchResults[0].profileURL;
    const match = profileURL.match(/linkedin\.com\/in\/([^/]+)/);
    if (match) {
      // This is a person's username, we need to search for company page
      // For now, return the company name in lowercase with hyphens
      return companyName.toLowerCase().replace(/\s+/g, "-");
    }
  }

  // Fallback: convert company name to likely LinkedIn username format
  return companyName.toLowerCase().replace(/\s+/g, "-").replace(/[^a-z0-9-]/g, "");
}

/**
 * Analyze LinkedIn profile for improvement opportunities
 */
export function analyzeLinkedInProfile(profile: LinkedInPersonProfile): {
  score: number;
  improvements: string[];
} {
  const improvements: string[] = [];
  let score = 100;

  // Check headline
  if (!profile.headline || profile.headline.length < 50) {
    improvements.push("Headline is too short - should be 50-120 characters with keywords");
    score -= 10;
  }

  // Check summary
  if (!profile.summary || profile.summary.length < 200) {
    improvements.push("Summary is missing or too short - should be 200-2000 characters");
    score -= 15;
  }

  // Check profile picture
  if (!profile.profilePicture) {
    improvements.push("Missing professional profile picture");
    score -= 10;
  }

  // Check skills
  if (!profile.skills || profile.skills.length < 5) {
    improvements.push("Add more skills (aim for 10-50 relevant skills)");
    score -= 10;
  }

  // Check endorsements
  const totalEndorsements = profile.skills?.reduce(
    (sum, skill) => sum + skill.endorsementsCount,
    0
  ) || 0;
  if (totalEndorsements < 10) {
    improvements.push("Request more skill endorsements from connections");
    score -= 5;
  }

  // Check experience
  if (!profile.position || profile.position.length === 0) {
    improvements.push("Add work experience with detailed descriptions");
    score -= 20;
  } else {
    const hasDescriptions = profile.position.some((pos) => pos.description);
    if (!hasDescriptions) {
      improvements.push("Add detailed descriptions to work experience");
      score -= 10;
    }
  }

  // Check education
  if (!profile.educations || profile.educations.length === 0) {
    improvements.push("Add education history");
    score -= 10;
  }

  // Check premium status
  if (!profile.isPremium) {
    improvements.push("Consider LinkedIn Premium for better visibility and InMail credits");
    score -= 5;
  }

  return {
    score: Math.max(0, score),
    improvements,
  };
}

/**
 * Generate LinkedIn profile optimization recommendations
 */
export function generateLinkedInOptimizations(
  profile: LinkedInPersonProfile,
  industry: string
): {
  headline: string;
  summary: string;
  skills: string[];
  contentStrategy: string[];
} {
  const currentRole = profile.position?.[0]?.title || "Professional";
  const company = profile.position?.[0]?.companyName || "";

  return {
    headline: `${currentRole} at ${company} | ${industry} Expert | Driving Innovation & Growth`,
    summary: `Experienced ${currentRole} with a proven track record in ${industry}. 

${profile.summary || "Passionate about delivering exceptional results and driving business growth through innovative solutions."}

Specialties: ${profile.skills?.slice(0, 5).map((s) => s.name).join(", ") || "Leadership, Strategy, Innovation"}

Open to: Networking, Collaboration, Speaking Opportunities

Let's connect!`,
    skills: [
      // Industry-specific skills based on category
      ...(industry === "Technology"
        ? ["Cloud Computing", "AI/ML", "DevOps", "Agile", "Software Architecture"]
        : industry === "Finance"
        ? ["Financial Analysis", "Risk Management", "Investment Strategy", "Portfolio Management", "Compliance"]
        : industry === "Healthcare"
        ? ["Patient Care", "Healthcare Management", "Medical Technology", "Clinical Research", "Health Policy"]
        : industry === "Retail"
        ? ["Customer Experience", "Inventory Management", "E-commerce", "Merchandising", "Supply Chain"]
        : ["Leadership", "Strategy", "Project Management", "Business Development", "Team Building"]),
    ],
    contentStrategy: [
      "Post 2-3 times per week about industry trends and insights",
      "Share company achievements and milestones",
      "Engage with connections' posts (like, comment, share)",
      "Publish long-form articles monthly on LinkedIn",
      "Use relevant hashtags (#" + industry.replace(/\s+/g, "") + ", #Leadership, #Innovation)",
      "Share case studies and success stories",
      "Participate in LinkedIn groups related to " + industry,
      "Host or attend LinkedIn Live sessions",
    ],
  };
}
