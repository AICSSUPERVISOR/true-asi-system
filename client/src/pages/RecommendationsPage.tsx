import { useEffect, useState } from "react";
import { useParams, useLocation } from "wouter";
import { trpc } from "../lib/trpc";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Loader2, Building2, TrendingUp, Target, Zap, ArrowLeft, Filter } from "lucide-react";
import { toast } from "sonner";
import RecommendationCard from "../components/RecommendationCard";
import type { Recommendation, DeeplinkAction } from "../components/RecommendationCard";
import { useWebSocketEvent } from "../contexts/WebSocketProvider";

export default function RecommendationsPage() {
  const { companyId } = useParams<{ companyId: string }>();
  const [, setLocation] = useLocation();
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [loadingStep, setLoadingStep] = useState(1);
  const [loadingMessage, setLoadingMessage] = useState("Fetching company data from Brreg.no...");

  // Real AI analysis integration
  const runAnalysisMutation = trpc.businessOrchestrator.runCompleteAnalysis.useMutation({
    onSuccess: (data) => {
      console.log("[RecommendationsPage] Analysis complete:", data);
      
      // Convert AI analysis to Recommendation format
      const convertedRecommendations: Recommendation[] = data.recommendations.map((rec: any, index: number) => ({
        id: `rec_${index}`,
        category: rec.category || "operations",
        action: rec.action || rec.recommendation,
        impact: rec.impact || "medium",
        difficulty: rec.difficulty || "medium",
        roi: rec.roi || "+25%",
        estimatedCost: rec.estimatedCost || "$100-500/mo",
        timeframe: rec.timeframe || "2-4 weeks",
        priority: rec.priority || 5,
        deeplinks: rec.deeplinks || [],
      }));
      
      setRecommendations(convertedRecommendations);
      setIsLoading(false);
      toast.success(`Generated ${convertedRecommendations.length} recommendations!`);
    },
    onError: (error) => {
      console.error("[RecommendationsPage] Analysis failed:", error);
      toast.error("Failed to generate recommendations. Showing example data.");
      
      // Fallback to mock data on error
      setTimeout(() => {
      setRecommendations([
        {
          id: "1",
          category: "revenue",
          action: "Implement CRM system to track all customer interactions and sales pipeline",
          impact: "high",
          difficulty: "medium",
          roi: "+35%",
          estimatedCost: "$50-200/mo",
          timeframe: "2-4 weeks",
          priority: 9,
          deeplinks: [
            {
              platform: "HubSpot CRM",
              category: "CRM",
              url: "https://www.hubspot.com/products/crm",
              description: "Free CRM with sales pipeline management",
              setupTime: "1 hour",
              cost: "Free",
            },
            {
              platform: "Salesforce",
              category: "CRM",
              url: "https://www.salesforce.com/form/signup/freetrial-sales/",
              description: "Enterprise CRM for sales and customer management",
              setupTime: "2 hours",
              cost: "From $25/user/month",
            },
          ],
        },
        {
          id: "2",
          category: "marketing",
          action: "Launch targeted LinkedIn advertising campaign to reach decision-makers in your industry",
          impact: "high",
          difficulty: "easy",
          roi: "+45%",
          estimatedCost: "$300-1000/mo",
          timeframe: "1-2 weeks",
          priority: 10,
          deeplinks: [
            {
              platform: "LinkedIn Ads",
              category: "Marketing",
              url: "https://www.linkedin.com/campaignmanager/",
              description: "Create targeted LinkedIn advertising campaigns",
              setupTime: "30 minutes",
              cost: "From $10/day",
            },
          ],
        },
        {
          id: "3",
          category: "operations",
          action: "Automate repetitive tasks with workflow automation tools to save 10+ hours per week",
          impact: "medium",
          difficulty: "easy",
          roi: "+25%",
          estimatedCost: "$0-50/mo",
          timeframe: "1 week",
          priority: 8,
          deeplinks: [
            {
              platform: "Zapier",
              category: "Automation",
              url: "https://zapier.com/",
              description: "Automate workflows between apps",
              setupTime: "1 hour",
              cost: "Free for 100 tasks/month",
            },
            {
              platform: "Make (Integromat)",
              category: "Automation",
              url: "https://www.make.com/",
              description: "Visual automation platform",
              setupTime: "1 hour",
              cost: "Free for 1,000 operations/month",
            },
          ],
        },
        {
          id: "4",
          category: "technology",
          action: "Improve website SEO to increase organic traffic by 50%",
          impact: "high",
          difficulty: "medium",
          roi: "+40%",
          estimatedCost: "$0-120/mo",
          timeframe: "4-8 weeks",
          priority: 9,
          deeplinks: [
            {
              platform: "Google Search Console",
              category: "SEO",
              url: "https://search.google.com/search-console/",
              description: "Monitor and optimize Google search presence",
              setupTime: "15 minutes",
              cost: "Free",
            },
            {
              platform: "SEMrush",
              category: "SEO Tools",
              url: "https://www.semrush.com/",
              description: "Comprehensive SEO and competitor analysis",
              setupTime: "30 minutes",
              cost: "From $119.95/month",
            },
          ],
        },
        {
          id: "5",
          category: "leadership",
          action: "Implement employee training program to improve skills and retention",
          impact: "medium",
          difficulty: "medium",
          roi: "+20%",
          estimatedCost: "$30-360/user/year",
          timeframe: "Ongoing",
          priority: 7,
          deeplinks: [
            {
              platform: "LinkedIn Learning",
              category: "Training",
              url: "https://learning.linkedin.com/",
              description: "Professional development courses",
              setupTime: "30 minutes",
              cost: "From $29.99/month",
            },
            {
              platform: "Udemy for Business",
              category: "Training",
              url: "https://business.udemy.com/",
              description: "Employee training platform",
              setupTime: "1 hour",
              cost: "From $360/user/year",
            },
          ],
        },
      ]);
      setIsLoading(false);
    }, 1500);
    },
  });

  // Listen for WebSocket progress updates
  useWebSocketEvent("analysis:progress", (data: any) => {
    console.log("[RecommendationsPage] Progress update:", data);
    if (data.companyId === companyId) {
      setLoadingStep(data.step);
      setLoadingMessage(data.message);
    }
  });

  // Listen for analysis completion
  useWebSocketEvent("analysis:complete", (data: any) => {
    console.log("[RecommendationsPage] Analysis complete:", data);
    if (data.companyId === companyId) {
      toast.success("Analysis complete! Generating recommendations...");
    }
  });

  // Trigger analysis on mount
  useEffect(() => {
    if (!companyId) return;
    
    // Start analysis
    runAnalysisMutation.mutate({
      companyId,
      orgnr: new URLSearchParams(window.location.search).get("orgnr") || "",
    });
  }, [companyId]);

  // Execution tracking mutation
  const trackExecutionMutation = trpc.executionTracking.trackExecution.useMutation({
    onSuccess: () => {
      toast.success("Execution tracked! Opening platform...");
    },
    onError: (error) => {
      console.error("[RecommendationsPage] Failed to track execution:", error);
      toast.error("Failed to track execution");
    },
  });

  const handleExecute = (recommendation: Recommendation, deeplink: DeeplinkAction) => {
    if (!companyId) {
      toast.error("Company ID not found");
      return;
    }
    
    console.log("Executing:", recommendation.action, "via", deeplink.platform);
    
    // Track execution
    trackExecutionMutation.mutate({
      companyId,
      recommendationId: recommendation.id || `rec_${Date.now()}`,
      recommendationAction: recommendation.action,
      recommendationCategory: recommendation.category as any,
      deeplinkPlatform: deeplink.platform,
      deeplinkUrl: deeplink.url,
    });
    
    // Open deeplink in new tab
    window.open(deeplink.url, "_blank");
  };

  const filteredRecommendations = selectedCategory
    ? recommendations.filter((rec) => rec.category === selectedCategory)
    : recommendations;

  const categories = [
    { value: "revenue", label: "Revenue", icon: TrendingUp, color: "emerald" },
    { value: "marketing", label: "Marketing", icon: Target, color: "purple" },
    { value: "operations", label: "Operations", icon: Zap, color: "orange" },
    { value: "technology", label: "Technology", icon: Building2, color: "cyan" },
    { value: "leadership", label: "Leadership", icon: Building2, color: "blue" },
  ];

  const highPriorityCount = recommendations.filter((r) => r.priority >= 8).length;
  const totalROI = recommendations.reduce((sum, rec) => {
    const roiMatch = rec.roi.match(/\+(\d+)%/);
    return sum + (roiMatch ? parseInt(roiMatch[1]) : 0);
  }, 0);

  if (isLoading) {
    const steps = [
      { id: 1, label: "Fetching company data", sublabel: "Brreg.no API" },
      { id: 2, label: "Analyzing financials", sublabel: "Proff.no data" },
      { id: 3, label: "Scraping website", sublabel: "AI-powered analysis" },
      { id: 4, label: "Gathering social data", sublabel: "LinkedIn integration" },
      { id: 5, label: "Running AI consensus", sublabel: "5 models in parallel" },
    ];

    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-950 via-purple-950 to-slate-950 flex items-center justify-center">
        <div className="text-center max-w-2xl mx-auto px-4">
          <Loader2 className="w-16 h-16 animate-spin text-purple-400 mx-auto mb-6" />
          <h2 className="text-3xl font-black text-white mb-2">Analyzing Your Business</h2>
          <p className="text-muted-foreground text-lg mb-8">Running multi-model AI consensus</p>
          
          {/* 5-Step Progress */}
          <div className="space-y-4">
            {steps.map((step) => (
              <div
                key={step.id}
                className={`flex items-center gap-4 p-4 rounded-lg transition-all duration-300 ${
                  step.id <= loadingStep
                    ? "bg-white/10 backdrop-blur-xl border-white/20"
                    : "bg-white/5 backdrop-blur-sm border-white/10"
                } border`}
              >
                <div
                  className={`w-10 h-10 rounded-full flex items-center justify-center font-bold ${
                    step.id < loadingStep
                      ? "bg-green-500 text-white"
                      : step.id === loadingStep
                      ? "bg-purple-500 text-white animate-pulse"
                      : "bg-white/10 text-muted-foreground"
                  }`}
                >
                  {step.id < loadingStep ? "✓" : step.id}
                </div>
                <div className="text-left flex-1">
                  <p className="text-white font-semibold">{step.label}</p>
                  <p className="text-muted-foreground text-sm">{step.sublabel}</p>
                </div>
                {step.id === loadingStep && (
                  <Loader2 className="w-5 h-5 animate-spin text-purple-400" />
                )}
              </div>
            ))}
          </div>
          
          <p className="text-muted-foreground text-sm mt-6">This usually takes 30-60 seconds</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-purple-950 to-slate-950">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <Button
            variant="ghost"
            onClick={() => setLocation("/company-lookup")}
            className="text-white hover:text-purple-400 mb-4"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Company Lookup
          </Button>

          <div className="flex items-start justify-between">
            <div>
              <h1 className="text-5xl font-black text-white tracking-tight mb-2">
                AI-Powered Recommendations
              </h1>
              <p className="text-xl text-muted-foreground">
                {recommendations.length} actionable strategies to grow your business
              </p>
            </div>

            <div className="text-right">
              <div className="text-4xl font-black text-transparent bg-clip-text bg-gradient-to-r from-green-400 to-emerald-600">
                +{totalROI}%
              </div>
              <div className="text-sm text-muted-foreground">Combined ROI Potential</div>
            </div>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <Card className="bg-white/5 backdrop-blur-xl border-white/10">
            <CardHeader>
              <CardTitle className="text-white text-lg">Total Recommendations</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-4xl font-black text-white">{recommendations.length}</div>
            </CardContent>
          </Card>

          <Card className="bg-white/5 backdrop-blur-xl border-white/10">
            <CardHeader>
              <CardTitle className="text-white text-lg">High Priority</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-4xl font-black text-green-400">{highPriorityCount}</div>
            </CardContent>
          </Card>

          <Card className="bg-white/5 backdrop-blur-xl border-white/10">
            <CardHeader>
              <CardTitle className="text-white text-lg">Avg. Priority</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-4xl font-black text-purple-400">
                {(recommendations.reduce((sum, r) => sum + r.priority, 0) / recommendations.length).toFixed(1)}/10
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Category Filter */}
        <div className="mb-6">
          <div className="flex items-center gap-2 mb-3">
            <Filter className="w-5 h-5 text-white" />
            <h3 className="text-lg font-semibold text-white">Filter by Category</h3>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button
              variant={selectedCategory === null ? "default" : "outline"}
              onClick={() => setSelectedCategory(null)}
              className={selectedCategory === null ? "bg-purple-600 hover:bg-purple-700" : ""}
            >
              All ({recommendations.length})
            </Button>
            {categories.map((cat) => (
              <Button
                key={cat.value}
                variant={selectedCategory === cat.value ? "default" : "outline"}
                onClick={() => setSelectedCategory(cat.value)}
                className={selectedCategory === cat.value ? `bg-${cat.color}-600 hover:bg-${cat.color}-700` : ""}
              >
                <cat.icon className="w-4 h-4 mr-2" />
                {cat.label} ({recommendations.filter((r) => r.category === cat.value).length})
              </Button>
            ))}
          </div>
        </div>

        {/* Recommendations Grid */}
        <div className="grid grid-cols-1 gap-6">
          {filteredRecommendations.map((recommendation) => (
            <RecommendationCard
              key={recommendation.id}
              recommendation={recommendation}
              onExecute={handleExecute}
            />
          ))}
        </div>

        {filteredRecommendations.length === 0 && (
          <div className="text-center py-12">
            <p className="text-muted-foreground text-lg">No recommendations in this category</p>
          </div>
        )}
      </div>
    </div>
  );
}
