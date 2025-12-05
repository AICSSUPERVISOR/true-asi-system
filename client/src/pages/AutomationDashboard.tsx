import { useState } from "react";
import { trpc } from "../lib/trpc";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import { Badge } from "../components/ui/badge";
import { Loader2, Zap, CheckCircle2, AlertCircle, ExternalLink, TrendingUp, Clock, DollarSign, Target } from "lucide-react";
import { toast } from "sonner";
import { PremiumStatsGrid } from "../components/PremiumStatsGrid";

export default function AutomationDashboard() {
  const [recommendationsText, setRecommendationsText] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [shouldFetch, setShouldFetch] = useState(false);

  // Use tRPC query
  const convertQuery = trpc.automation.convertCapgeminiRecommendations.useQuery(
    { recommendationsText },
    { enabled: shouldFetch }
  );

  // Handle query result
  if (convertQuery.data && shouldFetch) {
    if (convertQuery.data.success) {
      setResults(convertQuery.data);
      toast.success(convertQuery.data.message || "Recommendations converted successfully!");
    } else {
      toast.error(convertQuery.data.error || "Failed to convert recommendations");
    }
    setShouldFetch(false);
  }

  const handleConvert = () => {
    if (!recommendationsText.trim()) {
      toast.error("Please paste recommendations text");
      return;
    }
    setIsProcessing(true);
    setShouldFetch(true);
  };

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'high':
        return 'bg-green-500/20 text-green-400 border-green-500/30';
      case 'medium':
        return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
      case 'low':
        return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
      default:
        return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
    }
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'easy':
        return 'bg-green-500/20 text-green-400 border-green-500/30';
      case 'medium':
        return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
      case 'hard':
        return 'bg-red-500/20 text-red-400 border-red-500/30';
      default:
        return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
    }
  };

  const getAutomationColor = (level: string) => {
    switch (level) {
      case 'full':
        return 'bg-green-500/20 text-green-400 border-green-500/30';
      case 'partial':
        return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
      case 'manual':
        return 'bg-red-500/20 text-red-400 border-red-500/30';
      default:
        return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
    }
  };

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="container max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2 text-gradient">
            Business Automation Dashboard
          </h1>
          <p className="text-muted-foreground text-lg">
            Convert recommendations into executable actions with automatic platform integration
          </p>
        </div>

        {/* Input Section */}
        <Card className="card-glass mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Zap className="w-5 h-5 text-primary" />
              Paste Recommendations
            </CardTitle>
            <CardDescription>
              Paste your Capgemini-style recommendations below. The system will automatically map them to executable platforms.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <textarea
              value={recommendationsText}
              onChange={(e) => setRecommendationsText(e.target.value)}
              placeholder="OPERATIONS&#10;HIGH IMPACT&#10;MEDIUM&#10;Develop a comprehensive pricing strategy...&#10;&#10;OPERATIONS&#10;MEDIUM IMPACT&#10;EASY&#10;Implement a customer loyalty program..."
              className="w-full h-64 p-4 bg-background/50 border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary resize-none font-mono text-sm"
            />
            <div className="flex items-center justify-between mt-4">
              <p className="text-sm text-muted-foreground">
                {recommendationsText.split(/\n\n+/).filter(t => t.trim()).length} recommendations detected
              </p>
              <Button
                onClick={handleConvert}
                disabled={isProcessing || !recommendationsText.trim()}
                className="btn-primary"
              >
                {isProcessing ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Zap className="mr-2 h-4 w-4" />
                    Convert to Executable Actions
                  </>
                )}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Results Section */}
        {results && results.success && (
          <div className="space-y-8">
            {/* Statistics */}
            <PremiumStatsGrid
              stats={[
                {
                  title: "Total Recommendations",
                  value: results.stats.total.toString(),
                  icon: Target,
                  gradient: "from-blue-500 to-cyan-500",
                },
                {
                  title: "Automated",
                  value: (results.stats.fullyAutomated + results.stats.partiallyAutomated).toString(),
                  icon: CheckCircle2,
                  gradient: "from-green-500 to-emerald-500",
                },
                {
                  title: "Automation Coverage",
                  value: `${results.stats.coveragePercentage.toFixed(1)}%`,
                  icon: TrendingUp,
                  gradient: "from-purple-500 to-pink-500",
                },
                {
                  title: "Total Platforms",
                  value: results.stats.totalPlatforms.toString(),
                  icon: Zap,
                  gradient: "from-orange-500 to-red-500",
                },
              ]}
            />

            {/* Execution Plans */}
            <div>
              <h2 className="text-2xl font-bold mb-4">Execution Plans</h2>
              <div className="grid gap-6">
                {results.plans.map((plan: any, index: number) => {
                  const recommendation = results.recommendations[index];
                  return (
                    <Card key={plan.recommendationId} className="card-glass">
                      <CardHeader>
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <CardTitle className="text-xl mb-2">
                              {recommendation.title}
                            </CardTitle>
                            <div className="flex flex-wrap gap-2 mb-2">
                              <Badge className={getImpactColor(recommendation.impact)}>
                                {recommendation.impact} impact
                              </Badge>
                              <Badge className={getDifficultyColor(recommendation.difficulty)}>
                                {recommendation.difficulty}
                              </Badge>
                              <Badge className={getAutomationColor(plan.automationLevel)}>
                                {plan.automationLevel} automation
                              </Badge>
                              <Badge variant="outline">
                                Priority: {recommendation.priority}/10
                              </Badge>
                            </div>
                          </div>
                        </div>
                        <div className="grid grid-cols-3 gap-4 mt-4">
                          <div className="flex items-center gap-2 text-sm">
                            <Clock className="w-4 h-4 text-muted-foreground" />
                            <span>{plan.estimatedTime}</span>
                          </div>
                          <div className="flex items-center gap-2 text-sm">
                            <DollarSign className="w-4 h-4 text-muted-foreground" />
                            <span>{plan.totalCost}</span>
                          </div>
                          <div className="flex items-center gap-2 text-sm">
                            <TrendingUp className="w-4 h-4 text-muted-foreground" />
                            <span>{plan.expectedROI}</span>
                          </div>
                        </div>
                      </CardHeader>
                      <CardContent>
                        {/* Platforms */}
                        {plan.platforms.length > 0 && (
                          <div className="mb-6">
                            <h4 className="font-semibold mb-3 flex items-center gap-2">
                              <Zap className="w-4 h-4 text-primary" />
                              Recommended Platforms ({plan.platforms.length})
                            </h4>
                            <div className="grid gap-3">
                              {plan.platforms.map((platform: any) => (
                                <div
                                  key={platform.id}
                                  className="p-4 bg-background/50 border border-border rounded-lg hover:border-primary/50 transition-colors"
                                >
                                  <div className="flex items-start justify-between">
                                    <div className="flex-1">
                                      <h5 className="font-semibold mb-1">{platform.name}</h5>
                                      <p className="text-sm text-muted-foreground mb-2">
                                        {platform.description}
                                      </p>
                                      <div className="flex flex-wrap gap-2">
                                        <Badge variant="outline" className="text-xs">
                                          {platform.cost}
                                        </Badge>
                                        <Badge variant="outline" className="text-xs">
                                          {platform.setupTime}
                                        </Badge>
                                        <Badge variant="outline" className="text-xs">
                                          {platform.authType}
                                        </Badge>
                                      </div>
                                    </div>
                                    <Button
                                      size="sm"
                                      variant="outline"
                                      asChild
                                      className="ml-4"
                                    >
                                      <a href={platform.url} target="_blank" rel="noopener noreferrer">
                                        <ExternalLink className="w-4 h-4 mr-2" />
                                        Open
                                      </a>
                                    </Button>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Execution Steps */}
                        <div>
                          <h4 className="font-semibold mb-3 flex items-center gap-2">
                            <CheckCircle2 className="w-4 h-4 text-primary" />
                            Execution Steps ({plan.steps.length})
                          </h4>
                          <div className="space-y-3">
                            {plan.steps.map((step: any) => (
                              <div
                                key={step.stepNumber}
                                className="p-4 bg-background/50 border border-border rounded-lg"
                              >
                                <div className="flex items-start gap-3">
                                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center text-primary font-semibold">
                                    {step.stepNumber}
                                  </div>
                                  <div className="flex-1">
                                    <h5 className="font-semibold mb-1">{step.title}</h5>
                                    <p className="text-sm text-muted-foreground mb-2">
                                      {step.description}
                                    </p>
                                    <div className="flex items-center gap-4 text-xs text-muted-foreground mb-2">
                                      <span>⏱️ {step.estimatedTime}</span>
                                      <span>💰 {step.cost}</span>
                                      <span>
                                        {step.isAutomated ? (
                                          <span className="text-green-400">✅ Automated</span>
                                        ) : (
                                          <span className="text-yellow-400">⚠️ Manual</span>
                                        )}
                                      </span>
                                    </div>
                                    <ul className="text-sm space-y-1">
                                      {step.instructions.map((instruction: string, i: number) => (
                                        <li key={i} className="flex items-start gap-2">
                                          <span className="text-primary mt-1">•</span>
                                          <span className="text-muted-foreground">{instruction}</span>
                                        </li>
                                      ))}
                                    </ul>
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {/* Empty State */}
        {!results && (
          <Card className="card-glass">
            <CardContent className="py-12 text-center">
              <AlertCircle className="w-16 h-16 text-muted-foreground mx-auto mb-4" />
              <h3 className="text-xl font-semibold mb-2">No Recommendations Yet</h3>
              <p className="text-muted-foreground">
                Paste your recommendations above and click "Convert to Executable Actions" to get started.
              </p>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
