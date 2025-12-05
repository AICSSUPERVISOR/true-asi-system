import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { ExternalLink, TrendingUp, Clock, DollarSign, Zap, Target } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

export interface DeeplinkAction {
  platform: string;
  category: string;
  url: string;
  description: string;
  setupTime?: string;
  cost?: string;
}

export interface Recommendation {
  id?: string;
  category: "revenue" | "marketing" | "leadership" | "operations" | "technology";
  action: string;
  impact: "high" | "medium" | "low";
  difficulty: "easy" | "medium" | "hard";
  roi: string;
  estimatedCost?: string;
  timeframe?: string;
  deeplinks: DeeplinkAction[];
  priority: number;
}

interface RecommendationCardProps {
  recommendation: Recommendation;
  onExecute?: (recommendation: Recommendation, deeplink: DeeplinkAction) => void;
}

export default function RecommendationCard({ recommendation, onExecute }: RecommendationCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const impactColors = {
    high: "bg-green-500/20 text-green-400 border-green-500/30",
    medium: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
    low: "bg-blue-500/20 text-blue-400 border-blue-500/30",
  };

  const difficultyColors = {
    easy: "bg-green-500/20 text-green-400 border-green-500/30",
    medium: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
    hard: "bg-red-500/20 text-red-400 border-red-500/30",
  };

  const categoryColors = {
    revenue: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
    marketing: "bg-purple-500/20 text-purple-400 border-purple-500/30",
    leadership: "bg-blue-500/20 text-blue-400 border-blue-500/30",
    operations: "bg-orange-500/20 text-orange-400 border-orange-500/30",
    technology: "bg-cyan-500/20 text-cyan-400 border-cyan-500/30",
  };

  const categoryIcons = {
    revenue: TrendingUp,
    marketing: Target,
    leadership: Zap,
    operations: Clock,
    technology: ExternalLink,
  };

  const CategoryIcon = categoryIcons[recommendation.category];

  const handleExecute = (deeplink: DeeplinkAction) => {
    if (onExecute) {
      onExecute(recommendation, deeplink);
    }

    // Open deeplink in new tab
    window.open(deeplink.url, "_blank", "noopener,noreferrer");
    
    toast.success(`Opening ${deeplink.platform}...`, {
      description: `Starting: ${deeplink.description}`,
    });
  };

  return (
    <Card className="bg-white/5 backdrop-blur-xl border-white/10 hover:border-white/20 transition-all duration-300 hover:scale-[1.02] hover:shadow-2xl hover:shadow-purple-500/10">
      <CardHeader>
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
              <CategoryIcon className="w-5 h-5 text-purple-400" />
              <Badge className={`${categoryColors[recommendation.category]} border`}>
                {recommendation.category.toUpperCase()}
              </Badge>
              <Badge className={`${impactColors[recommendation.impact]} border`}>
                {recommendation.impact.toUpperCase()} IMPACT
              </Badge>
              <Badge className={`${difficultyColors[recommendation.difficulty]} border`}>
                {recommendation.difficulty.toUpperCase()}
              </Badge>
            </div>
            <CardTitle className="text-xl font-bold text-white tracking-tight">
              {recommendation.action}
            </CardTitle>
          </div>
          <div className="text-right">
            <div className="text-3xl font-black text-transparent bg-clip-text bg-gradient-to-r from-green-400 to-emerald-600">
              {recommendation.roi}
            </div>
            <div className="text-xs text-muted-foreground">Expected ROI</div>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Metrics */}
        <div className="grid grid-cols-3 gap-3">
          <div className="bg-white/5 rounded-lg p-3 border border-white/10">
            <div className="flex items-center gap-2 text-muted-foreground text-xs mb-1">
              <TrendingUp className="w-3 h-3" />
              Priority
            </div>
            <div className="text-lg font-bold text-white">{recommendation.priority}/10</div>
          </div>
          
          {recommendation.estimatedCost && (
            <div className="bg-white/5 rounded-lg p-3 border border-white/10">
              <div className="flex items-center gap-2 text-muted-foreground text-xs mb-1">
                <DollarSign className="w-3 h-3" />
                Cost
              </div>
              <div className="text-lg font-bold text-white">{recommendation.estimatedCost}</div>
            </div>
          )}

          {recommendation.timeframe && (
            <div className="bg-white/5 rounded-lg p-3 border border-white/10">
              <div className="flex items-center gap-2 text-muted-foreground text-xs mb-1">
                <Clock className="w-3 h-3" />
                Timeframe
              </div>
              <div className="text-lg font-bold text-white">{recommendation.timeframe}</div>
            </div>
          )}
        </div>

        {/* Deeplinks */}
        {recommendation.deeplinks && recommendation.deeplinks.length > 0 && (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-semibold text-white">
                Execute with ({recommendation.deeplinks.length} platform{recommendation.deeplinks.length > 1 ? "s" : ""})
              </h4>
              {recommendation.deeplinks.length > 3 && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setIsExpanded(!isExpanded)}
                  className="text-xs text-purple-400 hover:text-purple-300"
                >
                  {isExpanded ? "Show less" : "Show all"}
                </Button>
              )}
            </div>

            <div className="space-y-2">
              {(isExpanded ? recommendation.deeplinks : recommendation.deeplinks.slice(0, 3)).map((deeplink, index) => (
                <div
                  key={index}
                  className="bg-white/5 rounded-lg p-3 border border-white/10 hover:border-purple-500/30 transition-all duration-200 group"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <h5 className="font-semibold text-white text-sm">{deeplink.platform}</h5>
                        <Badge variant="outline" className="text-xs border-white/20 text-muted-foreground">
                          {deeplink.category}
                        </Badge>
                      </div>
                      <p className="text-xs text-muted-foreground mb-2">{deeplink.description}</p>
                      <div className="flex items-center gap-4 text-xs text-muted-foreground">
                        {deeplink.setupTime && (
                          <div className="flex items-center gap-1">
                            <Clock className="w-3 h-3" />
                            {deeplink.setupTime}
                          </div>
                        )}
                        {deeplink.cost && (
                          <div className="flex items-center gap-1">
                            <DollarSign className="w-3 h-3" />
                            {deeplink.cost}
                          </div>
                        )}
                      </div>
                    </div>
                    <Button
                      size="sm"
                      onClick={() => handleExecute(deeplink)}
                      className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white border-0 shadow-lg shadow-purple-500/20 group-hover:shadow-purple-500/40 transition-all duration-200"
                    >
                      <ExternalLink className="w-4 h-4 mr-1" />
                      Execute
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* No deeplinks fallback */}
        {(!recommendation.deeplinks || recommendation.deeplinks.length === 0) && (
          <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-3">
            <p className="text-sm text-yellow-400">
              No automated execution available. Manual implementation required.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
