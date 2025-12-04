import { useState } from "react";
import { trpc } from "@/lib/trpc";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  TrendingUp, 
  TrendingDown, 
  Lightbulb, 
  Target, 
  Brain, 
  Zap,
  CheckCircle2,
  AlertCircle,
  ArrowRight
} from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

export default function S7Comparison() {
  const [selectedQuestion, setSelectedQuestion] = useState<number>(1);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentComparison, setCurrentComparison] = useState<any>(null);

  const { data: mySubmissions } = trpc.s7.getMySubmissions.useQuery();
  const { data: comparisons } = trpc.s7Comparison.getMyComparisons.useQuery({
    questionNumber: selectedQuestion,
  });
  
  const analyzeMutation = trpc.s7Comparison.analyzeGap.useMutation();

  const handleAnalyze = async () => {
    const submission = mySubmissions?.find(s => s.questionNumber === selectedQuestion);
    if (!submission) {
      alert("No submission found for this question");
      return;
    }

    setIsAnalyzing(true);
    try {
      const result = await analyzeMutation.mutateAsync({
        questionNumber: selectedQuestion,
        submissionId: submission.id,
      });
      setCurrentComparison(result);
    } catch (error: any) {
      alert(error.message || "Failed to generate analysis");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const formatScore = (score: number | null) => {
    if (score === null) return "N/A";
    return score.toFixed(1);
  };

  const formatGap = (gap: number) => {
    const absGap = Math.abs(gap);
    return gap > 0 ? `+${absGap.toFixed(1)}` : gap < 0 ? absGap.toFixed(1) : "0.0";
  };

  const getGapColor = (gap: number) => {
    if (gap > 2) return "text-red-400";
    if (gap > 1) return "text-orange-400";
    if (gap > 0.5) return "text-yellow-400";
    return "text-green-400";
  };

  const getGapIcon = (gap: number) => {
    return gap > 0 ? <TrendingDown className="w-4 h-4" /> : <CheckCircle2 className="w-4 h-4" />;
  };

  const categories = [
    { key: "novelty", label: "Novelty & Originality", icon: Zap, color: "cyan" },
    { key: "coherence", label: "Logical Coherence", icon: Brain, color: "blue" },
    { key: "rigor", label: "Mathematical Rigor", icon: Target, color: "purple" },
    { key: "synthesis", label: "Cross-Domain Synthesis", icon: TrendingUp, color: "pink" },
    { key: "formalization", label: "Formalization Quality", icon: CheckCircle2, color: "orange" },
    { key: "depth", label: "Depth of Insight", icon: Lightbulb, color: "green" },
  ];

  const comparison = currentComparison || comparisons?.[0];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-950 text-white p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div className="text-center space-y-4">
          <h1 className="text-5xl font-bold bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400 bg-clip-text text-transparent">
            S-7 Answer Comparison Tool
          </h1>
          <p className="text-xl text-slate-300">
            AI-powered gap analysis with actionable recommendations
          </p>
        </div>

        {/* Question Selector */}
        <Card className="bg-slate-900/50 border-slate-700/50 backdrop-blur">
          <CardHeader>
            <CardTitle>Select Question</CardTitle>
            <CardDescription>
              Choose a question to compare your answer with top performers
            </CardDescription>
          </CardHeader>
          <CardContent className="flex gap-4">
            <Select value={selectedQuestion.toString()} onValueChange={(v) => setSelectedQuestion(parseInt(v))}>
              <SelectTrigger className="w-[200px] bg-slate-800/50">
                <SelectValue placeholder="Select question" />
              </SelectTrigger>
              <SelectContent>
                {Array.from({ length: 40 }, (_, i) => i + 1).map(q => (
                  <SelectItem key={q} value={q.toString()}>
                    Question {q}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button
              onClick={handleAnalyze}
              disabled={isAnalyzing}
              className="bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-600 hover:to-blue-600"
            >
              {isAnalyzing ? (
                <>
                  <Brain className="w-4 h-4 mr-2 animate-pulse" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Target className="w-4 h-4 mr-2" />
                  Analyze Gap
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {comparison && (
          <>
            {/* Score Comparison Overview */}
            <div className="grid md:grid-cols-2 gap-6">
              <Card className="bg-slate-900/50 border-cyan-500/30 backdrop-blur">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <TrendingDown className="w-5 h-5 text-cyan-400" />
                    Your Score
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-5xl font-bold text-cyan-400">
                    {formatScore(comparison.userScore)}
                  </div>
                  <div className="text-sm text-slate-400 mt-2">Average across all categories</div>
                </CardContent>
              </Card>

              <Card className="bg-slate-900/50 border-purple-500/30 backdrop-blur">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <TrendingUp className="w-5 h-5 text-purple-400" />
                    Top Performer Score
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-5xl font-bold text-purple-400">
                    {formatScore(comparison.topScore)}
                  </div>
                  <div className="text-sm text-slate-400 mt-2">Target score to reach</div>
                </CardContent>
              </Card>
            </div>

            {/* Category Gaps */}
            <Card className="bg-slate-900/50 border-slate-700/50 backdrop-blur">
              <CardHeader>
                <CardTitle>Performance Gap Analysis</CardTitle>
                <CardDescription>
                  Detailed breakdown of scoring differences by category
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {categories.map(cat => {
                    const gap = comparison.gaps?.[cat.key] || 0;
                    const recommendation = comparison.recommendations?.[cat.key];
                    
                    return (
                      <div
                        key={cat.key}
                        className="p-4 rounded-lg bg-slate-800/30 border border-slate-700/30 hover:border-cyan-500/30 transition-all"
                      >
                        <div className="flex items-center justify-between mb-3">
                          <div className="flex items-center gap-3">
                            <div className={`p-2 rounded-lg bg-${cat.color}-500/20`}>
                              <cat.icon className={`w-5 h-5 text-${cat.color}-400`} />
                            </div>
                            <div>
                              <div className="font-semibold">{cat.label}</div>
                              <div className="text-xs text-slate-400">Gap to close</div>
                            </div>
                          </div>
                          <div className="flex items-center gap-2">
                            <div className={`text-2xl font-bold ${getGapColor(gap)}`}>
                              {formatGap(gap)}
                            </div>
                            <div className={getGapColor(gap)}>
                              {getGapIcon(gap)}
                            </div>
                          </div>
                        </div>

                        {recommendation && (
                          <div className="mt-3 p-3 bg-slate-950/50 rounded border border-slate-700/30">
                            <div className="flex items-start gap-2">
                              <Lightbulb className="w-4 h-4 text-yellow-400 mt-1 flex-shrink-0" />
                              <div className="text-sm text-slate-300">{recommendation}</div>
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>

            {/* Overall Analysis */}
            {comparison.recommendations?.overall && (
              <Card className="bg-gradient-to-br from-blue-900/30 to-purple-900/30 border-blue-500/30 backdrop-blur">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Brain className="w-5 h-5 text-blue-400" />
                    Overall Analysis
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-slate-200 leading-relaxed">
                    {comparison.recommendations.overall}
                  </p>
                </CardContent>
              </Card>
            )}

            {/* Action Steps */}
            <Card className="bg-slate-900/50 border-green-500/30 backdrop-blur">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <CheckCircle2 className="w-5 h-5 text-green-400" />
                  Next Steps
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-start gap-3">
                    <div className="p-1 rounded-full bg-green-500/20 mt-1">
                      <ArrowRight className="w-4 h-4 text-green-400" />
                    </div>
                    <div>
                      <div className="font-semibold text-green-400">Focus on High-Gap Categories</div>
                      <div className="text-sm text-slate-300">
                        Prioritize improving categories with gaps &gt;2.0 points for maximum impact
                      </div>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="p-1 rounded-full bg-blue-500/20 mt-1">
                      <ArrowRight className="w-4 h-4 text-blue-400" />
                    </div>
                    <div>
                      <div className="font-semibold text-blue-400">Study Top-Ranked Answers</div>
                      <div className="text-sm text-slate-300">
                        Review the top-ranked answer to understand what makes it exceptional
                      </div>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="p-1 rounded-full bg-purple-500/20 mt-1">
                      <ArrowRight className="w-4 h-4 text-purple-400" />
                    </div>
                    <div>
                      <div className="font-semibold text-purple-400">Revise and Resubmit</div>
                      <div className="text-sm text-slate-300">
                        Apply the recommendations and submit an improved answer to track progress
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </>
        )}

        {/* Empty State */}
        {!comparison && !isAnalyzing && (
          <Card className="bg-slate-900/50 border-slate-700/50 backdrop-blur">
            <CardContent className="py-12 text-center">
              <AlertCircle className="w-16 h-16 mx-auto mb-4 text-slate-500" />
              <p className="text-slate-400">
                Select a question and click "Analyze Gap" to compare your answer with top performers
              </p>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
