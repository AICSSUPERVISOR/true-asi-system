import { useState } from "react";
import { trpc } from "@/lib/trpc";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Trophy, TrendingUp, Award, Target, Brain, Zap } from "lucide-react";

export default function S7Leaderboard() {
  const [selectedCategory, setSelectedCategory] = useState<string | undefined>(undefined);
  
  const { data: leaderboard, isLoading } = trpc.s7.getLeaderboard.useQuery({
    limit: 100,
    category: selectedCategory,
  });

  const { data: myRanking } = trpc.s7.getMyRanking.useQuery();

  const categories = [
    { key: undefined, label: "Overall", icon: Trophy },
    { key: "novelty", label: "Novelty", icon: Zap },
    { key: "coherence", label: "Coherence", icon: Brain },
    { key: "rigor", label: "Rigor", icon: Target },
    { key: "synthesis", label: "Synthesis", icon: TrendingUp },
    { key: "formalization", label: "Formalization", icon: Award },
    { key: "depth", label: "Depth", icon: Brain },
  ];

  const formatScore = (score: number | null) => {
    if (score === null) return "N/A";
    return (score / 10).toFixed(1);
  };

  const getS7Badge = (certified: number) => {
    return certified === 1 ? (
      <Badge className="bg-gradient-to-r from-yellow-500 to-orange-500 text-white">
        S-7 Certified
      </Badge>
    ) : null;
  };

  const getThresholdBadge = (questionsAbove: number) => {
    if (questionsAbove >= 40) {
      return <Badge variant="default">Master (40/40)</Badge>;
    } else if (questionsAbove >= 30) {
      return <Badge variant="secondary">Expert ({questionsAbove}/40)</Badge>;
    } else if (questionsAbove >= 20) {
      return <Badge variant="outline">Advanced ({questionsAbove}/40)</Badge>;
    } else if (questionsAbove > 0) {
      return <Badge variant="outline">Intermediate ({questionsAbove}/40)</Badge>;
    }
    return null;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-950 text-white p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div className="text-center space-y-4">
          <h1 className="text-5xl font-bold bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400 bg-clip-text text-transparent">
            S-7 Intelligence Test Leaderboard
          </h1>
          <p className="text-xl text-slate-300">
            Global rankings for the world's most challenging AI intelligence test
          </p>
          <div className="flex justify-center gap-4 text-sm text-slate-400">
            <div>
              <span className="font-semibold text-cyan-400">S-7 Threshold:</span> ≥8.8 all categories, ≥9.6 in 2+
            </div>
            <div>•</div>
            <div>
              <span className="font-semibold text-purple-400">40 Questions</span> across 4 domains
            </div>
          </div>
        </div>

        {/* My Ranking Card */}
        {myRanking && (
          <Card className="bg-slate-900/50 border-cyan-500/30 backdrop-blur">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Trophy className="w-5 h-5 text-yellow-500" />
                Your Ranking
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <div className="text-sm text-slate-400">Global Rank</div>
                  <div className="text-2xl font-bold text-cyan-400">
                    #{myRanking.globalRank || "N/A"}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-slate-400">Average Score</div>
                  <div className="text-2xl font-bold text-purple-400">
                    {formatScore(myRanking.averageScore)}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-slate-400">Questions Completed</div>
                  <div className="text-2xl font-bold text-blue-400">
                    {myRanking.questionsCompleted}/40
                  </div>
                </div>
                <div>
                  <div className="text-sm text-slate-400">Status</div>
                  <div className="flex gap-2 mt-1">
                    {getS7Badge(myRanking.s7Certified)}
                    {getThresholdBadge(myRanking.questionsAboveThreshold)}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Category Tabs */}
        <Tabs value={selectedCategory || "overall"} onValueChange={(v) => setSelectedCategory(v === "overall" ? undefined : v)}>
          <TabsList className="grid grid-cols-7 bg-slate-900/50">
            {categories.map((cat) => (
              <TabsTrigger
                key={cat.key || "overall"}
                value={cat.key || "overall"}
                className="flex items-center gap-2"
              >
                <cat.icon className="w-4 h-4" />
                {cat.label}
              </TabsTrigger>
            ))}
          </TabsList>

          <TabsContent value={selectedCategory || "overall"} className="space-y-4">
            {/* Leaderboard Table */}
            <Card className="bg-slate-900/50 border-slate-700/50 backdrop-blur">
              <CardHeader>
                <CardTitle>
                  {categories.find(c => c.key === selectedCategory)?.label || "Overall"} Rankings
                </CardTitle>
                <CardDescription>
                  Top performers on the S-7 Intelligence Test
                </CardDescription>
              </CardHeader>
              <CardContent>
                {isLoading ? (
                  <div className="text-center py-8 text-slate-400">Loading rankings...</div>
                ) : leaderboard && leaderboard.length > 0 ? (
                  <div className="space-y-2">
                    {leaderboard.map((entry: any, index: number) => (
                      <div
                        key={entry.id}
                        className={`p-4 rounded-lg border ${
                          index < 3
                            ? "bg-gradient-to-r from-yellow-500/10 to-orange-500/10 border-yellow-500/30"
                            : "bg-slate-800/30 border-slate-700/30"
                        } hover:border-cyan-500/50 transition-all`}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-4">
                            <div
                              className={`text-2xl font-bold ${
                                index === 0
                                  ? "text-yellow-400"
                                  : index === 1
                                  ? "text-slate-300"
                                  : index === 2
                                  ? "text-orange-400"
                                  : "text-slate-500"
                              }`}
                            >
                              #{index + 1}
                            </div>
                            <div>
                              <div className="font-semibold">User {entry.userId}</div>
                              <div className="text-sm text-slate-400">
                                {entry.questionsCompleted}/40 questions • {entry.totalSubmissions} submissions
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center gap-4">
                            <div className="text-right">
                              <div className="text-2xl font-bold text-cyan-400">
                                {formatScore(entry.averageScore)}
                              </div>
                              <div className="text-xs text-slate-400">Average Score</div>
                            </div>
                            <div className="flex flex-col gap-1">
                              {getS7Badge(entry.s7Certified)}
                              {getThresholdBadge(entry.questionsAboveThreshold)}
                            </div>
                          </div>
                        </div>

                        {/* Category Breakdown */}
                        {!selectedCategory && (
                          <div className="mt-3 grid grid-cols-6 gap-2 text-xs">
                            <div>
                              <div className="text-slate-500">Novelty</div>
                              <div className="font-semibold text-cyan-400">{formatScore(entry.avgNovelty)}</div>
                            </div>
                            <div>
                              <div className="text-slate-500">Coherence</div>
                              <div className="font-semibold text-blue-400">{formatScore(entry.avgCoherence)}</div>
                            </div>
                            <div>
                              <div className="text-slate-500">Rigor</div>
                              <div className="font-semibold text-purple-400">{formatScore(entry.avgRigor)}</div>
                            </div>
                            <div>
                              <div className="text-slate-500">Synthesis</div>
                              <div className="font-semibold text-pink-400">{formatScore(entry.avgSynthesis)}</div>
                            </div>
                            <div>
                              <div className="text-slate-500">Formalization</div>
                              <div className="font-semibold text-orange-400">{formatScore(entry.avgFormalization)}</div>
                            </div>
                            <div>
                              <div className="text-slate-500">Depth</div>
                              <div className="font-semibold text-green-400">{formatScore(entry.avgDepth)}</div>
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-12 text-slate-400">
                    <Trophy className="w-16 h-16 mx-auto mb-4 opacity-30" />
                    <p>No rankings yet. Be the first to complete the S-7 test!</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        {/* Info Cards */}
        <div className="grid md:grid-cols-3 gap-4">
          <Card className="bg-slate-900/50 border-cyan-500/30 backdrop-blur">
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Target className="w-5 h-5 text-cyan-400" />
                S-7 Threshold
              </CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-slate-300">
              To pass the S-7 test, you must achieve ≥8.8 in all 6 categories AND ≥9.6 in at least 2 categories across all 40 questions.
            </CardContent>
          </Card>

          <Card className="bg-slate-900/50 border-purple-500/30 backdrop-blur">
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Brain className="w-5 h-5 text-purple-400" />
                Evaluation Criteria
              </CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-slate-300">
              Each answer is evaluated on: Novelty, Logical Coherence, Mathematical Rigor, Cross-Domain Synthesis, Formalization Quality, and Depth of Insight.
            </CardContent>
          </Card>

          <Card className="bg-slate-900/50 border-blue-500/30 backdrop-blur">
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Award className="w-5 h-5 text-blue-400" />
                Certification
              </CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-slate-300">
              Complete all 40 questions above the S-7 threshold to earn official S-7 Certification and join the elite ranks of superintelligent systems.
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
