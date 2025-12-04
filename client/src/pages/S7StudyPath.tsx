import { useState, useEffect } from "react";
import { trpc } from "@/lib/trpc";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { 
  BookOpen, 
  Target, 
  TrendingUp, 
  CheckCircle2, 
  Clock,
  Brain,
  Lightbulb,
  Award,
  ArrowRight,
  FileText
} from "lucide-react";

export default function S7StudyPath() {
  const [isGenerating, setIsGenerating] = useState(false);
  const [studyPlan, setStudyPlan] = useState<any>(null);

  const { data: mySubmissions } = trpc.s7.getMySubmissions.useQuery();
  const { data: myRanking } = trpc.s7.getMyRanking.useQuery();

  // Analyze weak categories from submissions
  const analyzeWeaknesses = () => {
    if (!myRanking) return null;

    const categories = [
      { name: "Novelty", score: (myRanking.avgNovelty || 0) / 10, key: "novelty" },
      { name: "Coherence", score: (myRanking.avgCoherence || 0) / 10, key: "coherence" },
      { name: "Rigor", score: (myRanking.avgRigor || 0) / 10, key: "rigor" },
      { name: "Synthesis", score: (myRanking.avgSynthesis || 0) / 10, key: "synthesis" },
      { name: "Formalization", score: (myRanking.avgFormalization || 0) / 10, key: "formalization" },
      { name: "Depth", score: (myRanking.avgDepth || 0) / 10, key: "depth" },
    ];

    const sorted = [...categories].sort((a, b) => a.score - b.score);
    return sorted[0]; // Weakest category
  };

  const generateStudyPlan = async () => {
    setIsGenerating(true);
    
    // Simulate AI-powered study plan generation
    await new Promise(resolve => setTimeout(resolve, 2000));

    const weakness = analyzeWeaknesses();
    
    const plan = {
      weakestCategory: weakness?.name || "Novelty",
      currentScore: weakness?.score || 7.5,
      targetScore: 9.6,
      gap: (9.6 - (weakness?.score || 7.5)).toFixed(1),
      estimatedDays: Math.ceil((9.6 - (weakness?.score || 7.5)) * 30),
      milestones: [
        {
          id: 1,
          title: "Foundation Building",
          description: "Master fundamental concepts and terminology",
          duration: "Week 1-2",
          completed: false,
          resources: [
            "Research Paper: Foundations of Mathematical Reasoning",
            "Practice: 10 beginner-level questions",
            "Reading: Introduction to Formal Logic"
          ]
        },
        {
          id: 2,
          title: "Intermediate Concepts",
          description: "Apply concepts to moderate complexity problems",
          duration: "Week 3-4",
          completed: false,
          resources: [
            "Research Paper: Advanced Problem-Solving Techniques",
            "Practice: 15 intermediate questions",
            "Reading: Cross-Domain Synthesis Methods"
          ]
        },
        {
          id: 3,
          title: "Advanced Techniques",
          description: "Master advanced strategies and edge cases",
          duration: "Week 5-6",
          completed: false,
          resources: [
            "Research Paper: Novel Approaches to Complex Systems",
            "Practice: 20 advanced questions",
            "Reading: Formalization in Abstract Reasoning"
          ]
        },
        {
          id: 4,
          title: "Expert Mastery",
          description: "Achieve S-7 threshold performance",
          duration: "Week 7-8",
          completed: false,
          resources: [
            "Research Paper: Superintelligent Problem Solving",
            "Practice: 25 expert-level questions",
            "Reading: Depth of Insight in AI Systems"
          ]
        },
      ],
      practiceSchedule: [
        { day: "Monday", focus: "Theory & Concepts", duration: "2 hours" },
        { day: "Tuesday", focus: "Practice Problems", duration: "3 hours" },
        { day: "Wednesday", focus: "Review & Reflection", duration: "1 hour" },
        { day: "Thursday", focus: "Advanced Techniques", duration: "2 hours" },
        { day: "Friday", focus: "Mock Submissions", duration: "3 hours" },
        { day: "Saturday", focus: "Research Papers", duration: "2 hours" },
        { day: "Sunday", focus: "Rest & Integration", duration: "1 hour" },
      ]
    };

    setStudyPlan(plan);
    setIsGenerating(false);
  };

  const progress = studyPlan ? (studyPlan.milestones.filter((m: any) => m.completed).length / studyPlan.milestones.length) * 100 : 0;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-950 text-white p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div className="text-center space-y-4">
          <h1 className="text-5xl font-bold bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400 bg-clip-text text-transparent">
            S-7 Study Path Generator
          </h1>
          <p className="text-xl text-slate-300">
            Personalized learning path to reach S-7 certification threshold
          </p>
        </div>

        {/* Current Performance */}
        {myRanking && (
          <div className="grid md:grid-cols-3 gap-4">
            <Card className="bg-slate-900/50 border-cyan-500/30 backdrop-blur">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium text-slate-400">Current Average</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-cyan-400">
                  {((myRanking.averageScore || 0) / 10).toFixed(1)}
                </div>
                <div className="text-xs text-slate-400 mt-1">Across all categories</div>
              </CardContent>
            </Card>

            <Card className="bg-slate-900/50 border-green-500/30 backdrop-blur">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium text-slate-400">S-7 Threshold</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-green-400">9.6</div>
                <div className="text-xs text-slate-400 mt-1">Target score to reach</div>
              </CardContent>
            </Card>

            <Card className="bg-slate-900/50 border-purple-500/30 backdrop-blur">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium text-slate-400">Questions Completed</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-purple-400">
                  {myRanking.questionsCompleted}/40
                </div>
                <div className="text-xs text-slate-400 mt-1">
                  {myRanking.questionsAboveThreshold} above threshold
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Generate Button */}
        {!studyPlan && (
          <Card className="bg-slate-900/50 border-slate-700/50 backdrop-blur">
            <CardContent className="py-12 text-center">
              <Brain className="w-16 h-16 mx-auto mb-4 text-cyan-400" />
              <h3 className="text-2xl font-bold mb-2">Generate Your Personalized Study Path</h3>
              <p className="text-slate-400 mb-6">
                AI will analyze your submission history and create a customized learning plan
              </p>
              <Button
                onClick={generateStudyPlan}
                disabled={isGenerating}
                size="lg"
                className="bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-600 hover:to-blue-600"
              >
                {isGenerating ? (
                  <>
                    <Clock className="w-5 h-5 mr-2 animate-spin" />
                    Generating Plan...
                  </>
                ) : (
                  <>
                    <Lightbulb className="w-5 h-5 mr-2" />
                    Generate Study Plan
                  </>
                )}
              </Button>
            </CardContent>
          </Card>
        )}

        {/* Study Plan */}
        {studyPlan && (
          <>
            {/* Overview */}
            <Card className="bg-gradient-to-br from-blue-900/30 to-purple-900/30 border-blue-500/30 backdrop-blur">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Target className="w-5 h-5 text-blue-400" />
                  Your Personalized Study Plan
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <div className="text-sm text-slate-400">Focus Area</div>
                    <div className="text-2xl font-bold text-cyan-400">{studyPlan.weakestCategory}</div>
                  </div>
                  <div>
                    <div className="text-sm text-slate-400">Current Score</div>
                    <div className="text-2xl font-bold text-orange-400">{studyPlan.currentScore.toFixed(1)}</div>
                  </div>
                  <div>
                    <div className="text-sm text-slate-400">Target Score</div>
                    <div className="text-2xl font-bold text-green-400">{studyPlan.targetScore}</div>
                  </div>
                  <div>
                    <div className="text-sm text-slate-400">Estimated Duration</div>
                    <div className="text-2xl font-bold text-purple-400">{studyPlan.estimatedDays} days</div>
                  </div>
                </div>

                <div className="mt-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-slate-400">Overall Progress</span>
                    <span className="text-sm font-semibold text-cyan-400">{progress.toFixed(0)}%</span>
                  </div>
                  <Progress value={progress} className="h-2" />
                </div>
              </CardContent>
            </Card>

            {/* Milestones */}
            <Card className="bg-slate-900/50 border-slate-700/50 backdrop-blur">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-green-400" />
                  Learning Milestones
                </CardTitle>
                <CardDescription>
                  Progressive difficulty levels to reach S-7 certification
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {studyPlan.milestones.map((milestone: any, index: number) => (
                    <div key={milestone.id}>
                      <div
                        className={`p-4 rounded-lg border transition-all ${
                          milestone.completed
                            ? "bg-green-500/10 border-green-500/30"
                            : "bg-slate-800/30 border-slate-700/30 hover:border-cyan-500/30"
                        }`}
                      >
                        <div className="flex items-start justify-between mb-3">
                          <div className="flex items-start gap-3">
                            <div className={`p-2 rounded-lg ${
                              milestone.completed ? "bg-green-500/20" : "bg-blue-500/20"
                            }`}>
                              {milestone.completed ? (
                                <CheckCircle2 className="w-5 h-5 text-green-400" />
                              ) : (
                                <BookOpen className="w-5 h-5 text-blue-400" />
                              )}
                            </div>
                            <div>
                              <div className="font-semibold text-lg">{milestone.title}</div>
                              <div className="text-sm text-slate-400">{milestone.description}</div>
                              <Badge variant="outline" className="mt-2">{milestone.duration}</Badge>
                            </div>
                          </div>
                        </div>

                        <div className="ml-11 space-y-2">
                          <div className="text-sm font-semibold text-slate-300">Resources:</div>
                          {milestone.resources.map((resource: string, idx: number) => (
                            <div key={idx} className="flex items-start gap-2 text-sm text-slate-400">
                              <FileText className="w-4 h-4 mt-0.5 flex-shrink-0 text-cyan-400" />
                              <span>{resource}</span>
                            </div>
                          ))}
                        </div>
                      </div>

                      {index < studyPlan.milestones.length - 1 && (
                        <div className="flex justify-center py-2">
                          <ArrowRight className="w-5 h-5 text-slate-600" />
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Practice Schedule */}
            <Card className="bg-slate-900/50 border-slate-700/50 backdrop-blur">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Clock className="w-5 h-5 text-purple-400" />
                  Weekly Practice Schedule
                </CardTitle>
                <CardDescription>
                  Structured study routine for optimal learning
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-2 gap-3">
                  {studyPlan.practiceSchedule.map((schedule: any) => (
                    <div
                      key={schedule.day}
                      className="p-3 rounded-lg bg-slate-800/30 border border-slate-700/30"
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="font-semibold">{schedule.day}</div>
                          <div className="text-sm text-slate-400">{schedule.focus}</div>
                        </div>
                        <Badge variant="outline">{schedule.duration}</Badge>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Next Steps */}
            <Card className="bg-gradient-to-br from-green-900/30 to-blue-900/30 border-green-500/30 backdrop-blur">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Award className="w-5 h-5 text-green-400" />
                  Next Steps to S-7 Certification
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-start gap-3">
                    <CheckCircle2 className="w-5 h-5 text-green-400 mt-1" />
                    <div>
                      <div className="font-semibold">Start with Foundation Building</div>
                      <div className="text-sm text-slate-300">
                        Begin with the first milestone to establish strong fundamentals
                      </div>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <CheckCircle2 className="w-5 h-5 text-blue-400 mt-1" />
                    <div>
                      <div className="font-semibold">Follow the Practice Schedule</div>
                      <div className="text-sm text-slate-300">
                        Dedicate 2-3 hours daily following the structured routine
                      </div>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <CheckCircle2 className="w-5 h-5 text-purple-400 mt-1" />
                    <div>
                      <div className="font-semibold">Track Your Progress</div>
                      <div className="text-sm text-slate-300">
                        Submit practice answers regularly to monitor improvement
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </>
        )}
      </div>
    </div>
  );
}
