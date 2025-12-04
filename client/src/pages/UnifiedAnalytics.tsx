import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { 
  TrendingUp, TrendingDown, Users, Brain, Target, Award, 
  Calendar, Download, Filter, BarChart3, LineChart, PieChart 
} from "lucide-react";
import { BarChart, Bar, LineChart as RechartsLine, Line, PieChart as RechartsPie, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart } from "recharts";

export default function UnifiedAnalytics() {
  const [dateRange, setDateRange] = useState("7d");
  
  // Mock data for user progress tracking
  const progressData = [
    { date: "Dec 1", score: 7.2, submissions: 3 },
    { date: "Dec 2", score: 7.5, submissions: 5 },
    { date: "Dec 3", score: 7.8, submissions: 4 },
    { date: "Dec 4", score: 8.1, submissions: 6 },
    { date: "Dec 5", score: 8.4, submissions: 5 },
    { date: "Dec 6", score: 8.7, submissions: 7 },
    { date: "Dec 7", score: 9.0, submissions: 8 },
  ];

  // Agent utilization heatmap data
  const agentUtilization = [
    { hour: "00:00", utilization: 45 },
    { hour: "04:00", utilization: 30 },
    { hour: "08:00", utilization: 75 },
    { hour: "12:00", utilization: 90 },
    { hour: "16:00", utilization: 85 },
    { hour: "20:00", utilization: 60 },
    { hour: "23:00", utilization: 50 },
  ];

  // S-7 performance trends
  const s7Trends = [
    { category: "Novelty", current: 8.9, previous: 8.2, change: 8.5 },
    { category: "Coherence", current: 9.1, previous: 8.8, change: 3.4 },
    { category: "Rigor", current: 8.5, previous: 8.0, change: 6.3 },
    { category: "Synthesis", current: 9.3, previous: 9.0, change: 3.3 },
    { category: "Formalization", current: 8.7, previous: 8.3, change: 4.8 },
    { category: "Depth", current: 9.0, previous: 8.6, change: 4.7 },
  ];

  // Comparative analytics
  const compareData = [
    { metric: "Avg Score", you: 8.9, average: 7.5, top10: 9.4 },
    { metric: "Submissions", you: 38, average: 25, top10: 52 },
    { metric: "Study Hours", you: 45, average: 30, top10: 68 },
    { metric: "Completion", you: 95, average: 65, top10: 98 },
  ];

  // Achievement badges
  const badges = [
    { name: "S-7 Certified", earned: true, date: "Dec 1, 2025" },
    { name: "100 Submissions", earned: false, progress: 38 },
    { name: "Perfect Score", earned: false, progress: 0 },
    { name: "7-Day Streak", earned: true, date: "Dec 7, 2025" },
    { name: "Top 10%", earned: true, date: "Dec 5, 2025" },
    { name: "Collaboration Master", earned: false, progress: 60 },
  ];

  // Predictive success modeling
  const predictionData = [
    { week: "Week 1", actual: 7.2, predicted: 7.3 },
    { week: "Week 2", actual: 7.8, predicted: 7.9 },
    { week: "Week 3", actual: 8.4, predicted: 8.5 },
    { week: "Week 4", actual: 9.0, predicted: 9.1 },
    { week: "Week 5", actual: null, predicted: 9.4 },
    { week: "Week 6", actual: null, predicted: 9.6 },
    { week: "Week 7", actual: null, predicted: 9.7 },
    { week: "Week 8", actual: null, predicted: 9.8 },
  ];

  const COLORS = ["#06b6d4", "#8b5cf6", "#ec4899", "#f59e0b", "#10b981", "#3b82f6"];

  return (
    <div className="container mx-auto py-8 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600 bg-clip-text text-transparent">
            Advanced Analytics & Insights
          </h1>
          <p className="text-muted-foreground mt-2">
            Comprehensive performance tracking, predictions, and gamification
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm">
            <Filter className="h-4 w-4 mr-2" />
            Filter
          </Button>
          <Button variant="outline" size="sm">
            <Download className="h-4 w-4 mr-2" />
            Export PDF
          </Button>
        </div>
      </div>

      {/* Date Range Selector */}
      <div className="flex gap-2">
        {["7d", "30d", "90d", "1y", "all"].map((range) => (
          <Button
            key={range}
            variant={dateRange === range ? "default" : "outline"}
            size="sm"
            onClick={() => setDateRange(range)}
          >
            {range === "7d" && "7 Days"}
            {range === "30d" && "30 Days"}
            {range === "90d" && "90 Days"}
            {range === "1y" && "1 Year"}
            {range === "all" && "All Time"}
          </Button>
        ))}
      </div>

      {/* Key Metrics Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Current Score</CardTitle>
            <Target className="h-4 w-4 text-cyan-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">9.0/10</div>
            <p className="text-xs text-muted-foreground flex items-center gap-1 mt-1">
              <TrendingUp className="h-3 w-3 text-green-500" />
              +0.3 from last week
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Total Submissions</CardTitle>
            <BarChart3 className="h-4 w-4 text-blue-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">38</div>
            <p className="text-xs text-muted-foreground flex items-center gap-1 mt-1">
              <TrendingUp className="h-3 w-3 text-green-500" />
              +8 this week
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Study Hours</CardTitle>
            <Calendar className="h-4 w-4 text-purple-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">45h</div>
            <p className="text-xs text-muted-foreground flex items-center gap-1 mt-1">
              <TrendingUp className="h-3 w-3 text-green-500" />
              +12h this week
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Global Rank</CardTitle>
            <Award className="h-4 w-4 text-yellow-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">#12</div>
            <p className="text-xs text-muted-foreground flex items-center gap-1 mt-1">
              <TrendingUp className="h-3 w-4 text-green-500" />
              +5 positions
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Main Analytics Tabs */}
      <Tabs defaultValue="progress" className="space-y-4">
        <TabsList>
          <TabsTrigger value="progress">Progress Tracking</TabsTrigger>
          <TabsTrigger value="agents">Agent Utilization</TabsTrigger>
          <TabsTrigger value="trends">S-7 Trends</TabsTrigger>
          <TabsTrigger value="compare">Comparative</TabsTrigger>
          <TabsTrigger value="predict">Predictions</TabsTrigger>
          <TabsTrigger value="badges">Achievements</TabsTrigger>
        </TabsList>

        {/* Progress Tracking */}
        <TabsContent value="progress" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Your Progress Over Time</CardTitle>
              <CardDescription>Track your S-7 score improvements and submission frequency</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={350}>
                <AreaChart data={progressData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis yAxisId="left" domain={[0, 10]} />
                  <YAxis yAxisId="right" orientation="right" />
                  <Tooltip />
                  <Legend />
                  <Area yAxisId="left" type="monotone" dataKey="score" stroke="#06b6d4" fill="#06b6d4" fillOpacity={0.6} name="Average Score" />
                  <Area yAxisId="right" type="monotone" dataKey="submissions" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.6} name="Submissions" />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Agent Utilization */}
        <TabsContent value="agents" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Agent Utilization Heatmap</CardTitle>
              <CardDescription>Peak usage times for 250 specialized agents</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={agentUtilization}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="hour" />
                  <YAxis domain={[0, 100]} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="utilization" fill="#06b6d4" name="Utilization %" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        {/* S-7 Performance Trends */}
        <TabsContent value="trends" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>S-7 Category Performance Trends</CardTitle>
              <CardDescription>Track improvements across all 6 evaluation categories</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {s7Trends.map((trend) => (
                  <div key={trend.category} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="font-medium">{trend.category}</span>
                      <div className="flex items-center gap-2">
                        <span className="text-2xl font-bold">{trend.current}</span>
                        <Badge variant={trend.change > 0 ? "default" : "secondary"}>
                          {trend.change > 0 ? "+" : ""}{trend.change}%
                        </Badge>
                      </div>
                    </div>
                    <div className="h-2 bg-secondary rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-gradient-to-r from-cyan-500 to-blue-600 transition-all"
                        style={{ width: `${(trend.current / 10) * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Comparative Analytics */}
        <TabsContent value="compare" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Comparative Performance</CardTitle>
              <CardDescription>How you compare to average users and top 10%</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={compareData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="metric" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="you" fill="#06b6d4" name="You" />
                  <Bar dataKey="average" fill="#94a3b8" name="Average" />
                  <Bar dataKey="top10" fill="#8b5cf6" name="Top 10%" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Predictive Modeling */}
        <TabsContent value="predict" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Predictive Success Modeling</CardTitle>
              <CardDescription>AI-powered forecast of your future performance</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={350}>
                <RechartsLine data={predictionData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="week" />
                  <YAxis domain={[0, 10]} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="actual" stroke="#06b6d4" strokeWidth={2} name="Actual Score" />
                  <Line type="monotone" dataKey="predicted" stroke="#8b5cf6" strokeWidth={2} strokeDasharray="5 5" name="Predicted Score" />
                </RechartsLine>
              </ResponsiveContainer>
              <div className="mt-4 p-4 bg-secondary/50 rounded-lg">
                <p className="text-sm">
                  <strong>Prediction:</strong> Based on your current trajectory, you're projected to achieve a score of <strong>9.8/10</strong> by Week 8, 
                  placing you in the <strong>top 5%</strong> globally. Maintain your current study pace and focus on improving Rigor (+0.5 needed).
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Achievement Badges */}
        <TabsContent value="badges" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Achievement Badges & Gamification</CardTitle>
              <CardDescription>Unlock badges by reaching milestones and completing challenges</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {badges.map((badge) => (
                  <Card key={badge.name} className={badge.earned ? "border-cyan-500" : "opacity-60"}>
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        <Award className={`h-8 w-8 ${badge.earned ? "text-yellow-500" : "text-muted-foreground"}`} />
                        {badge.earned && <Badge>Earned</Badge>}
                      </div>
                      <CardTitle className="text-lg">{badge.name}</CardTitle>
                    </CardHeader>
                    <CardContent>
                      {badge.earned ? (
                        <p className="text-sm text-muted-foreground">Earned on {badge.date}</p>
                      ) : (
                        <div className="space-y-2">
                          <p className="text-sm text-muted-foreground">Progress: {badge.progress}%</p>
                          <div className="h-2 bg-secondary rounded-full overflow-hidden">
                            <div 
                              className="h-full bg-gradient-to-r from-cyan-500 to-blue-600"
                              style={{ width: `${badge.progress}%` }}
                            />
                          </div>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
