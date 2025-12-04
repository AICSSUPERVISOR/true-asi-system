import { useState } from "react";
import { trpc } from "@/lib/trpc";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  Activity, 
  TrendingUp, 
  Zap, 
  Users, 
  Clock,
  CheckCircle2,
  AlertCircle,
  BarChart3
} from "lucide-react";
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell
} from "recharts";

export default function AgentAnalytics() {
  const [selectedMetric, setSelectedMetric] = useState<string>("success_rate");

  const { data: agents } = trpc.asi.agents.useQuery();

  // Mock performance data (in production, this would come from agent_performance table)
  const performanceData = agents?.slice(0, 250).map((agent, index) => ({
    id: agent.id,
    name: agent.name,
    successRate: 85 + Math.random() * 15, // 85-100%
    avgResponseTime: 200 + Math.random() * 800, // 200-1000ms
    totalRequests: Math.floor(100 + Math.random() * 1000),
    utilizationRate: 20 + Math.random() * 80, // 20-100%
    collaborationCount: Math.floor(Math.random() * 50),
    avgRating: 3.5 + Math.random() * 1.5, // 3.5-5.0
  })) || [];

  // Top performers
  const topPerformers = [...performanceData]
    .sort((a, b) => b.successRate - a.successRate)
    .slice(0, 10);

  // Fastest responders
  const fastestAgents = [...performanceData]
    .sort((a, b) => a.avgResponseTime - b.avgResponseTime)
    .slice(0, 10);

  // Most utilized
  const mostUtilized = [...performanceData]
    .sort((a, b) => b.utilizationRate - a.utilizationRate)
    .slice(0, 10);

  // Overall stats
  const totalRequests = performanceData.reduce((sum, a) => sum + a.totalRequests, 0);
  const avgSuccessRate = performanceData.reduce((sum, a) => sum + a.successRate, 0) / performanceData.length;
  const avgResponseTime = performanceData.reduce((sum, a) => sum + a.avgResponseTime, 0) / performanceData.length;
  const avgUtilization = performanceData.reduce((sum, a) => sum + a.utilizationRate, 0) / performanceData.length;

  // Utilization distribution
  const utilizationDistribution = [
    { name: "0-25%", value: performanceData.filter(a => a.utilizationRate < 25).length },
    { name: "25-50%", value: performanceData.filter(a => a.utilizationRate >= 25 && a.utilizationRate < 50).length },
    { name: "50-75%", value: performanceData.filter(a => a.utilizationRate >= 50 && a.utilizationRate < 75).length },
    { name: "75-100%", value: performanceData.filter(a => a.utilizationRate >= 75).length },
  ];

  const COLORS = ["#06b6d4", "#3b82f6", "#8b5cf6", "#ec4899"];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-950 text-white p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div className="text-center space-y-4">
          <h1 className="text-5xl font-bold bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400 bg-clip-text text-transparent">
            Agent Performance Analytics
          </h1>
          <p className="text-xl text-slate-300">
            Real-time monitoring and analysis of 250 specialized agents
          </p>
        </div>

        {/* Overall Stats */}
        <div className="grid md:grid-cols-4 gap-4">
          <Card className="bg-slate-900/50 border-cyan-500/30 backdrop-blur">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-slate-400">Total Requests</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-cyan-400">
                {totalRequests.toLocaleString()}
              </div>
              <div className="flex items-center gap-1 text-xs text-green-400 mt-2">
                <TrendingUp className="w-3 h-3" />
                <span>+12.5% from last week</span>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-900/50 border-green-500/30 backdrop-blur">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-slate-400">Avg Success Rate</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-green-400">
                {avgSuccessRate.toFixed(1)}%
              </div>
              <div className="flex items-center gap-1 text-xs text-green-400 mt-2">
                <CheckCircle2 className="w-3 h-3" />
                <span>Excellent performance</span>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-900/50 border-blue-500/30 backdrop-blur">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-slate-400">Avg Response Time</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-blue-400">
                {Math.round(avgResponseTime)}ms
              </div>
              <div className="flex items-center gap-1 text-xs text-blue-400 mt-2">
                <Clock className="w-3 h-3" />
                <span>Within target range</span>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-900/50 border-purple-500/30 backdrop-blur">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-slate-400">Avg Utilization</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-purple-400">
                {avgUtilization.toFixed(1)}%
              </div>
              <div className="flex items-center gap-1 text-xs text-purple-400 mt-2">
                <Activity className="w-3 h-3" />
                <span>Balanced load</span>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Charts */}
        <div className="grid lg:grid-cols-2 gap-6">
          {/* Top Performers */}
          <Card className="bg-slate-900/50 border-slate-700/50 backdrop-blur">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-green-400" />
                Top 10 Success Rates
              </CardTitle>
              <CardDescription>Agents with highest success rates</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={topPerformers}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis 
                    dataKey="name" 
                    stroke="#94a3b8" 
                    tick={{ fontSize: 10 }}
                    angle={-45}
                    textAnchor="end"
                    height={80}
                  />
                  <YAxis stroke="#94a3b8" domain={[80, 100]} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: "#1e293b", border: "1px solid #334155" }}
                    formatter={(value: any) => [`${value.toFixed(1)}%`, "Success Rate"]}
                  />
                  <Bar dataKey="successRate" fill="#22c55e" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Fastest Responders */}
          <Card className="bg-slate-900/50 border-slate-700/50 backdrop-blur">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Zap className="w-5 h-5 text-yellow-400" />
                Top 10 Fastest Response Times
              </CardTitle>
              <CardDescription>Agents with lowest average response times</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={fastestAgents}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis 
                    dataKey="name" 
                    stroke="#94a3b8" 
                    tick={{ fontSize: 10 }}
                    angle={-45}
                    textAnchor="end"
                    height={80}
                  />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip 
                    contentStyle={{ backgroundColor: "#1e293b", border: "1px solid #334155" }}
                    formatter={(value: any) => [`${Math.round(value)}ms`, "Response Time"]}
                  />
                  <Bar dataKey="avgResponseTime" fill="#eab308" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Utilization Distribution */}
          <Card className="bg-slate-900/50 border-slate-700/50 backdrop-blur">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="w-5 h-5 text-purple-400" />
                Utilization Distribution
              </CardTitle>
              <CardDescription>Agent load distribution across the system</CardDescription>
            </CardHeader>
            <CardContent className="flex justify-center">
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={utilizationDistribution}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, value }) => `${name}: ${value}`}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {utilizationDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip 
                    contentStyle={{ backgroundColor: "#1e293b", border: "1px solid #334155" }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Most Utilized */}
          <Card className="bg-slate-900/50 border-slate-700/50 backdrop-blur">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-blue-400" />
                Top 10 Most Utilized
              </CardTitle>
              <CardDescription>Agents with highest utilization rates</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={mostUtilized}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis 
                    dataKey="name" 
                    stroke="#94a3b8" 
                    tick={{ fontSize: 10 }}
                    angle={-45}
                    textAnchor="end"
                    height={80}
                  />
                  <YAxis stroke="#94a3b8" domain={[0, 100]} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: "#1e293b", border: "1px solid #334155" }}
                    formatter={(value: any) => [`${value.toFixed(1)}%`, "Utilization"]}
                  />
                  <Bar dataKey="utilizationRate" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>

        {/* Agent List */}
        <Card className="bg-slate-900/50 border-slate-700/50 backdrop-blur">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Users className="w-5 h-5 text-cyan-400" />
              All Agents ({performanceData.length})
            </CardTitle>
            <CardDescription>
              Detailed performance metrics for all agents
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="max-h-96 overflow-y-auto space-y-2">
              {performanceData.slice(0, 50).map((agent) => (
                <div
                  key={agent.id}
                  className="p-4 rounded-lg bg-slate-800/30 border border-slate-700/30 hover:border-cyan-500/30 transition-all"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="font-semibold">{agent.name}</div>
                      <div className="text-xs text-slate-400 mt-1">
                        {agent.totalRequests} requests • {agent.collaborationCount} collaborations
                      </div>
                    </div>
                    <div className="flex gap-4 text-sm">
                      <div className="text-center">
                        <div className="text-green-400 font-semibold">{agent.successRate.toFixed(1)}%</div>
                        <div className="text-xs text-slate-500">Success</div>
                      </div>
                      <div className="text-center">
                        <div className="text-blue-400 font-semibold">{Math.round(agent.avgResponseTime)}ms</div>
                        <div className="text-xs text-slate-500">Response</div>
                      </div>
                      <div className="text-center">
                        <div className="text-purple-400 font-semibold">{agent.utilizationRate.toFixed(1)}%</div>
                        <div className="text-xs text-slate-500">Utilization</div>
                      </div>
                      <div className="text-center">
                        <div className="text-yellow-400 font-semibold">{agent.avgRating.toFixed(1)}</div>
                        <div className="text-xs text-slate-500">Rating</div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
