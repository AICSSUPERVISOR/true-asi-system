import { useState, useEffect } from "react";
import { useParams, useLocation } from "wouter";
import { trpc } from "../lib/trpc";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Loader2, ArrowLeft, TrendingUp, Target, CheckCircle2, Clock, DollarSign } from "lucide-react";
import { toast } from "sonner";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

export default function ExecutionROIDashboard() {
  const { companyId } = useParams<{ companyId: string }>();
  const [, setLocation] = useLocation();

  // Fetch execution history
  const { data: executions, isLoading } = trpc.executionTracking.getExecutionHistory.useQuery(
    { companyId: companyId! },
    { enabled: !!companyId }
  );

  // Fetch execution stats
  const { data: stats } = trpc.executionTracking.getExecutionStats.useQuery(
    { companyId },
    { enabled: !!companyId }
  );

  // Fetch company data
  const { data: company } = trpc.brreg.getCompanyById.useQuery(
    { companyId: companyId! },
    { enabled: !!companyId }
  );

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-950 via-purple-950 to-slate-950 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 animate-spin text-purple-400 mx-auto mb-4" />
          <p className="text-white text-lg font-semibold">Loading ROI dashboard...</p>
        </div>
      </div>
    );
  }

  // Prepare timeline data
  const timelineData = executions?.map((exec: any) => ({
    date: new Date(exec.executedAt).toLocaleDateString(),
    action: exec.recommendationAction.substring(0, 30) + "...",
    platform: exec.deeplinkPlatform,
    status: exec.status,
  })) || [];

  // Prepare ROI comparison data
  const roiData = executions
    ?.filter((exec: any) => exec.actualROI)
    .map((exec: any) => {
      const predictedMatch = exec.recommendationAction.match(/\+(\d+)%/);
      const predicted = predictedMatch ? parseInt(predictedMatch[1]) : 25;
      const actualMatch = exec.actualROI?.match(/\+(\d+)%/);
      const actual = actualMatch ? parseInt(actualMatch[1]) : 0;

      return {
        action: exec.deeplinkPlatform,
        predicted,
        actual,
      };
    }) || [];

  // Prepare completion rate data
  const completionData = [
    { name: "Completed", value: stats?.completed || 0, color: "#10b981" },
    { name: "In Progress", value: stats?.inProgress || 0, color: "#f59e0b" },
    { name: "Pending", value: stats?.pending || 0, color: "#6b7280" },
  ];

  // Calculate total revenue increase
  const totalRevenueIncrease = roiData.reduce((sum, item) => sum + item.actual, 0);
  const avgPredictedROI = roiData.length > 0 
    ? Math.round(roiData.reduce((sum, item) => sum + item.predicted, 0) / roiData.length)
    : 0;
  const avgActualROI = roiData.length > 0
    ? Math.round(roiData.reduce((sum, item) => sum + item.actual, 0) / roiData.length)
    : 0;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-purple-950 to-slate-950">
      {/* Header */}
      <div className="bg-white/5 backdrop-blur-xl border-b border-white/10">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setLocation(`/recommendations-ai/${companyId}`)}
                className="text-white hover:bg-white/10"
              >
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back to Recommendations
              </Button>
              <div>
                <h1 className="text-3xl font-black text-white tracking-tight">ROI Dashboard</h1>
                <p className="text-muted-foreground">{company?.company?.name || "Company"}</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-8">
        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <Card className="bg-white/5 backdrop-blur-xl border-white/10">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <Target className="w-4 h-4" />
                Total Executions
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-4xl font-black text-white">{stats?.total || 0}</div>
              <p className="text-xs text-muted-foreground mt-1">Recommendations executed</p>
            </CardContent>
          </Card>

          <Card className="bg-white/5 backdrop-blur-xl border-white/10">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4" />
                Completion Rate
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-4xl font-black text-green-400">{stats?.completionRate || 0}%</div>
              <p className="text-xs text-muted-foreground mt-1">{stats?.completed || 0} completed</p>
            </CardContent>
          </Card>

          <Card className="bg-white/5 backdrop-blur-xl border-white/10">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <TrendingUp className="w-4 h-4" />
                Avg Predicted ROI
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-4xl font-black text-purple-400">+{avgPredictedROI}%</div>
              <p className="text-xs text-muted-foreground mt-1">Expected return</p>
            </CardContent>
          </Card>

          <Card className="bg-white/5 backdrop-blur-xl border-white/10">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <DollarSign className="w-4 h-4" />
                Avg Actual ROI
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-4xl font-black text-green-400">+{avgActualROI}%</div>
              <p className="text-xs text-muted-foreground mt-1">Achieved return</p>
            </CardContent>
          </Card>
        </div>

        {/* Charts Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Actual vs Predicted ROI */}
          <Card className="bg-white/5 backdrop-blur-xl border-white/10">
            <CardHeader>
              <CardTitle className="text-xl font-bold text-white">Actual vs Predicted ROI</CardTitle>
              <p className="text-sm text-muted-foreground">Compare expected and achieved returns</p>
            </CardHeader>
            <CardContent>
              {roiData.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={roiData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                    <XAxis dataKey="action" stroke="#ffffff60" />
                    <YAxis stroke="#ffffff60" />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#1e293b",
                        border: "1px solid #ffffff20",
                        borderRadius: "8px",
                      }}
                    />
                    <Legend />
                    <Bar dataKey="predicted" fill="#a855f7" name="Predicted ROI %" />
                    <Bar dataKey="actual" fill="#10b981" name="Actual ROI %" />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex items-center justify-center h-[300px] text-muted-foreground">
                  No ROI data yet. Complete executions to see results.
                </div>
              )}
            </CardContent>
          </Card>

          {/* Completion Rate Donut Chart */}
          <Card className="bg-white/5 backdrop-blur-xl border-white/10">
            <CardHeader>
              <CardTitle className="text-xl font-bold text-white">Completion Status</CardTitle>
              <p className="text-sm text-muted-foreground">Track execution progress</p>
            </CardHeader>
            <CardContent>
              {stats && stats.total > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={completionData}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={100}
                      paddingAngle={5}
                      dataKey="value"
                    >
                      {completionData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#1e293b",
                        border: "1px solid #ffffff20",
                        borderRadius: "8px",
                      }}
                    />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex items-center justify-center h-[300px] text-muted-foreground">
                  No executions yet. Start executing recommendations!
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Execution Timeline */}
        <Card className="bg-white/5 backdrop-blur-xl border-white/10 mt-6">
          <CardHeader>
            <CardTitle className="text-xl font-bold text-white">Execution Timeline</CardTitle>
            <p className="text-sm text-muted-foreground">Recent recommendation executions</p>
          </CardHeader>
          <CardContent>
            {executions && executions.length > 0 ? (
              <div className="space-y-4">
                {executions.slice(0, 10).map((exec: any) => (
                  <div
                    key={exec.id}
                    className="flex items-center justify-between p-4 bg-white/5 rounded-lg border border-white/10"
                  >
                    <div className="flex-1">
                      <p className="text-white font-semibold">{exec.deeplinkPlatform}</p>
                      <p className="text-sm text-muted-foreground">{exec.recommendationAction}</p>
                      <p className="text-xs text-muted-foreground mt-1">
                        {new Date(exec.executedAt).toLocaleString()}
                      </p>
                    </div>
                    <Badge
                      className={
                        exec.status === "completed"
                          ? "bg-green-500/20 text-green-400"
                          : exec.status === "in-progress"
                          ? "bg-yellow-500/20 text-yellow-400"
                          : "bg-gray-500/20 text-gray-400"
                      }
                    >
                      {exec.status}
                    </Badge>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex items-center justify-center h-[200px] text-muted-foreground">
                No executions yet. Execute recommendations to see timeline.
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
