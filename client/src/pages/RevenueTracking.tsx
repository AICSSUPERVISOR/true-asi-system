import { useState } from "react";
import { trpc } from "@/lib/trpc";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { TrendingUp, Users, Globe, Linkedin, Share2, DollarSign, Download } from "lucide-react";

type TimeRange = "7d" | "30d" | "90d" | "1y";
type MetricType = "revenue" | "customers" | "traffic" | "linkedin" | "social" | "all";

export default function RevenueTracking() {
  const [timeRange, setTimeRange] = useState<TimeRange>("30d");
  const [metricType, setMetricType] = useState<MetricType>("all");
  const [analysisId] = useState<string>("1"); // Would come from URL params in production

  // Fetch tracking records
  const { data: records, isLoading } = trpc.revenueTracking.getTrackingRecords.useQuery({
    analysisId,
    startDate: getStartDate(timeRange),
    endDate: new Date().toISOString(),
  });

  // Fetch latest metrics
  const { data: latestMetrics } = trpc.revenueTracking.getLatestMetrics.useQuery({
    analysisId,
  });

  // Fetch ROI calculation
  const { data: roiData } = trpc.revenueTracking.calculateROI.useQuery({
    analysisId,
  });

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center">
        <div className="text-white text-xl">Loading revenue data...</div>
      </div>
    );
  }

  const chartData = records?.map(r => ({
    date: new Date(r.createdAt).toLocaleDateString(),
    revenue: r.revenue ? Number(r.revenue) / 100 : 0,
    customers: r.customers || 0,
    traffic: r.websiteTraffic || 0,
    linkedin: r.linkedinEngagement ? Number(r.linkedinEngagement) / 10 : 0,
    social: r.socialMediaFollowers || 0,
  })) || [];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 py-12 px-4">
      <div className="container mx-auto max-w-7xl">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">Revenue Tracking Dashboard</h1>
          <p className="text-slate-300">Monitor your business growth and ROI over time</p>
        </div>

        {/* Controls */}
        <div className="flex flex-wrap gap-4 mb-8">
          <Select value={timeRange} onValueChange={(v) => setTimeRange(v as TimeRange)}>
            <SelectTrigger className="w-[180px] bg-white/10 border-white/20 text-white">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="7d">Last 7 days</SelectItem>
              <SelectItem value="30d">Last 30 days</SelectItem>
              <SelectItem value="90d">Last 90 days</SelectItem>
              <SelectItem value="1y">Last year</SelectItem>
            </SelectContent>
          </Select>

          <Select value={metricType} onValueChange={(v) => setMetricType(v as MetricType)}>
            <SelectTrigger className="w-[180px] bg-white/10 border-white/20 text-white">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Metrics</SelectItem>
              <SelectItem value="revenue">Revenue Only</SelectItem>
              <SelectItem value="customers">Customers Only</SelectItem>
              <SelectItem value="traffic">Traffic Only</SelectItem>
              <SelectItem value="linkedin">LinkedIn Only</SelectItem>
              <SelectItem value="social">Social Media Only</SelectItem>
            </SelectContent>
          </Select>

          <Button variant="outline" className="bg-white/10 border-white/20 text-white hover:bg-white/20">
            <Download className="w-4 h-4 mr-2" />
            Export CSV
          </Button>
        </div>

        {/* Metrics Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
          <Card className="bg-white/10 backdrop-blur-lg border-white/20 p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-green-500/20 rounded-lg">
                <DollarSign className="w-6 h-6 text-green-400" />
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-white">
                  {latestMetrics?.current.revenue ? `${(Number(latestMetrics.current.revenue) / 100).toLocaleString()} NOK` : "N/A"}
                </div>
                <div className="text-sm text-slate-300">
                  {latestMetrics?.changes?.revenue ? (
                    <span className={latestMetrics.changes.revenue > 0 ? "text-green-400" : "text-red-400"}>
                      {latestMetrics.changes.revenue > 0 ? "+" : ""}{latestMetrics.changes.revenue.toFixed(1)}%
                    </span>
                  ) : "No change"}
                </div>
              </div>
            </div>
            <div className="text-slate-300">Total Revenue</div>
          </Card>

          <Card className="bg-white/10 backdrop-blur-lg border-white/20 p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-blue-500/20 rounded-lg">
                <Users className="w-6 h-6 text-blue-400" />
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-white">
                  {latestMetrics?.current.customers?.toLocaleString() || "N/A"}
                </div>
                <div className="text-sm text-slate-300">
                  {latestMetrics?.changes?.customers ? (
                    <span className={latestMetrics.changes.customers > 0 ? "text-green-400" : "text-red-400"}>
                      {latestMetrics.changes.customers > 0 ? "+" : ""}{latestMetrics.changes.customers.toFixed(1)}%
                    </span>
                  ) : "No change"}
                </div>
              </div>
            </div>
            <div className="text-slate-300">Total Customers</div>
          </Card>

          <Card className="bg-white/10 backdrop-blur-lg border-white/20 p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-purple-500/20 rounded-lg">
                <Globe className="w-6 h-6 text-purple-400" />
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-white">
                  {latestMetrics?.current.websiteTraffic?.toLocaleString() || "N/A"}
                </div>
                <div className="text-sm text-slate-300">
                  {latestMetrics?.changes?.websiteTraffic ? (
                    <span className={latestMetrics.changes.websiteTraffic > 0 ? "text-green-400" : "text-red-400"}>
                      {latestMetrics.changes.websiteTraffic > 0 ? "+" : ""}{latestMetrics.changes.websiteTraffic.toFixed(1)}%
                    </span>
                  ) : "No change"}
                </div>
              </div>
            </div>
            <div className="text-slate-300">Website Traffic</div>
          </Card>

          <Card className="bg-white/10 backdrop-blur-lg border-white/20 p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-blue-600/20 rounded-lg">
                <Linkedin className="w-6 h-6 text-blue-500" />
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-white">
                  {latestMetrics?.current.linkedinFollowers?.toLocaleString() || "N/A"}
                </div>
                <div className="text-sm text-slate-300">
                  {latestMetrics?.changes?.linkedinFollowers ? (
                    <span className={latestMetrics.changes.linkedinFollowers > 0 ? "text-green-400" : "text-red-400"}>
                      {latestMetrics.changes.linkedinFollowers > 0 ? "+" : ""}{latestMetrics.changes.linkedinFollowers.toFixed(1)}%
                    </span>
                  ) : "No change"}
                </div>
              </div>
            </div>
            <div className="text-slate-300">LinkedIn Engagement</div>
          </Card>

          <Card className="bg-white/10 backdrop-blur-lg border-white/20 p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-pink-500/20 rounded-lg">
                <Share2 className="w-6 h-6 text-pink-400" />
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-white">
                  {latestMetrics?.current.socialMediaFollowers?.toLocaleString() || "N/A"}
                </div>
                <div className="text-sm text-slate-300">
                  {latestMetrics?.changes?.socialMediaFollowers ? (
                    <span className={latestMetrics.changes.socialMediaFollowers > 0 ? "text-green-400" : "text-red-400"}>
                      {latestMetrics.changes.socialMediaFollowers > 0 ? "+" : ""}{latestMetrics.changes.socialMediaFollowers.toFixed(1)}%
                    </span>
                  ) : "No change"}
                </div>
              </div>
            </div>
            <div className="text-slate-300">Social Media Followers</div>
          </Card>

          <Card className="bg-white/10 backdrop-blur-lg border-white/20 p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-yellow-500/20 rounded-lg">
                <TrendingUp className="w-6 h-6 text-yellow-400" />
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-white">
                  {roiData?.roi ? `${roiData.roi.toFixed(1)}%` : "N/A"}
                </div>
                <div className="text-sm text-slate-300">
                  {roiData?.totalInvestment ? `${(roiData.totalInvestment / 100).toLocaleString()} NOK invested` : "No investment"}
                </div>
              </div>
            </div>
            <div className="text-slate-300">Return on Investment</div>
          </Card>
        </div>

        {/* Charts */}
        {(metricType === "all" || metricType === "revenue") && (
          <Card className="bg-white/10 backdrop-blur-lg border-white/20 p-6 mb-8">
            <h2 className="text-2xl font-bold text-white mb-6">Revenue Growth Over Time</h2>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="date" stroke="rgba(255,255,255,0.5)" />
                <YAxis stroke="rgba(255,255,255,0.5)" />
                <Tooltip
                  contentStyle={{ backgroundColor: "rgba(0,0,0,0.8)", border: "1px solid rgba(255,255,255,0.2)" }}
                  labelStyle={{ color: "#fff" }}
                />
                <Legend />
                <Line type="monotone" dataKey="revenue" stroke="#10b981" strokeWidth={2} name="Revenue (NOK)" />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        )}

        {(metricType === "all" || metricType === "customers") && (
          <Card className="bg-white/10 backdrop-blur-lg border-white/20 p-6 mb-8">
            <h2 className="text-2xl font-bold text-white mb-6">Customer Acquisition Trends</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="date" stroke="rgba(255,255,255,0.5)" />
                <YAxis stroke="rgba(255,255,255,0.5)" />
                <Tooltip
                  contentStyle={{ backgroundColor: "rgba(0,0,0,0.8)", border: "1px solid rgba(255,255,255,0.2)" }}
                  labelStyle={{ color: "#fff" }}
                />
                <Legend />
                <Bar dataKey="customers" fill="#3b82f6" name="Total Customers" />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        )}

        {(metricType === "all" || metricType === "traffic") && (
          <Card className="bg-white/10 backdrop-blur-lg border-white/20 p-6 mb-8">
            <h2 className="text-2xl font-bold text-white mb-6">Website Traffic Improvements</h2>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="date" stroke="rgba(255,255,255,0.5)" />
                <YAxis stroke="rgba(255,255,255,0.5)" />
                <Tooltip
                  contentStyle={{ backgroundColor: "rgba(0,0,0,0.8)", border: "1px solid rgba(255,255,255,0.2)" }}
                  labelStyle={{ color: "#fff" }}
                />
                <Legend />
                <Area type="monotone" dataKey="traffic" stroke="#a855f7" fill="#a855f7" fillOpacity={0.3} name="Website Traffic" />
              </AreaChart>
            </ResponsiveContainer>
          </Card>
        )}

        {(metricType === "all" || metricType === "linkedin") && (
          <Card className="bg-white/10 backdrop-blur-lg border-white/20 p-6 mb-8">
            <h2 className="text-2xl font-bold text-white mb-6">LinkedIn Engagement Metrics</h2>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="date" stroke="rgba(255,255,255,0.5)" />
                <YAxis stroke="rgba(255,255,255,0.5)" />
                <Tooltip
                  contentStyle={{ backgroundColor: "rgba(0,0,0,0.8)", border: "1px solid rgba(255,255,255,0.2)" }}
                  labelStyle={{ color: "#fff" }}
                />
                <Legend />
                <Line type="monotone" dataKey="linkedin" stroke="#0a66c2" strokeWidth={2} name="LinkedIn Engagement" />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        )}

        {(metricType === "all" || metricType === "social") && (
          <Card className="bg-white/10 backdrop-blur-lg border-white/20 p-6 mb-8">
            <h2 className="text-2xl font-bold text-white mb-6">Social Media Follower Growth</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="date" stroke="rgba(255,255,255,0.5)" />
                <YAxis stroke="rgba(255,255,255,0.5)" />
                <Tooltip
                  contentStyle={{ backgroundColor: "rgba(0,0,0,0.8)", border: "1px solid rgba(255,255,255,0.2)" }}
                  labelStyle={{ color: "#fff" }}
                />
                <Legend />
                <Bar dataKey="social" fill="#ec4899" name="Social Media Followers" />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        )}

        {/* ROI Summary */}
        {roiData && (
          <Card className="bg-white/10 backdrop-blur-lg border-white/20 p-6">
            <h2 className="text-2xl font-bold text-white mb-6">ROI Summary</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div>
                <div className="text-slate-300 mb-2">Total Investment</div>
                <div className="text-3xl font-bold text-white">
                  {(roiData.totalInvestment / 100).toLocaleString()} NOK
                </div>
              </div>
              <div>
                <div className="text-slate-300 mb-2">Total Return</div>
                <div className="text-3xl font-bold text-green-400">
                  {(roiData.totalRevenueIncrease / 100).toLocaleString()} NOK
                </div>
              </div>
              <div>
                <div className="text-slate-300 mb-2">ROI Percentage</div>
                <div className="text-3xl font-bold text-yellow-400">
                  {roiData.roi.toFixed(1)}%
                </div>
              </div>
            </div>
          </Card>
        )}
      </div>
    </div>
  );
}

function getStartDate(timeRange: TimeRange): string {
  const now = new Date();
  switch (timeRange) {
    case "7d":
      return new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000).toISOString();
    case "30d":
      return new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000).toISOString();
    case "90d":
      return new Date(now.getTime() - 90 * 24 * 60 * 60 * 1000).toISOString();
    case "1y":
      return new Date(now.getTime() - 365 * 24 * 60 * 60 * 1000).toISOString();
    default:
      return new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000).toISOString();
  }
}
