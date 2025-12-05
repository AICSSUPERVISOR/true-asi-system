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
import { useToast } from "@/hooks/use-toast";
import { LoadingSkeleton } from "@/components/LoadingSkeleton";
import { ConnectionStatus } from '@/components/ConnectionStatus';
import { NotificationCenter } from '@/components/NotificationCenter';
import { useRealtimeMetrics } from "@/contexts/WebSocketProvider";

type TimeRange = "7d" | "30d" | "90d" | "1y";
type MetricType = "revenue" | "customers" | "traffic" | "linkedin" | "social" | "all";

export default function RevenueTracking() {
  const [timeRange, setTimeRange] = useState<TimeRange>("30d");
  const [metricType, setMetricType] = useState<MetricType>("all");
  const [analysisId] = useState<string>("1"); // Would come from URL params in production
  const { toast } = useToast();

  // CSV Export function
  const exportToCSV = () => {
    if (!records || records.length === 0) {
      toast({ title: "No data to export", variant: "destructive" });
      return;
    }

    // Prepare CSV headers
    const headers = [
      "Date",
      "Revenue (NOK)",
      "Customers",
      "Website Traffic",
      "LinkedIn Followers",
      "LinkedIn Engagement (%)",
      "Social Media Followers",
      "Social Media Engagement (%)",
      "Review Rating",
    ];

    // Prepare CSV rows
    const rows = records.map(record => [
      new Date(record.createdAt).toLocaleDateString(),
      record.revenue ? (Number(record.revenue) / 100).toString() : "0",
      record.customers?.toString() || "0",
      record.websiteTraffic?.toString() || "0",
      record.linkedinFollowers?.toString() || "0",
      record.linkedinEngagement ? (Number(record.linkedinEngagement) / 10).toFixed(1) : "0",
      record.socialMediaFollowers?.toString() || "0",
      record.socialMediaEngagement ? (Number(record.socialMediaEngagement) / 10).toFixed(1) : "0",
      record.averageRating ? (Number(record.averageRating) / 10).toFixed(1) : "0",
    ]);

    // Combine headers and rows
    const csvContent = [
      headers.join(","),
      ...rows.map(row => row.join(",")),
    ].join("\n");

    // Create blob and download
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    const url = URL.createObjectURL(blob);
    link.setAttribute("href", url);
    link.setAttribute("download", `revenue_tracking_${new Date().toISOString().split("T")[0]}.csv`);
    link.style.visibility = "hidden";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    toast({ title: "CSV exported successfully" });
  };

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

  // Real-time metric updates
  const realtimeMetrics = useRealtimeMetrics(analysisId);
  
  // Use real-time metrics if available, otherwise use fetched data
  const displayMetrics = realtimeMetrics || latestMetrics;

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 py-12 px-4">
        <div className="container mx-auto max-w-7xl">
          <div className="mb-8">
            <div className="w-96 h-12 bg-white/10 rounded animate-pulse mb-2" />
            <div className="w-64 h-6 bg-white/10 rounded animate-pulse" />
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
            <LoadingSkeleton variant="metric" count={6} />
          </div>
          <LoadingSkeleton variant="chart" count={1} />
        </div>
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
          <div className="flex items-start justify-between">
            <div>
              <h1 className="text-5xl font-black text-white mb-2 tracking-tight">Revenue Tracking Dashboard</h1>
              <p className="text-slate-300 tracking-wider">Monitor your business growth and ROI over time</p>
            </div>
            <div className="flex items-center gap-2">
              <NotificationCenter />
              <ConnectionStatus />
            </div>
          </div>
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

          <Button variant="outline" className="bg-white/10 border-white/20 text-white hover:bg-white/20" onClick={exportToCSV}>
            <Download className="w-4 h-4 mr-2" />
            Export CSV
          </Button>
        </div>

        {/* Metrics Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
          <Card className="bg-white/5 backdrop-blur-xl border-white/10 hover:border-white/20 transition-all duration-300 hover:scale-[1.02] shadow-2xl hover:shadow-purple-500/20 p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-gradient-to-br from-green-500/30 to-emerald-500/30 rounded-xl shadow-lg">
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

          <Card className="bg-white/5 backdrop-blur-xl border-white/10 hover:border-white/20 transition-all duration-300 hover:scale-[1.02] shadow-2xl hover:shadow-purple-500/20 p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-gradient-to-br from-blue-500/30 to-cyan-500/30 rounded-xl shadow-lg">
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

          <Card className="bg-white/5 backdrop-blur-xl border-white/10 hover:border-white/20 transition-all duration-300 hover:scale-[1.02] shadow-2xl hover:shadow-purple-500/20 p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-gradient-to-br from-purple-500/30 to-violet-500/30 rounded-xl shadow-lg">
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

          <Card className="bg-white/5 backdrop-blur-xl border-white/10 hover:border-white/20 transition-all duration-300 hover:scale-[1.02] shadow-2xl hover:shadow-purple-500/20 p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-gradient-to-br from-blue-600/30 to-indigo-600/30 rounded-xl shadow-lg">
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

          <Card className="bg-white/5 backdrop-blur-xl border-white/10 hover:border-white/20 transition-all duration-300 hover:scale-[1.02] shadow-2xl hover:shadow-purple-500/20 p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-gradient-to-br from-pink-500/30 to-rose-500/30 rounded-xl shadow-lg">
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

          <Card className="bg-white/5 backdrop-blur-xl border-white/10 hover:border-white/20 transition-all duration-300 hover:scale-[1.02] shadow-2xl hover:shadow-purple-500/20 p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-gradient-to-br from-yellow-500/30 to-amber-500/30 rounded-xl shadow-lg">
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
          <Card className="bg-white/5 backdrop-blur-xl border-white/10 hover:border-white/20 transition-all duration-300 shadow-2xl hover:shadow-cyan-500/10 p-6 mb-8">
            <h2 className="text-3xl font-bold text-white mb-6 tracking-tight">Revenue Growth Over Time</h2>
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
          <Card className="bg-white/5 backdrop-blur-xl border-white/10 hover:border-white/20 transition-all duration-300 shadow-2xl hover:shadow-cyan-500/10 p-6 mb-8">
            <h2 className="text-3xl font-bold text-white mb-6 tracking-tight">Customer Acquisition Trends</h2>
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
          <Card className="bg-white/5 backdrop-blur-xl border-white/10 hover:border-white/20 transition-all duration-300 shadow-2xl hover:shadow-cyan-500/10 p-6 mb-8">
            <h2 className="text-3xl font-bold text-white mb-6 tracking-tight">Website Traffic Improvements</h2>
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
          <Card className="bg-white/5 backdrop-blur-xl border-white/10 hover:border-white/20 transition-all duration-300 shadow-2xl hover:shadow-cyan-500/10 p-6 mb-8">
            <h2 className="text-3xl font-bold text-white mb-6 tracking-tight">LinkedIn Engagement Metrics</h2>
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
          <Card className="bg-white/5 backdrop-blur-xl border-white/10 hover:border-white/20 transition-all duration-300 shadow-2xl hover:shadow-cyan-500/10 p-6 mb-8">
            <h2 className="text-3xl font-bold text-white mb-6 tracking-tight">Social Media Follower Growth</h2>
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
          <Card className="bg-white/5 backdrop-blur-xl border-white/10 hover:border-white/20 transition-all duration-300 hover:scale-[1.02] shadow-2xl hover:shadow-purple-500/20 p-6">
            <h2 className="text-3xl font-bold text-white mb-6 tracking-tight">ROI Summary</h2>
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
