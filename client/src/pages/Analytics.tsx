import { trpc } from "@/lib/trpc";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
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
  AreaChart,
  Area,
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
} from 'recharts';
import {
  TrendingUp,
  TrendingDown,
  DollarSign,
  Users,
  Zap,
  Clock,
  Activity,
  BarChart3,
} from "lucide-react";
import { useState, useMemo } from "react";

export default function Analytics() {
  const { data: metrics, isLoading } = trpc.asi.metrics.useQuery();
  const { data: status } = trpc.asi.status.useQuery();
  const [timeRange, setTimeRange] = useState("7d");

  // Generate sample time-series data
  const performanceData = useMemo(() => {
    const data = [];
    for (let i = 0; i < 24; i++) {
      data.push({
        time: `${i}:00`,
        requests: Math.floor(Math.random() * 5000) + 2000,
        success: Math.floor(Math.random() * 4800) + 1900,
        failed: Math.floor(Math.random() * 200) + 50,
        avgResponseTime: Math.floor(Math.random() * 300) + 100,
      });
    }
    return data;
  }, []);

  const costData = useMemo(() => {
    const data = [];
    const models = ['GPT-4', 'Claude', 'Gemini', 'DeepSeek', 'Grok'];
    for (const model of models) {
      data.push({
        model,
        cost: Math.floor(Math.random() * 5000) + 1000,
        requests: Math.floor(Math.random() * 10000) + 5000,
      });
    }
    return data;
  }, []);

  const agentActivityData = useMemo(() => {
    return [
      { name: 'Active', value: 250, color: '#10b981' },
      { name: 'Idle', value: 0, color: '#6b7280' },
    ];
  }, []);

  const usageByCapability = useMemo(() => {
    return [
      { capability: 'Reasoning', usage: 8500 },
      { capability: 'Coding', usage: 7200 },
      { capability: 'Analysis', usage: 6800 },
      { capability: 'Research', usage: 5400 },
      { capability: 'Writing', usage: 4900 },
    ];
  }, []);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <Activity className="w-12 h-12 animate-spin text-primary mx-auto mb-4" />
          <p className="text-muted-foreground">Loading analytics...</p>
        </div>
      </div>
    );
  }

  const totalCost = costData.reduce((sum, item) => sum + item.cost, 0);
  const totalRequests = performanceData.reduce((sum, item) => sum + item.requests, 0);
  const avgResponseTime = Math.floor(
    performanceData.reduce((sum, item) => sum + item.avgResponseTime, 0) / performanceData.length
  );

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="border-b border-border bg-card">
        <div className="container mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <BarChart3 className="w-8 h-8 text-primary" />
              <div>
                <h1 className="text-4xl font-bold mb-2">Analytics Dashboard</h1>
                <p className="text-muted-foreground">
                  Comprehensive performance metrics and cost analysis
                </p>
              </div>
            </div>
            <Select value={timeRange} onValueChange={setTimeRange}>
              <SelectTrigger className="w-40">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="24h">Last 24 Hours</SelectItem>
                <SelectItem value="7d">Last 7 Days</SelectItem>
                <SelectItem value="30d">Last 30 Days</SelectItem>
                <SelectItem value="90d">Last 90 Days</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-6 py-8">
        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <Card className="card-elevated p-6">
            <div className="flex items-center justify-between mb-4">
              <Zap className="w-10 h-10 text-primary" />
              <Badge className="badge-success">
                <TrendingUp className="w-3 h-3 mr-1" />
                +12%
              </Badge>
            </div>
            <div className="text-3xl font-bold text-gradient mb-1">
              {totalRequests.toLocaleString()}
            </div>
            <div className="text-sm text-muted-foreground">Total Requests</div>
            <div className="mt-2 text-xs text-success">
              vs. last period
            </div>
          </Card>

          <Card className="card-elevated p-6">
            <div className="flex items-center justify-between mb-4">
              <Clock className="w-10 h-10 text-secondary" />
              <Badge className="badge-success">
                <TrendingDown className="w-3 h-3 mr-1" />
                -8%
              </Badge>
            </div>
            <div className="text-3xl font-bold text-gradient mb-1">
              {avgResponseTime}ms
            </div>
            <div className="text-sm text-muted-foreground">Avg Response Time</div>
            <div className="mt-2 text-xs text-success">
              Faster than before
            </div>
          </Card>

          <Card className="card-elevated p-6">
            <div className="flex items-center justify-between mb-4">
              <DollarSign className="w-10 h-10 text-warning" />
              <Badge className="badge-warning">
                <TrendingUp className="w-3 h-3 mr-1" />
                +5%
              </Badge>
            </div>
            <div className="text-3xl font-bold text-gradient mb-1">
              ${totalCost.toLocaleString()}
            </div>
            <div className="text-sm text-muted-foreground">Total Cost</div>
            <div className="mt-2 text-xs text-muted-foreground">
              This period
            </div>
          </Card>

          <Card className="card-elevated p-6">
            <div className="flex items-center justify-between mb-4">
              <Users className="w-10 h-10 text-accent" />
              <Badge className="badge-success">
                100%
              </Badge>
            </div>
            <div className="text-3xl font-bold text-gradient mb-1">
              {metrics?.agents.active || 250}
            </div>
            <div className="text-sm text-muted-foreground">Active Agents</div>
            <div className="mt-2 text-xs text-success">
              All operational
            </div>
          </Card>
        </div>

        {/* Charts Row 1 */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <Card className="card-elevated p-6">
            <h3 className="text-xl font-bold mb-6">Request Volume (24h)</h3>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={performanceData}>
                <defs>
                  <linearGradient id="colorRequests" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#4299e1" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#4299e1" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="time" stroke="#888" />
                <YAxis stroke="#888" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1a1a1a',
                    border: '1px solid #333',
                    borderRadius: '8px',
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="requests"
                  stroke="#4299e1"
                  fillOpacity={1}
                  fill="url(#colorRequests)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </Card>

          <Card className="card-elevated p-6">
            <h3 className="text-xl font-bold mb-6">Success vs Failed Requests</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="time" stroke="#888" />
                <YAxis stroke="#888" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1a1a1a',
                    border: '1px solid #333',
                    borderRadius: '8px',
                  }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="success"
                  stroke="#10b981"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="failed"
                  stroke="#ef4444"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </div>

        {/* Charts Row 2 */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <Card className="card-elevated p-6">
            <h3 className="text-xl font-bold mb-6">Cost by Model</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={costData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="model" stroke="#888" />
                <YAxis stroke="#888" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1a1a1a',
                    border: '1px solid #333',
                    borderRadius: '8px',
                  }}
                />
                <Bar dataKey="cost" fill="#8b5cf6" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card className="card-elevated p-6">
            <h3 className="text-xl font-bold mb-6">Usage by Capability</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={usageByCapability} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis type="number" stroke="#888" />
                <YAxis dataKey="capability" type="category" stroke="#888" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1a1a1a',
                    border: '1px solid #333',
                    borderRadius: '8px',
                  }}
                />
                <Bar dataKey="usage" fill="#00d9ff" radius={[0, 8, 8, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>

        {/* ROI and Cost Analysis */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <Card className="card-elevated p-6">
            <h3 className="text-xl font-bold mb-6">Agent Activity</h3>
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie
                  data={agentActivityData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {agentActivityData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div className="text-center mt-4">
              <div className="text-3xl font-bold text-gradient">100%</div>
              <div className="text-sm text-muted-foreground">Operational</div>
            </div>
          </Card>

          <Card className="card-elevated p-6 lg:col-span-2">
            <h3 className="text-xl font-bold mb-6">ROI Metrics</h3>
            <div className="space-y-4">
              <div className="flex items-center justify-between p-4 bg-primary/10 rounded-lg">
                <div>
                  <div className="text-sm text-muted-foreground mb-1">Cost per Request</div>
                  <div className="text-2xl font-bold text-gradient">
                    ${(totalCost / totalRequests).toFixed(4)}
                  </div>
                </div>
                <Badge className="badge-success">Optimized</Badge>
              </div>

              <div className="flex items-center justify-between p-4 bg-secondary/10 rounded-lg">
                <div>
                  <div className="text-sm text-muted-foreground mb-1">Efficiency Score</div>
                  <div className="text-2xl font-bold text-gradient">98.5%</div>
                </div>
                <Badge className="badge-success">Excellent</Badge>
              </div>

              <div className="flex items-center justify-between p-4 bg-accent/10 rounded-lg">
                <div>
                  <div className="text-sm text-muted-foreground mb-1">Projected Monthly Cost</div>
                  <div className="text-2xl font-bold text-gradient">
                    ${(totalCost * 30).toLocaleString()}
                  </div>
                </div>
                <Badge className="badge-warning">Monitor</Badge>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
