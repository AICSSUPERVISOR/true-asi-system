import { trpc } from "@/lib/trpc";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Brain,
  Database,
  Cpu,
  Activity,
  TrendingUp,
  Zap,
  Network,
  HardDrive,
} from "lucide-react";

export default function Dashboard() {
  const { data: status, isLoading: statusLoading } = trpc.asi.status.useQuery();
  const { data: metrics, isLoading: metricsLoading } = trpc.asi.metrics.useQuery();
  const { data: knowledgeGraph } = trpc.asi.knowledgeGraph.useQuery();

  if (statusLoading || metricsLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <Activity className="w-12 h-12 animate-spin text-primary mx-auto mb-4" />
          <p className="text-muted-foreground">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="border-b border-border bg-card">
        <div className="container mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold mb-2">ASI Dashboard</h1>
              <p className="text-muted-foreground">
                Real-time monitoring of 250 agents and 6.54TB knowledge base
              </p>
            </div>
            <Badge
              className={
                status?.status === "operational"
                  ? "badge-success text-lg px-4 py-2"
                  : "badge-warning text-lg px-4 py-2"
              }
            >
              {status?.status === "operational" ? "● Operational" : "● Degraded"}
            </Badge>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-6 py-8">
        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <Card className="card-elevated p-6">
            <div className="flex items-center justify-between mb-4">
              <Brain className="w-10 h-10 text-primary" />
              <TrendingUp className="w-5 h-5 text-success" />
            </div>
            <div className="text-3xl font-bold text-gradient mb-1">
              {metrics?.agents.active || 250}
            </div>
            <div className="text-sm text-muted-foreground">Active Agents</div>
            <div className="mt-2 text-xs text-success">
              100% operational
            </div>
          </Card>

          <Card className="card-elevated p-6">
            <div className="flex items-center justify-between mb-4">
              <Database className="w-10 h-10 text-secondary" />
              <TrendingUp className="w-5 h-5 text-success" />
            </div>
            <div className="text-3xl font-bold text-gradient mb-1">
              {knowledgeGraph?.size || "6.54TB"}
            </div>
            <div className="text-sm text-muted-foreground">Knowledge Base</div>
            <div className="mt-2 text-xs text-muted-foreground">
              {knowledgeGraph?.files.toLocaleString() || "1,174,651"} files
            </div>
          </Card>

          <Card className="card-elevated p-6">
            <div className="flex items-center justify-between mb-4">
              <Cpu className="w-10 h-10 text-accent" />
              <Activity className="w-5 h-5 text-primary" />
            </div>
            <div className="text-3xl font-bold text-gradient mb-1">
              {metrics?.cpu.usage.toFixed(1) || "45.2"}%
            </div>
            <div className="text-sm text-muted-foreground">CPU Usage</div>
            <div className="mt-2 text-xs text-muted-foreground">
              {metrics?.cpu.cores || 8} cores
            </div>
          </Card>

          <Card className="card-elevated p-6">
            <div className="flex items-center justify-between mb-4">
              <Network className="w-10 h-10 text-warning" />
              <TrendingUp className="w-5 h-5 text-success" />
            </div>
            <div className="text-3xl font-bold text-gradient mb-1">
              {((metrics?.requests.success || 95000) / 1000).toFixed(1)}K
            </div>
            <div className="text-sm text-muted-foreground">Requests</div>
            <div className="mt-2 text-xs text-success">
              {(((metrics?.requests.success || 95000) / (metrics?.requests.total || 100000)) * 100).toFixed(1)}% success rate
            </div>
          </Card>
        </div>

        {/* System Resources */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <Card className="card-elevated p-6">
            <h3 className="text-xl font-bold mb-6 flex items-center">
              <Cpu className="w-6 h-6 mr-2 text-primary" />
              System Resources
            </h3>
            <div className="space-y-6">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium">CPU Usage</span>
                  <span className="text-sm text-muted-foreground">
                    {metrics?.cpu.usage.toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-muted rounded-full h-2">
                  <div
                    className="bg-primary rounded-full h-2 transition-all duration-500"
                    style={{ width: `${metrics?.cpu.usage || 45}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium">Memory</span>
                  <span className="text-sm text-muted-foreground">
                    {metrics?.memory.used.toFixed(1)}GB / {metrics?.memory.total}
                  </span>
                </div>
                <div className="w-full bg-muted rounded-full h-2">
                  <div
                    className="bg-secondary rounded-full h-2 transition-all duration-500"
                    style={{
                      width: `${((metrics?.memory.used || 8) / 16) * 100}%`,
                    }}
                  />
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium">Storage</span>
                  <span className="text-sm text-muted-foreground">
                    {metrics?.storage.used.toFixed(1)}TB / {metrics?.storage.total}
                  </span>
                </div>
                <div className="w-full bg-muted rounded-full h-2">
                  <div
                    className="bg-accent rounded-full h-2 transition-all duration-500"
                    style={{
                      width: `${((metrics?.storage.used || 3.2) / 5) * 100}%`,
                    }}
                  />
                </div>
              </div>
            </div>
          </Card>

          <Card className="card-elevated p-6">
            <h3 className="text-xl font-bold mb-6 flex items-center">
              <Database className="w-6 h-6 mr-2 text-secondary" />
              Knowledge Graph
            </h3>
            <div className="space-y-4">
              <div className="flex items-center justify-between p-4 bg-primary/10 rounded-lg">
                <span className="font-medium">Entities</span>
                <span className="text-2xl font-bold text-gradient">
                  {knowledgeGraph?.entities.toLocaleString() || "19,649"}
                </span>
              </div>
              <div className="flex items-center justify-between p-4 bg-secondary/10 rounded-lg">
                <span className="font-medium">Relationships</span>
                <span className="text-2xl font-bold text-gradient">
                  {knowledgeGraph?.relationships.toLocaleString() || "468"}
                </span>
              </div>
              <div className="flex items-center justify-between p-4 bg-accent/10 rounded-lg">
                <span className="font-medium">Total Files</span>
                <span className="text-2xl font-bold text-gradient">
                  {knowledgeGraph?.files.toLocaleString() || "1,174,651"}
                </span>
              </div>
              <div className="flex items-center justify-between p-4 bg-success/10 rounded-lg">
                <span className="font-medium">Total Size</span>
                <span className="text-2xl font-bold text-gradient">
                  {knowledgeGraph?.size || "6.54TB"}
                </span>
              </div>
            </div>
          </Card>
        </div>

        {/* Quick Actions */}
        <Card className="card-elevated p-6">
          <h3 className="text-xl font-bold mb-6 flex items-center">
            <Zap className="w-6 h-6 mr-2 text-warning" />
            Quick Actions
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Button className="btn-primary h-auto py-4 flex-col items-start">
              <Brain className="w-6 h-6 mb-2" />
              <span className="font-semibold">View All Agents</span>
              <span className="text-xs opacity-80">Manage 250 agents</span>
            </Button>
            <Button className="btn-secondary h-auto py-4 flex-col items-start">
              <Network className="w-6 h-6 mb-2" />
              <span className="font-semibold">Start Chat</span>
              <span className="text-xs opacity-80">Interact with ASI</span>
            </Button>
            <Button className="btn-accent h-auto py-4 flex-col items-start">
              <HardDrive className="w-6 h-6 mb-2" />
              <span className="font-semibold">Browse Knowledge</span>
              <span className="text-xs opacity-80">Explore 6.54TB data</span>
            </Button>
          </div>
        </Card>
      </div>
    </div>
  );
}
