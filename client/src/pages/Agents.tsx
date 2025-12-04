import { trpc } from "@/lib/trpc";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Brain, Search, Filter, Activity, CheckCircle2 } from "lucide-react";
import { useState } from "react";
import AgentModal from "@/components/AgentModal";

export default function Agents() {
  const { data: agents, isLoading } = trpc.asi.agents.useQuery();
  const [searchTerm, setSearchTerm] = useState("");
  type Agent = NonNullable<typeof agents>[number];
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleAgentClick = (agent: Agent) => {
    setSelectedAgent(agent);
    setIsModalOpen(true);
  };

  const filteredAgents = agents?.filter((agent) =>
    agent.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <Activity className="w-12 h-12 animate-spin text-primary mx-auto mb-4" />
          <p className="text-muted-foreground">Loading agents...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="border-b border-border bg-card">
        <div className="container mx-auto px-6 py-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h1 className="text-4xl font-bold mb-2">ASI Agents</h1>
              <p className="text-muted-foreground">
                Manage and monitor all 250 specialized agents
              </p>
            </div>
            <Badge className="badge-success text-lg px-4 py-2">
              {agents?.length || 250} Active
            </Badge>
          </div>

          {/* Search and Filters */}
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
              <Input
                type="text"
                placeholder="Search agents..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10"
              />
            </div>
            <Button variant="outline" className="gap-2">
              <Filter className="w-4 h-4" />
              Filters
            </Button>
          </div>
        </div>
      </div>

      {/* Agents Grid */}
      <div className="container mx-auto px-6 py-8">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {filteredAgents?.map((agent) => (
            <Card
              key={agent.id}
              className="card-elevated p-6 hover:shadow-2xl transition-all duration-300 group cursor-pointer"
            >
              <div className="flex items-start justify-between mb-4">
                <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center group-hover:scale-110 transition-transform">
                  <Brain className="w-6 h-6 text-primary" />
                </div>
                <Badge className="badge-success">
                  <CheckCircle2 className="w-3 h-3 mr-1" />
                  {agent.status}
                </Badge>
              </div>

              <h3 className="text-lg font-bold mb-2">{agent.name}</h3>

              <div className="space-y-2 mb-4">
                <div className="text-sm text-muted-foreground">Capabilities:</div>
                <div className="flex flex-wrap gap-2">
                  {agent.capabilities.map((cap, i) => (
                    <Badge
                      key={i}
                      variant="outline"
                      className="text-xs"
                    >
                      {cap}
                    </Badge>
                  ))}
                </div>
              </div>

              <div className="text-xs text-muted-foreground">
                Last active: {new Date(agent.lastActive).toLocaleString()}
              </div>

              <Button
                className="w-full mt-4 btn-primary"
                size="sm"
                onClick={() => handleAgentClick(agent)}
              >
                Interact
              </Button>
            </Card>
          ))}
        </div>

        {filteredAgents?.length === 0 && (
          <div className="text-center py-12">
            <Brain className="w-16 h-16 text-muted-foreground mx-auto mb-4 opacity-50" />
            <p className="text-muted-foreground">No agents found matching your search.</p>
          </div>
        )}
      </div>

      {/* Agent Modal */}
      <AgentModal
        agent={selectedAgent as any}
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
      />
    </div>
  );
}