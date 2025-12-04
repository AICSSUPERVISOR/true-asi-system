import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Book,
  Code,
  Rocket,
  Shield,
  Zap,
  Search,
  ExternalLink,
  Copy,
  CheckCircle2,
} from "lucide-react";
import { useState } from "react";
import { Streamdown } from "streamdown";

const API_EXAMPLES = {
  chat: `// Chat with ASI
const response = await trpc.asi.chat.mutate({
  message: "Explain quantum computing",
  model: "gpt-4",
  agentId: 42
});

console.log(response.message);`,

  agents: `// Get all agents
const agents = await trpc.asi.agents.useQuery();

// Filter active agents
const activeAgents = agents.filter(a => a.status === "active");

console.log(\`\${activeAgents.length} agents ready\`);`,

  metrics: `// Get system metrics
const metrics = await trpc.asi.metrics.useQuery();

console.log({
  cpu: metrics.cpu.usage,
  memory: metrics.memory.used,
  agents: metrics.agents.active
});`,

  knowledge: `// Query knowledge graph
const graph = await trpc.asi.knowledgeGraph.useQuery();

console.log({
  entities: graph.entities,
  relationships: graph.relationships,
  size: graph.size
});`,
};

export default function Documentation() {
  const [searchTerm, setSearchTerm] = useState("");
  const [copiedCode, setCopiedCode] = useState<string | null>(null);

  const copyToClipboard = (code: string, id: string) => {
    navigator.clipboard.writeText(code);
    setCopiedCode(id);
    setTimeout(() => setCopiedCode(null), 2000);
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="border-b border-border bg-card">
        <div className="container mx-auto px-6 py-8">
          <div className="flex items-center gap-4 mb-6">
            <Book className="w-10 h-10 text-primary" />
            <div>
              <h1 className="text-4xl font-bold mb-2">Documentation</h1>
              <p className="text-muted-foreground">
                Complete guide to building with TRUE ASI
              </p>
            </div>
          </div>

          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
            <Input
              type="text"
              placeholder="Search documentation..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10"
            />
          </div>
        </div>
      </div>

      <div className="container mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Sidebar */}
          <div className="lg:col-span-1">
            <Card className="card-elevated p-4 sticky top-6">
              <h3 className="font-bold mb-4">Quick Links</h3>
              <nav className="space-y-2">
                <a href="#getting-started" className="block py-2 px-3 rounded hover:bg-primary/10 transition-colors">
                  Getting Started
                </a>
                <a href="#api-reference" className="block py-2 px-3 rounded hover:bg-primary/10 transition-colors">
                  API Reference
                </a>
                <a href="#agents" className="block py-2 px-3 rounded hover:bg-primary/10 transition-colors">
                  Agent System
                </a>
                <a href="#knowledge-graph" className="block py-2 px-3 rounded hover:bg-primary/10 transition-colors">
                  Knowledge Graph
                </a>
                <a href="#best-practices" className="block py-2 px-3 rounded hover:bg-primary/10 transition-colors">
                  Best Practices
                </a>
                <a href="#examples" className="block py-2 px-3 rounded hover:bg-primary/10 transition-colors">
                  Examples
                </a>
              </nav>
            </Card>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3 space-y-8">
            {/* Getting Started */}
            <section id="getting-started">
              <h2 className="text-3xl font-bold mb-6">Getting Started</h2>
              
              <Card className="card-elevated p-6 mb-6">
                <h3 className="text-xl font-bold mb-4">What is TRUE ASI?</h3>
                <Streamdown>
{`TRUE ASI is an **artificial superintelligence system** with:

- **250 specialized agents** for different tasks
- **6.54TB knowledge base** with 1.17M files
- **Real-time access** to all leading AI models
- **19,649 entities** in the knowledge graph
- **468 relationships** connecting concepts

Built to **outcompete every AI system on the planet**.`}
                </Streamdown>
              </Card>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <Card className="card-elevated p-6 text-center">
                  <Rocket className="w-12 h-12 text-primary mx-auto mb-4" />
                  <h3 className="font-bold mb-2">Quick Start</h3>
                  <p className="text-sm text-muted-foreground mb-4">
                    Get up and running in minutes
                  </p>
                  <Button className="btn-primary w-full" size="sm">
                    Start Tutorial
                  </Button>
                </Card>

                <Card className="card-elevated p-6 text-center">
                  <Code className="w-12 h-12 text-secondary mx-auto mb-4" />
                  <h3 className="font-bold mb-2">API Reference</h3>
                  <p className="text-sm text-muted-foreground mb-4">
                    Complete API documentation
                  </p>
                  <Button className="btn-secondary w-full" size="sm">
                    View APIs
                  </Button>
                </Card>

                <Card className="card-elevated p-6 text-center">
                  <Shield className="w-12 h-12 text-accent mx-auto mb-4" />
                  <h3 className="font-bold mb-2">Best Practices</h3>
                  <p className="text-sm text-muted-foreground mb-4">
                    Learn production patterns
                  </p>
                  <Button className="btn-accent w-full" size="sm">
                    Read Guide
                  </Button>
                </Card>
              </div>
            </section>

            {/* API Reference */}
            <section id="api-reference">
              <h2 className="text-3xl font-bold mb-6">API Reference</h2>

              <Tabs defaultValue="chat" className="w-full">
                <TabsList className="grid w-full grid-cols-4">
                  <TabsTrigger value="chat">Chat</TabsTrigger>
                  <TabsTrigger value="agents">Agents</TabsTrigger>
                  <TabsTrigger value="metrics">Metrics</TabsTrigger>
                  <TabsTrigger value="knowledge">Knowledge</TabsTrigger>
                </TabsList>

                <TabsContent value="chat">
                  <Card className="card-elevated p-6">
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <h3 className="text-xl font-bold">asi.chat</h3>
                        <p className="text-sm text-muted-foreground">
                          Send messages to ASI agents
                        </p>
                      </div>
                      <Badge className="badge-success">Authenticated</Badge>
                    </div>

                    <div className="bg-muted/50 rounded-lg p-4 mb-4 relative">
                      <Button
                        variant="ghost"
                        size="sm"
                        className="absolute top-2 right-2"
                        onClick={() => copyToClipboard(API_EXAMPLES.chat, 'chat')}
                      >
                        {copiedCode === 'chat' ? (
                          <CheckCircle2 className="w-4 h-4 text-success" />
                        ) : (
                          <Copy className="w-4 h-4" />
                        )}
                      </Button>
                      <pre className="text-sm overflow-x-auto">
                        <code>{API_EXAMPLES.chat}</code>
                      </pre>
                    </div>

                    <h4 className="font-bold mb-2">Parameters</h4>
                    <div className="space-y-2 mb-4">
                      <div className="flex items-start gap-2">
                        <code className="text-sm bg-primary/10 px-2 py-1 rounded">message</code>
                        <span className="text-sm text-muted-foreground">
                          (string) - The message to send
                        </span>
                      </div>
                      <div className="flex items-start gap-2">
                        <code className="text-sm bg-primary/10 px-2 py-1 rounded">model</code>
                        <span className="text-sm text-muted-foreground">
                          (string, optional) - AI model to use
                        </span>
                      </div>
                      <div className="flex items-start gap-2">
                        <code className="text-sm bg-primary/10 px-2 py-1 rounded">agentId</code>
                        <span className="text-sm text-muted-foreground">
                          (number, optional) - Specific agent ID
                        </span>
                      </div>
                    </div>

                    <h4 className="font-bold mb-2">Response</h4>
                    <div className="bg-muted/50 rounded-lg p-4">
                      <pre className="text-sm overflow-x-auto">
                        <code>{`{
  success: boolean,
  message: string,
  model: string,
  agentId?: number
}`}</code>
                      </pre>
                    </div>
                  </Card>
                </TabsContent>

                <TabsContent value="agents">
                  <Card className="card-elevated p-6">
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <h3 className="text-xl font-bold">asi.agents</h3>
                        <p className="text-sm text-muted-foreground">
                          Get list of all 250 agents
                        </p>
                      </div>
                      <Badge className="badge-info">Public</Badge>
                    </div>

                    <div className="bg-muted/50 rounded-lg p-4 mb-4 relative">
                      <Button
                        variant="ghost"
                        size="sm"
                        className="absolute top-2 right-2"
                        onClick={() => copyToClipboard(API_EXAMPLES.agents, 'agents')}
                      >
                        {copiedCode === 'agents' ? (
                          <CheckCircle2 className="w-4 h-4 text-success" />
                        ) : (
                          <Copy className="w-4 h-4" />
                        )}
                      </Button>
                      <pre className="text-sm overflow-x-auto">
                        <code>{API_EXAMPLES.agents}</code>
                      </pre>
                    </div>

                    <h4 className="font-bold mb-2">Response</h4>
                    <div className="bg-muted/50 rounded-lg p-4">
                      <pre className="text-sm overflow-x-auto">
                        <code>{`Array<{
  id: number,
  name: string,
  status: "active" | "idle",
  capabilities: string[],
  lastActive: Date
}>`}</code>
                      </pre>
                    </div>
                  </Card>
                </TabsContent>

                <TabsContent value="metrics">
                  <Card className="card-elevated p-6">
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <h3 className="text-xl font-bold">asi.metrics</h3>
                        <p className="text-sm text-muted-foreground">
                          Get system performance metrics
                        </p>
                      </div>
                      <Badge className="badge-success">Authenticated</Badge>
                    </div>

                    <div className="bg-muted/50 rounded-lg p-4 mb-4 relative">
                      <Button
                        variant="ghost"
                        size="sm"
                        className="absolute top-2 right-2"
                        onClick={() => copyToClipboard(API_EXAMPLES.metrics, 'metrics')}
                      >
                        {copiedCode === 'metrics' ? (
                          <CheckCircle2 className="w-4 h-4 text-success" />
                        ) : (
                          <Copy className="w-4 h-4" />
                        )}
                      </Button>
                      <pre className="text-sm overflow-x-auto">
                        <code>{API_EXAMPLES.metrics}</code>
                      </pre>
                    </div>
                  </Card>
                </TabsContent>

                <TabsContent value="knowledge">
                  <Card className="card-elevated p-6">
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <h3 className="text-xl font-bold">asi.knowledgeGraph</h3>
                        <p className="text-sm text-muted-foreground">
                          Access knowledge graph statistics
                        </p>
                      </div>
                      <Badge className="badge-info">Public</Badge>
                    </div>

                    <div className="bg-muted/50 rounded-lg p-4 mb-4 relative">
                      <Button
                        variant="ghost"
                        size="sm"
                        className="absolute top-2 right-2"
                        onClick={() => copyToClipboard(API_EXAMPLES.knowledge, 'knowledge')}
                      >
                        {copiedCode === 'knowledge' ? (
                          <CheckCircle2 className="w-4 h-4 text-success" />
                        ) : (
                          <Copy className="w-4 h-4" />
                        )}
                      </Button>
                      <pre className="text-sm overflow-x-auto">
                        <code>{API_EXAMPLES.knowledge}</code>
                      </pre>
                    </div>
                  </Card>
                </TabsContent>
              </Tabs>
            </section>

            {/* Best Practices */}
            <section id="best-practices">
              <h2 className="text-3xl font-bold mb-6">Best Practices</h2>

              <div className="space-y-4">
                <Card className="card-elevated p-6">
                  <div className="flex items-start gap-4">
                    <Zap className="w-8 h-8 text-warning flex-shrink-0 mt-1" />
                    <div>
                      <h3 className="text-lg font-bold mb-2">Optimize Agent Selection</h3>
                      <p className="text-muted-foreground mb-4">
                        Choose the right agent for your task. Use reasoning agents for complex logic,
                        coding agents for development tasks, and analysis agents for data processing.
                      </p>
                      <Button variant="outline" size="sm" className="gap-2">
                        Learn More
                        <ExternalLink className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                </Card>

                <Card className="card-elevated p-6">
                  <div className="flex items-start gap-4">
                    <Shield className="w-8 h-8 text-success flex-shrink-0 mt-1" />
                    <div>
                      <h3 className="text-lg font-bold mb-2">Handle Rate Limits</h3>
                      <p className="text-muted-foreground mb-4">
                        Implement exponential backoff and request queuing to handle API rate limits
                        gracefully. Monitor your usage in the Analytics dashboard.
                      </p>
                      <Button variant="outline" size="sm" className="gap-2">
                        View Examples
                        <ExternalLink className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                </Card>

                <Card className="card-elevated p-6">
                  <div className="flex items-start gap-4">
                    <Code className="w-8 h-8 text-primary flex-shrink-0 mt-1" />
                    <div>
                      <h3 className="text-lg font-bold mb-2">Error Handling</h3>
                      <p className="text-muted-foreground mb-4">
                        Always implement proper error handling. Check response.success before using
                        data and provide fallback mechanisms for critical operations.
                      </p>
                      <Button variant="outline" size="sm" className="gap-2">
                        See Patterns
                        <ExternalLink className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                </Card>
              </div>
            </section>
          </div>
        </div>
      </div>
    </div>
  );
}
