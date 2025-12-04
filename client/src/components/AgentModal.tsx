import { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import {
  Bot,
  MessageSquare,
  Activity,
  TrendingUp,
  Send,
  Loader2,
} from "lucide-react";
import { trpc } from "@/lib/trpc";
import { Streamdown } from "streamdown";

interface Agent {
  id: number;
  name: string;
  status: "active" | "idle";
  capabilities: string[];
  lastActive: Date;
}

interface AgentModalProps {
  agent: Agent | null;
  isOpen: boolean;
  onClose: () => void;
}

export default function AgentModal({ agent, isOpen, onClose }: AgentModalProps) {
  const [testMessage, setTestMessage] = useState("");
  const [chatMessage, setChatMessage] = useState("");
  const [testResult, setTestResult] = useState<string | null>(null);
  const [chatHistory, setChatHistory] = useState<Array<{ role: string; content: string }>>([]);

  const chatMutation = trpc.asi.chat.useMutation();

  // Generate sample performance data
  const performanceData = Array.from({ length: 24 }, (_, i) => ({
    hour: `${i}:00`,
    requests: Math.floor(Math.random() * 100) + 50,
    success: Math.floor(Math.random() * 95) + 48,
    avgTime: Math.floor(Math.random() * 200) + 100,
  }));

  const handleTest = async () => {
    if (!testMessage.trim() || !agent) return;

    setTestResult("Testing agent capability...");

    try {
      const result = await chatMutation.mutateAsync({
        message: testMessage,
        agentId: agent.id,
      });

      setTestResult(result.message || "Test completed successfully!");
    } catch (error) {
      setTestResult(`Test failed: ${error instanceof Error ? error.message : "Unknown error"}`);
    }
  };

  const handleChat = async () => {
    if (!chatMessage.trim() || !agent) return;

    const userMessage = chatMessage;
    setChatMessage("");
    setChatHistory((prev) => [...prev, { role: "user", content: userMessage }]);

    try {
      const result = await chatMutation.mutateAsync({
        message: userMessage,
        agentId: agent.id,
      });

      setChatHistory((prev) => [
        ...prev,
        { role: "assistant", content: result.message || "No response" },
      ]);
    } catch (error) {
      setChatHistory((prev) => [
        ...prev,
        {
          role: "error",
          content: `Error: ${error instanceof Error ? error.message : "Unknown error"}`,
        },
      ]);
    }
  };

  if (!agent) return null;

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Bot className="w-8 h-8 text-primary" />
              <div>
                <DialogTitle className="text-2xl">{agent.name}</DialogTitle>
                <div className="flex items-center gap-2 mt-1">
                  <Badge className={agent.status === "active" ? "badge-success" : "badge-secondary"}>
                    {agent.status}
                  </Badge>
                  <span className="text-sm text-muted-foreground">
                    Agent #{agent.id}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </DialogHeader>

        <Tabs defaultValue="overview" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="performance">Performance</TabsTrigger>
            <TabsTrigger value="test">Test</TabsTrigger>
            <TabsTrigger value="chat">Chat</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-4">
            <Card className="card-elevated p-6">
              <h3 className="text-lg font-bold mb-4">Capabilities</h3>
              <div className="flex flex-wrap gap-2">
                {agent.capabilities.map((cap, idx) => (
                  <Badge key={idx} className="badge-info">
                    {cap}
                  </Badge>
                ))}
              </div>
            </Card>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Card className="card-elevated p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Activity className="w-5 h-5 text-primary" />
                  <span className="text-sm text-muted-foreground">Status</span>
                </div>
                <div className="text-2xl font-bold text-gradient">
                  {agent.status === "active" ? "Active" : "Idle"}
                </div>
              </Card>

              <Card className="card-elevated p-4">
                <div className="flex items-center gap-2 mb-2">
                  <TrendingUp className="w-5 h-5 text-success" />
                  <span className="text-sm text-muted-foreground">Success Rate</span>
                </div>
                <div className="text-2xl font-bold text-gradient">98.5%</div>
              </Card>

              <Card className="card-elevated p-4">
                <div className="flex items-center gap-2 mb-2">
                  <MessageSquare className="w-5 h-5 text-secondary" />
                  <span className="text-sm text-muted-foreground">Avg Response</span>
                </div>
                <div className="text-2xl font-bold text-gradient">145ms</div>
              </Card>
            </div>

            <Card className="card-elevated p-6">
              <h3 className="text-lg font-bold mb-2">Last Active</h3>
              <p className="text-muted-foreground">
                {new Date(agent.lastActive).toLocaleString()}
              </p>
            </Card>
          </TabsContent>

          {/* Performance Tab */}
          <TabsContent value="performance" className="space-y-4">
            <Card className="card-elevated p-6">
              <h3 className="text-lg font-bold mb-4">Request Volume (24h)</h3>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={performanceData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                  <XAxis dataKey="hour" stroke="#888" />
                  <YAxis stroke="#888" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#1a1a1a",
                      border: "1px solid #333",
                      borderRadius: "8px",
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="requests"
                    stroke="#4299e1"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </Card>

            <Card className="card-elevated p-6">
              <h3 className="text-lg font-bold mb-4">Success Rate (24h)</h3>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={performanceData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                  <XAxis dataKey="hour" stroke="#888" />
                  <YAxis stroke="#888" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#1a1a1a",
                      border: "1px solid #333",
                      borderRadius: "8px",
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="success"
                    stroke="#10b981"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          </TabsContent>

          {/* Test Tab */}
          <TabsContent value="test" className="space-y-4">
            <Card className="card-elevated p-6">
              <h3 className="text-lg font-bold mb-4">Test Agent Capability</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Send a test message to verify the agent's capabilities and response quality.
              </p>

              <div className="space-y-4">
                <Input
                  placeholder="Enter test message..."
                  value={testMessage}
                  onChange={(e) => setTestMessage(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      handleTest();
                    }
                  }}
                />

                <Button
                  onClick={handleTest}
                  disabled={!testMessage.trim() || chatMutation.isPending}
                  className="btn-primary w-full"
                >
                  {chatMutation.isPending ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Testing...
                    </>
                  ) : (
                    "Run Test"
                  )}
                </Button>

                {testResult && (
                  <Card className="card-elevated p-4 bg-primary/5">
                    <h4 className="font-bold mb-2">Test Result:</h4>
                    <Streamdown>{testResult}</Streamdown>
                  </Card>
                )}
              </div>
            </Card>
          </TabsContent>

          {/* Chat Tab */}
          <TabsContent value="chat" className="space-y-4">
            <Card className="card-elevated p-6">
              <h3 className="text-lg font-bold mb-4">Direct Chat</h3>

              {/* Chat History */}
              <div className="h-96 overflow-y-auto mb-4 space-y-4 p-4 bg-muted/20 rounded-lg">
                {chatHistory.length === 0 ? (
                  <div className="text-center text-muted-foreground py-8">
                    Start a conversation with {agent.name}
                  </div>
                ) : (
                  chatHistory.map((msg, idx) => (
                    <div
                      key={idx}
                      className={`flex ${
                        msg.role === "user" ? "justify-end" : "justify-start"
                      }`}
                    >
                      <div
                        className={`max-w-[80%] p-4 rounded-lg ${
                          msg.role === "user"
                            ? "bg-primary text-primary-foreground"
                            : msg.role === "error"
                            ? "bg-destructive/20 text-destructive"
                            : "bg-card"
                        }`}
                      >
                        <Streamdown>{msg.content}</Streamdown>
                      </div>
                    </div>
                  ))
                )}
              </div>

              {/* Chat Input */}
              <div className="flex gap-2">
                <Input
                  placeholder="Type your message..."
                  value={chatMessage}
                  onChange={(e) => setChatMessage(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      handleChat();
                    }
                  }}
                  disabled={chatMutation.isPending}
                />
                <Button
                  onClick={handleChat}
                  disabled={!chatMessage.trim() || chatMutation.isPending}
                  className="btn-primary"
                >
                  {chatMutation.isPending ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Send className="w-4 h-4" />
                  )}
                </Button>
              </div>
            </Card>
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
}
