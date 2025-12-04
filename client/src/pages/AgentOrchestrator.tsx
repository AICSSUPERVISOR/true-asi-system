import { useState } from "react";
import { trpc } from "@/lib/trpc";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  Network, 
  Play, 
  Plus, 
  X, 
  ArrowRight, 
  Zap, 
  Brain, 
  Code, 
  Database,
  CheckCircle2,
  Clock
} from "lucide-react";
import { Textarea } from "@/components/ui/textarea";

interface SelectedAgent {
  id: number;
  name: string;
  capabilities: string[];
  role: string;
}

interface WorkflowStep {
  id: string;
  agentId: number;
  agentName: string;
  input: string;
  output?: string;
  status: "pending" | "running" | "complete" | "error";
  duration?: number;
}

export default function AgentOrchestrator() {
  const [selectedAgents, setSelectedAgents] = useState<SelectedAgent[]>([]);
  const [workflowSteps, setWorkflowSteps] = useState<WorkflowStep[]>([]);
  const [question, setQuestion] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const [activeStep, setActiveStep] = useState<number | null>(null);

  const { data: agents } = trpc.asi.agents.useQuery();
  const chatMutation = trpc.asi.chat.useMutation();

  // Predefined agent roles for collaboration
  const agentRoles = [
    { id: 0, name: "Research Coordinator", icon: Database, color: "cyan" },
    { id: 1, name: "Mathematical Analyst", icon: Brain, color: "purple" },
    { id: 2, name: "Code Synthesizer", icon: Code, color: "blue" },
    { id: 3, name: "Logic Validator", icon: CheckCircle2, color: "green" },
    { id: 4, name: "Integration Specialist", icon: Network, color: "orange" },
  ];

  const addAgent = (agentId: number) => {
    if (selectedAgents.length >= 5) {
      alert("Maximum 5 agents allowed in a workflow");
      return;
    }

    const role = agentRoles[selectedAgents.length];
    const agent = agents?.find(a => a.id === agentId);
    
    if (agent && !selectedAgents.find(a => a.id === agentId)) {
      setSelectedAgents([...selectedAgents, {
        ...agent,
        role: role.name,
      }]);
    }
  };

  const removeAgent = (agentId: number) => {
    setSelectedAgents(selectedAgents.filter(a => a.id !== agentId));
  };

  const runWorkflow = async () => {
    if (selectedAgents.length < 2) {
      alert("Please select at least 2 agents for collaboration");
      return;
    }

    if (!question.trim()) {
      alert("Please enter a question or task");
      return;
    }

    setIsRunning(true);
    setWorkflowSteps([]);

    // Create workflow steps
    const steps: WorkflowStep[] = selectedAgents.map((agent, index) => ({
      id: `step-${index}`,
      agentId: agent.id,
      agentName: agent.name,
      input: index === 0 ? question : "Output from previous agent",
      status: "pending",
    }));

    setWorkflowSteps(steps);

    // Execute workflow sequentially
    let previousOutput = question;

    for (let i = 0; i < steps.length; i++) {
      setActiveStep(i);
      setWorkflowSteps(prev => prev.map((step, idx) => 
        idx === i ? { ...step, status: "running" } : step
      ));

      const startTime = Date.now();

      try {
        const result = await chatMutation.mutateAsync({
          message: `${selectedAgents[i].role}: ${previousOutput}`,
          model: "gpt-4",
          agentId: selectedAgents[i].id,
        });

        const duration = Date.now() - startTime;
        const output = result.message || "No response";

        setWorkflowSteps(prev => prev.map((step, idx) => 
          idx === i ? { 
            ...step, 
            status: "complete", 
            output,
            duration 
          } : step
        ));

        previousOutput = output;
      } catch (error) {
        setWorkflowSteps(prev => prev.map((step, idx) => 
          idx === i ? { 
            ...step, 
            status: "error", 
            output: "Error: Failed to execute step" 
          } : step
        ));
        break;
      }
    }

    setIsRunning(false);
    setActiveStep(null);
  };

  const clearWorkflow = () => {
    setSelectedAgents([]);
    setWorkflowSteps([]);
    setQuestion("");
    setActiveStep(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-950 text-white p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div className="text-center space-y-4">
          <h1 className="text-5xl font-bold bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400 bg-clip-text text-transparent">
            Agent Collaboration Orchestrator
          </h1>
          <p className="text-xl text-slate-300">
            Combine multiple specialized agents to solve complex S-7 questions
          </p>
        </div>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Agent Selection Panel */}
          <Card className="bg-slate-900/50 border-slate-700/50 backdrop-blur lg:col-span-1">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Plus className="w-5 h-5 text-cyan-400" />
                Select Agents (2-5)
              </CardTitle>
              <CardDescription>
                Choose agents to collaborate on your task
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Selected Agents */}
              <div className="space-y-2">
                <div className="text-sm font-semibold text-slate-400">Selected ({selectedAgents.length}/5)</div>
                {selectedAgents.length === 0 ? (
                  <div className="text-sm text-slate-500 py-4 text-center">
                    No agents selected
                  </div>
                ) : (
                  selectedAgents.map((agent, index) => {
                    const roleInfo = agentRoles[index];
                    return (
                      <div
                        key={agent.id}
                        className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg border border-slate-700/30"
                      >
                        <div className="flex items-center gap-3">
                          <div className={`p-2 rounded-lg bg-${roleInfo.color}-500/20`}>
                            <roleInfo.icon className={`w-4 h-4 text-${roleInfo.color}-400`} />
                          </div>
                          <div>
                            <div className="font-semibold text-sm">{agent.name}</div>
                            <div className="text-xs text-slate-400">{agent.role}</div>
                          </div>
                        </div>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => removeAgent(agent.id)}
                          className="h-8 w-8 p-0"
                        >
                          <X className="w-4 h-4" />
                        </Button>
                      </div>
                    );
                  })
                )}
              </div>

              {/* Available Agents */}
              {selectedAgents.length < 5 && (
                <div className="space-y-2">
                  <div className="text-sm font-semibold text-slate-400">Available Agents</div>
                  <div className="max-h-64 overflow-y-auto space-y-1">
                    {agents?.slice(0, 20).map(agent => (
                      <Button
                        key={agent.id}
                        variant="ghost"
                        size="sm"
                        onClick={() => addAgent(agent.id)}
                        disabled={selectedAgents.some(a => a.id === agent.id)}
                        className="w-full justify-start text-left h-auto py-2"
                      >
                        <div>
                          <div className="font-semibold text-sm">{agent.name}</div>
                          <div className="text-xs text-slate-400">
                            {agent.capabilities.slice(0, 2).join(", ")}
                          </div>
                        </div>
                      </Button>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Workflow Builder & Execution */}
          <Card className="bg-slate-900/50 border-slate-700/50 backdrop-blur lg:col-span-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Network className="w-5 h-5 text-purple-400" />
                Collaborative Workflow
              </CardTitle>
              <CardDescription>
                Define the task and execute the multi-agent workflow
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Question Input */}
              <div className="space-y-2">
                <label className="text-sm font-semibold text-slate-300">
                  Question or Task
                </label>
                <Textarea
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="Enter an S-7 question or complex task that requires multiple agents to solve..."
                  className="min-h-[100px] bg-slate-800/50 border-slate-700/50"
                  disabled={isRunning}
                />
              </div>

              {/* Control Buttons */}
              <div className="flex gap-2">
                <Button
                  onClick={runWorkflow}
                  disabled={isRunning || selectedAgents.length < 2 || !question.trim()}
                  className="flex-1 bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-600 hover:to-blue-600"
                >
                  {isRunning ? (
                    <>
                      <Clock className="w-4 h-4 mr-2 animate-spin" />
                      Running Workflow...
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4 mr-2" />
                      Run Workflow
                    </>
                  )}
                </Button>
                <Button
                  onClick={clearWorkflow}
                  disabled={isRunning}
                  variant="outline"
                >
                  Clear
                </Button>
              </div>

              {/* Workflow Visualization */}
              {workflowSteps.length > 0 && (
                <div className="space-y-3 mt-6">
                  <div className="text-sm font-semibold text-slate-300">Workflow Execution</div>
                  {workflowSteps.map((step, index) => (
                    <div key={step.id}>
                      <div
                        className={`p-4 rounded-lg border transition-all ${
                          step.status === "complete"
                            ? "bg-green-500/10 border-green-500/30"
                            : step.status === "running"
                            ? "bg-blue-500/10 border-blue-500/30 animate-pulse"
                            : step.status === "error"
                            ? "bg-red-500/10 border-red-500/30"
                            : "bg-slate-800/30 border-slate-700/30"
                        }`}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-3">
                            <div className="text-lg font-bold text-slate-500">#{index + 1}</div>
                            <div>
                              <div className="font-semibold">{step.agentName}</div>
                              <div className="text-xs text-slate-400">
                                {selectedAgents[index]?.role}
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center gap-2">
                            {step.duration && (
                              <Badge variant="outline" className="text-xs">
                                {step.duration}ms
                              </Badge>
                            )}
                            {step.status === "complete" && (
                              <CheckCircle2 className="w-5 h-5 text-green-400" />
                            )}
                            {step.status === "running" && (
                              <Clock className="w-5 h-5 text-blue-400 animate-spin" />
                            )}
                          </div>
                        </div>

                        {step.output && (
                          <div className="mt-3 p-3 bg-slate-950/50 rounded text-sm text-slate-300">
                            {step.output.slice(0, 200)}
                            {step.output.length > 200 && "..."}
                          </div>
                        )}
                      </div>

                      {index < workflowSteps.length - 1 && (
                        <div className="flex justify-center py-2">
                          <ArrowRight className="w-5 h-5 text-slate-600" />
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Info Cards */}
        <div className="grid md:grid-cols-3 gap-4">
          <Card className="bg-slate-900/50 border-cyan-500/30 backdrop-blur">
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Zap className="w-5 h-5 text-cyan-400" />
                Sequential Processing
              </CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-slate-300">
              Agents process the task sequentially, with each agent building on the output of the previous one for comprehensive analysis.
            </CardContent>
          </Card>

          <Card className="bg-slate-900/50 border-purple-500/30 backdrop-blur">
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Brain className="w-5 h-5 text-purple-400" />
                Specialized Roles
              </CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-slate-300">
              Each agent is assigned a specialized role (Research, Analysis, Synthesis, Validation, Integration) for optimal collaboration.
            </CardContent>
          </Card>

          <Card className="bg-slate-900/50 border-blue-500/30 backdrop-blur">
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Network className="w-5 h-5 text-blue-400" />
                Real-time Monitoring
              </CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-slate-300">
              Watch the workflow execute in real-time with status updates, performance metrics, and intermediate results from each agent.
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
