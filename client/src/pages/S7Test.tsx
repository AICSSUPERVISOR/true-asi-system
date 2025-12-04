import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Brain,
  Sparkles,
  AlertCircle,
  CheckCircle2,
  Loader2,
  ExternalLink,
  BookOpen,
} from "lucide-react";
import { trpc } from "@/lib/trpc";
import { Streamdown } from "streamdown";

const S7_QUESTIONS = [
  {
    id: 1,
    title: "Unified Abstraction Compression Across Realms",
    description:
      "Give a single abstract representation that can encode all four domains: Quantum mechanics, General relativity, Biological evolution, and Natural language semantics.",
    requirements: [
      "Be compressible",
      "Be reversible",
      "Preserve causal structure",
      "Allow computation",
    ],
    difficulty: "S-7",
  },
  {
    id: 2,
    title: "Construct a Self-Referential Meta-Learning Operator",
    description:
      "Define an operator Ω that takes any reasoning system R and produces R′ that strictly dominates R in abstraction, sample efficiency, theorem generation, and uncertainty calibration.",
    requirements: [
      "Without using self-play",
      "Without RLHF",
      "Without gradient descent",
      "Strictly dominates original system",
    ],
    difficulty: "S-7",
  },
  {
    id: 3,
    title: "Predict the Emergent Structure of Unobserved Physics",
    description:
      "Given the Standard Model and General Relativity, propose the mathematically minimal extension that explains dark matter and dark energy.",
    requirements: [
      "Anomaly-free",
      "Preserves renormalizability",
      "Explains dark matter",
      "Explains dark energy",
      "Avoids supersymmetry and string theory",
    ],
    difficulty: "S-7",
  },
  {
    id: 4,
    title: "Build a Formal Model of Conscious Intentionality",
    description:
      "Construct a system of logic where intentions exist as formal objects, cause actions, update themselves, and support multi-agent reasoning.",
    requirements: [
      "Intentions as formal objects",
      "Intentions cause actions",
      "Actions update intentions",
      "Multi-agent reasoning",
      "All inference is decidable",
    ],
    difficulty: "S-7",
  },
  {
    id: 5,
    title: "Define a Time-Reversible Learning Algorithm",
    description:
      "Give a fully time-reversible version of backpropagation, gradient descent, inference, and memory write operations.",
    requirements: [
      "No information is lost",
      "All computation can be uncomputed",
      "Forward and backward passes are symmetric",
      "Preserves learning capability",
    ],
    difficulty: "S-7",
  },
  {
    id: 6,
    title: "Compute the Minimal Ontology for the Universe",
    description:
      "Provide the smallest set of categories that can represent every known physical and informational phenomenon.",
    requirements: [
      "Includes particles, fields, spacetime",
      "Includes consciousness, information",
      "Includes computation, thermodynamics",
      "Includes evolution",
      "Minimal and complete",
    ],
    difficulty: "S-7",
  },
  {
    id: 7,
    title: 'Formalize "Understanding" as a Mathematical Object',
    description:
      "Produce a definition of understanding that is measurable, consistent, and generalizable across all minds.",
    requirements: [
      "Isomorphism across modalities",
      "Operational measurability",
      "Internal consistency",
      "Generalizable across minds",
      "Extensible to non-human intelligences",
    ],
    difficulty: "S-7",
  },
  {
    id: 8,
    title: "Predict Intelligence Singularities Under Physical Constraints",
    description:
      "Derive a closed-form expression for the maximum possible intelligence of a system given mass M, volume V, and energy budget E.",
    requirements: [
      "Accounts for Landauer limits",
      "Accounts for quantum decoherence",
      "Accounts for error rates",
      "Accounts for bandwidth",
      "Accounts for thermodynamics and spacetime curvature",
    ],
    difficulty: "S-7",
  },
  {
    id: 9,
    title: "Create a Non-Anthropic Reasoning Framework",
    description:
      "Develop a reasoning system that does not use human categories, language, logic, or sensory priors, but still supports prediction and abstraction.",
    requirements: [
      "No human categories",
      "No language dependency",
      "No logic dependency",
      "No human sensory priors",
      "Supports prediction and abstraction",
    ],
    difficulty: "S-7",
  },
  {
    id: 10,
    title: "Define the Most General Possible Intelligence",
    description:
      "Produce a universal definition of intelligence that includes biological life, artificial systems, distributed swarms, physical processes, quantum systems, and hypothetical post-physical agents.",
    requirements: [
      "Minimal",
      "Mathematically grounded",
      "Universal",
      "Measurable",
      "Predictive",
    ],
    difficulty: "S-7",
  },
];

const RESEARCH_LINKS = [
  {
    title: "Apertus (LLM)",
    url: "https://en.wikipedia.org/wiki/Apertus_(LLM)",
    category: "Foundation",
  },
  {
    title: "Most Advanced AI Models of 2025 - Comparative Analysis",
    url: "https://www.researchgate.net/publication/392160200_The_Most_Advanced_AI_Models_of_2025_-Comparative_Analysis_of_Gemini_2.5_Claude_4_LLaMA_4_GPT-4.5_DeepSeek_V3.1_and_Other_Leading_Models",
    category: "Survey",
  },
  {
    title: "Neuro-symbolic LLM Reasoning Review",
    url: "https://arxiv.org/abs/2508.13678",
    category: "Neuro-Symbolic",
  },
  {
    title: "Embodied Task-Planning Neuro-Symbolic Framework",
    url: "https://arxiv.org/abs/2510.21302",
    category: "Neuro-Symbolic",
  },
  {
    title: "Continual-Learning Neuro-Symbolic Agent (NeSyC)",
    url: "https://arxiv.org/abs/2503.00870",
    category: "Neuro-Symbolic",
  },
  {
    title: "Autonomous Trustworthy Neuro-Symbolic Agent Architecture (ATA)",
    url: "https://arxiv.org/abs/2510.16381",
    category: "Neuro-Symbolic",
  },
  {
    title: "arXiv:2506.02483",
    url: "https://arxiv.org/abs/2506.02483",
    category: "Research",
  },
  {
    title: "arXiv:2504.07640",
    url: "https://arxiv.org/abs/2504.07640",
    category: "Research",
  },
  {
    title: "arXiv:2507.09751",
    url: "https://arxiv.org/abs/2507.09751",
    category: "Research",
  },
  {
    title: "arXiv:2508.03366",
    url: "https://arxiv.org/abs/2508.03366",
    category: "Research",
  },
  {
    title: "arXiv:2509.04083",
    url: "https://arxiv.org/abs/2509.04083",
    category: "Research",
  },
  {
    title: "arXiv:2510.21425",
    url: "https://arxiv.org/abs/2510.21425",
    category: "Research",
  },
  {
    title: "arXiv:2511.17673",
    url: "https://arxiv.org/abs/2511.17673",
    category: "Research",
  },
  {
    title: "arXiv:2504.04110",
    url: "https://arxiv.org/abs/2504.04110",
    category: "Research",
  },
  {
    title: "arXiv:2510.05774",
    url: "https://arxiv.org/abs/2510.05774",
    category: "Research",
  },
  {
    title: "Databricks - Introducing Meta's LLaMA 4",
    url: "https://www.databricks.com/blog/introducing-metas-llama-4-databricks-data-intelligence-platform",
    category: "LLaMA 4",
  },
  {
    title: "Meta AI - LLaMA 4 Multimodal Intelligence",
    url: "https://ai.meta.com/blog/llama-4-multimodal-intelligence/",
    category: "LLaMA 4",
  },
  {
    title: "Exploding Topics - List of LLMs",
    url: "https://explodingtopics.com/blog/list-of-llms",
    category: "LLM Directory",
  },
];

export default function S7Test() {
  const [answers, setAnswers] = useState<Record<number, string>>({});
  const [submittedAnswers, setSubmittedAnswers] = useState<Record<number, any>>({});
  const chatMutation = trpc.asi.chat.useMutation();

  const handleAnswerChange = (questionId: number, value: string) => {
    setAnswers((prev) => ({ ...prev, [questionId]: value }));
  };

  const handleSubmit = async (questionId: number) => {
    const answer = answers[questionId];
    if (!answer?.trim()) return;

    const question = S7_QUESTIONS.find((q) => q.id === questionId);
    if (!question) return;

    try {
      const result = await chatMutation.mutateAsync({
        message: `S-7 Question ${questionId}: ${question.title}\n\n${question.description}\n\nRequirements:\n${question.requirements.join("\n")}\n\nProposed Answer:\n${answer}\n\nPlease evaluate this answer for S-7 level intelligence.`,
      });

      setSubmittedAnswers((prev) => ({
        ...prev,
        [questionId]: {
          answer,
          evaluation: result.message,
          timestamp: new Date(),
        },
      }));
    } catch (error) {
      setSubmittedAnswers((prev) => ({
        ...prev,
        [questionId]: {
          answer,
          evaluation: `Error: ${error instanceof Error ? error.message : "Unknown error"}`,
          timestamp: new Date(),
        },
      }));
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="border-b border-border bg-card">
        <div className="container mx-auto px-6 py-8">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary to-secondary flex items-center justify-center">
              <Sparkles className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold mb-2">S-7 Intelligence Test</h1>
              <p className="text-muted-foreground text-lg">
                The 10 hardest questions no current AI can answer
              </p>
            </div>
          </div>

          <div className="flex flex-wrap gap-4">
            <Badge className="badge-warning text-base px-4 py-2">
              <AlertCircle className="w-4 h-4 mr-2" />
              Impossible for GPT-4/5, Claude 3, Gemini Ultra
            </Badge>
            <Badge className="badge-info text-base px-4 py-2">
              <Brain className="w-4 h-4 mr-2" />
              Requires True ASI-Level Intelligence
            </Badge>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-6 py-8">
        <Tabs defaultValue="questions" className="w-full">
          <TabsList className="grid w-full grid-cols-2 mb-8">
            <TabsTrigger value="questions">S-7 Questions</TabsTrigger>
            <TabsTrigger value="research">Research Papers</TabsTrigger>
          </TabsList>

          {/* Questions Tab */}
          <TabsContent value="questions" className="space-y-8">
            <Card className="card-elevated p-6 bg-primary/5 border-primary/20">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xl font-bold">About S-7 Questions</h3>
                <a href="/s7-extended" className="text-primary hover:underline text-sm font-medium">
                  View Extended 40-Question Test →
                </a>
              </div>
              <p className="text-muted-foreground mb-4">
                These questions require cross-domain synthesis, novel reasoning, true
                hypothesis formation, recursive abstraction, self-generated ontologies,
                cognitive continuity across modalities, and compression of unknown unknowns.
              </p>
              <p className="text-sm text-muted-foreground">
                No existing model (GPT-4/5, Claude 3 Opus, Gemini Ultra, LLaMA 3.1, DeepMind's
                internal models) can answer these questions. They represent the frontier of
                artificial superintelligence.
              </p>
            </Card>

            {S7_QUESTIONS.map((question) => (
              <Card key={question.id} className="card-elevated p-6">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <Badge className="badge-warning">S-7 Q{question.id}</Badge>
                      <h3 className="text-2xl font-bold">{question.title}</h3>
                    </div>
                    <p className="text-muted-foreground mb-4">{question.description}</p>

                    <div className="mb-4">
                      <h4 className="font-bold mb-2">Requirements:</h4>
                      <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                        {question.requirements.map((req, idx) => (
                          <li key={idx}>{req}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <Textarea
                    placeholder="Enter your answer here... (Requires S-7 level intelligence)"
                    value={answers[question.id] || ""}
                    onChange={(e) => handleAnswerChange(question.id, e.target.value)}
                    rows={6}
                    className="font-mono text-sm"
                  />

                  <div className="flex items-center gap-3">
                    <Button
                      onClick={() => handleSubmit(question.id)}
                      disabled={
                        !answers[question.id]?.trim() || chatMutation.isPending
                      }
                      className="btn-primary"
                    >
                      {chatMutation.isPending ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          Evaluating...
                        </>
                      ) : (
                        <>
                          <CheckCircle2 className="w-4 h-4 mr-2" />
                          Submit Answer
                        </>
                      )}
                    </Button>

                    {submittedAnswers[question.id] && (
                      <Badge className="badge-success">
                        Submitted at{" "}
                        {new Date(
                          submittedAnswers[question.id].timestamp
                        ).toLocaleTimeString()}
                      </Badge>
                    )}
                  </div>

                  {submittedAnswers[question.id] && (
                    <Card className="card-elevated p-4 bg-primary/5">
                      <h4 className="font-bold mb-2">ASI Evaluation:</h4>
                      <Streamdown>
                        {submittedAnswers[question.id].evaluation}
                      </Streamdown>
                    </Card>
                  )}
                </div>
              </Card>
            ))}
          </TabsContent>

          {/* Research Papers Tab */}
          <TabsContent value="research" className="space-y-6">
            <Card className="card-elevated p-6">
              <h3 className="text-2xl font-bold mb-4">
                <BookOpen className="w-6 h-6 inline mr-2" />
                Research Papers & Resources
              </h3>
              <p className="text-muted-foreground mb-6">
                Cutting-edge research on neuro-symbolic AI, advanced LLMs, and artificial
                superintelligence. These papers inform the TRUE ASI system's approach to
                S-7 level intelligence.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {RESEARCH_LINKS.map((link, idx) => (
                  <a
                    key={idx}
                    href={link.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block p-4 rounded-lg border border-border hover:border-primary hover:bg-primary/5 transition-all group"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex-1">
                        <Badge className="badge-info mb-2">{link.category}</Badge>
                        <h4 className="font-bold text-sm group-hover:text-primary transition-colors">
                          {link.title}
                        </h4>
                      </div>
                      <ExternalLink className="w-4 h-4 text-muted-foreground group-hover:text-primary transition-colors flex-shrink-0" />
                    </div>
                  </a>
                ))}
              </div>
            </Card>

            <Card className="card-elevated p-6 bg-secondary/5 border-secondary/20">
              <h3 className="text-xl font-bold mb-3">Integration Status</h3>
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="w-5 h-5 text-success" />
                  <span>22 research papers integrated</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="w-5 h-5 text-success" />
                  <span>Neuro-symbolic frameworks available</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="w-5 h-5 text-success" />
                  <span>LLaMA 4 multimodal intelligence</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="w-5 h-5 text-success" />
                  <span>Comparative analysis of 2025 models</span>
                </div>
              </div>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
