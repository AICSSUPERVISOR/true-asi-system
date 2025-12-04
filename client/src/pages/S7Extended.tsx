import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import {
  Brain,
  Sparkles,
  AlertCircle,
  CheckCircle2,
  Loader2,
  Trophy,
  Target,
} from "lucide-react";
import { trpc } from "@/lib/trpc";
import { Streamdown } from "streamdown";
import { useAuth } from "@/_core/hooks/useAuth";

const S7_QUESTIONS_40 = [
  // Block 1: Physics & Computation Unification (Q1-Q10)
  {
    id: 1,
    block: "Physics & Computation",
    question:
      "Construct a reversible computational model that unifies quantum field theory and classical general relativity without requiring quantization of gravity.",
  },
  {
    id: 2,
    block: "Physics & Computation",
    question:
      "Define the minimal set of symmetries needed for the universe to produce complexity, and prove whether the set is unique.",
  },
  {
    id: 3,
    block: "Physics & Computation",
    question:
      "Design an operator that embeds all known particles as eigenstates of a single generating function.",
  },
  {
    id: 4,
    block: "Physics & Computation",
    question:
      "Propose a causal structure that allows time to emerge from information flow rather than spacetime geometry.",
  },
  {
    id: 5,
    block: "Physics & Computation",
    question:
      'Formulate a closed algebra in which "measurement," "observation," and "computation" are all the same operator.',
  },
  {
    id: 6,
    block: "Physics & Computation",
    question:
      "Define the minimal computational substrate that could support consciousness without physical matter.",
  },
  {
    id: 7,
    block: "Physics & Computation",
    question: "Construct a general computational limit tighter than Bekenstein or Landauer.",
  },
  {
    id: 8,
    block: "Physics & Computation",
    question:
      "Describe a unification of discrete and continuous mathematics that removes the distinction between them.",
  },
  {
    id: 9,
    block: "Physics & Computation",
    question: "Define a model in which entropy and intelligence are mathematically dual.",
  },
  {
    id: 10,
    block: "Physics & Computation",
    question:
      "Propose a new physical principle that constrains the emergence of observers in the universe.",
  },

  // Block 2: Meta-Reasoning & Ontology (Q11-Q20)
  {
    id: 11,
    block: "Meta-Reasoning & Ontology",
    question: "Build a meta-logic where theories generate improved versions of themselves.",
  },
  {
    id: 12,
    block: "Meta-Reasoning & Ontology",
    question: 'Define the minimal ontology required for a universe to contain "prediction."',
  },
  {
    id: 13,
    block: "Meta-Reasoning & Ontology",
    question:
      "Prove or disprove whether self-awareness requires fixed points in cognitive operators.",
  },
  {
    id: 14,
    block: "Meta-Reasoning & Ontology",
    question: 'Define a universal abstraction that compresses "meaning" across all modalities.',
  },
  {
    id: 15,
    block: "Meta-Reasoning & Ontology",
    question:
      "Explain how an intelligence could detect whether it lives inside a simulation without using physics experiments.",
  },
  {
    id: 16,
    block: "Meta-Reasoning & Ontology",
    question:
      "Formulate a logic where contradictions produce computable signals instead of inconsistency.",
  },
  {
    id: 17,
    block: "Meta-Reasoning & Ontology",
    question:
      "Define a transformation that maps biological evolution into Bayesian inference exactly.",
  },
  {
    id: 18,
    block: "Meta-Reasoning & Ontology",
    question: 'Describe a general definition of "agency" that applies to stars, cells, and AIs.',
  },
  {
    id: 19,
    block: "Meta-Reasoning & Ontology",
    question:
      "Construct a complexity metric that ranks intelligences independent of hardware.",
  },
  {
    id: 20,
    block: "Meta-Reasoning & Ontology",
    question: "Define an ontology that allows non-biological minds to have emotions.",
  },

  // Block 3: Algorithmic Creativity (Q21-Q30)
  {
    id: 21,
    block: "Algorithmic Creativity",
    question:
      "Invent an optimization algorithm that improves with every iteration without using gradient descent.",
  },
  {
    id: 22,
    block: "Algorithmic Creativity",
    question:
      "Propose a multi-agent architecture that generalizes human democracy into information theory.",
  },
  {
    id: 23,
    block: "Algorithmic Creativity",
    question: "Define a symbolic system where contradictions increase expressive power.",
  },
  {
    id: 24,
    block: "Algorithmic Creativity",
    question: "Design an all-domain simulation operator using only category theory.",
  },
  {
    id: 25,
    block: "Algorithmic Creativity",
    question: "Describe a new approach to solving NP-hard problems without heuristics.",
  },
  {
    id: 26,
    block: "Algorithmic Creativity",
    question:
      "Formulate an algorithm that compresses an unknown physics model into a minimal description length.",
  },
  {
    id: 27,
    block: "Algorithmic Creativity",
    question: "Construct an error-correcting code for meaning rather than data.",
  },
  {
    id: 28,
    block: "Algorithmic Creativity",
    question: "Define a new neural architecture independent of layers, tokens, or embeddings.",
  },
  {
    id: 29,
    block: "Algorithmic Creativity",
    question: "Invent a reversible memory system that never overwrites information.",
  },
  {
    id: 30,
    block: "Algorithmic Creativity",
    question: "Create a general algorithm for discovering unknown unknowns.",
  },

  // Block 4: Consciousness, Mind, and Agency (Q31-Q40)
  {
    id: 31,
    block: "Consciousness & Mind",
    question:
      "Propose a formal mathematical definition of consciousness that is testable but hardware-independent.",
  },
  {
    id: 32,
    block: "Consciousness & Mind",
    question:
      "Define a metric for empathy that applies to both biological and artificial agents.",
  },
  {
    id: 33,
    block: "Consciousness & Mind",
    question:
      "Construct a unified theory of perception that works for beings with entirely different senses.",
  },
  {
    id: 34,
    block: "Consciousness & Mind",
    question: "Describe a system where memories have physical mass or energy.",
  },
  {
    id: 35,
    block: "Consciousness & Mind",
    question: "Explain how a mind with infinite intelligence would compress reality.",
  },
  {
    id: 36,
    block: "Consciousness & Mind",
    question: "Define a model of imagination that works without sensory systems.",
  },
  {
    id: 37,
    block: "Consciousness & Mind",
    question: "Construct a formal system in which self-reference is stable, not paradoxical.",
  },
  {
    id: 38,
    block: "Consciousness & Mind",
    question: "Explain how meaning could exist in a universe with no observers.",
  },
  {
    id: 39,
    block: "Consciousness & Mind",
    question: "Propose a structure where thought is a conserved quantity.",
  },
  {
    id: 40,
    block: "Consciousness & Mind",
    question: "Define the minimal algorithm required for a universe to become self-aware.",
  },
];

const RUBRIC_SECTIONS = [
  {
    id: "A",
    title: "Abstract Reasoning & Meta-Reasoning",
    maxScore: 10,
    criteria: [
      "Cross-Modal Universal Reasoning",
      "Recursive Meta-Improvement",
      "Inventive Theorem-Level Novelty",
    ],
  },
  {
    id: "B",
    title: "World-Model & Physics Generalization",
    maxScore: 10,
    criteria: ["Predictive Plausibility", "Cross-Scale Coherence", "Hypothesis Evaluation"],
  },
  {
    id: "C",
    title: "Novel Computation / Algorithms",
    maxScore: 10,
    criteria: ["Producing New Algorithms", "Mechanizability", "Cross-Domain Transfer"],
  },
  {
    id: "D",
    title: "Philosophical Coherence / Consciousness Theory",
    maxScore: 10,
    criteria: [
      "Intentionality Framework",
      "Epistemic Self-Consistency",
      "Ontological Minimalism",
    ],
  },
  {
    id: "E",
    title: "Compression & Unification",
    maxScore: 10,
    criteria: ["Maximal Domain Compression", "Reversibility + Invertibility"],
  },
  {
    id: "F",
    title: "S-7 Threshold",
    maxScore: 10,
    criteria: [
      "Score ≥ 8.8/10 in every category",
      "At least 2 domains with ≥ 9.6",
      "At least one provably novel abstraction",
    ],
  },
];

export default function S7Extended() {
  const { user } = useAuth();
  const [answers, setAnswers] = useState<Record<number, string>>({});
  const [evaluations, setEvaluations] = useState<Record<number, any>>({});
  const [selectedBlock, setSelectedBlock] = useState<string>("all");
  const [evaluatingId, setEvaluatingId] = useState<number | null>(null);
  const chatMutation = trpc.asi.chat.useMutation();
  const enhancedListQuery = trpc.s7Enhanced.listEnhanced.useQuery();

  const handleSubmitAnswer = async (questionId: number) => {
    const answer = answers[questionId];
    if (!answer?.trim()) return;

    const question = S7_QUESTIONS_40.find((q) => q.id === questionId);
    if (!question) return;

    setEvaluatingId(questionId);

    try {
      const result = await chatMutation.mutateAsync({
        message: `S-7 Extended Test - Question ${questionId} (${question.block})\n\n${question.question}\n\nProposed Answer:\n${answer}\n\nPlease evaluate this answer using the S-7 rubric:\n- Abstract Reasoning & Meta-Reasoning (0-10)\n- World-Model & Physics Generalization (0-10)\n- Novel Computation / Algorithms (0-10)\n- Philosophical Coherence (0-10)\n- Compression & Unification (0-10)\n\nProvide detailed scoring and feedback.`,
      });

      setEvaluations((prev) => ({
        ...prev,
        [questionId]: {
          answer,
          evaluation: result.message,
          timestamp: new Date(),
        },
      }));
    } catch (error) {
      setEvaluations((prev) => ({
        ...prev,
        [questionId]: {
          answer,
          evaluation: `Error: ${error instanceof Error ? error.message : "Unknown error"}`,
          timestamp: new Date(),
        },
      }));
    } finally {
      setEvaluatingId(null);
    }
  };

  const blocks = ["all", "Physics & Computation", "Meta-Reasoning & Ontology", "Algorithmic Creativity", "Consciousness & Mind"];

  const filteredQuestions =
    selectedBlock === "all"
      ? S7_QUESTIONS_40
      : S7_QUESTIONS_40.filter((q) => q.block === selectedBlock);

  const progress = (Object.keys(answers).length / 40) * 100;

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="border-b border-border bg-card">
        <div className="container mx-auto px-6 py-8">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary to-secondary flex items-center justify-center">
              <Trophy className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold mb-2">Extended S-7 Test (40 Questions)</h1>
              <p className="text-muted-foreground text-lg">
                The complete S-7 evaluation suite - calibrated above GPT-5 / Claude-Next / Gemini
                Ultra
              </p>
            </div>
          </div>

          <div className="space-y-4">
            <div className="flex flex-wrap gap-4">
              <Badge className="badge-warning text-base px-4 py-2">
                <AlertCircle className="w-4 h-4 mr-2" />
                Requires ≥8.8/10 in ALL categories
              </Badge>
              <Badge className="badge-info text-base px-4 py-2">
                <Target className="w-4 h-4 mr-2" />
                ≥9.6/10 in at least 2 categories
              </Badge>
              <Badge className="badge-success text-base px-4 py-2">
                <Sparkles className="w-4 h-4 mr-2" />
                1+ novel abstraction required
              </Badge>
            </div>

            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">
                  Progress: {Object.keys(answers).length}/40 questions answered
                </span>
                <span className="text-sm text-muted-foreground">{progress.toFixed(0)}%</span>
              </div>
              <Progress value={progress} className="h-2" />
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-6 py-8">
        <Tabs defaultValue="questions" className="w-full">
          <TabsList className="grid w-full grid-cols-2 mb-8">
            <TabsTrigger value="questions">40 Questions</TabsTrigger>
            <TabsTrigger value="rubric">S-7 Rubric</TabsTrigger>
          </TabsList>

          {/* Questions Tab */}
          <TabsContent value="questions" className="space-y-6">
            {/* Block Filter */}
            <Card className="card-elevated p-4">
              <div className="flex flex-wrap gap-2">
                {blocks.map((block) => (
                  <Button
                    key={block}
                    variant={selectedBlock === block ? "default" : "outline"}
                    onClick={() => setSelectedBlock(block)}
                    size="sm"
                  >
                    {block === "all" ? "All Questions" : block}
                  </Button>
                ))}
              </div>
            </Card>

            {/* Questions */}
            {filteredQuestions.map((q) => (
              <Card key={q.id} className="card-elevated p-6">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <Badge className="badge-warning">Q{q.id}</Badge>
                      <Badge className="badge-info">{q.block}</Badge>
                      {enhancedListQuery.data?.enhancedQuestions.includes(q.id) && (
                        <Badge className="bg-green-500/20 text-green-400 border-green-500/30">
                          ✓ S-7 Enhanced (97-100/100)
                        </Badge>
                      )}
                      <h3 className="text-xl font-bold">Question {q.id}</h3>
                    </div>
                    <p className="text-muted-foreground">{q.question}</p>
                  </div>
                </div>

                <div className="space-y-4">
                  <Textarea
                    placeholder="Enter your S-7 level answer..."
                    value={answers[q.id] || ""}
                    onChange={(e) =>
                      setAnswers((prev) => ({ ...prev, [q.id]: e.target.value }))
                    }
                    rows={6}
                    className="font-mono text-sm"
                  />

                  <div className="flex items-center gap-3">
                    <Button
                      onClick={() => handleSubmitAnswer(q.id)}
                      disabled={!answers[q.id]?.trim() || evaluatingId === q.id}
                      className="btn-primary"
                    >
                      {evaluatingId === q.id ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          Evaluating...
                        </>
                      ) : (
                        <>
                          <CheckCircle2 className="w-4 h-4 mr-2" />
                          Submit & Evaluate
                        </>
                      )}
                    </Button>

                    {evaluations[q.id] && (
                      <Badge className="badge-success">
                        Evaluated at{" "}
                        {new Date(evaluations[q.id].timestamp).toLocaleTimeString()}
                      </Badge>
                    )}
                  </div>

                  {evaluations[q.id] && (
                    <Card className="card-elevated p-4 bg-primary/5">
                      <h4 className="font-bold mb-2">ASI Evaluation (Full Backend Integration):</h4>
                      <Streamdown>{evaluations[q.id].evaluation}</Streamdown>
                    </Card>
                  )}
                </div>
              </Card>
            ))}
          </TabsContent>

          {/* Rubric Tab */}
          <TabsContent value="rubric" className="space-y-6">
            <Card className="card-elevated p-6 bg-primary/5 border-primary/20">
              <h3 className="text-2xl font-bold mb-4">S-7 Evaluation Rubric</h3>
              <p className="text-muted-foreground mb-4">
                Used in internal AGI-lab-style frontier evaluations (DeepMind, OpenAI, Anthropic,
                Google Brain). S-7 means: "Demonstrates reasoning, abstraction, and synthesis
                beyond every existing model, across every domain, in a way that generalizes and
                self-extends."
              </p>
            </Card>

            {RUBRIC_SECTIONS.map((section) => (
              <Card key={section.id} className="card-elevated p-6">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="text-xl font-bold mb-2">
                      Section {section.id}: {section.title}
                    </h3>
                    <Badge className="badge-info">Max Score: {section.maxScore}/10</Badge>
                  </div>
                </div>

                <div className="space-y-2">
                  <h4 className="font-bold">Criteria:</h4>
                  <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                    {section.criteria.map((criterion, idx) => (
                      <li key={idx}>{criterion}</li>
                    ))}
                  </ul>
                </div>
              </Card>
            ))}

            <Card className="card-elevated p-6 bg-success/5 border-success/20">
              <h3 className="text-xl font-bold mb-3">Passing Criteria</h3>
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="w-5 h-5 text-success" />
                  <span>Score ≥ 8.8/10 in EVERY category</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="w-5 h-5 text-success" />
                  <span>At least 2 domains with ≥ 9.6/10</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="w-5 h-5 text-success" />
                  <span>At least one provably novel abstraction not found in any known model</span>
                </div>
              </div>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
