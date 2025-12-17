import { useState, useRef, useEffect } from "react";
import { trpc } from "@/lib/trpc";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { 
  Send, 
  Bot, 
  User, 
  Sparkles, 
  FileText, 
  Building2, 
  Calculator,
  AlertTriangle,
  CheckCircle2,
  Zap,
  MessageSquare,
  RefreshCw,
  Loader2
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { Streamdown } from "streamdown";
import { toast } from "sonner";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  actions?: Array<{
    type: string;
    label: string;
    success?: boolean;
    params?: Record<string, unknown>;
    result?: unknown;
  }>;
}

const QUICK_PROMPTS = [
  { icon: FileText, label: "Klassifiser dokument", prompt: "Kan du hjelpe meg å klassifisere og kontere et nytt dokument?" },
  { icon: Calculator, label: "MVA-beregning", prompt: "Kan du beregne MVA for inneværende termin?" },
  { icon: Building2, label: "Kredittsjekk", prompt: "Kan du gjøre en kredittsjekk på en leverandør?" },
  { icon: AlertTriangle, label: "Risikovurdering", prompt: "Gi meg en risikovurdering av porteføljen min" },
];

export default function Chat() {
  const [selectedCompanyId, setSelectedCompanyId] = useState<number | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const { data: companies } = trpc.company.list.useQuery();
  const { data: selectedCompany } = trpc.company.get.useQuery(
    { id: selectedCompanyId! },
    { enabled: !!selectedCompanyId }
  );

  // Auto-select first company
  useEffect(() => {
    if (!selectedCompanyId && companies && companies.length > 0) {
      setSelectedCompanyId(companies[0].id);
    }
  }, [companies, selectedCompanyId]);

  const [sessionId] = useState(() => `session-${Date.now()}-${Math.random().toString(36).slice(2)}`);

  const chatMutation = trpc.chat.send.useMutation({
    onSuccess: (result) => {
      setMessages((prev) => [
        ...prev,
        {
          id: `assistant-${Date.now()}`,
          role: "assistant",
          content: result.message,
          timestamp: new Date(),
          actions: result.actions,
        },
      ]);
      setIsLoading(false);
    },
    onError: (error) => {
      toast.error(`Feil: ${error.message}`);
      setIsLoading(false);
    },
  });

  const handleSend = () => {
    if (!input.trim() || !selectedCompanyId || isLoading) return;

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    chatMutation.mutate({
      companyId: selectedCompanyId,
      message: userMessage.content,
      sessionId,
    });
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleQuickPrompt = (prompt: string) => {
    if (selectedCompanyId) {
      setInput(prompt);
      inputRef.current?.focus();
    }
  };

  const clearChat = () => {
    setMessages([]);
    toast.success("Chat nullstilt");
  };

  // Scroll to bottom on new messages
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)]">
      {/* Premium Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-white flex items-center gap-3">
            <div className="rounded-xl bg-gradient-to-br from-emerald-500 to-teal-500 p-2.5 shadow-lg">
              <Bot className="h-6 w-6 text-white" />
            </div>
            AI Regnskapsassistent
          </h1>
          <p className="text-slate-500 mt-1">
            Autonom hjelp med regnskap, skatt og revisjon
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Select
            value={selectedCompanyId?.toString() || ""}
            onValueChange={(v) => {
              setSelectedCompanyId(Number(v));
              setMessages([]);
            }}
          >
            <SelectTrigger className="w-[250px]">
              <SelectValue placeholder="Velg selskap..." />
            </SelectTrigger>
            <SelectContent>
              {companies?.map((company) => (
                <SelectItem key={company.id} value={company.id.toString()}>
                  {company.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button variant="outline" size="icon" onClick={clearChat} title="Nullstill chat">
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex gap-6 min-h-0">
        {/* Chat Messages */}
        <Card className="flex-1 flex flex-col border-0 bg-gradient-to-br from-slate-50 to-white dark:from-slate-900 dark:to-slate-800 shadow-lg overflow-hidden">
          {/* Company Context Banner */}
          {selectedCompany && (
            <div className="px-4 py-3 bg-emerald-50 dark:bg-emerald-900/20 border-b border-emerald-100 dark:border-emerald-800">
              <div className="flex items-center gap-3">
                <Building2 className="h-5 w-5 text-emerald-600" />
                <div>
                  <p className="font-medium text-slate-900 dark:text-white">
                    {selectedCompany.name}
                  </p>
                  <p className="text-sm text-slate-500">
                    Org.nr: {selectedCompany.orgNumber}
                    {selectedCompany.forvaltRating && (
                      <Badge className="ml-2 bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400">
                        Rating: {selectedCompany.forvaltRating}
                      </Badge>
                    )}
                  </p>
                </div>
              </div>
            </div>
          )}

          <ScrollArea className="flex-1 p-6" ref={scrollRef}>
            {messages.length === 0 ? (
              <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="h-full flex flex-col items-center justify-center text-center"
              >
                <div className="rounded-2xl bg-gradient-to-br from-emerald-500 to-teal-500 p-4 mb-6 shadow-xl">
                  <Bot className="h-12 w-12 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-2">
                  Hei! Jeg er Skatt-Flow
                </h3>
                <p className="text-slate-500 max-w-md mb-8">
                  Din autonome regnskapsassistent. Jeg kan hjelpe deg med dokumentklassifisering,
                  MVA-beregninger, kredittsjekk, og mye mer.
                </p>

                <div className="grid grid-cols-2 gap-3 max-w-lg">
                  {QUICK_PROMPTS.map((prompt, i) => (
                    <Button
                      key={i}
                      variant="outline"
                      className="justify-start text-left h-auto py-3 px-4 hover:bg-emerald-50 hover:border-emerald-200 dark:hover:bg-emerald-900/20"
                      onClick={() => handleQuickPrompt(prompt.prompt)}
                    >
                      <prompt.icon className="h-4 w-4 mr-2 text-emerald-500 flex-shrink-0" />
                      <span className="text-sm">{prompt.label}</span>
                    </Button>
                  ))}
                </div>
              </motion.div>
            ) : (
              <div className="space-y-6">
                <AnimatePresence>
                  {messages.map((message) => (
                    <motion.div
                      key={message.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -20 }}
                      transition={{ duration: 0.3 }}
                    >
                      <ChatMessageBubble message={message} />
                    </motion.div>
                  ))}
                </AnimatePresence>

                {isLoading && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex items-start gap-4"
                  >
                    <div className="rounded-xl bg-gradient-to-br from-emerald-500 to-teal-500 p-2.5 shadow-lg">
                      <Bot className="h-5 w-5 text-white" />
                    </div>
                    <div className="flex-1 rounded-2xl rounded-tl-none bg-white dark:bg-slate-800 p-4 shadow-sm border border-slate-200 dark:border-slate-700">
                      <div className="flex items-center gap-2">
                        <Loader2 className="h-4 w-4 animate-spin text-emerald-500" />
                        <span className="text-sm text-slate-500">Skatt-Flow tenker...</span>
                      </div>
                    </div>
                  </motion.div>
                )}
              </div>
            )}
          </ScrollArea>

          {/* Input Area */}
          <div className="p-4 border-t border-slate-200 dark:border-slate-700">
            {!selectedCompanyId && (
              <div className="mb-4 p-3 rounded-lg bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 text-amber-700 dark:text-amber-400 text-sm flex items-center gap-2">
                <AlertTriangle className="h-4 w-4" />
                Velg et selskap for å starte samtalen
              </div>
            )}
            <div className="flex gap-3">
              <Textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={selectedCompanyId ? "Still et spørsmål om regnskap..." : "Velg et selskap først..."}
                disabled={!selectedCompanyId || isLoading}
                className="min-h-[60px] max-h-[120px] resize-none"
              />
              <Button 
                onClick={handleSend}
                disabled={!input.trim() || !selectedCompanyId || isLoading}
                className="h-auto px-6 bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-600 hover:to-teal-600"
              >
                <Send className="h-5 w-5" />
              </Button>
            </div>
            <p className="text-xs text-slate-400 mt-2">
              Trykk Enter for å sende, Shift+Enter for ny linje
            </p>
          </div>
        </Card>

        {/* Quick Actions Sidebar */}
        <div className="w-80 space-y-4 hidden lg:block">
          <Card className="border-0 bg-gradient-to-br from-slate-50 to-white dark:from-slate-900 dark:to-slate-800 shadow-lg">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center gap-2">
                <Zap className="h-4 w-4 text-amber-500" />
                Hurtigkommandoer
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {QUICK_PROMPTS.map((prompt, index) => (
                <Button
                  key={index}
                  variant="outline"
                  className="w-full justify-start gap-3 h-auto py-3 px-4"
                  onClick={() => handleQuickPrompt(prompt.prompt)}
                  disabled={!selectedCompanyId}
                >
                  <prompt.icon className="h-4 w-4 text-slate-500" />
                  <span className="text-sm">{prompt.label}</span>
                </Button>
              ))}
            </CardContent>
          </Card>

          <Card className="border-0 bg-gradient-to-br from-emerald-50 to-teal-50 dark:from-emerald-900/20 dark:to-teal-900/20 shadow-lg">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center gap-2">
                <Sparkles className="h-4 w-4 text-emerald-500" />
                AI-kapabiliteter
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <CapabilityItem label="Dokumentklassifisering" enabled />
              <CapabilityItem label="Automatisk kontering" enabled />
              <CapabilityItem label="MVA-beregning" enabled />
              <CapabilityItem label="Kredittsjekk" enabled />
              <CapabilityItem label="SAF-T validering" enabled />
              <CapabilityItem label="Risikovurdering" enabled />
            </CardContent>
          </Card>

          <Card className="border-0 bg-gradient-to-br from-slate-50 to-white dark:from-slate-900 dark:to-slate-800 shadow-lg">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center gap-2">
                <MessageSquare className="h-4 w-4 text-blue-500" />
                Samtalestatistikk
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-slate-500">Meldinger</span>
                <span className="font-medium">{messages.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Handlinger utført</span>
                <span className="font-medium">
                  {messages.reduce((acc, m) => acc + (m.actions?.filter(a => a.success).length || 0), 0)}
                </span>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// COMPONENT DEFINITIONS
// ============================================================================

interface ChatMessageBubbleProps {
  message: ChatMessage;
}

function ChatMessageBubble({ message }: ChatMessageBubbleProps) {
  const isUser = message.role === "user";

  return (
    <div className={`flex items-start gap-4 ${isUser ? "flex-row-reverse" : ""}`}>
      {isUser ? (
        <Avatar className="h-10 w-10 flex-shrink-0">
          <AvatarFallback className="bg-slate-200 dark:bg-slate-700">
            <User className="h-5 w-5 text-slate-600 dark:text-slate-300" />
          </AvatarFallback>
        </Avatar>
      ) : (
        <div className="rounded-xl bg-gradient-to-br from-emerald-500 to-teal-500 p-2.5 shadow-lg flex-shrink-0">
          <Bot className="h-5 w-5 text-white" />
        </div>
      )}
      <div className={`flex-1 max-w-[80%] ${isUser ? "text-right" : ""}`}>
        <div
          className={`inline-block rounded-2xl p-4 shadow-sm ${
            isUser
              ? "rounded-tr-none bg-emerald-500 text-white"
              : "rounded-tl-none bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700"
          }`}
        >
          {isUser ? (
            <p className="text-sm whitespace-pre-wrap">{message.content}</p>
          ) : (
            <div className="prose prose-sm dark:prose-invert max-w-none">
              <Streamdown>{message.content}</Streamdown>
            </div>
          )}
        </div>

        {/* Actions performed */}
        {message.actions && message.actions.length > 0 && (
          <div className="mt-3 space-y-2">
            {message.actions.map((action, index) => (
              <div
                key={index}
                className={`inline-flex items-center gap-2 rounded-lg px-3 py-1.5 text-xs ${
                  action.success !== false
                    ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400"
                    : "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400"
                }`}
              >
                {action.success !== false ? (
                  <CheckCircle2 className="h-3 w-3" />
                ) : (
                  <AlertTriangle className="h-3 w-3" />
                )}
                {action.label}
              </div>
            ))}
          </div>
        )}

        <p className={`mt-2 text-xs text-slate-400 ${isUser ? "text-right" : ""}`}>
          {message.timestamp.toLocaleTimeString("nb-NO", { hour: "2-digit", minute: "2-digit" })}
        </p>
      </div>
    </div>
  );
}

interface CapabilityItemProps {
  label: string;
  enabled: boolean;
}

function CapabilityItem({ label, enabled }: CapabilityItemProps) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-sm text-slate-600 dark:text-slate-400">{label}</span>
      {enabled ? (
        <Badge className="bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400 text-xs">
          Aktiv
        </Badge>
      ) : (
        <Badge variant="secondary" className="text-xs">
          Inaktiv
        </Badge>
      )}
    </div>
  );
}
