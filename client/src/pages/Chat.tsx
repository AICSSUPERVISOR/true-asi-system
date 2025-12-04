import { trpc } from "@/lib/trpc";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Brain, Send, Loader2, Sparkles } from "lucide-react";
import { useState } from "react";
import { Streamdown } from "streamdown";

interface Message {
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [selectedModel, setSelectedModel] = useState("gpt-4");

  const { data: models } = trpc.asi.models.useQuery();
  const chatMutation = trpc.asi.chat.useMutation({
    onSuccess: (data) => {
      if (data.success) {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: data.message,
            timestamp: new Date(),
          },
        ]);
      }
    },
  });

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage: Message = {
      role: "user",
      content: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");

    await chatMutation.mutateAsync({
      message: input,
      model: selectedModel,
    });
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Header */}
      <div className="border-b border-border bg-card">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <Brain className="w-8 h-8 text-primary" />
                <div>
                  <h1 className="text-2xl font-bold">ASI Chat</h1>
                  <p className="text-sm text-muted-foreground">
                    Powered by 250 agents & 6.54TB knowledge
                  </p>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-4">
              <Select value={selectedModel} onValueChange={setSelectedModel}>
                <SelectTrigger className="w-48">
                  <SelectValue placeholder="Select model" />
                </SelectTrigger>
                <SelectContent>
                  {models?.models.map((model: any) => (
                    <SelectItem key={model.id} value={model.id}>
                      {model.name || model.id}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Badge className="badge-success">Connected</Badge>
            </div>
          </div>
        </div>
      </div>

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto">
        <div className="container mx-auto px-6 py-8 max-w-4xl">
          {messages.length === 0 ? (
            <div className="text-center py-12">
              <Sparkles className="w-16 h-16 text-primary mx-auto mb-4 animate-pulse-glow" />
              <h2 className="text-2xl font-bold mb-2">Welcome to TRUE ASI Chat</h2>
              <p className="text-muted-foreground max-w-md mx-auto">
                Start a conversation with our artificial superintelligence system.
                Ask anything - from complex reasoning to code generation.
              </p>
            </div>
          ) : (
            <div className="space-y-6">
              {messages.map((message, i) => (
                <div
                  key={i}
                  className={`flex gap-4 ${
                    message.role === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  {message.role === "assistant" && (
                    <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
                      <Brain className="w-5 h-5 text-primary" />
                    </div>
                  )}
                  <Card
                    className={`max-w-2xl p-4 ${
                      message.role === "user"
                        ? "bg-primary text-primary-foreground"
                        : "card-elevated"
                    }`}
                  >
                    {message.role === "assistant" ? (
                      <Streamdown>{message.content}</Streamdown>
                    ) : (
                      <p className="whitespace-pre-wrap">{message.content}</p>
                    )}
                    <div
                      className={`text-xs mt-2 ${
                        message.role === "user"
                          ? "text-primary-foreground/70"
                          : "text-muted-foreground"
                      }`}
                    >
                      {message.timestamp.toLocaleTimeString()}
                    </div>
                  </Card>
                  {message.role === "user" && (
                    <div className="w-10 h-10 rounded-full bg-secondary/10 flex items-center justify-center flex-shrink-0">
                      <span className="text-sm font-bold text-secondary">You</span>
                    </div>
                  )}
                </div>
              ))}
              {chatMutation.isPending && (
                <div className="flex gap-4">
                  <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
                    <Brain className="w-5 h-5 text-primary" />
                  </div>
                  <Card className="card-elevated p-4">
                    <Loader2 className="w-5 h-5 animate-spin text-primary" />
                  </Card>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Input Area */}
      <div className="border-t border-border bg-card">
        <div className="container mx-auto px-6 py-4 max-w-4xl">
          <div className="flex gap-4">
            <Textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder="Ask anything... (Shift+Enter for new line)"
              className="min-h-[60px] max-h-[200px] resize-none"
              disabled={chatMutation.isPending}
            />
            <Button
              onClick={handleSend}
              disabled={!input.trim() || chatMutation.isPending}
              className="btn-primary h-[60px] px-6"
            >
              {chatMutation.isPending ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
            </Button>
          </div>
          <div className="text-xs text-muted-foreground mt-2">
            Model: {selectedModel} • Press Enter to send, Shift+Enter for new line
          </div>
        </div>
      </div>
    </div>
  );
}
