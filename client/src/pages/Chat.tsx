import { trpc } from "@/lib/trpc";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { 
  Brain, 
  Send, 
  Loader2, 
  Sparkles, 
  Upload, 
  Mic, 
  MicOff,
  Image as ImageIcon,
  FileText,
  Home,
  X,
} from "lucide-react";
import { useState, useRef, useEffect } from "react";
import { Streamdown } from "streamdown";
import { Link } from "wouter";
import { useToast } from "@/hooks/use-toast";

interface Message {
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: Date;
  model?: string;
  files?: Array<{ name: string; type: string; url: string }>;
}

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [selectedModel, setSelectedModel] = useState("true-asi-ultra");
  const [isRecording, setIsRecording] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<Array<{ name: string; type: string; url: string }>>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  // Fetch available models
  const { data: modelsData } = trpc.trueASIUltra.getModels.useQuery();
  
  // Chat mutation
  const chatMutation = trpc.trueASIUltra.chat.useMutation({
    onSuccess: (data) => {
      if (data.success) {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: data.content || "",
            timestamp: new Date(),
            model: "model" in data ? data.model : selectedModel,
          },
        ]);
        
        // Show success toast for TRUE ASI Ultra
        if ("modelsUsed" in data && data.modelsUsed) {
          toast({
            title: "TRUE ASI Ultra Response",
            description: `Synthesized from ${data.successfulModels}/${data.totalModels} models: ${data.modelsUsed.join(", ")}`,
          });
        }
      }
    },
    onError: (error) => {
      toast({
        title: "Error",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Handle file upload
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const newFiles: Array<{ name: string; type: string; url: string }> = [];

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      
      // Create object URL for preview
      const url = URL.createObjectURL(file);
      
      newFiles.push({
        name: file.name,
        type: file.type,
        url,
      });
    }

    setUploadedFiles((prev) => [...prev, ...newFiles]);
    
    toast({
      title: "Files Uploaded",
      description: `${newFiles.length} file(s) ready to send`,
    });
  };

  // Remove uploaded file
  const removeFile = (index: number) => {
    setUploadedFiles((prev) => prev.filter((_, i) => i !== index));
  };

  // Handle voice recording
  const toggleRecording = async () => {
    if (isRecording) {
      // Stop recording
      mediaRecorderRef.current?.stop();
      setIsRecording(false);
    } else {
      // Start recording
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mediaRecorder = new MediaRecorder(stream);
        mediaRecorderRef.current = mediaRecorder;

        const audioChunks: Blob[] = [];
        mediaRecorder.ondataavailable = (event) => {
          audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
          const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
          const url = URL.createObjectURL(audioBlob);
          
          setUploadedFiles((prev) => [
            ...prev,
            {
              name: `voice-recording-${Date.now()}.webm`,
              type: "audio/webm",
              url,
            },
          ]);

          toast({
            title: "Voice Recording Complete",
            description: "Audio file ready to send",
          });

          // Stop all tracks
          stream.getTracks().forEach((track) => track.stop());
        };

        mediaRecorder.start();
        setIsRecording(true);
        
        toast({
          title: "Recording Started",
          description: "Speak now...",
        });
      } catch (error) {
        toast({
          title: "Microphone Error",
          description: "Could not access microphone",
          variant: "destructive",
        });
      }
    }
  };

  // Handle send message
  const handleSend = async () => {
    if (!input.trim() && uploadedFiles.length === 0) return;

    const userMessage: Message = {
      role: "user",
      content: input,
      timestamp: new Date(),
      files: uploadedFiles.length > 0 ? uploadedFiles : undefined,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setUploadedFiles([]);

    // Prepare messages for API
    const apiMessages = messages.map((m) => ({
      role: m.role,
      content: m.content,
    }));

    apiMessages.push({
      role: "user",
      content: input || "Analyze the uploaded files",
    });

    await chatMutation.mutateAsync({
      messages: apiMessages,
      model: selectedModel,
      options: {
        temperature: 0.7,
        max_tokens: 4096,
      },
    });
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-950 flex flex-col">
      {/* Header */}
      <div className="border-b border-white/10 bg-white/5 backdrop-blur-xl">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link href="/">
                <Button variant="ghost" size="icon" className="text-cyan-400 hover:text-cyan-300">
                  <Home className="w-5 h-5" />
                </Button>
              </Link>
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
                  <Brain className="w-7 h-7 text-white" />
                </div>
                <div>
                  <h1 className="text-2xl font-black tracking-tight text-white">TRUE ASI Chat</h1>
                  <p className="text-sm text-cyan-400">
                    193 Models • 6.54TB Knowledge • 250 Agents
                  </p>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-4">
              <Select value={selectedModel} onValueChange={setSelectedModel}>
                <SelectTrigger className="w-64 bg-white/5 border-white/10 text-white">
                  <SelectValue placeholder="Select model" />
                </SelectTrigger>
                <SelectContent>
                  {modelsData?.models.map((model) => (
                    <SelectItem key={model.id} value={model.id}>
                      <div className="flex items-center gap-2">
                        <span>{model.name}</span>
                        {model.id === "true-asi-ultra" && (
                          <Badge className="bg-gradient-to-r from-cyan-500 to-blue-600 text-white text-xs">
                            Ultra
                          </Badge>
                        )}
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Badge className="bg-green-500/20 text-green-400 border-green-500/30">
                Connected
              </Badge>
            </div>
          </div>
        </div>
      </div>

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto">
        <div className="container mx-auto px-6 py-8 max-w-5xl">
          {messages.length === 0 ? (
            <div className="text-center py-16">
              <div className="w-24 h-24 rounded-2xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center mx-auto mb-6 shadow-2xl shadow-cyan-500/30 animate-float">
                <Sparkles className="w-12 h-12 text-white animate-pulse-glow" />
              </div>
              <h2 className="text-4xl font-black tracking-tight text-white mb-3">
                Welcome to TRUE ASI Ultra
              </h2>
              <p className="text-lg text-slate-300 max-w-2xl mx-auto mb-8">
                Experience artificial superintelligence powered by 193 AI models working in parallel.
                Upload files, use voice input, and get responses synthesized from the world's best AI systems.
              </p>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-3xl mx-auto">
                <Card className="bg-white/5 backdrop-blur-xl border-white/10 p-6">
                  <Brain className="w-8 h-8 text-cyan-400 mb-3" />
                  <h3 className="text-white font-bold mb-2">Multi-Model Consensus</h3>
                  <p className="text-sm text-slate-400">
                    Responses synthesized from GPT-4, Claude, Gemini, Llama, and more
                  </p>
                </Card>
                <Card className="bg-white/5 backdrop-blur-xl border-white/10 p-6">
                  <Upload className="w-8 h-8 text-blue-400 mb-3" />
                  <h3 className="text-white font-bold mb-2">File Upload</h3>
                  <p className="text-sm text-slate-400">
                    Upload images, documents, audio, video - all formats supported
                  </p>
                </Card>
                <Card className="bg-white/5 backdrop-blur-xl border-white/10 p-6">
                  <Mic className="w-8 h-8 text-purple-400 mb-3" />
                  <h3 className="text-white font-bold mb-2">Voice Input</h3>
                  <p className="text-sm text-slate-400">
                    Record voice messages and get AI-powered transcription
                  </p>
                </Card>
              </div>
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
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center flex-shrink-0 shadow-lg shadow-cyan-500/20">
                      <Brain className="w-5 h-5 text-white" />
                    </div>
                  )}
                  <div className="max-w-3xl">
                    <Card
                      className={`p-5 ${
                        message.role === "user"
                          ? "bg-gradient-to-br from-cyan-500 to-blue-600 text-white border-0 shadow-lg shadow-cyan-500/20"
                          : "bg-white/5 backdrop-blur-xl border-white/10"
                      }`}
                    >
                      {message.files && message.files.length > 0 && (
                        <div className="mb-3 flex flex-wrap gap-2">
                          {message.files.map((file, idx) => (
                            <div
                              key={idx}
                              className="flex items-center gap-2 bg-white/10 rounded-lg px-3 py-2"
                            >
                              {file.type.startsWith("image/") ? (
                                <ImageIcon className="w-4 h-4" />
                              ) : file.type.startsWith("audio/") ? (
                                <Mic className="w-4 h-4" />
                              ) : (
                                <FileText className="w-4 h-4" />
                              )}
                              <span className="text-sm">{file.name}</span>
                            </div>
                          ))}
                        </div>
                      )}
                      {message.role === "assistant" ? (
                        <div className="text-white">
                          <Streamdown>{message.content}</Streamdown>
                        </div>
                      ) : (
                        <p className="whitespace-pre-wrap">{message.content}</p>
                      )}
                      <div
                        className={`text-xs mt-3 flex items-center gap-2 ${
                          message.role === "user"
                            ? "text-white/70"
                            : "text-slate-400"
                        }`}
                      >
                        <span>{message.timestamp.toLocaleTimeString()}</span>
                        {message.model && (
                          <>
                            <span>•</span>
                            <span>{message.model}</span>
                          </>
                        )}
                      </div>
                    </Card>
                  </div>
                  {message.role === "user" && (
                    <div className="w-10 h-10 rounded-xl bg-white/10 flex items-center justify-center flex-shrink-0">
                      <span className="text-white font-bold">You</span>
                    </div>
                  )}
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
      </div>

      {/* Input Area */}
      <div className="border-t border-white/10 bg-white/5 backdrop-blur-xl">
        <div className="container mx-auto px-6 py-4 max-w-5xl">
          {/* Uploaded Files Preview */}
          {uploadedFiles.length > 0 && (
            <div className="mb-3 flex flex-wrap gap-2">
              {uploadedFiles.map((file, idx) => (
                <div
                  key={idx}
                  className="flex items-center gap-2 bg-white/10 rounded-lg px-3 py-2 border border-white/10"
                >
                  {file.type.startsWith("image/") ? (
                    <ImageIcon className="w-4 h-4 text-cyan-400" />
                  ) : file.type.startsWith("audio/") ? (
                    <Mic className="w-4 h-4 text-purple-400" />
                  ) : (
                    <FileText className="w-4 h-4 text-blue-400" />
                  )}
                  <span className="text-sm text-white">{file.name}</span>
                  <button
                    onClick={() => removeFile(idx)}
                    className="ml-2 text-white/50 hover:text-white"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>
          )}

          <div className="flex gap-3">
            {/* File Upload */}
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept="*/*"
              onChange={handleFileUpload}
              className="hidden"
            />
            <Button
              variant="outline"
              size="icon"
              onClick={() => fileInputRef.current?.click()}
              className="bg-white/5 border-white/10 hover:bg-white/10 text-white"
            >
              <Upload className="w-5 h-5" />
            </Button>

            {/* Voice Recording */}
            <Button
              variant="outline"
              size="icon"
              onClick={toggleRecording}
              className={`border-white/10 ${
                isRecording
                  ? "bg-red-500 hover:bg-red-600 text-white animate-pulse"
                  : "bg-white/5 hover:bg-white/10 text-white"
              }`}
            >
              {isRecording ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
            </Button>

            {/* Text Input */}
            <Textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder="Ask anything... (Shift+Enter for new line)"
              className="flex-1 min-h-[60px] max-h-[200px] bg-white/5 border-white/10 text-white placeholder:text-slate-400 focus:border-cyan-500"
              disabled={chatMutation.isPending}
            />

            {/* Send Button */}
            <Button
              onClick={handleSend}
              disabled={chatMutation.isPending || (!input.trim() && uploadedFiles.length === 0)}
              className="bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 text-white shadow-lg shadow-cyan-500/20"
              size="icon"
            >
              {chatMutation.isPending ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
            </Button>
          </div>

          <p className="text-xs text-slate-400 mt-3 text-center">
            TRUE ASI Ultra may produce inaccurate information. Verify important facts.
          </p>
        </div>
      </div>
    </div>
  );
}
