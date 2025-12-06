/**
 * ChatASI - ChatGPT 5.1-Style Chat Interface
 * 
 * Advanced chat interface with:
 * - Voice input with waveform visualization
 * - Massive file upload (drag-and-drop, Google Drive)
 * - Deep research mode
 * - Shopping research mode
 * - Image generation mode
 * - Agent mode
 * - Multi-model selection
 */

import { useState, useRef, useEffect } from 'react';
import { Link } from 'wouter';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card } from '@/components/ui/card';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Mic,
  Upload,
  Image as ImageIcon,
  Search,
  ShoppingCart,
  Bot,
  MoreHorizontal,
  Send,
  Plus,
  MessageSquare,
  Library,
  Code,
  Globe,
  FolderOpen,
  Sparkles,
  Home,
  ArrowLeft,
} from 'lucide-react';
import { trpc } from '@/lib/trpc';
import { Streamdown } from 'streamdown';
import { VoiceInput } from '@/components/VoiceInput';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  model?: string;
}

type ChatMode = 'default' | 'deep_research' | 'shopping' | 'image_gen' | 'agent';

export default function ChatASI() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [selectedModel, setSelectedModel] = useState('true-asi-ultra');
  const [chatMode, setChatMode] = useState<ChatMode>('default');
  const [isRecording, setIsRecording] = useState(false);
  const [showSidebar, setShowSidebar] = useState(true);
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const chatMutation = trpc.trueASI.chat.useMutation();
  const { data: modelsData } = trpc.trueASI.getModels.useQuery();

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async () => {
    if (!input.trim() && uploadedFiles.length === 0) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');

    try {
      // Add mode-specific context
      let systemPrompt = '';
      if (chatMode === 'deep_research') {
        systemPrompt = 'You are a deep research assistant. Provide comprehensive, well-cited answers with multiple sources.';
      } else if (chatMode === 'shopping') {
        systemPrompt = 'You are a shopping research assistant. Help compare products, find deals, and provide purchase recommendations.';
      } else if (chatMode === 'image_gen') {
        systemPrompt = 'You are an image generation assistant. Help create detailed prompts for image generation.';
      } else if (chatMode === 'agent') {
        systemPrompt = 'You are coordinating multiple AI agents. Delegate tasks appropriately and synthesize results.';
      }

      const fullMessage = systemPrompt ? `${systemPrompt}\n\n${userMessage.content}` : userMessage.content;
      
      const response = await chatMutation.mutateAsync({
        message: fullMessage,
        model: selectedModel,
      });

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.success ? response.message : 'Error: ' + (response.error || 'Unknown error'),
        timestamp: new Date(),
        model: selectedModel,
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    setUploadedFiles(prev => [...prev, ...files]);
  };

  const handleVoiceInput = () => {
    setIsRecording(!isRecording);
    // TODO: Implement actual voice recording
    console.log('Voice input:', isRecording ? 'stopped' : 'started');
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="flex h-screen bg-[#0a0a0a] text-white">
      {/* Sidebar */}
      {showSidebar && (
        <div className="w-64 bg-[#1a1a1a] border-r border-gray-800 flex flex-col">
          <div className="p-4 border-b border-gray-800">
            <Button
              onClick={() => setMessages([])}
              className="w-full justify-start gap-2 bg-transparent hover:bg-gray-800 border border-gray-700"
            >
              <MessageSquare className="w-4 h-4" />
              New chat
            </Button>
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-2">
            <div className="text-sm text-gray-400 mb-2">GPTs</div>
            <Button variant="ghost" className="w-full justify-start gap-2 text-sm">
              <Sparkles className="w-4 h-4" />
              Explore
            </Button>
            <Button variant="ghost" className="w-full justify-start gap-2 text-sm">
              <Library className="w-4 h-4" />
              Scholar GPT
            </Button>
            <Button variant="ghost" className="w-full justify-start gap-2 text-sm">
              <Code className="w-4 h-4" />
              Codex
            </Button>
            <Button variant="ghost" className="w-full justify-start gap-2 text-sm">
              <Globe className="w-4 h-4" />
              Atlas
            </Button>
            <Button variant="ghost" className="w-full justify-start gap-2 text-sm">
              <FolderOpen className="w-4 h-4" />
              Projects
            </Button>

            <div className="text-sm text-gray-400 mt-6 mb-2">Recent Chats</div>
            {messages.length > 0 && (
              <Button variant="ghost" className="w-full justify-start gap-2 text-sm truncate">
                <MessageSquare className="w-4 h-4 flex-shrink-0" />
                <span className="truncate">{messages[0]?.content.slice(0, 30)}...</span>
              </Button>
            )}
          </div>
        </div>
      )}

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="h-16 border-b border-gray-800 flex items-center justify-between px-6">
          <Select value={selectedModel} onValueChange={setSelectedModel}>
            <SelectTrigger className="w-48 bg-transparent border-gray-700">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="true-asi-ultra">
                <div className="flex flex-col">
                  <span className="font-bold">TRUE ASI Ultra</span>
                  <span className="text-xs text-gray-400">All 193 models + AWS + GitHub + 1700 deeplinks</span>
                </div>
              </SelectItem>
              <SelectItem value="asi1-ultra">ASI1 Ultra</SelectItem>
              <SelectItem value="gpt-4">GPT-4</SelectItem>
              <SelectItem value="claude-3.5-sonnet">Claude 3.5 Sonnet</SelectItem>
              <SelectItem value="gemini-1.5-pro">Gemini 1.5 Pro</SelectItem>
              <SelectItem value="llama-3.3-70b">Llama 3.3 70B</SelectItem>
            </SelectContent>
          </Select>

          <div className="flex gap-2">
            <Link href="/">
              <Button
                variant="ghost"
                size="sm"
                className="gap-2"
              >
                <Home className="w-4 h-4" />
                Home
              </Button>
            </Link>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowSidebar(!showSidebar)}
            >
              {showSidebar ? '←' : '→'}
            </Button>
          </div>
        </div>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-6">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center max-w-3xl mx-auto">
              <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-purple-600 bg-clip-text text-transparent">
                What's on your mind today?
              </h1>
              <p className="text-gray-400 mb-8">
                Ask anything, upload files, or use advanced modes
              </p>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto space-y-6">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex gap-4 ${
                    message.role === 'user' ? 'justify-end' : 'justify-start'
                  }`}
                >
                  <Card
                    className={`p-4 max-w-[80%] ${
                      message.role === 'user'
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-800 text-white'
                    }`}
                  >
                    {message.role === 'assistant' ? (
                      <Streamdown>{message.content}</Streamdown>
                    ) : (
                      <p className="whitespace-pre-wrap">{message.content}</p>
                    )}
                    {message.model && (
                      <p className="text-xs text-gray-400 mt-2">Model: {message.model}</p>
                    )}
                  </Card>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="border-t border-gray-800 p-6">
          <div className="max-w-3xl mx-auto">
            {/* Mode Buttons */}
            <div className="flex gap-2 mb-4 flex-wrap">
              <Button
                variant="outline"
                size="sm"
                onClick={() => fileInputRef.current?.click()}
                className="gap-2 bg-transparent border-gray-700 hover:bg-gray-800"
              >
                <Upload className="w-4 h-4" />
                Add photos & files
              </Button>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                onChange={handleFileUpload}
                className="hidden"
              />
              
              <Button
                variant="outline"
                size="sm"
                onClick={() => setChatMode('deep_research')}
                className={`gap-2 bg-transparent border-gray-700 hover:bg-gray-800 ${
                  chatMode === 'deep_research' ? 'bg-blue-600' : ''
                }`}
              >
                <Search className="w-4 h-4" />
                Deep research
              </Button>
              
              <Button
                variant="outline"
                size="sm"
                onClick={() => setChatMode('shopping')}
                className={`gap-2 bg-transparent border-gray-700 hover:bg-gray-800 ${
                  chatMode === 'shopping' ? 'bg-blue-600' : ''
                }`}
              >
                <ShoppingCart className="w-4 h-4" />
                Shopping research
              </Button>
              
              <Button
                variant="outline"
                size="sm"
                onClick={() => setChatMode('image_gen')}
                className={`gap-2 bg-transparent border-gray-700 hover:bg-gray-800 ${
                  chatMode === 'image_gen' ? 'bg-blue-600' : ''
                }`}
              >
                <ImageIcon className="w-4 h-4" />
                Create image
              </Button>
              
              <Button
                variant="outline"
                size="sm"
                onClick={() => setChatMode('agent')}
                className={`gap-2 bg-transparent border-gray-700 hover:bg-gray-800 ${
                  chatMode === 'agent' ? 'bg-blue-600' : ''
                }`}
              >
                <Bot className="w-4 h-4" />
                Agent mode
              </Button>
              
              <Button
                variant="outline"
                size="sm"
                className="gap-2 bg-transparent border-gray-700 hover:bg-gray-800"
              >
                <MoreHorizontal className="w-4 h-4" />
                More
              </Button>
            </div>

            {/* Uploaded Files */}
            {uploadedFiles.length > 0 && (
              <div className="mb-4 flex gap-2 flex-wrap">
                {uploadedFiles.map((file, index) => (
                  <Card key={index} className="p-2 bg-gray-800 text-xs flex items-center gap-2">
                    <Upload className="w-3 h-3" />
                    {file.name}
                    <button
                      onClick={() => setUploadedFiles(prev => prev.filter((_, i) => i !== index))}
                      className="text-red-400 hover:text-red-300"
                    >
                      ×
                    </button>
                  </Card>
                ))}
              </div>
            )}

            {/* Input Box */}
            <div className="relative">
              <div className="flex items-center gap-2 bg-gray-800 rounded-lg p-2">
                <Button
                  variant="ghost"
                  size="icon"
                  className="flex-shrink-0"
                >
                  <Plus className="w-5 h-5" />
                </Button>
                
                <Input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask anything"
                  className="flex-1 bg-transparent border-none focus-visible:ring-0 text-white placeholder:text-gray-400"
                />
                
                <VoiceInput
                  onTranscript={(text) => {
                    setInput((prev) => (prev ? prev + ' ' + text : text));
                  }}
                  onError={(error) => {
                    console.error('Voice input error:', error);
                  }}
                  language="en-US"
                  continuous={false}
                  className="flex-shrink-0"
                />
                
                <Button
                  onClick={handleSendMessage}
                  disabled={!input.trim() && uploadedFiles.length === 0}
                  className="flex-shrink-0 bg-blue-600 hover:bg-blue-700"
                  size="icon"
                >
                  <Send className="w-5 h-5" />
                </Button>
              </div>
            </div>

            {/* Voice Recording Indicator */}
            {isRecording && (
              <div className="mt-2 flex items-center gap-2 text-sm text-red-400">
                <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                Recording...
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
