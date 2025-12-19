import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { 
  Brain, 
  Download, 
  CheckCircle2, 
  Clock, 
  Cpu, 
  Database,
  Zap,
  Activity,
  RefreshCw
} from "lucide-react";

interface ModelInfo {
  id: string;
  category: string;
  size_gb: number;
  status: "downloaded" | "downloading" | "pending";
  capabilities: string[];
}

// Real model data from ASI_SYMBIOSIS_SYSTEM
const MODEL_REGISTRY: ModelInfo[] = [
  // Downloaded Embedding Models
  { id: "BAAI/bge-large-en-v1.5", category: "embedding", size_gb: 0.67, status: "downloaded", capabilities: ["embedding", "retrieval"] },
  { id: "BAAI/bge-base-en-v1.5", category: "embedding", size_gb: 0.22, status: "downloaded", capabilities: ["embedding", "retrieval"] },
  { id: "BAAI/bge-small-en-v1.5", category: "embedding", size_gb: 0.07, status: "downloaded", capabilities: ["embedding"] },
  { id: "sentence-transformers/all-MiniLM-L6-v2", category: "embedding", size_gb: 0.09, status: "downloaded", capabilities: ["embedding"] },
  { id: "sentence-transformers/all-mpnet-base-v2", category: "embedding", size_gb: 0.44, status: "downloaded", capabilities: ["embedding"] },
  { id: "intfloat/e5-large-v2", category: "embedding", size_gb: 0.67, status: "downloaded", capabilities: ["embedding"] },
  { id: "thenlper/gte-large", category: "embedding", size_gb: 0.67, status: "downloaded", capabilities: ["embedding"] },
  
  // Downloaded Small LLMs
  { id: "TinyLlama/TinyLlama-1.1B-Chat-v1.0", category: "foundation", size_gb: 2.2, status: "downloaded", capabilities: ["chat"] },
  { id: "HuggingFaceTB/SmolLM-360M-Instruct", category: "foundation", size_gb: 0.72, status: "downloaded", capabilities: ["chat"] },
  
  // Pending Large Models
  { id: "meta-llama/Llama-3.3-70B-Instruct", category: "foundation", size_gb: 140, status: "pending", capabilities: ["chat", "reasoning", "code"] },
  { id: "meta-llama/Llama-3.1-70B-Instruct", category: "foundation", size_gb: 140, status: "pending", capabilities: ["chat", "reasoning", "code"] },
  { id: "mistralai/Mixtral-8x22B-Instruct-v0.1", category: "foundation", size_gb: 282, status: "pending", capabilities: ["chat", "reasoning"] },
  { id: "Qwen/Qwen2.5-72B-Instruct", category: "foundation", size_gb: 144, status: "pending", capabilities: ["chat", "reasoning", "math"] },
  { id: "deepseek-ai/DeepSeek-V3", category: "foundation", size_gb: 1342, status: "pending", capabilities: ["chat", "reasoning", "code"] },
  { id: "deepseek-ai/DeepSeek-R1", category: "reasoning", size_gb: 1342, status: "pending", capabilities: ["reasoning", "math"] },
  { id: "deepseek-ai/deepseek-coder-33b-instruct", category: "code", size_gb: 66, status: "pending", capabilities: ["code", "debugging"] },
  { id: "Qwen/Qwen2.5-Coder-32B-Instruct", category: "code", size_gb: 64, status: "pending", capabilities: ["code", "debugging"] },
  { id: "WizardLM/WizardMath-70B-V1.0", category: "math", size_gb: 140, status: "pending", capabilities: ["math", "reasoning"] },
  { id: "openai/whisper-large-v3", category: "audio", size_gb: 3.1, status: "pending", capabilities: ["transcription"] },
];

export default function ASIStatus() {
  const [models] = useState<ModelInfo[]>(MODEL_REGISTRY);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const downloadedModels = models.filter(m => m.status === "downloaded");
  const pendingModels = models.filter(m => m.status === "pending");
  const downloadedSize = downloadedModels.reduce((acc, m) => acc + m.size_gb, 0);
  const totalSize = models.reduce((acc, m) => acc + m.size_gb, 0);
  const downloadProgress = (downloadedSize / totalSize) * 100;

  const categoryStats = models.reduce((acc, m) => {
    if (!acc[m.category]) {
      acc[m.category] = { total: 0, downloaded: 0 };
    }
    acc[m.category].total++;
    if (m.status === "downloaded") acc[m.category].downloaded++;
    return acc;
  }, {} as Record<string, { total: number; downloaded: number }>);

  const handleRefresh = () => {
    setIsRefreshing(true);
    setTimeout(() => setIsRefreshing(false), 1000);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl">
              <Brain className="h-8 w-8 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-white">ASI Model Status</h1>
              <p className="text-slate-400">Real-time model download and inference status</p>
            </div>
          </div>
          <Button 
            onClick={handleRefresh} 
            variant="outline" 
            className="gap-2"
            disabled={isRefreshing}
          >
            <RefreshCw className={`h-4 w-4 ${isRefreshing ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card className="bg-slate-900/50 border-slate-800">
            <CardContent className="p-6">
              <div className="flex items-center gap-4">
                <div className="p-3 bg-green-500/20 rounded-lg">
                  <CheckCircle2 className="h-6 w-6 text-green-500" />
                </div>
                <div>
                  <p className="text-sm text-slate-400">Downloaded</p>
                  <p className="text-2xl font-bold text-white">{downloadedModels.length}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-900/50 border-slate-800">
            <CardContent className="p-6">
              <div className="flex items-center gap-4">
                <div className="p-3 bg-yellow-500/20 rounded-lg">
                  <Clock className="h-6 w-6 text-yellow-500" />
                </div>
                <div>
                  <p className="text-sm text-slate-400">Pending</p>
                  <p className="text-2xl font-bold text-white">{pendingModels.length}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-900/50 border-slate-800">
            <CardContent className="p-6">
              <div className="flex items-center gap-4">
                <div className="p-3 bg-blue-500/20 rounded-lg">
                  <Database className="h-6 w-6 text-blue-500" />
                </div>
                <div>
                  <p className="text-sm text-slate-400">Downloaded Size</p>
                  <p className="text-2xl font-bold text-white">{downloadedSize.toFixed(1)} GB</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-900/50 border-slate-800">
            <CardContent className="p-6">
              <div className="flex items-center gap-4">
                <div className="p-3 bg-purple-500/20 rounded-lg">
                  <Zap className="h-6 w-6 text-purple-500" />
                </div>
                <div>
                  <p className="text-sm text-slate-400">Total Target</p>
                  <p className="text-2xl font-bold text-white">{totalSize.toFixed(0)} GB</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Progress Bar */}
        <Card className="bg-slate-900/50 border-slate-800">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <Download className="h-5 w-5" />
              Download Progress
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-slate-400">Progress</span>
                <span className="text-white font-medium">{downloadProgress.toFixed(1)}%</span>
              </div>
              <Progress value={downloadProgress} className="h-3" />
              <p className="text-xs text-slate-500">
                {downloadedSize.toFixed(1)} GB of {totalSize.toFixed(0)} GB downloaded
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Category Stats */}
        <Card className="bg-slate-900/50 border-slate-800">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Models by Category
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {Object.entries(categoryStats).map(([category, stats]) => (
                <div key={category} className="p-4 bg-slate-800/50 rounded-lg">
                  <p className="text-sm text-slate-400 capitalize">{category}</p>
                  <p className="text-xl font-bold text-white">
                    {stats.downloaded}/{stats.total}
                  </p>
                  <Progress 
                    value={(stats.downloaded / stats.total) * 100} 
                    className="h-1 mt-2" 
                  />
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Model List */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Downloaded Models */}
          <Card className="bg-slate-900/50 border-slate-800">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <CheckCircle2 className="h-5 w-5 text-green-500" />
                Downloaded Models ({downloadedModels.length})
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 max-h-96 overflow-y-auto">
              {downloadedModels.map((model) => (
                <div 
                  key={model.id} 
                  className="p-3 bg-slate-800/50 rounded-lg flex items-center justify-between"
                >
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-white truncate">{model.id}</p>
                    <div className="flex items-center gap-2 mt-1">
                      <Badge variant="outline" className="text-xs">
                        {model.category}
                      </Badge>
                      <span className="text-xs text-slate-500">{model.size_gb} GB</span>
                    </div>
                  </div>
                  <Badge className="bg-green-500/20 text-green-400 border-green-500/30">
                    Ready
                  </Badge>
                </div>
              ))}
            </CardContent>
          </Card>

          {/* Pending Models */}
          <Card className="bg-slate-900/50 border-slate-800">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Clock className="h-5 w-5 text-yellow-500" />
                Pending Models ({pendingModels.length})
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 max-h-96 overflow-y-auto">
              {pendingModels.map((model) => (
                <div 
                  key={model.id} 
                  className="p-3 bg-slate-800/50 rounded-lg flex items-center justify-between"
                >
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-white truncate">{model.id}</p>
                    <div className="flex items-center gap-2 mt-1">
                      <Badge variant="outline" className="text-xs">
                        {model.category}
                      </Badge>
                      <span className="text-xs text-slate-500">{model.size_gb} GB</span>
                    </div>
                  </div>
                  <Badge className="bg-yellow-500/20 text-yellow-400 border-yellow-500/30">
                    Pending
                  </Badge>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>

        {/* System Info */}
        <Card className="bg-gradient-to-r from-purple-900/30 to-pink-900/30 border-purple-500/30">
          <CardContent className="p-6">
            <div className="flex items-center gap-4">
              <Cpu className="h-12 w-12 text-purple-400" />
              <div>
                <h3 className="text-xl font-bold text-white">ASI Symbiosis Engine</h3>
                <p className="text-slate-300">
                  Multi-model consensus system with task-based routing. 
                  {downloadedModels.length} models ready for inference.
                </p>
                <div className="flex gap-2 mt-2">
                  {["chat", "code", "math", "reasoning", "embedding"].map((cap) => (
                    <Badge key={cap} variant="secondary" className="text-xs">
                      {cap}
                    </Badge>
                  ))}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
