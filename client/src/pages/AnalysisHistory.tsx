import { useState } from "react";
import { trpc } from "@/lib/trpc";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { Search, Download, Trash2, TrendingUp, Filter, FileText } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { LoadingSkeleton } from "@/components/LoadingSkeleton";

type SortField = "date" | "company" | "industry" | "score";
type SortOrder = "asc" | "desc";

export default function AnalysisHistory() {
  const [searchQuery, setSearchQuery] = useState("");
  const [industryFilter, setIndustryFilter] = useState<string>("all");
  const [sortField, setSortField] = useState<SortField>("date");
  const [sortOrder, setSortOrder] = useState<SortOrder>("desc");
  const [selectedAnalyses, setSelectedAnalyses] = useState<string[]>([]);
  const { toast } = useToast();

  // Fetch analyses
  const { data: analyses, isLoading, refetch } = trpc.analysisHistory.getMyAnalyses.useQuery({
    limit: 100,
    offset: 0,
  });

  // Fetch stats
  const { data: stats } = trpc.analysisHistory.getMyStats.useQuery();

  // Delete mutation
  const deleteMutation = trpc.analysisHistory.deleteAnalysis.useMutation({
    onSuccess: () => {
      toast({ title: "Success", description: "Analysis deleted successfully" });
      refetch();
    },
    onError: (error) => {
      toast({ title: "Error", description: error.message, variant: "destructive" });
    },
  });

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 py-12 px-4">
        <div className="container mx-auto max-w-7xl">
          <div className="mb-8">
            <div className="w-80 h-12 bg-white/10 rounded animate-pulse mb-2" />
            <div className="w-64 h-6 bg-white/10 rounded animate-pulse" />
          </div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <LoadingSkeleton variant="metric" count={4} />
          </div>
          <LoadingSkeleton variant="chart" count={1} />
          <div className="mt-8">
            <LoadingSkeleton variant="table" count={5} />
          </div>
        </div>
      </div>
    );
  }

  const analysesList = analyses || [];

  // Filter and sort analyses
  const filteredAnalyses = analysesList
    .filter((a) => {
      const matchesSearch = a.companyName.toLowerCase().includes(searchQuery.toLowerCase()) ||
                           a.organizationNumber.includes(searchQuery);
      const matchesIndustry = industryFilter === "all" || a.industryCategory === industryFilter;
      return matchesSearch && matchesIndustry;
    })
    .sort((a, b) => {
      let comparison = 0;
      switch (sortField) {
        case "date":
          comparison = new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime();
          break;
        case "company":
          comparison = a.companyName.localeCompare(b.companyName);
          break;
        case "industry":
          comparison = (a.industryCategory || "").localeCompare(b.industryCategory || "");
          break;
        case "score":
          comparison = a.digitalMaturityScore - b.digitalMaturityScore;
          break;
      }
      return sortOrder === "asc" ? comparison : -comparison;
    });

  // Prepare trend chart data
  const trendData = analysesList
    .sort((a, b) => new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime())
    .map((a) => ({
      date: new Date(a.createdAt).toLocaleDateString(),
      score: a.digitalMaturityScore,
      company: a.companyName,
    }));

  // Get unique industries for filter
  const industries = Array.from(new Set(analysesList.map((a) => a.industryCategory).filter(Boolean))) as string[];

  const handleDelete = (id: string) => {
    if (confirm("Are you sure you want to delete this analysis?")) {
      deleteMutation.mutate({ analysisId: id });
    }
  };

  const handleExportPDF = (analysisId: string) => {
    toast({ title: "Coming Soon", description: "PDF export functionality will be available soon" });
  };

  const handleExportAllCSV = () => {
    const csv = [
      ["Date", "Company", "Org Number", "Industry", "Digital Maturity Score", "Position", "Status"],
      ...filteredAnalyses.map((a) => [
        new Date(a.createdAt).toLocaleDateString(),
        a.companyName,
        a.organizationNumber,
        a.industryCategory || "N/A",
        a.digitalMaturityScore.toString(),
        a.competitivePosition,
        "completed",
      ]),
    ]
      .map((row) => row.join(","))
      .join("\\n");

    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `analysis-history-${new Date().toISOString().split("T")[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);

    toast({ title: "Success", description: "CSV exported successfully" });
  };

  const toggleSelection = (id: string) => {
    setSelectedAnalyses((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 py-12 px-4">
      <div className="container mx-auto max-w-7xl">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-5xl font-black text-white mb-2 tracking-tight">Analysis History</h1>
          <p className="text-slate-300">View and manage all your business assessments</p>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <Card className="bg-white/5 backdrop-blur-xl border-white/10 hover:border-white/20 transition-all duration-300 hover:scale-[1.02] shadow-2xl hover:shadow-purple-500/20 p-6">
            <div className="text-slate-300 mb-2">Total Analyses</div>
            <div className="text-3xl font-bold text-white">{stats?.totalAnalyses || 0}</div>
          </Card>
          <Card className="bg-white/5 backdrop-blur-xl border-white/10 hover:border-white/20 transition-all duration-300 hover:scale-[1.02] shadow-2xl hover:shadow-purple-500/20 p-6">
            <div className="text-slate-300 mb-2">Average Score</div>
            <div className="text-3xl font-bold text-white">
              {stats?.avgDigitalMaturity?.toFixed(1) || "N/A"}
            </div>
          </Card>
          <Card className="bg-white/5 backdrop-blur-xl border-white/10 hover:border-white/20 transition-all duration-300 hover:scale-[1.02] shadow-2xl hover:shadow-purple-500/20 p-6">
            <div className="text-slate-300 mb-2">Top Industry</div>
            <div className="text-xl font-bold text-white">
              {stats?.topIndustries?.[0]?.industry || "N/A"}
            </div>
          </Card>
          <Card className="bg-white/5 backdrop-blur-xl border-white/10 hover:border-white/20 transition-all duration-300 hover:scale-[1.02] shadow-2xl hover:shadow-purple-500/20 p-6">
            <div className="text-slate-300 mb-2">Leaders</div>
            <div className="text-3xl font-bold text-green-400">
              {stats?.positionCounts?.leader || 0}
            </div>
          </Card>
        </div>

        {/* Trend Chart */}
        {trendData.length > 1 && (
          <Card className="bg-white/5 backdrop-blur-xl border-white/10 hover:border-white/20 transition-all duration-300 shadow-2xl hover:shadow-cyan-500/10 p-6 mb-8">
            <h2 className="text-3xl font-bold text-white mb-6 tracking-tight flex items-center gap-2">
              <TrendingUp className="w-6 h-6" />
              Digital Maturity Score Trend
            </h2>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={trendData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="date" stroke="rgba(255,255,255,0.5)" />
                <YAxis stroke="rgba(255,255,255,0.5)" domain={[0, 100]} />
                <Tooltip
                  contentStyle={{ backgroundColor: "rgba(0,0,0,0.8)", border: "1px solid rgba(255,255,255,0.2)" }}
                  labelStyle={{ color: "#fff" }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="score"
                  stroke="#10b981"
                  strokeWidth={2}
                  name="Digital Maturity Score"
                  dot={{ fill: "#10b981", r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        )}

        {/* Filters and Search */}
        <Card className="bg-white/10 backdrop-blur-lg border-white/20 p-6 mb-8">
          <div className="flex flex-wrap gap-4">
            <div className="flex-1 min-w-[200px]">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400" />
                <Input
                  placeholder="Search by company name or org number..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10 bg-white/5 border-white/20 text-white placeholder:text-slate-400"
                />
              </div>
            </div>

            <Select value={industryFilter} onValueChange={setIndustryFilter}>
              <SelectTrigger className="w-[200px] bg-white/5 border-white/20 text-white">
                <Filter className="w-4 h-4 mr-2" />
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Industries</SelectItem>
                {industries.map((industry) => (
                  <SelectItem key={industry} value={industry}>
                    {industry}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <Select value={sortField} onValueChange={(v) => setSortField(v as SortField)}>
              <SelectTrigger className="w-[180px] bg-white/5 border-white/20 text-white">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="date">Sort by Date</SelectItem>
                <SelectItem value="company">Sort by Company</SelectItem>
                <SelectItem value="industry">Sort by Industry</SelectItem>
                <SelectItem value="score">Sort by Score</SelectItem>
              </SelectContent>
            </Select>

            <Button
              variant="outline"
              onClick={() => setSortOrder(sortOrder === "asc" ? "desc" : "asc")}
              className="bg-white/5 border-white/20 text-white hover:bg-white/10"
            >
              {sortOrder === "asc" ? "↑" : "↓"}
            </Button>

            <Button
              variant="outline"
              onClick={handleExportAllCSV}
              className="bg-white/5 border-white/20 text-white hover:bg-white/10"
            >
              <Download className="w-4 h-4 mr-2" />
              Export CSV
            </Button>
          </div>
        </Card>

        {/* Table */}
        <Card className="bg-white/10 backdrop-blur-lg border-white/20 overflow-hidden">
          <Table>
            <TableHeader>
              <TableRow className="border-white/10 hover:bg-white/5">
                <TableHead className="text-slate-300">Date</TableHead>
                <TableHead className="text-slate-300">Company</TableHead>
                <TableHead className="text-slate-300">Org Number</TableHead>
                <TableHead className="text-slate-300">Industry</TableHead>
                <TableHead className="text-slate-300">Score</TableHead>
                <TableHead className="text-slate-300">Position</TableHead>
                <TableHead className="text-slate-300">Status</TableHead>
                <TableHead className="text-slate-300">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredAnalyses.map((analysis: any) => (
                <TableRow
                  key={analysis.id}
                  className="border-white/10 hover:bg-white/5 cursor-pointer"
                  onClick={() => toggleSelection(analysis.id)}
                >
                  <TableCell className="text-white">
                    {new Date(analysis.createdAt).toLocaleDateString()}
                  </TableCell>
                  <TableCell className="text-white font-medium">{analysis.companyName}</TableCell>
                  <TableCell className="text-slate-300">{analysis.organizationNumber}</TableCell>
                  <TableCell className="text-slate-300">{analysis.industryCategory || "N/A"}</TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <div
                        className={`w-12 h-12 rounded-full flex items-center justify-center font-bold ${
                          analysis.digitalMaturityScore >= 80
                            ? "bg-green-500/20 text-green-400"
                            : analysis.digitalMaturityScore >= 60
                            ? "bg-yellow-500/20 text-yellow-400"
                            : "bg-red-500/20 text-red-400"
                        }`}
                      >
                        {analysis.digitalMaturityScore}
                      </div>
                    </div>
                  </TableCell>
                  <TableCell>
                    <span
                      className={`px-3 py-1 rounded-full text-sm font-medium ${
                        analysis.competitivePosition === "leader"
                          ? "bg-green-500/20 text-green-400"
                          : analysis.competitivePosition === "challenger"
                          ? "bg-blue-500/20 text-blue-400"
                          : analysis.competitivePosition === "follower"
                          ? "bg-yellow-500/20 text-yellow-400"
                          : "bg-slate-500/20 text-slate-400"
                      }`}
                    >
                      {analysis.competitivePosition}
                    </span>
                  </TableCell>
                  <TableCell>
                    <span className="px-3 py-1 rounded-full text-sm font-medium bg-green-500/20 text-green-400">
                      completed
                    </span>
                  </TableCell>
                  <TableCell>
                    <div className="flex gap-2" onClick={(e) => e.stopPropagation()}>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => handleExportPDF(analysis.id)}
                        className="text-white hover:bg-white/10"
                      >
                        <FileText className="w-4 h-4" />
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => handleDelete(analysis.id)}
                        className="text-red-400 hover:bg-red-500/10"
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>

          {filteredAnalyses.length === 0 && (
            <div className="text-center py-12 text-slate-300">
              No analyses found. Try adjusting your filters.
            </div>
          )}
        </Card>

        {/* Selected Analyses Info */}
        {selectedAnalyses.length > 0 && (
          <Card className="bg-white/10 backdrop-blur-lg border-white/20 p-6 mt-8">
            <div className="flex items-center justify-between">
              <div className="text-white">
                <span className="font-bold">{selectedAnalyses.length}</span> analyses selected
              </div>
              <div className="flex gap-4">
                <Button
                  variant="outline"
                  onClick={() => setSelectedAnalyses([])}
                  className="bg-white/5 border-white/20 text-white hover:bg-white/10"
                >
                  Clear Selection
                </Button>
                <Button
                  variant="outline"
                  className="bg-white/5 border-white/20 text-white hover:bg-white/10"
                >
                  Compare Selected
                </Button>
              </div>
            </div>
          </Card>
        )}
      </div>
    </div>
  );
}
