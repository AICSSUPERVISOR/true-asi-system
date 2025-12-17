import { useAuth } from "@/_core/hooks/useAuth";
import { trpc } from "@/lib/trpc";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { toast } from "sonner";
import {
  BarChart3,
  PieChart,
  TrendingUp,
  Download,
  Calendar,
  FileSpreadsheet,
  FileText,
  Building2,
  DollarSign,
  ArrowUpRight,
  ArrowDownRight,
} from "lucide-react";

export default function Reports() {
  const { user } = useAuth();
  const [selectedCompanyId, setSelectedCompanyId] = useState<number | null>(null);
  const [selectedPeriod, setSelectedPeriod] = useState("current_year");

  const { data: companies, isLoading: companiesLoading } = trpc.company.list.useQuery();

  if (!user) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <p className="text-muted-foreground">Vennligst logg inn for å se denne siden.</p>
      </div>
    );
  }

  const reportTypes = [
    {
      id: "balance",
      title: "Balanse",
      description: "Eiendeler, gjeld og egenkapital",
      icon: BarChart3,
      color: "text-blue-500",
    },
    {
      id: "profit_loss",
      title: "Resultatregnskap",
      description: "Inntekter og kostnader",
      icon: TrendingUp,
      color: "text-green-500",
    },
    {
      id: "vat_summary",
      title: "MVA-oversikt",
      description: "Utgående og inngående MVA",
      icon: PieChart,
      color: "text-purple-500",
    },
    {
      id: "ledger",
      title: "Hovedbok",
      description: "Alle posteringer per konto",
      icon: FileSpreadsheet,
      color: "text-orange-500",
    },
    {
      id: "trial_balance",
      title: "Saldobalanse",
      description: "Saldo per konto",
      icon: DollarSign,
      color: "text-emerald-500",
    },
    {
      id: "journal",
      title: "Journalrapport",
      description: "Alle bilag i perioden",
      icon: FileText,
      color: "text-cyan-500",
    },
  ];

  const handleGenerateReport = (reportId: string) => {
    if (!selectedCompanyId) {
      toast.error("Velg et selskap først");
      return;
    }
    toast.info(`Genererer ${reportTypes.find(r => r.id === reportId)?.title}...`);
    // In production, this would call a tRPC mutation to generate the report
  };

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-3xl font-bold">Rapporter</h1>
          <p className="text-muted-foreground">Generer finansielle rapporter og analyser</p>
        </div>
        <div className="flex gap-3">
          <Select value={selectedCompanyId?.toString() || ""} onValueChange={(v) => setSelectedCompanyId(parseInt(v))}>
            <SelectTrigger className="w-[200px]">
              <Building2 className="mr-2 h-4 w-4" />
              <SelectValue placeholder="Velg selskap" />
            </SelectTrigger>
            <SelectContent>
              {companies?.map((company) => (
                <SelectItem key={company.id} value={company.id.toString()}>
                  {company.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Select value={selectedPeriod} onValueChange={setSelectedPeriod}>
            <SelectTrigger className="w-[180px]">
              <Calendar className="mr-2 h-4 w-4" />
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="current_month">Denne måneden</SelectItem>
              <SelectItem value="last_month">Forrige måned</SelectItem>
              <SelectItem value="current_quarter">Dette kvartalet</SelectItem>
              <SelectItem value="last_quarter">Forrige kvartal</SelectItem>
              <SelectItem value="current_year">Dette året</SelectItem>
              <SelectItem value="last_year">Forrige år</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Quick Stats */}
      {selectedCompanyId && (
        <div className="grid gap-4 md:grid-cols-4">
          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Omsetning</p>
                  <p className="text-2xl font-bold">kr 1 234 567</p>
                </div>
                <div className="flex items-center text-green-500">
                  <ArrowUpRight className="h-4 w-4" />
                  <span className="text-sm">+12%</span>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Resultat</p>
                  <p className="text-2xl font-bold">kr 234 567</p>
                </div>
                <div className="flex items-center text-green-500">
                  <ArrowUpRight className="h-4 w-4" />
                  <span className="text-sm">+8%</span>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">MVA skyldig</p>
                  <p className="text-2xl font-bold">kr 45 678</p>
                </div>
                <div className="flex items-center text-red-500">
                  <ArrowDownRight className="h-4 w-4" />
                  <span className="text-sm">-5%</span>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Bilag</p>
                  <p className="text-2xl font-bold">156</p>
                </div>
                <div className="flex items-center text-green-500">
                  <ArrowUpRight className="h-4 w-4" />
                  <span className="text-sm">+23</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Report Types */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {reportTypes.map((report) => (
          <Card key={report.id} className="hover:border-emerald-500/50 transition-colors cursor-pointer">
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className={`p-2 rounded-lg bg-muted ${report.color}`}>
                  <report.icon className="h-5 w-5" />
                </div>
                <div>
                  <CardTitle className="text-lg">{report.title}</CardTitle>
                  <CardDescription>{report.description}</CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  className="flex-1"
                  onClick={() => handleGenerateReport(report.id)}
                >
                  <FileText className="mr-2 h-4 w-4" />
                  Vis
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleGenerateReport(report.id)}
                >
                  <Download className="h-4 w-4" />
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* SAF-T Export */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileSpreadsheet className="h-5 w-5 text-emerald-500" />
            SAF-T Eksport
          </CardTitle>
          <CardDescription>
            Eksporter regnskapsdata i SAF-T format for Skatteetaten
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-sm text-muted-foreground">
                SAF-T (Standard Audit File - Tax) er et standardisert format for utveksling av regnskapsdata.
                Alle bokføringspliktige virksomheter må kunne levere SAF-T-fil på forespørsel fra Skatteetaten.
              </p>
            </div>
            <Button className="bg-emerald-600 hover:bg-emerald-700 whitespace-nowrap">
              <Download className="mr-2 h-4 w-4" />
              Generer SAF-T
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
