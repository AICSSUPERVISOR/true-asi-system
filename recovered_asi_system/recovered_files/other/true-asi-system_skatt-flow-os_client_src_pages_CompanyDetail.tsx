import { useState } from "react";
import { useParams, Link } from "wouter";
import { trpc } from "@/lib/trpc";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Skeleton } from "@/components/ui/skeleton";
import { Separator } from "@/components/ui/separator";
import {
  Building2,
  ArrowLeft,
  RefreshCw,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  FileText,
  BookOpen,
  FileCheck,
  ExternalLink,
} from "lucide-react";
import { toast } from "sonner";

export default function CompanyDetail() {
  const params = useParams<{ id: string }>();
  const companyId = parseInt(params.id || "0");

  const { data: company, isLoading, refetch } = trpc.company.get.useQuery(
    { id: companyId },
    { enabled: !!companyId }
  );

  // Forvalt history would come from a separate query - using company data for now

  const refreshMutation = trpc.company.refreshForvaltData.useMutation({
    onSuccess: () => {
      toast.success("Forvalt-data oppdatert");
      refetch();
    },
    onError: (error) => {
      toast.error(`Kunne ikke oppdatere: ${error.message}`);
    },
  });

  if (isLoading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-10 w-48" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  if (!company) {
    return (
      <div className="flex flex-col items-center justify-center py-12">
        <Building2 className="h-12 w-12 text-slate-300 mb-4" />
        <h3 className="text-lg font-medium text-slate-900 dark:text-white mb-2">
          Selskap ikke funnet
        </h3>
        <Link href="/companies">
          <Button variant="outline">
            <ArrowLeft className="h-4 w-4 mr-2" />
            Tilbake til selskaper
          </Button>
        </Link>
      </div>
    );
  }

  const riskStyles: Record<string, { bg: string; icon: typeof CheckCircle }> = {
    HIGH: { bg: "bg-red-100 text-red-700", icon: AlertTriangle },
    MEDIUM: { bg: "bg-amber-100 text-amber-700", icon: AlertTriangle },
    LOW: { bg: "bg-emerald-100 text-emerald-700", icon: CheckCircle },
  };

  const riskConfig = riskStyles[company.forvaltRiskClass || "LOW"] || riskStyles.LOW;
  const RiskIcon = riskConfig.icon;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Link href="/companies">
          <Button variant="ghost" size="icon">
            <ArrowLeft className="h-5 w-5" />
          </Button>
        </Link>
        <div className="flex-1">
          <h1 className="text-3xl font-bold tracking-tight text-slate-900 dark:text-white">
            {company.name}
          </h1>
          <p className="text-slate-500">Org.nr: {company.orgNumber}</p>
        </div>
        <Button
          variant="outline"
          onClick={() => refreshMutation.mutate({ id: company.id })}
          disabled={refreshMutation.isPending}
        >
          <RefreshCw className={`h-4 w-4 mr-2 ${refreshMutation.isPending ? "animate-spin" : ""}`} />
          Oppdater Forvalt-data
        </Button>
      </div>

      {/* Overview Cards */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card className="border-slate-200 dark:border-slate-800">
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className={`h-10 w-10 rounded-lg flex items-center justify-center ${riskConfig.bg}`}>
                <RiskIcon className="h-5 w-5" />
              </div>
              <div>
                <p className="text-sm text-slate-500">Risikoklasse</p>
                <p className="text-xl font-bold">
                  {company.forvaltRiskClass === "HIGH"
                    ? "Høy"
                    : company.forvaltRiskClass === "MEDIUM"
                    ? "Middels"
                    : "Lav"}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border-slate-200 dark:border-slate-800">
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-lg bg-blue-100 flex items-center justify-center">
                <TrendingUp className="h-5 w-5 text-blue-700" />
              </div>
              <div>
                <p className="text-sm text-slate-500">Rating</p>
                <p className="text-xl font-bold">{company.forvaltRating || "-"}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border-slate-200 dark:border-slate-800">
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-lg bg-purple-100 flex items-center justify-center">
                <TrendingDown className="h-5 w-5 text-purple-700" />
              </div>
              <div>
                <p className="text-sm text-slate-500">Kredittscore</p>
                <p className="text-xl font-bold">{company.forvaltCreditScore ?? "-"}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border-slate-200 dark:border-slate-800">
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-lg bg-slate-100 flex items-center justify-center">
                <Building2 className="h-5 w-5 text-slate-700" />
              </div>
              <div>
                <p className="text-sm text-slate-500">System</p>
                <p className="text-xl font-bold">
                  {company.externalRegnskapSystem?.replace("_", " ") || "-"}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Tabs */}
      <Tabs defaultValue="info">
        <TabsList>
          <TabsTrigger value="info">Informasjon</TabsTrigger>
          <TabsTrigger value="forvalt">Forvalt-historikk</TabsTrigger>
          <TabsTrigger value="quick-actions">Hurtighandlinger</TabsTrigger>
        </TabsList>

        <TabsContent value="info" className="mt-6">
          <Card className="border-slate-200 dark:border-slate-800">
            <CardHeader>
              <CardTitle>Selskapsinformasjon</CardTitle>
              <CardDescription>Detaljert informasjon fra Brønnøysundregistrene og Forvalt</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-6 md:grid-cols-2">
                <div className="space-y-4">
                  <div>
                    <p className="text-sm text-slate-500">Adresse</p>
                    <p className="font-medium">{company.address || "-"}</p>
                  </div>
                  <div>
                    <p className="text-sm text-slate-500">Postnummer og sted</p>
                    <p className="font-medium">
                      {company.postalCode} {company.city}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-slate-500">Opprettet i systemet</p>
                    <p className="font-medium">
                      {new Date(company.createdAt).toLocaleDateString("nb-NO")}
                    </p>
                  </div>
                </div>

                <div className="space-y-4">
                  <div>
                    <p className="text-sm text-slate-500">Regnskapssystem</p>
                    <p className="font-medium">
                      {company.externalRegnskapSystem?.replace("_", " ") || "Ikke tilkoblet"}
                    </p>
                  </div>
                  {company.externalRegnskapCompanyId && (
                    <div>
                      <p className="text-sm text-slate-500">Selskaps-ID i system</p>
                      <p className="font-medium font-mono">{company.externalRegnskapCompanyId}</p>
                    </div>
                  )}
                  <div>
                    <p className="text-sm text-slate-500">Sist oppdatert</p>
                    <p className="font-medium">
                      {new Date(company.updatedAt).toLocaleString("nb-NO")}
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="forvalt" className="mt-6">
          <Card className="border-slate-200 dark:border-slate-800">
            <CardHeader>
              <CardTitle>Forvalt-historikk</CardTitle>
              <CardDescription>Historiske snapshots fra Forvalt</CardDescription>
            </CardHeader>
            <CardContent>
              {company ? (
                <div className="space-y-4">
                  <div className="p-4 border border-slate-200 dark:border-slate-700 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <p className="font-medium">
                        {new Date(company.updatedAt).toLocaleString("nb-NO")}
                      </p>
                      <Badge variant="outline">{company.forvaltRating || "-"}</Badge>
                    </div>
                    <div className="grid grid-cols-3 gap-4 text-sm">
                      <div>
                        <p className="text-slate-500">Kredittscore</p>
                        <p className="font-medium">{company.forvaltCreditScore ?? "-"}</p>
                      </div>
                      <div>
                        <p className="text-slate-500">Risikoklasse</p>
                        <p className="font-medium">{company.forvaltRiskClass || "-"}</p>
                      </div>
                      <div>
                        <p className="text-slate-500">Status</p>
                        <p className="font-medium">Aktiv</p>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <p className="text-slate-500 text-center py-8">
                  Ingen Forvalt-historikk tilgjengelig
                </p>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="quick-actions" className="mt-6">
          <div className="grid gap-4 md:grid-cols-3">
            <Link href="/accounting">
              <Card className="border-slate-200 dark:border-slate-800 hover:shadow-md transition-shadow cursor-pointer h-full">
                <CardContent className="pt-6">
                  <div className="flex items-center gap-3 mb-3">
                    <div className="h-10 w-10 rounded-lg bg-blue-100 flex items-center justify-center">
                      <FileText className="h-5 w-5 text-blue-700" />
                    </div>
                    <h3 className="font-medium">Regnskapsdokumenter</h3>
                  </div>
                  <p className="text-sm text-slate-500">
                    Last opp og behandle bilag for dette selskapet
                  </p>
                </CardContent>
              </Card>
            </Link>

            <Link href="/ledger">
              <Card className="border-slate-200 dark:border-slate-800 hover:shadow-md transition-shadow cursor-pointer h-full">
                <CardContent className="pt-6">
                  <div className="flex items-center gap-3 mb-3">
                    <div className="h-10 w-10 rounded-lg bg-emerald-100 flex items-center justify-center">
                      <BookOpen className="h-5 w-5 text-emerald-700" />
                    </div>
                    <h3 className="font-medium">Hovedbok</h3>
                  </div>
                  <p className="text-sm text-slate-500">
                    Se og filtrer hovedbokposter
                  </p>
                </CardContent>
              </Card>
            </Link>

            <Link href="/filings">
              <Card className="border-slate-200 dark:border-slate-800 hover:shadow-md transition-shadow cursor-pointer h-full">
                <CardContent className="pt-6">
                  <div className="flex items-center gap-3 mb-3">
                    <div className="h-10 w-10 rounded-lg bg-purple-100 flex items-center justify-center">
                      <FileCheck className="h-5 w-5 text-purple-700" />
                    </div>
                    <h3 className="font-medium">Innleveringer</h3>
                  </div>
                  <p className="text-sm text-slate-500">
                    MVA-meldinger og SAF-T eksport
                  </p>
                </CardContent>
              </Card>
            </Link>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
