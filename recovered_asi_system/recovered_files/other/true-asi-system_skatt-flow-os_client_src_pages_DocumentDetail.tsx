import { useAuth } from "@/_core/hooks/useAuth";
import { trpc } from "@/lib/trpc";
import { useParams, useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { toast } from "sonner";
import {
  ArrowLeft,
  FileText,
  Calendar,
  Building2,
  Eye,
  Download,
  Clock,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  RefreshCw,
  Sparkles,
  FileCheck,
} from "lucide-react";

export default function DocumentDetail() {
  const { user } = useAuth();
  const params = useParams<{ id: string }>();
  const [, navigate] = useLocation();
  const documentId = parseInt(params.id || "0");

  const { data: document, isLoading, refetch } = trpc.document.get.useQuery(
    { id: documentId },
    { enabled: !!documentId }
  );

  const processMutation = trpc.document.process.useMutation({
    onSuccess: () => {
      toast.success("Dokument behandlet med AI");
      refetch();
    },
    onError: (error: { message: string }) => {
      toast.error(`Feil ved behandling: ${error.message}`);
    },
  });

  if (!user) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <p className="text-muted-foreground">Vennligst logg inn for å se denne siden.</p>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="space-y-6 p-6">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-[400px] w-full" />
      </div>
    );
  }

  if (!document) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] gap-4">
        <XCircle className="h-16 w-16 text-muted-foreground" />
        <p className="text-muted-foreground">Dokument ikke funnet</p>
        <Button variant="outline" onClick={() => navigate("/accounting")}>
          <ArrowLeft className="mr-2 h-4 w-4" />
          Tilbake til regnskap
        </Button>
      </div>
    );
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "NEW":
        return <Badge variant="secondary"><Clock className="mr-1 h-3 w-3" />Ny</Badge>;
      case "PROCESSING":
        return <Badge variant="outline" className="bg-blue-500/10 text-blue-600"><RefreshCw className="mr-1 h-3 w-3 animate-spin" />Behandler</Badge>;
      case "PENDING_APPROVAL":
        return <Badge variant="outline" className="bg-yellow-500/10 text-yellow-600"><AlertTriangle className="mr-1 h-3 w-3" />Venter godkjenning</Badge>;
      case "APPROVED":
        return <Badge variant="outline" className="bg-green-500/10 text-green-600"><CheckCircle2 className="mr-1 h-3 w-3" />Godkjent</Badge>;
      case "POSTED":
        return <Badge variant="outline" className="bg-emerald-500/10 text-emerald-600"><FileCheck className="mr-1 h-3 w-3" />Bokført</Badge>;
      case "REJECTED":
        return <Badge variant="destructive"><XCircle className="mr-1 h-3 w-3" />Avvist</Badge>;
      default:
        return <Badge variant="secondary">{status}</Badge>;
    }
  };

  const getSourceTypeLabel = (type: string) => {
    switch (type) {
      case "INVOICE_SUPPLIER":
        return "Leverandørfaktura";
      case "INVOICE_CUSTOMER":
        return "Kundefaktura";
      case "RECEIPT":
        return "Kvittering";
      case "BANK_STATEMENT":
        return "Kontoutskrift";
      case "OTHER":
        return "Annet";
      default:
        return type;
    }
  };

  const formatDate = (date: Date | string | null) => {
    if (!date) return "-";
    return new Date(date).toLocaleDateString("nb-NO", {
      year: "numeric",
      month: "long",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const extractedData = document.parsedJson as Record<string, unknown> | null;
  const suggestedEntry = {
    account: document.suggestedAccount,
    vatCode: document.suggestedVatCode,
    description: document.suggestedDescription,
    amount: document.suggestedAmount,
  } as Record<string, unknown> | null;

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="icon" onClick={() => navigate("/accounting")}>
            <ArrowLeft className="h-5 w-5" />
          </Button>
          <div>
            <h1 className="text-2xl font-bold">{document.originalFileName || "Dokument"}</h1>
            <p className="text-muted-foreground">
              {getSourceTypeLabel(document.sourceType)} • Lastet opp {formatDate(document.createdAt)}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {getStatusBadge(document.status)}
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Document Preview */}
        <Card className="lg:row-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Eye className="h-5 w-5" />
              Forhåndsvisning
            </CardTitle>
          </CardHeader>
          <CardContent>
            {document.originalFileUrl ? (
              <div className="aspect-[3/4] bg-muted rounded-lg overflow-hidden">
                {document.originalFileUrl.endsWith(".pdf") ? (
                  <iframe
                    src={document.originalFileUrl}
                    className="w-full h-full"
                    title="Document preview"
                  />
                ) : (
                  <img
                    src={document.originalFileUrl}
                    alt={document.originalFileName || "Document"}
                    className="w-full h-full object-contain"
                  />
                )}
              </div>
            ) : (
              <div className="aspect-[3/4] bg-muted rounded-lg flex items-center justify-center">
                <FileText className="h-16 w-16 text-muted-foreground" />
              </div>
            )}
            <div className="mt-4 flex gap-2">
              {document.originalFileUrl && (
                <Button variant="outline" asChild className="flex-1">
                  <a href={document.originalFileUrl} target="_blank" rel="noopener noreferrer">
                    <Download className="mr-2 h-4 w-4" />
                    Last ned
                  </a>
                </Button>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Document Info */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="h-5 w-5" />
              Dokumentdetaljer
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-muted-foreground">Type</p>
                <p className="font-medium">{getSourceTypeLabel(document.sourceType)}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Status</p>
                <div className="mt-1">{getStatusBadge(document.status)}</div>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Filnavn</p>
                <p className="font-medium truncate">{document.originalFileName || "-"}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Opprettet</p>
                <p className="font-medium">{formatDate(document.createdAt)}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Extracted Data */}
        {extractedData && Object.keys(extractedData).length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Sparkles className="h-5 w-5 text-amber-500" />
                AI-ekstraherte data
              </CardTitle>
              <CardDescription>Automatisk uttrukket fra dokumentet</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {Object.entries(extractedData).map(([key, value]) => (
                  <div key={key} className="flex justify-between items-center py-2 border-b last:border-0">
                    <span className="text-muted-foreground capitalize">
                      {key.replace(/_/g, " ")}
                    </span>
                    <span className="font-medium">
                      {typeof value === "number"
                        ? value.toLocaleString("nb-NO")
                        : String(value)}
                    </span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Suggested Entry */}
        {suggestedEntry && Object.keys(suggestedEntry).length > 0 && (
          <Card className="border-emerald-500/50">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileCheck className="h-5 w-5 text-emerald-500" />
                Foreslått bokføring
              </CardTitle>
              <CardDescription>AI-generert bilagsforslag</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {Object.entries(suggestedEntry).map(([key, value]) => (
                  <div key={key} className="flex justify-between items-center py-2 border-b last:border-0">
                    <span className="text-muted-foreground capitalize">
                      {key.replace(/_/g, " ")}
                    </span>
                    <span className="font-medium">
                      {typeof value === "number"
                        ? value.toLocaleString("nb-NO", { style: "currency", currency: "NOK" })
                        : String(value)}
                    </span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Actions */}
      <Card>
        <CardHeader>
          <CardTitle>Handlinger</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-wrap gap-3">
          {document.status === "NEW" && (
            <Button
              className="bg-emerald-600 hover:bg-emerald-700"
              onClick={() => processMutation.mutate({ id: documentId })}
              disabled={processMutation.isPending}
            >
              <Sparkles className="mr-2 h-4 w-4" />
              {processMutation.isPending ? "Behandler..." : "Behandle med AI"}
            </Button>
          )}

          {document.status === "PROCESSED" && (
            <>
              <Button className="bg-emerald-600 hover:bg-emerald-700">
                <CheckCircle2 className="mr-2 h-4 w-4" />
                Godkjenn og bokfør
              </Button>
              <Button variant="destructive">
                <XCircle className="mr-2 h-4 w-4" />
                Avvis
              </Button>
            </>
          )}

          <Button variant="outline" onClick={() => refetch()}>
            <RefreshCw className="mr-2 h-4 w-4" />
            Oppdater
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}
