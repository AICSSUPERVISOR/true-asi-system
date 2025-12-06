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
  Send,
  Download,
  Clock,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  RefreshCw,
  ExternalLink,
} from "lucide-react";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";

export default function FilingDetail() {
  const { user } = useAuth();
  const params = useParams<{ id: string }>();
  const [, navigate] = useLocation();
  const filingId = parseInt(params.id || "0");

  const { data: filing, isLoading, refetch } = trpc.filing.get.useQuery(
    { id: filingId },
    { enabled: !!filingId }
  );

  const submitMutation = trpc.filing.submitToAltinn.useMutation({
    onSuccess: () => {
      toast.success("Innlevering sendt til Altinn");
      refetch();
    },
    onError: (error: { message: string }) => {
      toast.error(`Feil ved innsending: ${error.message}`);
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

  if (!filing) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] gap-4">
        <XCircle className="h-16 w-16 text-muted-foreground" />
        <p className="text-muted-foreground">Innlevering ikke funnet</p>
        <Button variant="outline" onClick={() => navigate("/filings")}>
          <ArrowLeft className="mr-2 h-4 w-4" />
          Tilbake til innleveringer
        </Button>
      </div>
    );
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "DRAFT":
        return <Badge variant="secondary"><Clock className="mr-1 h-3 w-3" />Utkast</Badge>;
      case "PENDING":
        return <Badge variant="outline" className="bg-yellow-500/10 text-yellow-600"><RefreshCw className="mr-1 h-3 w-3" />Venter</Badge>;
      case "SUBMITTED":
        return <Badge variant="outline" className="bg-blue-500/10 text-blue-600"><Send className="mr-1 h-3 w-3" />Sendt</Badge>;
      case "ACCEPTED":
        return <Badge variant="outline" className="bg-green-500/10 text-green-600"><CheckCircle2 className="mr-1 h-3 w-3" />Godkjent</Badge>;
      case "REJECTED":
        return <Badge variant="destructive"><XCircle className="mr-1 h-3 w-3" />Avvist</Badge>;
      default:
        return <Badge variant="secondary">{status}</Badge>;
    }
  };

  const getFilingTypeLabel = (type: string) => {
    switch (type) {
      case "MVA_MELDING":
        return "MVA-melding";
      case "SAFT":
        return "SAF-T Eksport";
      case "A_MELDING":
        return "A-melding";
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
    });
  };

  const summaryData = filing.summaryJson as Record<string, unknown> | null;

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="icon" onClick={() => navigate("/filings")}>
            <ArrowLeft className="h-5 w-5" />
          </Button>
          <div>
            <h1 className="text-2xl font-bold">{getFilingTypeLabel(filing.filingType)}</h1>
            <p className="text-muted-foreground">
              Periode: {formatDate(filing.periodStart)} - {formatDate(filing.periodEnd)}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {getStatusBadge(filing.status)}
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {/* Filing Info */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="h-5 w-5" />
              Innleveringsdetaljer
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-muted-foreground">Type</p>
                <p className="font-medium">{getFilingTypeLabel(filing.filingType)}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Status</p>
                <div className="mt-1">{getStatusBadge(filing.status)}</div>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Periode start</p>
                <p className="font-medium">{formatDate(filing.periodStart)}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Periode slutt</p>
                <p className="font-medium">{formatDate(filing.periodEnd)}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Opprettet</p>
                <p className="font-medium">{formatDate(filing.createdAt)}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Sendt</p>
                <p className="font-medium">{filing.submittedAt ? formatDate(filing.submittedAt) : "-"}</p>
              </div>
            </div>

            {filing.altinnReference && (
              <>
                <Separator />
                <div>
                  <p className="text-sm text-muted-foreground">Altinn referanse</p>
                  <div className="flex items-center gap-2 mt-1">
                    <code className="bg-muted px-2 py-1 rounded text-sm">{filing.altinnReference}</code>
                    <Button variant="ghost" size="sm" asChild>
                      <a
                        href={`https://www.altinn.no/skjemaoversikt/skatteetaten/${filing.altinnReference}`}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        <ExternalLink className="h-4 w-4" />
                      </a>
                    </Button>
                  </div>
                </div>
              </>
            )}
          </CardContent>
        </Card>

        {/* Summary */}
        {summaryData && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Building2 className="h-5 w-5" />
                Sammendrag
              </CardTitle>
              <CardDescription>Beregnet data for perioden</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {Object.entries(summaryData).map(([key, value]) => (
                  <div key={key} className="flex justify-between items-center py-2 border-b last:border-0">
                    <span className="text-muted-foreground capitalize">
                      {key.replace(/_/g, " ")}
                    </span>
                    <span className="font-medium">
                      {typeof value === "number"
                        ? value.toLocaleString("nb-NO", {
                            style: key.includes("amount") || key.includes("vat") ? "currency" : "decimal",
                            currency: "NOK",
                          })
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
          {filing.status === "DRAFT" && (
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button className="bg-emerald-600 hover:bg-emerald-700">
                  <Send className="mr-2 h-4 w-4" />
                  Send til Altinn
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Bekreft innsending</AlertDialogTitle>
                  <AlertDialogDescription>
                    Er du sikker på at du vil sende denne {getFilingTypeLabel(filing.filingType).toLowerCase()} til Altinn?
                    Denne handlingen kan ikke angres.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Avbryt</AlertDialogCancel>
                  <AlertDialogAction
                    onClick={() => submitMutation.mutate({ id: filingId, confirmed: true })}
                    disabled={submitMutation.isPending}
                  >
                    {submitMutation.isPending ? "Sender..." : "Bekreft innsending"}
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          )}

          <Button variant="outline">
            <Download className="mr-2 h-4 w-4" />
            Last ned XML
          </Button>

          <Button variant="outline" onClick={() => refetch()}>
            <RefreshCw className="mr-2 h-4 w-4" />
            Oppdater status
          </Button>
        </CardContent>
      </Card>

      {/* Warnings */}
      {filing.status === "ERROR" && (
        <Card className="border-destructive">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-destructive">
              <AlertTriangle className="h-5 w-5" />
              Avvist av Altinn
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground">
              Denne innleveringen ble avvist. Vennligst sjekk feilmeldingen fra Altinn og korriger dataene før du sender på nytt.
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
