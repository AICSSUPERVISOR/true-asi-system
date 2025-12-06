import { useState } from "react";
import { trpc } from "@/lib/trpc";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  FileCheck,
  Plus,
  Send,
  Clock,
  CheckCircle,
  AlertTriangle,
  FileText,
  Download,
  Eye,
} from "lucide-react";
import { toast } from "sonner";

export default function Filings() {
  const [selectedCompanyId, setSelectedCompanyId] = useState<number | null>(null);
  const [activeTab, setActiveTab] = useState("all");
  const [isCreateMVAOpen, setIsCreateMVAOpen] = useState(false);
  const [isCreateSAFTOpen, setIsCreateSAFTOpen] = useState(false);

  const { data: companies } = trpc.company.list.useQuery();

  // Auto-select first company
  if (!selectedCompanyId && companies && companies.length > 0) {
    setSelectedCompanyId(companies[0].id);
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-slate-900 dark:text-white">
            Innleveringer
          </h1>
          <p className="text-slate-500 dark:text-slate-400">
            MVA-meldinger, SAF-T eksport og Altinn-innsendinger
          </p>
        </div>

        <div className="flex items-center gap-3">
          <Select
            value={selectedCompanyId?.toString() || ""}
            onValueChange={(v) => setSelectedCompanyId(Number(v))}
          >
            <SelectTrigger className="w-[200px]">
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
        </div>
      </div>

      {selectedCompanyId ? (
        <>
          {/* Action buttons */}
          <div className="flex gap-3">
            <Button
              onClick={() => setIsCreateMVAOpen(true)}
              className="bg-emerald-600 hover:bg-emerald-700 text-white"
            >
              <Plus className="h-4 w-4 mr-2" />
              Ny MVA-melding
            </Button>
            <Button variant="outline" onClick={() => setIsCreateSAFTOpen(true)}>
              <FileText className="h-4 w-4 mr-2" />
              Generer SAF-T
            </Button>
          </div>

          {/* Tabs */}
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList>
              <TabsTrigger value="all">Alle</TabsTrigger>
              <TabsTrigger value="MVA_MELDING">MVA-meldinger</TabsTrigger>
              <TabsTrigger value="SAF_T">SAF-T</TabsTrigger>
              <TabsTrigger value="A_MELDING_SUMMARY">A-meldinger</TabsTrigger>
            </TabsList>

            <TabsContent value="all" className="mt-6">
              <FilingsList companyId={selectedCompanyId} />
            </TabsContent>
            <TabsContent value="MVA_MELDING" className="mt-6">
              <FilingsList companyId={selectedCompanyId} type="MVA_MELDING" />
            </TabsContent>
            <TabsContent value="SAF_T" className="mt-6">
              <FilingsList companyId={selectedCompanyId} type="SAF_T" />
            </TabsContent>
            <TabsContent value="A_MELDING_SUMMARY" className="mt-6">
              <FilingsList companyId={selectedCompanyId} type="A_MELDING_SUMMARY" />
            </TabsContent>
          </Tabs>

          {/* Dialogs */}
          <CreateMVADialog
            companyId={selectedCompanyId}
            open={isCreateMVAOpen}
            onOpenChange={setIsCreateMVAOpen}
          />
          <CreateSAFTDialog
            companyId={selectedCompanyId}
            open={isCreateSAFTOpen}
            onOpenChange={setIsCreateSAFTOpen}
          />
        </>
      ) : (
        <Card className="border-slate-200 dark:border-slate-800">
          <CardContent className="flex flex-col items-center justify-center py-12">
            <FileCheck className="h-12 w-12 text-slate-300 mb-4" />
            <h3 className="text-lg font-medium text-slate-900 dark:text-white mb-2">
              Velg et selskap
            </h3>
            <p className="text-slate-500 text-center max-w-sm">
              Velg et selskap fra listen over for å se og administrere innleveringer.
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

interface FilingsListProps {
  companyId: number;
  type?: "MVA_MELDING" | "A_MELDING_SUMMARY" | "SAF_T" | "ARSREGNSKAP" | "OTHER";
}

function FilingsList({ companyId, type }: FilingsListProps) {
  const [selectedFiling, setSelectedFiling] = useState<number | null>(null);
  const [confirmSubmit, setConfirmSubmit] = useState<number | null>(null);

  const { data: filings, isLoading, refetch } = trpc.filing.list.useQuery({
    companyId,
    type,
  });

  const submitMutation = trpc.filing.submitToAltinn.useMutation({
    onSuccess: (result) => {
      toast.success(`Innsendt til Altinn. Referanse: ${result.reference}`);
      refetch();
    },
    onError: (error) => {
      toast.error(`Innsending feilet: ${error.message}`);
    },
  });

  const statusConfig = {
    DRAFT: { icon: Clock, label: "Utkast", color: "bg-slate-100 text-slate-700" },
    READY_FOR_REVIEW: { icon: Eye, label: "Klar for gjennomgang", color: "bg-amber-100 text-amber-700" },
    SUBMITTED: { icon: CheckCircle, label: "Innsendt", color: "bg-emerald-100 text-emerald-700" },
    ERROR: { icon: AlertTriangle, label: "Feil", color: "bg-red-100 text-red-700" },
  };

  const typeLabels: Record<string, string> = {
    MVA_MELDING: "MVA-melding",
    A_MELDING_SUMMARY: "A-melding",
    SAF_T: "SAF-T",
    ARSREGNSKAP: "Årsregnskap",
    OTHER: "Annet",
  };

  return (
    <div className="space-y-4">
      {isLoading ? (
        <div className="space-y-3">
          {[1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-24 w-full" />
          ))}
        </div>
      ) : filings && filings.length > 0 ? (
        <div className="space-y-3">
          {filings.map((filing) => {
            const StatusIcon = statusConfig[filing.status].icon;
            return (
              <Card
                key={filing.id}
                className="border-slate-200 dark:border-slate-800 hover:shadow-sm transition-shadow"
              >
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <div className="h-12 w-12 rounded-lg bg-slate-100 dark:bg-slate-800 flex items-center justify-center">
                        <FileCheck className="h-6 w-6 text-slate-500" />
                      </div>
                      <div>
                        <p className="font-medium text-slate-900 dark:text-white">
                          {typeLabels[filing.filingType]}
                        </p>
                        <div className="flex items-center gap-2 text-sm text-slate-500">
                          <span>
                            {new Date(filing.periodStart).toLocaleDateString("nb-NO")} -{" "}
                            {new Date(filing.periodEnd).toLocaleDateString("nb-NO")}
                          </span>
                          {filing.altinnReference && (
                            <>
                              <span>•</span>
                              <span>Ref: {filing.altinnReference}</span>
                            </>
                          )}
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center gap-3">
                      <Badge className={statusConfig[filing.status].color}>
                        <StatusIcon className="h-3 w-3 mr-1" />
                        {statusConfig[filing.status].label}
                      </Badge>

                      <div className="flex gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => setSelectedFiling(filing.id)}
                        >
                          <Eye className="h-4 w-4" />
                        </Button>

                        {filing.status === "DRAFT" || filing.status === "READY_FOR_REVIEW" ? (
                          <Button
                            size="sm"
                            onClick={() => setConfirmSubmit(filing.id)}
                            className="bg-emerald-600 hover:bg-emerald-700 text-white"
                          >
                            <Send className="h-4 w-4 mr-2" />
                            Send til Altinn
                          </Button>
                        ) : null}
                      </div>
                    </div>
                  </div>

                  {filing.errorMessage && (
                    <div className="mt-3 p-3 bg-red-50 dark:bg-red-950/50 rounded-lg">
                      <p className="text-sm text-red-700 dark:text-red-400">
                        {filing.errorMessage}
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>
            );
          })}
        </div>
      ) : (
        <Card className="border-slate-200 dark:border-slate-800">
          <CardContent className="flex flex-col items-center justify-center py-12">
            <FileCheck className="h-12 w-12 text-slate-300 mb-4" />
            <h3 className="text-lg font-medium text-slate-900 dark:text-white mb-2">
              Ingen innleveringer
            </h3>
            <p className="text-slate-500 text-center max-w-sm">
              Det finnes ingen innleveringer for dette selskapet ennå.
            </p>
          </CardContent>
        </Card>
      )}

      {/* View Filing Dialog */}
      {selectedFiling && (
        <ViewFilingDialog
          filingId={selectedFiling}
          open={!!selectedFiling}
          onOpenChange={(open) => !open && setSelectedFiling(null)}
        />
      )}

      {/* Confirm Submit Dialog */}
      <AlertDialog open={!!confirmSubmit} onOpenChange={(open) => !open && setConfirmSubmit(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Bekreft innsending til Altinn</AlertDialogTitle>
            <AlertDialogDescription>
              Er du sikker på at du vil sende inn denne innleveringen til Altinn? Denne handlingen
              kan ikke angres.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Avbryt</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                if (confirmSubmit) {
                  submitMutation.mutate({ id: confirmSubmit, confirmed: true });
                  setConfirmSubmit(null);
                }
              }}
              className="bg-emerald-600 hover:bg-emerald-700"
            >
              {submitMutation.isPending ? "Sender..." : "Bekreft og send"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

function ViewFilingDialog({
  filingId,
  open,
  onOpenChange,
}: {
  filingId: number;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const { data: filing } = trpc.filing.get.useQuery({ id: filingId });

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Innleveringsdetaljer</DialogTitle>
          <DialogDescription>
            {filing?.filingType} for perioden{" "}
            {filing && new Date(filing.periodStart).toLocaleDateString("nb-NO")} -{" "}
            {filing && new Date(filing.periodEnd).toLocaleDateString("nb-NO")}
          </DialogDescription>
        </DialogHeader>

        {filing && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label className="text-slate-500">Status</Label>
                <p className="font-medium">{filing.status}</p>
              </div>
              <div>
                <Label className="text-slate-500">Altinn-referanse</Label>
                <p className="font-medium">{filing.altinnReference || "-"}</p>
              </div>
              <div>
                <Label className="text-slate-500">Opprettet</Label>
                <p className="font-medium">
                  {new Date(filing.createdAt).toLocaleString("nb-NO")}
                </p>
              </div>
              <div>
                <Label className="text-slate-500">Innsendt</Label>
                <p className="font-medium">
                  {filing.submittedAt
                    ? new Date(filing.submittedAt).toLocaleString("nb-NO")
                    : "-"}
                </p>
              </div>
            </div>

            {filing.summaryJson ? (
              <div>
                <Label className="text-slate-500">Sammendrag</Label>
                <div className="mt-2 p-4 bg-slate-50 dark:bg-slate-800 rounded-lg">
                  <pre className="text-sm whitespace-pre-wrap">
                    {JSON.stringify(filing.summaryJson as Record<string, unknown>, null, 2)}
                  </pre>
                </div>
              </div>
            ) : null}
          </div>
        )}

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Lukk
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function CreateMVADialog({
  companyId,
  open,
  onOpenChange,
}: {
  companyId: number;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const [year, setYear] = useState(new Date().getFullYear().toString());
  const [term, setTerm] = useState("1");

  const utils = trpc.useUtils();
  const createMutation = trpc.filing.generateMVADraft.useMutation({
    onSuccess: (result) => {
      toast.success("MVA-melding utkast opprettet");
      utils.filing.list.invalidate();
      onOpenChange(false);
    },
    onError: (error) => {
      toast.error(`Kunne ikke opprette MVA-melding: ${error.message}`);
    },
  });

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Generer MVA-melding</DialogTitle>
          <DialogDescription>
            Velg år og termin for å generere et MVA-melding utkast basert på hovedbokdata.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label>År</Label>
              <Input
                type="number"
                value={year}
                onChange={(e) => setYear(e.target.value)}
                min={2020}
                max={2030}
              />
            </div>
            <div className="space-y-2">
              <Label>Termin</Label>
              <Select value={term} onValueChange={setTerm}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1">1 (jan-feb)</SelectItem>
                  <SelectItem value="2">2 (mar-apr)</SelectItem>
                  <SelectItem value="3">3 (mai-jun)</SelectItem>
                  <SelectItem value="4">4 (jul-aug)</SelectItem>
                  <SelectItem value="5">5 (sep-okt)</SelectItem>
                  <SelectItem value="6">6 (nov-des)</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Avbryt
          </Button>
          <Button
            onClick={() =>
              createMutation.mutate({
                companyId,
                year: parseInt(year),
                term: parseInt(term),
              })
            }
            disabled={createMutation.isPending}
            className="bg-emerald-600 hover:bg-emerald-700"
          >
            {createMutation.isPending ? "Genererer..." : "Generer utkast"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function CreateSAFTDialog({
  companyId,
  open,
  onOpenChange,
}: {
  companyId: number;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const [periodStart, setPeriodStart] = useState("");
  const [periodEnd, setPeriodEnd] = useState("");

  const utils = trpc.useUtils();
  const createMutation = trpc.filing.generateSAFT.useMutation({
    onSuccess: (result) => {
      toast.success("SAF-T eksport generert");
      utils.filing.list.invalidate();
      onOpenChange(false);
    },
    onError: (error) => {
      toast.error(`Kunne ikke generere SAF-T: ${error.message}`);
    },
  });

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Generer SAF-T eksport</DialogTitle>
          <DialogDescription>
            Velg periode for å generere en SAF-T fil fra regnskapssystemet.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label>Fra dato</Label>
              <Input
                type="date"
                value={periodStart}
                onChange={(e) => setPeriodStart(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label>Til dato</Label>
              <Input
                type="date"
                value={periodEnd}
                onChange={(e) => setPeriodEnd(e.target.value)}
              />
            </div>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Avbryt
          </Button>
          <Button
            onClick={() =>
              createMutation.mutate({
                companyId,
                periodStart,
                periodEnd,
              })
            }
            disabled={createMutation.isPending || !periodStart || !periodEnd}
            className="bg-emerald-600 hover:bg-emerald-700"
          >
            {createMutation.isPending ? "Genererer..." : "Generer SAF-T"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
