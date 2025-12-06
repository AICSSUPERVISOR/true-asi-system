import { useState, useRef } from "react";
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  FileText,
  Upload,
  CheckCircle,
  XCircle,
  Clock,
  Eye,
  Wand2,
  FileCheck,
} from "lucide-react";
import { toast } from "sonner";

export default function Accounting() {
  const [selectedCompanyId, setSelectedCompanyId] = useState<number | null>(null);
  const [activeTab, setActiveTab] = useState("new");

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
            Regnskapsdokumenter
          </h1>
          <p className="text-slate-500 dark:text-slate-400">
            Last opp, behandle og bokfør bilag
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
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList>
            <TabsTrigger value="new">Nye</TabsTrigger>
            <TabsTrigger value="processed">Behandlet</TabsTrigger>
            <TabsTrigger value="posted">Bokført</TabsTrigger>
            <TabsTrigger value="rejected">Avvist</TabsTrigger>
          </TabsList>

          <TabsContent value="new" className="mt-6">
            <DocumentList companyId={selectedCompanyId} status="NEW" />
          </TabsContent>
          <TabsContent value="processed" className="mt-6">
            <DocumentList companyId={selectedCompanyId} status="PROCESSED" />
          </TabsContent>
          <TabsContent value="posted" className="mt-6">
            <DocumentList companyId={selectedCompanyId} status="POSTED" />
          </TabsContent>
          <TabsContent value="rejected" className="mt-6">
            <DocumentList companyId={selectedCompanyId} status="REJECTED" />
          </TabsContent>
        </Tabs>
      ) : (
        <Card className="border-slate-200 dark:border-slate-800">
          <CardContent className="flex flex-col items-center justify-center py-12">
            <FileText className="h-12 w-12 text-slate-300 mb-4" />
            <h3 className="text-lg font-medium text-slate-900 dark:text-white mb-2">
              Velg et selskap
            </h3>
            <p className="text-slate-500 text-center max-w-sm">
              Velg et selskap fra listen over for å se og administrere regnskapsdokumenter.
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

interface DocumentListProps {
  companyId: number;
  status: "NEW" | "PROCESSED" | "POSTED" | "REJECTED";
}

function DocumentList({ companyId, status }: DocumentListProps) {
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [selectedDoc, setSelectedDoc] = useState<number | null>(null);

  const { data: documents, isLoading, refetch } = trpc.document.list.useQuery({
    companyId,
    status,
  });

  const statusConfig = {
    NEW: { icon: Clock, label: "Ny", color: "bg-blue-100 text-blue-700" },
    PROCESSED: { icon: Wand2, label: "Behandlet", color: "bg-amber-100 text-amber-700" },
    POSTED: { icon: CheckCircle, label: "Bokført", color: "bg-emerald-100 text-emerald-700" },
    REJECTED: { icon: XCircle, label: "Avvist", color: "bg-red-100 text-red-700" },
  };

  return (
    <div className="space-y-4">
      {status === "NEW" && (
        <div className="flex justify-end">
          <Button
            onClick={() => setIsUploadOpen(true)}
            className="bg-emerald-600 hover:bg-emerald-700 text-white"
          >
            <Upload className="h-4 w-4 mr-2" />
            Last opp bilag
          </Button>
        </div>
      )}

      {isLoading ? (
        <div className="space-y-3">
          {[1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-20 w-full" />
          ))}
        </div>
      ) : documents && documents.length > 0 ? (
        <div className="space-y-3">
          {documents.map((doc) => (
            <Card
              key={doc.id}
              className="border-slate-200 dark:border-slate-800 hover:shadow-sm transition-shadow"
            >
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className="h-10 w-10 rounded-lg bg-slate-100 dark:bg-slate-800 flex items-center justify-center">
                      <FileText className="h-5 w-5 text-slate-500" />
                    </div>
                    <div>
                      <p className="font-medium text-slate-900 dark:text-white">
                        {doc.originalFileName || `Dokument #${doc.id}`}
                      </p>
                      <div className="flex items-center gap-2 text-sm text-slate-500">
                        <span>{doc.sourceType.replace("_", " ")}</span>
                        <span>•</span>
                        <span>{new Date(doc.createdAt).toLocaleDateString("nb-NO")}</span>
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-3">
                    <Badge className={statusConfig[doc.status].color}>
                      {statusConfig[doc.status].label}
                    </Badge>

                    <div className="flex gap-2">
                      {doc.originalFileUrl && (
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => window.open(doc.originalFileUrl!, "_blank")}
                        >
                          <Eye className="h-4 w-4" />
                        </Button>
                      )}
                      {status === "NEW" && (
                        <ProcessButton docId={doc.id} onSuccess={refetch} />
                      )}
                      {status === "PROCESSED" && (
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => setSelectedDoc(doc.id)}
                        >
                          <FileCheck className="h-4 w-4 mr-2" />
                          Godkjenn
                        </Button>
                      )}
                    </div>
                  </div>
                </div>

                {doc.suggestedAccount && status === "PROCESSED" && (
                  <div className="mt-3 p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
                    <p className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
                      AI-forslag:
                    </p>
                    <div className="grid grid-cols-4 gap-4 text-sm">
                      <div>
                        <span className="text-slate-500">Konto:</span>{" "}
                        <span className="font-medium">{doc.suggestedAccount}</span>
                      </div>
                      <div>
                        <span className="text-slate-500">MVA:</span>{" "}
                        <span className="font-medium">{doc.suggestedVatCode}</span>
                      </div>
                      <div>
                        <span className="text-slate-500">Beløp:</span>{" "}
                        <span className="font-medium">
                          {doc.suggestedAmount?.toLocaleString("nb-NO")} kr
                        </span>
                      </div>
                      <div className="col-span-1">
                        <span className="text-slate-500">Beskrivelse:</span>{" "}
                        <span className="font-medium">{doc.suggestedDescription}</span>
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <Card className="border-slate-200 dark:border-slate-800">
          <CardContent className="flex flex-col items-center justify-center py-12">
            <FileText className="h-12 w-12 text-slate-300 mb-4" />
            <h3 className="text-lg font-medium text-slate-900 dark:text-white mb-2">
              Ingen dokumenter
            </h3>
            <p className="text-slate-500 text-center max-w-sm">
              {status === "NEW"
                ? "Ingen nye dokumenter å behandle. Last opp et bilag for å komme i gang."
                : `Ingen dokumenter med status "${statusConfig[status].label}".`}
            </p>
          </CardContent>
        </Card>
      )}

      <UploadDialog
        companyId={companyId}
        open={isUploadOpen}
        onOpenChange={setIsUploadOpen}
        onSuccess={refetch}
      />

      {selectedDoc && (
        <ApproveDialog
          docId={selectedDoc}
          open={!!selectedDoc}
          onOpenChange={(open) => !open && setSelectedDoc(null)}
          onSuccess={refetch}
        />
      )}
    </div>
  );
}

function ProcessButton({ docId, onSuccess }: { docId: number; onSuccess: () => void }) {
  const processMutation = trpc.document.process.useMutation({
    onSuccess: () => {
      toast.success("Dokument behandlet med AI");
      onSuccess();
    },
    onError: (error) => {
      toast.error(`Behandling feilet: ${error.message}`);
    },
  });

  return (
    <Button
      variant="outline"
      size="sm"
      onClick={() => processMutation.mutate({ id: docId })}
      disabled={processMutation.isPending}
    >
      <Wand2 className={`h-4 w-4 mr-2 ${processMutation.isPending ? "animate-pulse" : ""}`} />
      {processMutation.isPending ? "Behandler..." : "Behandle"}
    </Button>
  );
}

interface UploadDialogProps {
  companyId: number;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSuccess: () => void;
}

function UploadDialog({ companyId, open, onOpenChange, onSuccess }: UploadDialogProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [sourceType, setSourceType] = useState<string>("INVOICE_SUPPLIER");

  const uploadMutation = trpc.document.upload.useMutation({
    onSuccess: () => {
      toast.success("Dokument lastet opp");
      setSelectedFile(null);
      onSuccess();
      onOpenChange(false);
    },
    onError: (error) => {
      toast.error(`Opplasting feilet: ${error.message}`);
    },
  });

  const handleUpload = async () => {
    if (!selectedFile) return;

    const reader = new FileReader();
    reader.onload = () => {
      const base64 = (reader.result as string).split(",")[1];
      uploadMutation.mutate({
        companyId,
        sourceType: sourceType as "INVOICE_SUPPLIER" | "INVOICE_CUSTOMER" | "RECEIPT" | "CONTRACT" | "OTHER",
        fileName: selectedFile.name,
        fileContent: base64,
        contentType: selectedFile.type,
      });
    };
    reader.readAsDataURL(selectedFile);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Last opp bilag</DialogTitle>
          <DialogDescription>
            Last opp en faktura, kvittering eller annet regnskapsbilag.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div className="space-y-2">
            <Label>Dokumenttype</Label>
            <Select value={sourceType} onValueChange={setSourceType}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="INVOICE_SUPPLIER">Leverandørfaktura</SelectItem>
                <SelectItem value="INVOICE_CUSTOMER">Kundefaktura</SelectItem>
                <SelectItem value="RECEIPT">Kvittering</SelectItem>
                <SelectItem value="CONTRACT">Kontrakt</SelectItem>
                <SelectItem value="OTHER">Annet</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label>Fil</Label>
            <div
              className="border-2 border-dashed border-slate-200 dark:border-slate-700 rounded-lg p-6 text-center cursor-pointer hover:border-emerald-500 transition-colors"
              onClick={() => fileInputRef.current?.click()}
            >
              {selectedFile ? (
                <div className="flex items-center justify-center gap-2">
                  <FileText className="h-5 w-5 text-emerald-600" />
                  <span className="text-slate-900 dark:text-white">{selectedFile.name}</span>
                </div>
              ) : (
                <div>
                  <Upload className="h-8 w-8 text-slate-400 mx-auto mb-2" />
                  <p className="text-slate-500">Klikk for å velge fil</p>
                  <p className="text-xs text-slate-400 mt-1">PDF, PNG, JPG opp til 10MB</p>
                </div>
              )}
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.png,.jpg,.jpeg"
              className="hidden"
              onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
            />
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Avbryt
          </Button>
          <Button
            onClick={handleUpload}
            disabled={!selectedFile || uploadMutation.isPending}
            className="bg-emerald-600 hover:bg-emerald-700"
          >
            {uploadMutation.isPending ? "Laster opp..." : "Last opp"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

interface ApproveDialogProps {
  docId: number;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSuccess: () => void;
}

function ApproveDialog({ docId, open, onOpenChange, onSuccess }: ApproveDialogProps) {
  const { data: doc } = trpc.document.get.useQuery({ id: docId });

  const [account, setAccount] = useState("");
  const [vatCode, setVatCode] = useState("");
  const [description, setDescription] = useState("");
  const [amount, setAmount] = useState("");

  // Populate with suggestions when doc loads
  if (doc && !account && doc.suggestedAccount) {
    setAccount(doc.suggestedAccount);
    setVatCode(doc.suggestedVatCode || "");
    setDescription(doc.suggestedDescription || "");
    setAmount(doc.suggestedAmount?.toString() || "");
  }

  const approveMutation = trpc.document.approve.useMutation({
    onSuccess: () => {
      toast.success("Dokument godkjent og bokført");
      onSuccess();
      onOpenChange(false);
    },
    onError: (error) => {
      toast.error(`Godkjenning feilet: ${error.message}`);
    },
  });

  const rejectMutation = trpc.document.reject.useMutation({
    onSuccess: () => {
      toast.success("Dokument avvist");
      onSuccess();
      onOpenChange(false);
    },
    onError: (error) => {
      toast.error(`Avvisning feilet: ${error.message}`);
    },
  });

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Godkjenn bokføring</DialogTitle>
          <DialogDescription>
            Bekreft eller juster AI-forslaget før bokføring.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="account">Konto</Label>
              <Input
                id="account"
                value={account}
                onChange={(e) => setAccount(e.target.value)}
                placeholder="F.eks. 6000"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="vatCode">MVA-kode</Label>
              <Input
                id="vatCode"
                value={vatCode}
                onChange={(e) => setVatCode(e.target.value)}
                placeholder="F.eks. 1"
              />
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="amount">Beløp</Label>
            <Input
              id="amount"
              type="number"
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
              placeholder="0.00"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="description">Beskrivelse</Label>
            <Input
              id="description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Beskrivelse av transaksjonen"
            />
          </div>
        </div>

        <DialogFooter className="flex justify-between">
          <Button
            variant="destructive"
            onClick={() =>
              rejectMutation.mutate({ id: docId, reason: "Manuelt avvist" })
            }
            disabled={rejectMutation.isPending}
          >
            Avvis
          </Button>
          <div className="flex gap-2">
            <Button variant="outline" onClick={() => onOpenChange(false)}>
              Avbryt
            </Button>
            <Button
              onClick={() =>
                approveMutation.mutate({
                  id: docId,
                  account,
                  vatCode,
                  description,
                  amount: parseFloat(amount) || 0,
                })
              }
              disabled={approveMutation.isPending || !account || !amount}
              className="bg-emerald-600 hover:bg-emerald-700"
            >
              {approveMutation.isPending ? "Bokfører..." : "Godkjenn og bokfør"}
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
