import { useState } from "react";
import { trpc } from "@/lib/trpc";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
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
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  FolderOpen,
  Plus,
  FileText,
  Download,
  Eye,
  Wand2,
  Copy,
} from "lucide-react";
import { toast } from "sonner";
import { Streamdown } from "streamdown";

export default function Documents() {
  const [activeTab, setActiveTab] = useState("templates");
  const [selectedCompanyId, setSelectedCompanyId] = useState<number | null>(null);

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
            Dokumenter
          </h1>
          <p className="text-slate-500 dark:text-slate-400">
            Maler og AI-genererte dokumenter
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

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="templates">Maler</TabsTrigger>
          <TabsTrigger value="generated">Genererte dokumenter</TabsTrigger>
        </TabsList>

        <TabsContent value="templates" className="mt-6">
          <TemplatesList companyId={selectedCompanyId} />
        </TabsContent>
        <TabsContent value="generated" className="mt-6">
          {selectedCompanyId ? (
            <GeneratedDocsList companyId={selectedCompanyId} />
          ) : (
            <Card className="border-slate-200 dark:border-slate-800">
              <CardContent className="flex flex-col items-center justify-center py-12">
                <FolderOpen className="h-12 w-12 text-slate-300 mb-4" />
                <h3 className="text-lg font-medium text-slate-900 dark:text-white mb-2">
                  Velg et selskap
                </h3>
                <p className="text-slate-500 text-center max-w-sm">
                  Velg et selskap for å se genererte dokumenter.
                </p>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}

function TemplatesList({ companyId }: { companyId: number | null }) {
  const [selectedCategory, setSelectedCategory] = useState<string>("all");
  const [selectedTemplate, setSelectedTemplate] = useState<number | null>(null);
  const [isGenerateOpen, setIsGenerateOpen] = useState(false);

  const { data: templates, isLoading } = trpc.template.list.useQuery({
    category: selectedCategory === "all" ? undefined : selectedCategory as "CONTRACT" | "HR" | "LEGAL" | "FINANCIAL" | "GOVERNANCE" | "OTHER",
  });

  const categoryLabels: Record<string, string> = {
    CONTRACT: "Kontrakter",
    HR: "HR",
    LEGAL: "Juridisk",
    FINANCIAL: "Finans",
    GOVERNANCE: "Styredokumenter",
    OTHER: "Annet",
  };

  return (
    <div className="space-y-4">
      {/* Filter */}
      <div className="flex gap-3">
        <Select value={selectedCategory} onValueChange={setSelectedCategory}>
          <SelectTrigger className="w-[200px]">
            <SelectValue placeholder="Alle kategorier" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">Alle kategorier</SelectItem>
            <SelectItem value="CONTRACT">Kontrakter</SelectItem>
            <SelectItem value="HR">HR</SelectItem>
            <SelectItem value="LEGAL">Juridisk</SelectItem>
            <SelectItem value="FINANCIAL">Finans</SelectItem>
            <SelectItem value="GOVERNANCE">Styredokumenter</SelectItem>
            <SelectItem value="OTHER">Annet</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Templates Grid */}
      {isLoading ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {[1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-40 w-full" />
          ))}
        </div>
      ) : templates && templates.length > 0 ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {templates.map((template) => (
            <Card
              key={template.id}
              className="border-slate-200 dark:border-slate-800 hover:shadow-md transition-shadow cursor-pointer"
              onClick={() => setSelectedTemplate(template.id)}
            >
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-3">
                    <div className="h-10 w-10 rounded-lg bg-slate-100 dark:bg-slate-800 flex items-center justify-center">
                      <FileText className="h-5 w-5 text-slate-500" />
                    </div>
                    <div>
                      <CardTitle className="text-base">{template.name}</CardTitle>
                      <CardDescription>{template.language?.toUpperCase()}</CardDescription>
                    </div>
                  </div>
                  <Badge variant="outline">{categoryLabels[template.category]}</Badge>
                </div>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-slate-500 line-clamp-2">
                  {template.description || "Ingen beskrivelse"}
                </p>
                <div className="flex gap-2 mt-4">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation();
                      setSelectedTemplate(template.id);
                    }}
                  >
                    <Eye className="h-4 w-4 mr-2" />
                    Vis
                  </Button>
                  {companyId && (
                    <Button
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedTemplate(template.id);
                        setIsGenerateOpen(true);
                      }}
                      className="bg-emerald-600 hover:bg-emerald-700 text-white"
                    >
                      <Wand2 className="h-4 w-4 mr-2" />
                      Generer
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <Card className="border-slate-200 dark:border-slate-800">
          <CardContent className="flex flex-col items-center justify-center py-12">
            <FolderOpen className="h-12 w-12 text-slate-300 mb-4" />
            <h3 className="text-lg font-medium text-slate-900 dark:text-white mb-2">
              Ingen maler funnet
            </h3>
            <p className="text-slate-500 text-center max-w-sm">
              Det finnes ingen dokumentmaler i denne kategorien.
            </p>
          </CardContent>
        </Card>
      )}

      {/* View Template Dialog */}
      {selectedTemplate && !isGenerateOpen && (
        <ViewTemplateDialog
          templateId={selectedTemplate}
          open={!!selectedTemplate && !isGenerateOpen}
          onOpenChange={(open) => !open && setSelectedTemplate(null)}
          onGenerate={() => setIsGenerateOpen(true)}
          canGenerate={!!companyId}
        />
      )}

      {/* Generate Document Dialog */}
      {selectedTemplate && isGenerateOpen && companyId && (
        <GenerateDocumentDialog
          templateId={selectedTemplate}
          companyId={companyId}
          open={isGenerateOpen}
          onOpenChange={(open) => {
            setIsGenerateOpen(open);
            if (!open) setSelectedTemplate(null);
          }}
        />
      )}
    </div>
  );
}

function ViewTemplateDialog({
  templateId,
  open,
  onOpenChange,
  onGenerate,
  canGenerate,
}: {
  templateId: number;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onGenerate: () => void;
  canGenerate: boolean;
}) {
  const { data: template } = trpc.template.get.useQuery({ id: templateId });

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl max-h-[80vh]">
        <DialogHeader>
          <DialogTitle>{template?.name}</DialogTitle>
          <DialogDescription>{template?.description}</DialogDescription>
        </DialogHeader>

        {template && (
          <ScrollArea className="h-[50vh]">
            <div className="prose prose-slate dark:prose-invert max-w-none p-4">
              <Streamdown>{template.bodyMarkdown}</Streamdown>
            </div>
          </ScrollArea>
        )}

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Lukk
          </Button>
          {canGenerate && (
            <Button
              onClick={onGenerate}
              className="bg-emerald-600 hover:bg-emerald-700"
            >
              <Wand2 className="h-4 w-4 mr-2" />
              Generer dokument
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function GenerateDocumentDialog({
  templateId,
  companyId,
  open,
  onOpenChange,
}: {
  templateId: number;
  companyId: number;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const [title, setTitle] = useState("");
  const [variables, setVariables] = useState<Record<string, string>>({});
  const [generatedDoc, setGeneratedDoc] = useState<string | null>(null);

  const { data: template } = trpc.template.get.useQuery({ id: templateId });

  const generateMutation = trpc.template.generate.useMutation({
    onSuccess: (result) => {
      setGeneratedDoc(result.outputMarkdown);
      toast.success("Dokument generert");
    },
    onError: (error) => {
      toast.error(`Generering feilet: ${error.message}`);
    },
  });

  const handleGenerate = () => {
    generateMutation.mutate({
      templateId,
      companyId,
      title: title || `${template?.name} - ${new Date().toLocaleDateString("nb-NO")}`,
      variables,
    });
  };

  const copyToClipboard = () => {
    if (generatedDoc) {
      navigator.clipboard.writeText(generatedDoc);
      toast.success("Kopiert til utklippstavle");
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[90vh]">
        <DialogHeader>
          <DialogTitle>
            {generatedDoc ? "Generert dokument" : `Generer: ${template?.name}`}
          </DialogTitle>
          <DialogDescription>
            {generatedDoc
              ? "Dokumentet er generert. Du kan kopiere eller laste ned innholdet."
              : "Fyll inn variablene for å generere dokumentet."}
          </DialogDescription>
        </DialogHeader>

        {!generatedDoc ? (
          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Dokumenttittel</Label>
              <Input
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder={`${template?.name} - ${new Date().toLocaleDateString("nb-NO")}`}
              />
            </div>

            {template?.variablesJson && Object.keys(template.variablesJson as Record<string, string>).length > 0 ? (
              <div className="space-y-3">
                <Label>Variabler</Label>
                {Object.entries(template.variablesJson as Record<string, string>).map(([key, description]) => (
                  <div key={key} className="space-y-1">
                    <Label className="text-sm text-slate-500">{key}</Label>
                    <Input
                      value={variables[key] || ""}
                      onChange={(e) =>
                        setVariables((prev) => ({ ...prev, [key]: e.target.value }))
                      }
                      placeholder={String(description)}
                    />
                  </div>
                ))}
              </div>
            ) : null}

            <DialogFooter>
              <Button variant="outline" onClick={() => onOpenChange(false)}>
                Avbryt
              </Button>
              <Button
                onClick={handleGenerate}
                disabled={generateMutation.isPending}
                className="bg-emerald-600 hover:bg-emerald-700"
              >
                {generateMutation.isPending ? "Genererer..." : "Generer"}
              </Button>
            </DialogFooter>
          </div>
        ) : (
          <div className="space-y-4">
            <ScrollArea className="h-[50vh] border rounded-lg">
              <div className="prose prose-slate dark:prose-invert max-w-none p-4">
                <Streamdown>{generatedDoc}</Streamdown>
              </div>
            </ScrollArea>

            <DialogFooter>
              <Button variant="outline" onClick={copyToClipboard}>
                <Copy className="h-4 w-4 mr-2" />
                Kopier
              </Button>
              <Button
                onClick={() => onOpenChange(false)}
                className="bg-emerald-600 hover:bg-emerald-700"
              >
                Ferdig
              </Button>
            </DialogFooter>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}

function GeneratedDocsList({ companyId }: { companyId: number }) {
  const [selectedDoc, setSelectedDoc] = useState<number | null>(null);

  const { data: documents, isLoading } = trpc.generatedDoc.list.useQuery({ companyId });

  return (
    <div className="space-y-4">
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
              className="border-slate-200 dark:border-slate-800 hover:shadow-sm transition-shadow cursor-pointer"
              onClick={() => setSelectedDoc(doc.id)}
            >
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className="h-10 w-10 rounded-lg bg-slate-100 dark:bg-slate-800 flex items-center justify-center">
                      <FileText className="h-5 w-5 text-slate-500" />
                    </div>
                    <div>
                      <p className="font-medium text-slate-900 dark:text-white">
                        {doc.title}
                      </p>
                      <p className="text-sm text-slate-500">
                        {new Date(doc.createdAt).toLocaleDateString("nb-NO")}
                      </p>
                    </div>
                  </div>

                  <Button variant="outline" size="sm">
                    <Eye className="h-4 w-4 mr-2" />
                    Vis
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <Card className="border-slate-200 dark:border-slate-800">
          <CardContent className="flex flex-col items-center justify-center py-12">
            <FolderOpen className="h-12 w-12 text-slate-300 mb-4" />
            <h3 className="text-lg font-medium text-slate-900 dark:text-white mb-2">
              Ingen genererte dokumenter
            </h3>
            <p className="text-slate-500 text-center max-w-sm">
              Du har ikke generert noen dokumenter for dette selskapet ennå.
            </p>
          </CardContent>
        </Card>
      )}

      {/* View Generated Document Dialog */}
      {selectedDoc && (
        <ViewGeneratedDocDialog
          docId={selectedDoc}
          open={!!selectedDoc}
          onOpenChange={(open) => !open && setSelectedDoc(null)}
        />
      )}
    </div>
  );
}

function ViewGeneratedDocDialog({
  docId,
  open,
  onOpenChange,
}: {
  docId: number;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const { data: doc } = trpc.generatedDoc.get.useQuery({ id: docId });

  const copyToClipboard = () => {
    if (doc?.outputMarkdown) {
      navigator.clipboard.writeText(doc.outputMarkdown);
      toast.success("Kopiert til utklippstavle");
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl max-h-[80vh]">
        <DialogHeader>
          <DialogTitle>{doc?.title}</DialogTitle>
          <DialogDescription>
            Generert {doc && new Date(doc.createdAt).toLocaleString("nb-NO")}
          </DialogDescription>
        </DialogHeader>

        {doc && (
          <ScrollArea className="h-[50vh]">
            <div className="prose prose-slate dark:prose-invert max-w-none p-4">
              <Streamdown>{doc.outputMarkdown}</Streamdown>
            </div>
          </ScrollArea>
        )}

        <DialogFooter>
          <Button variant="outline" onClick={copyToClipboard}>
            <Copy className="h-4 w-4 mr-2" />
            Kopier
          </Button>
          <Button onClick={() => onOpenChange(false)}>Lukk</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
