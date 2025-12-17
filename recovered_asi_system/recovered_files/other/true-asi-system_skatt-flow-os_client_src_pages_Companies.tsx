import { useState } from "react";
import { trpc } from "@/lib/trpc";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Building2, Plus, Search, RefreshCw, ExternalLink } from "lucide-react";
import { Link } from "wouter";
import { toast } from "sonner";

export default function Companies() {
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");

  const { data: companies, isLoading, refetch } = trpc.company.list.useQuery();

  const filteredCompanies = companies?.filter(
    (company) =>
      company.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      company.orgNumber.includes(searchQuery)
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-slate-900 dark:text-white">
            Selskaper
          </h1>
          <p className="text-slate-500 dark:text-slate-400">
            Administrer selskaper og se Forvalt-data
          </p>
        </div>
        <Dialog open={isAddDialogOpen} onOpenChange={setIsAddDialogOpen}>
          <DialogTrigger asChild>
            <Button className="bg-emerald-600 hover:bg-emerald-700 text-white">
              <Plus className="h-4 w-4 mr-2" />
              Legg til selskap
            </Button>
          </DialogTrigger>
          <AddCompanyDialog onClose={() => setIsAddDialogOpen(false)} onSuccess={refetch} />
        </Dialog>
      </div>

      {/* Search */}
      <div className="relative max-w-md">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
        <Input
          placeholder="Søk etter navn eller org.nr..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="pl-10"
        />
      </div>

      {/* Companies Grid */}
      {isLoading ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {[1, 2, 3].map((i) => (
            <Card key={i} className="border-slate-200 dark:border-slate-800">
              <CardHeader>
                <Skeleton className="h-6 w-48" />
                <Skeleton className="h-4 w-32 mt-2" />
              </CardHeader>
              <CardContent>
                <Skeleton className="h-20 w-full" />
              </CardContent>
            </Card>
          ))}
        </div>
      ) : filteredCompanies && filteredCompanies.length > 0 ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {filteredCompanies.map((company) => (
            <CompanyCard key={company.id} company={company} />
          ))}
        </div>
      ) : (
        <Card className="border-slate-200 dark:border-slate-800">
          <CardContent className="flex flex-col items-center justify-center py-12">
            <Building2 className="h-12 w-12 text-slate-300 mb-4" />
            <h3 className="text-lg font-medium text-slate-900 dark:text-white mb-2">
              Ingen selskaper funnet
            </h3>
            <p className="text-slate-500 text-center max-w-sm mb-4">
              {searchQuery
                ? "Ingen selskaper matcher søket ditt."
                : "Du har ikke lagt til noen selskaper ennå. Klikk på knappen over for å legge til ditt første selskap."}
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

interface CompanyCardProps {
  company: {
    id: number;
    name: string;
    orgNumber: string;
    city?: string | null;
    forvaltRating?: string | null;
    forvaltCreditScore?: number | null;
    forvaltRiskClass?: string | null;
    externalRegnskapSystem?: string | null;
  };
}

function CompanyCard({ company }: CompanyCardProps) {
  const utils = trpc.useUtils();
  const refreshMutation = trpc.company.refreshForvaltData.useMutation({
    onSuccess: () => {
      toast.success("Forvalt-data oppdatert");
      utils.company.list.invalidate();
    },
    onError: (error) => {
      toast.error(`Kunne ikke oppdatere: ${error.message}`);
    },
  });

  const riskStyles: Record<string, string> = {
    HIGH: "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-400",
    MEDIUM: "bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-400",
    LOW: "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-400",
  };

  return (
    <Card className="border-slate-200 dark:border-slate-800 hover:shadow-md transition-shadow">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-lg bg-slate-100 dark:bg-slate-800 flex items-center justify-center">
              <Building2 className="h-5 w-5 text-slate-500" />
            </div>
            <div>
              <CardTitle className="text-base">{company.name}</CardTitle>
              <CardDescription>Org.nr: {company.orgNumber}</CardDescription>
            </div>
          </div>
          {company.forvaltRiskClass && (
            <span
              className={`text-xs px-2 py-1 rounded-full ${
                riskStyles[company.forvaltRiskClass] || riskStyles.LOW
              }`}
            >
              {company.forvaltRiskClass === "HIGH"
                ? "Høy"
                : company.forvaltRiskClass === "MEDIUM"
                ? "Middels"
                : "Lav"}
            </span>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <p className="text-slate-500">Rating</p>
            <p className="font-medium text-slate-900 dark:text-white">
              {company.forvaltRating || "-"}
            </p>
          </div>
          <div>
            <p className="text-slate-500">Kredittscore</p>
            <p className="font-medium text-slate-900 dark:text-white">
              {company.forvaltCreditScore ?? "-"}
            </p>
          </div>
          <div>
            <p className="text-slate-500">By</p>
            <p className="font-medium text-slate-900 dark:text-white">
              {company.city || "-"}
            </p>
          </div>
          <div>
            <p className="text-slate-500">System</p>
            <p className="font-medium text-slate-900 dark:text-white">
              {company.externalRegnskapSystem || "-"}
            </p>
          </div>
        </div>

        <div className="flex gap-2 pt-2">
          <Link href={`/companies/${company.id}`} className="flex-1">
            <Button variant="outline" size="sm" className="w-full">
              <ExternalLink className="h-4 w-4 mr-2" />
              Detaljer
            </Button>
          </Link>
          <Button
            variant="outline"
            size="sm"
            onClick={() => refreshMutation.mutate({ id: company.id })}
            disabled={refreshMutation.isPending}
          >
            <RefreshCw className={`h-4 w-4 ${refreshMutation.isPending ? "animate-spin" : ""}`} />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

function AddCompanyDialog({
  onClose,
  onSuccess,
}: {
  onClose: () => void;
  onSuccess: () => void;
}) {
  const [step, setStep] = useState<"lookup" | "confirm">("lookup");
  const [orgNumber, setOrgNumber] = useState("");
  const [lookupData, setLookupData] = useState<{
    company: { name: string; address?: string; city?: string; postalCode?: string };
    credit: { rating?: string; creditScore?: number; riskClass?: string };
  } | null>(null);
  const [regnskapSystem, setRegnskapSystem] = useState<string>("");
  const [regnskapCompanyId, setRegnskapCompanyId] = useState("");

  const lookupMutation = trpc.company.lookupFromForvalt.useMutation({
    onSuccess: (data) => {
      setLookupData(data);
      setStep("confirm");
    },
    onError: (error) => {
      toast.error(`Kunne ikke finne selskap: ${error.message}`);
    },
  });

  const createMutation = trpc.company.create.useMutation({
    onSuccess: () => {
      toast.success("Selskap opprettet");
      onSuccess();
      onClose();
    },
    onError: (error) => {
      toast.error(`Kunne ikke opprette selskap: ${error.message}`);
    },
  });

  const handleLookup = () => {
    if (orgNumber.length < 9) {
      toast.error("Org.nr må være minst 9 siffer");
      return;
    }
    lookupMutation.mutate({ orgNumber });
  };

  const handleCreate = () => {
    if (!lookupData) return;

    createMutation.mutate({
      orgNumber,
      name: lookupData.company.name,
      address: lookupData.company.address,
      city: lookupData.company.city,
      postalCode: lookupData.company.postalCode,
      externalRegnskapSystem: regnskapSystem as "TRIPLETEX" | "POWEROFFICE" | "FIKEN" | "VISMA_EACCOUNTING" | "OTHER" | undefined,
      externalRegnskapCompanyId: regnskapCompanyId || undefined,
    });
  };

  return (
    <DialogContent className="sm:max-w-md">
      <DialogHeader>
        <DialogTitle>
          {step === "lookup" ? "Legg til selskap" : "Bekreft selskapsinformasjon"}
        </DialogTitle>
        <DialogDescription>
          {step === "lookup"
            ? "Skriv inn organisasjonsnummer for å hente selskapsinformasjon fra Forvalt."
            : "Bekreft informasjonen og koble til regnskapssystem."}
        </DialogDescription>
      </DialogHeader>

      {step === "lookup" ? (
        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="orgNumber">Organisasjonsnummer</Label>
            <Input
              id="orgNumber"
              placeholder="123456789"
              value={orgNumber}
              onChange={(e) => setOrgNumber(e.target.value.replace(/\s/g, ""))}
              maxLength={11}
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={onClose}>
              Avbryt
            </Button>
            <Button
              onClick={handleLookup}
              disabled={lookupMutation.isPending || orgNumber.length < 9}
              className="bg-emerald-600 hover:bg-emerald-700"
            >
              {lookupMutation.isPending ? "Søker..." : "Søk"}
            </Button>
          </DialogFooter>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="p-4 bg-slate-50 dark:bg-slate-800 rounded-lg space-y-2">
            <p className="font-medium text-slate-900 dark:text-white">
              {lookupData?.company.name}
            </p>
            <p className="text-sm text-slate-500">Org.nr: {orgNumber}</p>
            {lookupData?.company.city && (
              <p className="text-sm text-slate-500">{lookupData.company.city}</p>
            )}
            {lookupData?.credit.rating && (
              <p className="text-sm text-slate-500">
                Rating: {lookupData.credit.rating} | Score: {lookupData.credit.creditScore}
              </p>
            )}
          </div>

          <div className="space-y-2">
            <Label>Regnskapssystem (valgfritt)</Label>
            <Select value={regnskapSystem} onValueChange={setRegnskapSystem}>
              <SelectTrigger>
                <SelectValue placeholder="Velg system..." />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="TRIPLETEX">Tripletex</SelectItem>
                <SelectItem value="POWEROFFICE">PowerOffice</SelectItem>
                <SelectItem value="FIKEN">Fiken</SelectItem>
                <SelectItem value="VISMA_EACCOUNTING">Visma eAccounting</SelectItem>
                <SelectItem value="OTHER">Annet</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {regnskapSystem && (
            <div className="space-y-2">
              <Label htmlFor="regnskapCompanyId">Selskaps-ID i regnskapssystem</Label>
              <Input
                id="regnskapCompanyId"
                placeholder="F.eks. 12345"
                value={regnskapCompanyId}
                onChange={(e) => setRegnskapCompanyId(e.target.value)}
              />
            </div>
          )}

          <DialogFooter>
            <Button variant="outline" onClick={() => setStep("lookup")}>
              Tilbake
            </Button>
            <Button
              onClick={handleCreate}
              disabled={createMutation.isPending}
              className="bg-emerald-600 hover:bg-emerald-700"
            >
              {createMutation.isPending ? "Oppretter..." : "Opprett selskap"}
            </Button>
          </DialogFooter>
        </div>
      )}
    </DialogContent>
  );
}
