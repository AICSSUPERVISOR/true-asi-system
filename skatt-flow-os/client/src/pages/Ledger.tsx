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
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Skeleton } from "@/components/ui/skeleton";
import { BookOpen, Download, Filter, Search } from "lucide-react";

export default function Ledger() {
  const [selectedCompanyId, setSelectedCompanyId] = useState<number | null>(null);
  const [periodStart, setPeriodStart] = useState("");
  const [periodEnd, setPeriodEnd] = useState("");
  const [searchQuery, setSearchQuery] = useState("");

  const { data: companies } = trpc.company.list.useQuery();

  // Auto-select first company
  if (!selectedCompanyId && companies && companies.length > 0) {
    setSelectedCompanyId(companies[0].id);
  }

  const { data: entries, isLoading } = trpc.ledger.list.useQuery(
    {
      companyId: selectedCompanyId!,
      periodStart: periodStart || undefined,
      periodEnd: periodEnd || undefined,
    },
    { enabled: !!selectedCompanyId }
  );

  const filteredEntries = entries?.filter(
    (entry) =>
      entry.description?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      entry.debitAccount?.includes(searchQuery) ||
      entry.creditAccount?.includes(searchQuery) ||
      entry.voucherNumber?.includes(searchQuery)
  );

  // Calculate totals
  const totals = filteredEntries?.reduce(
    (acc, entry) => ({
      debit: acc.debit + (entry.amount || 0),
      credit: acc.credit,
    }),
    { debit: 0, credit: 0 }
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-slate-900 dark:text-white">
            Hovedbok
          </h1>
          <p className="text-slate-500 dark:text-slate-400">
            Se og filtrer hovedbokposter
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
          {/* Filters */}
          <Card className="border-slate-200 dark:border-slate-800">
            <CardHeader className="pb-3">
              <CardTitle className="text-base flex items-center gap-2">
                <Filter className="h-4 w-4" />
                Filtre
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-4">
                <div className="space-y-2">
                  <Label>Søk</Label>
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
                    <Input
                      placeholder="Søk i beskrivelse, konto..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="pl-10"
                    />
                  </div>
                </div>
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
                <div className="space-y-2">
                  <Label>&nbsp;</Label>
                  <Button variant="outline" className="w-full">
                    <Download className="h-4 w-4 mr-2" />
                    Eksporter
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Ledger Table */}
          <Card className="border-slate-200 dark:border-slate-800">
            <CardHeader>
              <CardTitle className="text-lg">Hovedbokposter</CardTitle>
              <CardDescription>
                {filteredEntries?.length || 0} poster funnet
              </CardDescription>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <div className="space-y-3">
                  {[1, 2, 3, 4, 5].map((i) => (
                    <Skeleton key={i} className="h-12 w-full" />
                  ))}
                </div>
              ) : filteredEntries && filteredEntries.length > 0 ? (
                <div className="rounded-lg border border-slate-200 dark:border-slate-800 overflow-hidden">
                  <Table>
                    <TableHeader>
                      <TableRow className="bg-slate-50 dark:bg-slate-800/50">
                        <TableHead className="w-[100px]">Dato</TableHead>
                        <TableHead className="w-[100px]">Bilag</TableHead>
                        <TableHead>Beskrivelse</TableHead>
                        <TableHead className="w-[100px]">Debet</TableHead>
                        <TableHead className="w-[100px]">Kredit</TableHead>
                        <TableHead className="w-[120px] text-right">Beløp</TableHead>
                        <TableHead className="w-[80px]">MVA</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {filteredEntries.map((entry) => (
                        <TableRow key={entry.id}>
                          <TableCell className="font-mono text-sm">
                            {new Date(entry.entryDate).toLocaleDateString("nb-NO")}
                          </TableCell>
                          <TableCell className="font-mono text-sm">
                            {entry.voucherNumber || "-"}
                          </TableCell>
                          <TableCell>{entry.description || "-"}</TableCell>
                          <TableCell className="font-mono text-sm">
                            {entry.debitAccount || "-"}
                          </TableCell>
                          <TableCell className="font-mono text-sm">
                            {entry.creditAccount || "-"}
                          </TableCell>
                          <TableCell className="text-right font-mono">
                            {(entry.amount / 100).toLocaleString("nb-NO", {
                              minimumFractionDigits: 2,
                            })}
                          </TableCell>
                          <TableCell className="font-mono text-sm">
                            {entry.vatCode || "-"}
                          </TableCell>
                        </TableRow>
                      ))}
                      {/* Totals row */}
                      <TableRow className="bg-slate-50 dark:bg-slate-800/50 font-medium">
                        <TableCell colSpan={5} className="text-right">
                          Sum:
                        </TableCell>
                        <TableCell className="text-right font-mono">
                          {((totals?.debit || 0) / 100).toLocaleString("nb-NO", {
                            minimumFractionDigits: 2,
                          })}
                        </TableCell>
                        <TableCell />
                      </TableRow>
                    </TableBody>
                  </Table>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-12">
                  <BookOpen className="h-12 w-12 text-slate-300 mb-4" />
                  <h3 className="text-lg font-medium text-slate-900 dark:text-white mb-2">
                    Ingen poster funnet
                  </h3>
                  <p className="text-slate-500 text-center max-w-sm">
                    Det finnes ingen hovedbokposter for valgt periode og filter.
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </>
      ) : (
        <Card className="border-slate-200 dark:border-slate-800">
          <CardContent className="flex flex-col items-center justify-center py-12">
            <BookOpen className="h-12 w-12 text-slate-300 mb-4" />
            <h3 className="text-lg font-medium text-slate-900 dark:text-white mb-2">
              Velg et selskap
            </h3>
            <p className="text-slate-500 text-center max-w-sm">
              Velg et selskap fra listen over for å se hovedboken.
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
