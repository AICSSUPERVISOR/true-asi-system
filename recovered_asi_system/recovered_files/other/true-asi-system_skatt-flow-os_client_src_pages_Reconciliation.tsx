import { useAuth } from "@/_core/hooks/useAuth";
import { trpc } from "@/lib/trpc";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Progress } from "@/components/ui/progress";
import { toast } from "sonner";
import {
  Landmark,
  ArrowLeftRight,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Upload,
  Download,
  RefreshCw,
  Building2,
  Calendar,
  DollarSign,
  Link2,
  Unlink,
} from "lucide-react";

type BankTransaction = {
  id: number;
  date: Date;
  description: string;
  amount: number;
  matched: boolean;
  matchedEntryId?: number;
};

type LedgerTransaction = {
  id: number;
  date: Date;
  description: string;
  amount: number;
  account: string;
  matched: boolean;
  matchedBankId?: number;
};

export default function Reconciliation() {
  const { user } = useAuth();
  const [selectedCompanyId, setSelectedCompanyId] = useState<number | null>(null);
  const [selectedBankAccount, setSelectedBankAccount] = useState<string | null>(null);
  const [selectedBankTx, setSelectedBankTx] = useState<number[]>([]);
  const [selectedLedgerTx, setSelectedLedgerTx] = useState<number[]>([]);

  const { data: companies } = trpc.company.list.useQuery();

  if (!user) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <p className="text-muted-foreground">Vennligst logg inn for å se denne siden.</p>
      </div>
    );
  }

  // Mock data - in production this would come from tRPC
  const bankTransactions: BankTransaction[] = [
    { id: 1, date: new Date("2024-12-01"), description: "Betaling fra kunde ABC AS", amount: 125000, matched: true, matchedEntryId: 101 },
    { id: 2, date: new Date("2024-12-02"), description: "Strøm november", amount: -4500, matched: true, matchedEntryId: 102 },
    { id: 3, date: new Date("2024-12-03"), description: "Overføring fra konto", amount: 50000, matched: false },
    { id: 4, date: new Date("2024-12-04"), description: "Innkjøp kontorrekvisita", amount: -2340, matched: false },
    { id: 5, date: new Date("2024-12-05"), description: "Husleie desember", amount: -15000, matched: true, matchedEntryId: 103 },
  ];

  const ledgerTransactions: LedgerTransaction[] = [
    { id: 101, date: new Date("2024-12-01"), description: "Faktura 2024-156 ABC AS", amount: 125000, account: "1920", matched: true, matchedBankId: 1 },
    { id: 102, date: new Date("2024-12-02"), description: "Strøm november", amount: -4500, account: "6340", matched: true, matchedBankId: 2 },
    { id: 103, date: new Date("2024-12-05"), description: "Husleie desember", amount: -15000, account: "6300", matched: true, matchedBankId: 5 },
    { id: 104, date: new Date("2024-12-03"), description: "Faktura 2024-157 XYZ AS", amount: 87500, account: "1920", matched: false },
    { id: 105, date: new Date("2024-12-06"), description: "Kontorrekvisita", amount: -2340, account: "6860", matched: false },
  ];

  const matchedCount = bankTransactions.filter(t => t.matched).length;
  const totalCount = bankTransactions.length;
  const matchPercentage = (matchedCount / totalCount) * 100;

  const bankBalance = bankTransactions.reduce((sum, t) => sum + t.amount, 0);
  const ledgerBalance = ledgerTransactions.reduce((sum, t) => sum + t.amount, 0);
  const difference = bankBalance - ledgerBalance;

  const handleMatch = () => {
    if (selectedBankTx.length === 0 || selectedLedgerTx.length === 0) {
      toast.error("Velg minst én transaksjon fra hver side");
      return;
    }
    toast.success("Transaksjoner koblet sammen");
    setSelectedBankTx([]);
    setSelectedLedgerTx([]);
  };

  const formatAmount = (amount: number) => {
    return amount.toLocaleString("nb-NO", { style: "currency", currency: "NOK" });
  };

  const formatDate = (date: Date) => {
    return date.toLocaleDateString("nb-NO", { day: "2-digit", month: "2-digit", year: "numeric" });
  };

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <ArrowLeftRight className="h-8 w-8 text-emerald-500" />
            Bankavstemming
          </h1>
          <p className="text-muted-foreground">Avstem banktransaksjoner mot hovedbok</p>
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
          <Select value={selectedBankAccount || ""} onValueChange={setSelectedBankAccount}>
            <SelectTrigger className="w-[180px]">
              <Landmark className="mr-2 h-4 w-4" />
              <SelectValue placeholder="Velg bankkonto" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="1920">1920 - Driftskonto</SelectItem>
              <SelectItem value="1921">1921 - Skattetrekk</SelectItem>
              <SelectItem value="1922">1922 - MVA-konto</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Stats */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-emerald-500/10">
                <CheckCircle2 className="h-5 w-5 text-emerald-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Avstemt</p>
                <p className="text-2xl font-bold">{matchedCount}/{totalCount}</p>
              </div>
            </div>
            <Progress value={matchPercentage} className="mt-3 h-2" />
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-blue-500/10">
                <Landmark className="h-5 w-5 text-blue-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Banksaldo</p>
                <p className="text-2xl font-bold">{formatAmount(bankBalance)}</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-purple-500/10">
                <DollarSign className="h-5 w-5 text-purple-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Bokført saldo</p>
                <p className="text-2xl font-bold">{formatAmount(ledgerBalance)}</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className={difference !== 0 ? "border-yellow-500/50" : "border-emerald-500/50"}>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className={`p-2 rounded-lg ${difference !== 0 ? "bg-yellow-500/10" : "bg-emerald-500/10"}`}>
                {difference !== 0 ? (
                  <AlertTriangle className="h-5 w-5 text-yellow-500" />
                ) : (
                  <CheckCircle2 className="h-5 w-5 text-emerald-500" />
                )}
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Differanse</p>
                <p className={`text-2xl font-bold ${difference !== 0 ? "text-yellow-500" : "text-emerald-500"}`}>
                  {formatAmount(difference)}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Action Bar */}
      <div className="flex flex-wrap gap-3">
        <Button variant="outline">
          <Upload className="mr-2 h-4 w-4" />
          Importer kontoutskrift
        </Button>
        <Button variant="outline">
          <RefreshCw className="mr-2 h-4 w-4" />
          Hent fra bank
        </Button>
        <Button
          className="bg-emerald-600 hover:bg-emerald-700"
          onClick={handleMatch}
          disabled={selectedBankTx.length === 0 || selectedLedgerTx.length === 0}
        >
          <Link2 className="mr-2 h-4 w-4" />
          Koble valgte ({selectedBankTx.length} + {selectedLedgerTx.length})
        </Button>
        <Button variant="outline" className="ml-auto">
          <Download className="mr-2 h-4 w-4" />
          Eksporter rapport
        </Button>
      </div>

      {/* Two-column reconciliation view */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Bank Transactions */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Landmark className="h-5 w-5 text-blue-500" />
              Banktransaksjoner
            </CardTitle>
            <CardDescription>Transaksjoner fra kontoutskrift</CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[40px]"></TableHead>
                  <TableHead>Dato</TableHead>
                  <TableHead>Beskrivelse</TableHead>
                  <TableHead className="text-right">Beløp</TableHead>
                  <TableHead className="w-[80px]">Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {bankTransactions.map((tx) => (
                  <TableRow key={tx.id} className={tx.matched ? "opacity-60" : ""}>
                    <TableCell>
                      <Checkbox
                        checked={selectedBankTx.includes(tx.id)}
                        onCheckedChange={(checked) => {
                          if (checked) {
                            setSelectedBankTx([...selectedBankTx, tx.id]);
                          } else {
                            setSelectedBankTx(selectedBankTx.filter(id => id !== tx.id));
                          }
                        }}
                        disabled={tx.matched}
                      />
                    </TableCell>
                    <TableCell className="font-mono text-sm">{formatDate(tx.date)}</TableCell>
                    <TableCell>{tx.description}</TableCell>
                    <TableCell className={`text-right font-mono ${tx.amount >= 0 ? "text-green-600" : "text-red-600"}`}>
                      {formatAmount(tx.amount)}
                    </TableCell>
                    <TableCell>
                      {tx.matched ? (
                        <Badge variant="outline" className="bg-emerald-500/10 text-emerald-600">
                          <Link2 className="mr-1 h-3 w-3" />
                          Koblet
                        </Badge>
                      ) : (
                        <Badge variant="secondary">
                          <Unlink className="mr-1 h-3 w-3" />
                          Åpen
                        </Badge>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>

        {/* Ledger Transactions */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <DollarSign className="h-5 w-5 text-purple-500" />
              Hovedbok
            </CardTitle>
            <CardDescription>Posteringer i regnskapet</CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[40px]"></TableHead>
                  <TableHead>Dato</TableHead>
                  <TableHead>Beskrivelse</TableHead>
                  <TableHead className="text-right">Beløp</TableHead>
                  <TableHead className="w-[80px]">Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {ledgerTransactions.map((tx) => (
                  <TableRow key={tx.id} className={tx.matched ? "opacity-60" : ""}>
                    <TableCell>
                      <Checkbox
                        checked={selectedLedgerTx.includes(tx.id)}
                        onCheckedChange={(checked) => {
                          if (checked) {
                            setSelectedLedgerTx([...selectedLedgerTx, tx.id]);
                          } else {
                            setSelectedLedgerTx(selectedLedgerTx.filter(id => id !== tx.id));
                          }
                        }}
                        disabled={tx.matched}
                      />
                    </TableCell>
                    <TableCell className="font-mono text-sm">{formatDate(tx.date)}</TableCell>
                    <TableCell>
                      <div>
                        {tx.description}
                        <span className="text-xs text-muted-foreground ml-2">({tx.account})</span>
                      </div>
                    </TableCell>
                    <TableCell className={`text-right font-mono ${tx.amount >= 0 ? "text-green-600" : "text-red-600"}`}>
                      {formatAmount(tx.amount)}
                    </TableCell>
                    <TableCell>
                      {tx.matched ? (
                        <Badge variant="outline" className="bg-emerald-500/10 text-emerald-600">
                          <Link2 className="mr-1 h-3 w-3" />
                          Koblet
                        </Badge>
                      ) : (
                        <Badge variant="secondary">
                          <Unlink className="mr-1 h-3 w-3" />
                          Åpen
                        </Badge>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
