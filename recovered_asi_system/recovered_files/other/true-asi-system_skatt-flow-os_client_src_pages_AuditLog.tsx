import { useAuth } from "@/_core/hooks/useAuth";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import {
  Shield,
  Search,
  Filter,
  Download,
  User,
  FileText,
  Building2,
  Clock,
  Eye,
  Edit,
  Trash2,
  Plus,
  Send,
  CheckCircle,
  AlertTriangle,
} from "lucide-react";

type AuditLogEntry = {
  id: number;
  timestamp: Date;
  userId: number;
  userName: string;
  action: string;
  entityType: string;
  entityId: number;
  entityName: string;
  details: string;
  ipAddress: string;
};

export default function AuditLog() {
  const { user } = useAuth();
  const [searchQuery, setSearchQuery] = useState("");
  const [actionFilter, setActionFilter] = useState("all");
  const [entityFilter, setEntityFilter] = useState("all");

  if (!user) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <p className="text-muted-foreground">Vennligst logg inn for å se denne siden.</p>
      </div>
    );
  }

  // Mock audit log data - in production this would come from tRPC
  const auditLogs: AuditLogEntry[] = [
    {
      id: 1,
      timestamp: new Date(Date.now() - 1000 * 60 * 5),
      userId: 1,
      userName: "Lucas Bjelland",
      action: "CREATE",
      entityType: "DOCUMENT",
      entityId: 123,
      entityName: "Faktura-2024-001.pdf",
      details: "Lastet opp nytt dokument",
      ipAddress: "192.168.1.1",
    },
    {
      id: 2,
      timestamp: new Date(Date.now() - 1000 * 60 * 30),
      userId: 1,
      userName: "Lucas Bjelland",
      action: "UPDATE",
      entityType: "COMPANY",
      entityId: 1,
      entityName: "Innovatech Global AS",
      details: "Oppdaterte selskapsinformasjon",
      ipAddress: "192.168.1.1",
    },
    {
      id: 3,
      timestamp: new Date(Date.now() - 1000 * 60 * 60),
      userId: 1,
      userName: "Lucas Bjelland",
      action: "SUBMIT",
      entityType: "FILING",
      entityId: 45,
      entityName: "MVA-melding Q4 2024",
      details: "Sendt til Altinn",
      ipAddress: "192.168.1.1",
    },
    {
      id: 4,
      timestamp: new Date(Date.now() - 1000 * 60 * 120),
      userId: 1,
      userName: "Lucas Bjelland",
      action: "APPROVE",
      entityType: "DOCUMENT",
      entityId: 122,
      entityName: "Kvittering-butikk.jpg",
      details: "Godkjent og bokført",
      ipAddress: "192.168.1.1",
    },
    {
      id: 5,
      timestamp: new Date(Date.now() - 1000 * 60 * 180),
      userId: 1,
      userName: "Lucas Bjelland",
      action: "DELETE",
      entityType: "DOCUMENT",
      entityId: 100,
      entityName: "Duplikat-faktura.pdf",
      details: "Slettet duplikat dokument",
      ipAddress: "192.168.1.1",
    },
  ];

  const getActionIcon = (action: string) => {
    switch (action) {
      case "CREATE":
        return <Plus className="h-4 w-4 text-green-500" />;
      case "UPDATE":
        return <Edit className="h-4 w-4 text-blue-500" />;
      case "DELETE":
        return <Trash2 className="h-4 w-4 text-red-500" />;
      case "VIEW":
        return <Eye className="h-4 w-4 text-gray-500" />;
      case "SUBMIT":
        return <Send className="h-4 w-4 text-purple-500" />;
      case "APPROVE":
        return <CheckCircle className="h-4 w-4 text-emerald-500" />;
      default:
        return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
    }
  };

  const getActionBadge = (action: string) => {
    switch (action) {
      case "CREATE":
        return <Badge variant="outline" className="bg-green-500/10 text-green-600">Opprettet</Badge>;
      case "UPDATE":
        return <Badge variant="outline" className="bg-blue-500/10 text-blue-600">Oppdatert</Badge>;
      case "DELETE":
        return <Badge variant="destructive">Slettet</Badge>;
      case "VIEW":
        return <Badge variant="secondary">Vist</Badge>;
      case "SUBMIT":
        return <Badge variant="outline" className="bg-purple-500/10 text-purple-600">Sendt</Badge>;
      case "APPROVE":
        return <Badge variant="outline" className="bg-emerald-500/10 text-emerald-600">Godkjent</Badge>;
      default:
        return <Badge variant="secondary">{action}</Badge>;
    }
  };

  const getEntityIcon = (entityType: string) => {
    switch (entityType) {
      case "DOCUMENT":
        return <FileText className="h-4 w-4" />;
      case "COMPANY":
        return <Building2 className="h-4 w-4" />;
      case "FILING":
        return <Send className="h-4 w-4" />;
      case "USER":
        return <User className="h-4 w-4" />;
      default:
        return <FileText className="h-4 w-4" />;
    }
  };

  const formatTimestamp = (date: Date) => {
    return date.toLocaleString("nb-NO", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  };

  const filteredLogs = auditLogs.filter((log) => {
    if (actionFilter !== "all" && log.action !== actionFilter) return false;
    if (entityFilter !== "all" && log.entityType !== entityFilter) return false;
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      return (
        log.userName.toLowerCase().includes(query) ||
        log.entityName.toLowerCase().includes(query) ||
        log.details.toLowerCase().includes(query)
      );
    }
    return true;
  });

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Shield className="h-8 w-8 text-emerald-500" />
            Revisjonslogg
          </h1>
          <p className="text-muted-foreground">
            Spor alle handlinger i systemet for Bokføringsloven-samsvar
          </p>
        </div>
        <Button variant="outline">
          <Download className="mr-2 h-4 w-4" />
          Eksporter logg
        </Button>
      </div>

      {/* Filters */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-col gap-4 md:flex-row">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Søk i logg..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>
            <Select value={actionFilter} onValueChange={setActionFilter}>
              <SelectTrigger className="w-[150px]">
                <Filter className="mr-2 h-4 w-4" />
                <SelectValue placeholder="Handling" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">Alle handlinger</SelectItem>
                <SelectItem value="CREATE">Opprettet</SelectItem>
                <SelectItem value="UPDATE">Oppdatert</SelectItem>
                <SelectItem value="DELETE">Slettet</SelectItem>
                <SelectItem value="VIEW">Vist</SelectItem>
                <SelectItem value="SUBMIT">Sendt</SelectItem>
                <SelectItem value="APPROVE">Godkjent</SelectItem>
              </SelectContent>
            </Select>
            <Select value={entityFilter} onValueChange={setEntityFilter}>
              <SelectTrigger className="w-[150px]">
                <Filter className="mr-2 h-4 w-4" />
                <SelectValue placeholder="Type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">Alle typer</SelectItem>
                <SelectItem value="DOCUMENT">Dokumenter</SelectItem>
                <SelectItem value="COMPANY">Selskaper</SelectItem>
                <SelectItem value="FILING">Innleveringer</SelectItem>
                <SelectItem value="USER">Brukere</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Log Table */}
      <Card>
        <CardHeader>
          <CardTitle>Aktivitetslogg</CardTitle>
          <CardDescription>
            Viser de siste {filteredLogs.length} hendelsene
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[180px]">Tidspunkt</TableHead>
                <TableHead>Bruker</TableHead>
                <TableHead>Handling</TableHead>
                <TableHead>Type</TableHead>
                <TableHead>Objekt</TableHead>
                <TableHead>Detaljer</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredLogs.map((log) => (
                <TableRow key={log.id}>
                  <TableCell className="font-mono text-sm">
                    <div className="flex items-center gap-2">
                      <Clock className="h-3 w-3 text-muted-foreground" />
                      {formatTimestamp(log.timestamp)}
                    </div>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <User className="h-4 w-4 text-muted-foreground" />
                      {log.userName}
                    </div>
                  </TableCell>
                  <TableCell>{getActionBadge(log.action)}</TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      {getEntityIcon(log.entityType)}
                      <span className="text-muted-foreground">{log.entityType}</span>
                    </div>
                  </TableCell>
                  <TableCell className="font-medium">{log.entityName}</TableCell>
                  <TableCell className="text-muted-foreground">{log.details}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* Compliance Info */}
      <Card className="border-emerald-500/50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="h-5 w-5 text-emerald-500" />
            Bokføringsloven § 13
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">
            I henhold til Bokføringsloven § 13 skal alle regnskapsmessige disposisjoner kunne dokumenteres.
            Denne revisjonsloggen oppfyller kravene til sporbarhet ved å registrere alle handlinger
            med tidspunkt, bruker og detaljer. Loggen oppbevares i minimum 5 år.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
