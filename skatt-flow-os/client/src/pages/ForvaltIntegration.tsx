import { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { 
  Search, Building2, TrendingUp, TrendingDown, AlertTriangle, 
  CheckCircle, XCircle, Users, Briefcase, FileText, Scale,
  Home, Eye, Bell, RefreshCw, ExternalLink, Shield, Banknote
} from "lucide-react";
import { trpc } from "@/lib/trpc";
import { toast } from "sonner";

export default function ForvaltIntegration() {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedOrgNr, setSelectedOrgNr] = useState<string | null>(null);

  // Market statistics
  const marketStats = trpc.forvalt.getMarketStats.useQuery(undefined, {
    refetchInterval: 300000, // Refresh every 5 minutes
  });

  // Company search
  const searchMutation = trpc.forvalt.searchCompanies.useMutation({
    onError: (error) => toast.error(`Søkefeil: ${error.message}`),
  });

  // Company details
  const companyDetails = trpc.forvalt.getCompanyDetails.useQuery(
    { orgNr: selectedOrgNr! },
    { enabled: !!selectedOrgNr }
  );

  const handleSearch = () => {
    if (searchQuery.trim()) {
      searchMutation.mutate({ query: searchQuery });
    }
  };

  const getRatingColor = (rating: string) => {
    switch (rating) {
      case 'A+':
      case 'A':
        return 'bg-emerald-500';
      case 'B':
        return 'bg-yellow-500';
      case 'C':
        return 'bg-orange-500';
      case 'D':
        return 'bg-red-500';
      default:
        return 'bg-gray-500';
    }
  };

  const getRatingText = (rating: string) => {
    switch (rating) {
      case 'A+':
        return 'Særdeles lav risiko';
      case 'A':
        return 'Meget lav risiko';
      case 'B':
        return 'Moderat risiko';
      case 'C':
        return 'Høy risiko';
      case 'D':
        return 'Meget høy risiko';
      default:
        return 'Ukjent';
    }
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">Forvalt Integrasjon</h1>
            <p className="text-muted-foreground">
              Kredittsjekk, regnskap og bedriftsinformasjon fra Proff Forvalt
            </p>
          </div>
          <Badge variant="outline" className="text-emerald-600 border-emerald-600">
            <CheckCircle className="h-3 w-3 mr-1" />
            Premium Tilkoblet
          </Badge>
        </div>

        {/* Market Statistics */}
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Nyetableringer</CardTitle>
              <TrendingUp className="h-4 w-4 text-emerald-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {marketStats.data?.nyetableringerSisteDogn || 0}
              </div>
              <p className="text-xs text-muted-foreground">
                {marketStats.data?.nyetableringer30Dager || 0} siste 30 dager
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Konkurser</CardTitle>
              <TrendingDown className="h-4 w-4 text-red-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {marketStats.data?.konkurserSisteDogn || 0}
              </div>
              <p className="text-xs text-muted-foreground">
                {marketStats.data?.konkurser30Dager || 0} siste 30 dager
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Betalingsanmerkninger</CardTitle>
              <AlertTriangle className="h-4 w-4 text-yellow-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {(marketStats.data?.betalingsanmerkningerAntall || 0).toLocaleString('nb-NO')}
              </div>
              <p className="text-xs text-muted-foreground">
                {((marketStats.data?.betalingsanmerkningerBelop || 0) / 1000000000).toFixed(1)} mrd NOK
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Aktive Selskaper</CardTitle>
              <Building2 className="h-4 w-4 text-blue-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {((marketStats.data?.aktiveSelskaperTotal || 0) / 1000000).toFixed(1)} mill
              </div>
              <p className="text-xs text-muted-foreground">
                {((marketStats.data?.inaktiveSelskaperTotal || 0) / 1000000).toFixed(1)} mill inaktive
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Search Section */}
        <Card>
          <CardHeader>
            <CardTitle>Søk etter selskap</CardTitle>
            <CardDescription>
              Søk på firmanavn, organisasjonsnummer eller person
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex gap-2">
              <Input
                placeholder="Firmanavn, org.nr. eller person..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                className="flex-1"
              />
              <Button onClick={handleSearch} disabled={searchMutation.isPending}>
                <Search className="h-4 w-4 mr-2" />
                {searchMutation.isPending ? 'Søker...' : 'Søk'}
              </Button>
            </div>

            {/* Search Results */}
            {searchMutation.data && searchMutation.data.length > 0 && (
              <div className="mt-4">
                <h4 className="font-medium mb-2">Søkeresultater</h4>
                <ScrollArea className="h-[200px]">
                  <div className="space-y-2">
                    {searchMutation.data.map((company) => (
                      <div
                        key={company.orgNr}
                        className="flex items-center justify-between p-3 rounded-lg border hover:bg-accent cursor-pointer"
                        onClick={() => setSelectedOrgNr(company.orgNr)}
                      >
                        <div>
                          <p className="font-medium">{company.navn}</p>
                          <p className="text-sm text-muted-foreground">
                            Org.nr: {company.orgNr}
                          </p>
                        </div>
                        <Badge variant={company.status === 'Aktivt' ? 'default' : 'secondary'}>
                          {company.status}
                        </Badge>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Company Details */}
        {selectedOrgNr && companyDetails.data && (
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-2xl">{companyDetails.data.basic.navn}</CardTitle>
                  <CardDescription>
                    Org.nr: {companyDetails.data.basic.orgNr} • {companyDetails.data.basic.organisasjonsform}
                  </CardDescription>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant={companyDetails.data.basic.status === 'Aktivt' ? 'default' : 'destructive'}>
                    {companyDetails.data.basic.status}
                  </Badge>
                  {companyDetails.data.creditRating && (
                    <Badge className={getRatingColor(companyDetails.data.creditRating.rating)}>
                      {companyDetails.data.creditRating.rating}
                    </Badge>
                  )}
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="info">
                <TabsList className="grid w-full grid-cols-6">
                  <TabsTrigger value="info">
                    <Building2 className="h-4 w-4 mr-2" />
                    Info
                  </TabsTrigger>
                  <TabsTrigger value="credit">
                    <Shield className="h-4 w-4 mr-2" />
                    Kreditt
                  </TabsTrigger>
                  <TabsTrigger value="financials">
                    <Banknote className="h-4 w-4 mr-2" />
                    Regnskap
                  </TabsTrigger>
                  <TabsTrigger value="roles">
                    <Users className="h-4 w-4 mr-2" />
                    Roller
                  </TabsTrigger>
                  <TabsTrigger value="owners">
                    <Briefcase className="h-4 w-4 mr-2" />
                    Eiere
                  </TabsTrigger>
                  <TabsTrigger value="legal">
                    <Scale className="h-4 w-4 mr-2" />
                    Juridisk
                  </TabsTrigger>
                </TabsList>

                {/* Basic Info Tab */}
                <TabsContent value="info" className="space-y-4">
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <h4 className="font-medium">Kontaktinformasjon</h4>
                      <div className="text-sm space-y-1">
                        <p><span className="text-muted-foreground">Adresse:</span> {companyDetails.data.basic.forretningsadresse.gate}</p>
                        <p><span className="text-muted-foreground">Poststed:</span> {companyDetails.data.basic.forretningsadresse.postnr} {companyDetails.data.basic.forretningsadresse.poststed}</p>
                        <p><span className="text-muted-foreground">Telefon:</span> {companyDetails.data.basic.telefon || 'Ikke oppgitt'}</p>
                        <p><span className="text-muted-foreground">Nettside:</span> {companyDetails.data.basic.internett || 'Ikke oppgitt'}</p>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <h4 className="font-medium">Selskapsinformasjon</h4>
                      <div className="text-sm space-y-1">
                        <p><span className="text-muted-foreground">Stiftet:</span> {companyDetails.data.basic.stiftelsesdato}</p>
                        <p><span className="text-muted-foreground">Aksjekapital:</span> {companyDetails.data.basic.aksjekapital.toLocaleString('nb-NO')} NOK</p>
                        <p><span className="text-muted-foreground">Ansatte:</span> {companyDetails.data.basic.antallAnsatte}</p>
                        <p><span className="text-muted-foreground">Bransje:</span> {companyDetails.data.basic.naceBransje}</p>
                      </div>
                    </div>
                  </div>
                </TabsContent>

                {/* Credit Rating Tab */}
                <TabsContent value="credit" className="space-y-4">
                  {companyDetails.data.creditRating ? (
                    <div className="space-y-6">
                      <div className="flex items-center gap-6">
                        <div className={`w-24 h-24 rounded-full flex items-center justify-center text-white text-3xl font-bold ${getRatingColor(companyDetails.data.creditRating.rating)}`}>
                          {companyDetails.data.creditRating.rating}
                        </div>
                        <div>
                          <h3 className="text-xl font-bold">{getRatingText(companyDetails.data.creditRating.rating)}</h3>
                          <p className="text-muted-foreground">Score: {companyDetails.data.creditRating.score}/100</p>
                          <p className="text-sm">Konkursrisiko: {companyDetails.data.creditRating.konkursrisiko}%</p>
                          <p className="text-sm">Kredittramme: {companyDetails.data.creditRating.kredittramme.toLocaleString('nb-NO')} NOK</p>
                        </div>
                      </div>

                      <Separator />

                      <div className="grid gap-4 md:grid-cols-4">
                        {Object.entries(companyDetails.data.creditRating.vurderinger).map(([key, value]) => (
                          <div key={key} className="space-y-2">
                            <p className="text-sm font-medium capitalize">{key.replace(/([A-Z])/g, ' $1').trim()}</p>
                            <Progress value={value * 20} className="h-2" />
                            <p className="text-xs text-muted-foreground">{value}/5</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <p className="text-muted-foreground">Ingen kredittinformasjon tilgjengelig</p>
                  )}
                </TabsContent>

                {/* Financials Tab */}
                <TabsContent value="financials">
                  {companyDetails.data.financials.length > 0 ? (
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>År</TableHead>
                          <TableHead className="text-right">Driftsinntekter</TableHead>
                          <TableHead className="text-right">Driftsresultat</TableHead>
                          <TableHead className="text-right">Resultat før skatt</TableHead>
                          <TableHead className="text-right">Sum eiendeler</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {companyDetails.data.financials.map((year) => (
                          <TableRow key={year.regnskapsaar}>
                            <TableCell className="font-medium">{year.regnskapsaar}</TableCell>
                            <TableCell className="text-right">
                              {year.sumDriftsinntekter.toLocaleString('nb-NO')} {year.valutakode}
                            </TableCell>
                            <TableCell className="text-right">
                              {year.driftsresultat.toLocaleString('nb-NO')} {year.valutakode}
                            </TableCell>
                            <TableCell className="text-right">
                              {year.ordinaertResultatForSkatt.toLocaleString('nb-NO')} {year.valutakode}
                            </TableCell>
                            <TableCell className="text-right">
                              {year.sumEiendeler.toLocaleString('nb-NO')} {year.valutakode}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  ) : (
                    <p className="text-muted-foreground">Ingen regnskapsinformasjon tilgjengelig</p>
                  )}
                </TabsContent>

                {/* Roles Tab */}
                <TabsContent value="roles">
                  {companyDetails.data.roles.length > 0 ? (
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Rolle</TableHead>
                          <TableHead>Navn</TableHead>
                          <TableHead>Fra dato</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {companyDetails.data.roles.map((role: { rolle: string; navn: string; fraDate?: string }, idx: number) => (
                          <TableRow key={idx}>
                            <TableCell className="font-medium">{role.rolle}</TableCell>
                            <TableCell>{role.navn}</TableCell>
                            <TableCell>{role.fraDate || '-'}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  ) : (
                    <p className="text-muted-foreground">Ingen rolleinformasjon tilgjengelig</p>
                  )}
                </TabsContent>

                {/* Owners Tab */}
                <TabsContent value="owners">
                  {companyDetails.data.shareholders.length > 0 ? (
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Aksjonær</TableHead>
                          <TableHead className="text-right">Antall aksjer</TableHead>
                          <TableHead className="text-right">Andel</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {companyDetails.data.shareholders.map((sh: { navn: string; antallAksjer: number; andel: number }, idx: number) => (
                          <TableRow key={idx}>
                            <TableCell className="font-medium">{sh.navn}</TableCell>
                            <TableCell className="text-right">{sh.antallAksjer.toLocaleString('nb-NO')}</TableCell>
                            <TableCell className="text-right">{sh.andel.toFixed(2)}%</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  ) : (
                    <p className="text-muted-foreground">Ingen aksjonærinformasjon tilgjengelig</p>
                  )}
                </TabsContent>

                {/* Legal Tab */}
                <TabsContent value="legal" className="space-y-4">
                  <div className="grid gap-4 md:grid-cols-2">
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-base">Betalingsanmerkninger</CardTitle>
                      </CardHeader>
                      <CardContent>
                        {companyDetails.data.paymentRemarks ? (
                          <div className="space-y-2">
                            <Badge variant="destructive">
                              <XCircle className="h-3 w-3 mr-1" />
                              Har anmerkninger
                            </Badge>
                          </div>
                        ) : (
                          <Badge variant="outline" className="text-emerald-600">
                            <CheckCircle className="h-3 w-3 mr-1" />
                            Ingen anmerkninger
                          </Badge>
                        )}
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader>
                        <CardTitle className="text-base">Saker i domstolene</CardTitle>
                      </CardHeader>
                      <CardContent>
                        {companyDetails.data.courtCases.length > 0 ? (
                          <div className="space-y-2">
                            {companyDetails.data.courtCases.slice(0, 3).map((c: { type: string; dato: string; domstol: string }, idx: number) => (
                              <div key={idx} className="text-sm">
                                <p className="font-medium">{c.type}</p>
                                <p className="text-muted-foreground">{c.dato} - {c.domstol}</p>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <Badge variant="outline" className="text-emerald-600">
                            <CheckCircle className="h-3 w-3 mr-1" />
                            Ingen aktive saker
                          </Badge>
                        )}
                      </CardContent>
                    </Card>
                  </div>
                </TabsContent>
              </Tabs>

              {/* Actions */}
              <div className="flex gap-2 mt-6 pt-4 border-t">
                <Button variant="outline" size="sm">
                  <Eye className="h-4 w-4 mr-2" />
                  Legg til overvåking
                </Button>
                <Button variant="outline" size="sm">
                  <FileText className="h-4 w-4 mr-2" />
                  Last ned rapport
                </Button>
                <Button variant="outline" size="sm">
                  <ExternalLink className="h-4 w-4 mr-2" />
                  Åpne i Forvalt
                </Button>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </DashboardLayout>
  );
}
