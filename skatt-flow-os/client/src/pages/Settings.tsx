import { useState } from "react";
import { useAuth } from "@/_core/hooks/useAuth";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import {
  User,
  Building2,
  Key,
  Bell,
  Shield,
  Link2,
  CheckCircle,
  AlertTriangle,
} from "lucide-react";
import { toast } from "sonner";

export default function Settings() {
  const { user } = useAuth();
  const [activeTab, setActiveTab] = useState("profile");

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-slate-900 dark:text-white">
          Innstillinger
        </h1>
        <p className="text-slate-500 dark:text-slate-400">
          Administrer konto og systeminnstillinger
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-5 max-w-2xl">
          <TabsTrigger value="profile">Profil</TabsTrigger>
          <TabsTrigger value="integrations">Integrasjoner</TabsTrigger>
          <TabsTrigger value="notifications">Varsler</TabsTrigger>
          <TabsTrigger value="security">Sikkerhet</TabsTrigger>
          <TabsTrigger value="api">API</TabsTrigger>
        </TabsList>

        {/* Profile Tab */}
        <TabsContent value="profile" className="mt-6 space-y-6">
          <Card className="border-slate-200 dark:border-slate-800">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <User className="h-5 w-5" />
                Profilinformasjon
              </CardTitle>
              <CardDescription>
                Din personlige informasjon og kontoinnstillinger
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <Label>Navn</Label>
                  <Input value={user?.name || ""} disabled />
                </div>
                <div className="space-y-2">
                  <Label>E-post</Label>
                  <Input value={user?.email || ""} disabled />
                </div>
              </div>

              <div className="space-y-2">
                <Label>Rolle</Label>
                <div className="flex items-center gap-2">
                  <Badge
                    className={
                      user?.role === "admin"
                        ? "bg-emerald-100 text-emerald-700"
                        : "bg-slate-100 text-slate-700"
                    }
                  >
                    {user?.role === "admin" ? "Administrator" : "Bruker"}
                  </Badge>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Integrations Tab */}
        <TabsContent value="integrations" className="mt-6 space-y-6">
          <Card className="border-slate-200 dark:border-slate-800">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Link2 className="h-5 w-5" />
                Regnskapssystemer
              </CardTitle>
              <CardDescription>
                Koble til eksterne regnskapssystemer for automatisk synkronisering
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <IntegrationItem
                name="Tripletex"
                description="Koble til Tripletex for automatisk import av bilag og hovedbok"
                connected={false}
              />
              <Separator />
              <IntegrationItem
                name="PowerOffice"
                description="Synkroniser med PowerOffice Go for sanntidsdata"
                connected={false}
              />
              <Separator />
              <IntegrationItem
                name="Fiken"
                description="Integrer med Fiken for små og mellomstore bedrifter"
                connected={false}
              />
              <Separator />
              <IntegrationItem
                name="Visma eAccounting"
                description="Koble til Visma for omfattende regnskapsfunksjoner"
                connected={false}
              />
            </CardContent>
          </Card>

          <Card className="border-slate-200 dark:border-slate-800">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Building2 className="h-5 w-5" />
                Offentlige tjenester
              </CardTitle>
              <CardDescription>
                Integrasjoner med norske offentlige tjenester
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <IntegrationItem
                name="Altinn"
                description="Send MVA-meldinger og andre skjemaer direkte til Altinn"
                connected={false}
              />
              <Separator />
              <IntegrationItem
                name="Forvalt / Proff"
                description="Hent selskapsinformasjon, kredittvurdering og risikoanalyse"
                connected={true}
              />
            </CardContent>
          </Card>
        </TabsContent>

        {/* Notifications Tab */}
        <TabsContent value="notifications" className="mt-6 space-y-6">
          <Card className="border-slate-200 dark:border-slate-800">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Bell className="h-5 w-5" />
                Varselinnstillinger
              </CardTitle>
              <CardDescription>
                Velg hvilke varsler du ønsker å motta
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <NotificationSetting
                title="Nye dokumenter"
                description="Varsle når nye bilag lastes opp"
                defaultChecked={true}
              />
              <Separator />
              <NotificationSetting
                title="AI-behandling fullført"
                description="Varsle når AI har behandlet et dokument"
                defaultChecked={true}
              />
              <Separator />
              <NotificationSetting
                title="Innleveringsfrister"
                description="Påminnelser om kommende MVA- og andre frister"
                defaultChecked={true}
              />
              <Separator />
              <NotificationSetting
                title="Forvalt-oppdateringer"
                description="Varsle ved endringer i selskapets kredittvurdering"
                defaultChecked={false}
              />
              <Separator />
              <NotificationSetting
                title="Ukentlig sammendrag"
                description="Motta en ukentlig oppsummering per e-post"
                defaultChecked={false}
              />
            </CardContent>
          </Card>
        </TabsContent>

        {/* Security Tab */}
        <TabsContent value="security" className="mt-6 space-y-6">
          <Card className="border-slate-200 dark:border-slate-800">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Shield className="h-5 w-5" />
                Sikkerhetsinnstillinger
              </CardTitle>
              <CardDescription>
                Administrer sikkerhet og tilgangskontroll
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">To-faktor autentisering</p>
                  <p className="text-sm text-slate-500">
                    Legg til et ekstra sikkerhetslag på kontoen din
                  </p>
                </div>
                <Button variant="outline">Aktiver</Button>
              </div>
              <Separator />
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Aktive økter</p>
                  <p className="text-sm text-slate-500">
                    Se og avslutt aktive pålogginger
                  </p>
                </div>
                <Button variant="outline">Administrer</Button>
              </div>
              <Separator />
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Aktivitetslogg</p>
                  <p className="text-sm text-slate-500">
                    Se historikk over handlinger i systemet
                  </p>
                </div>
                <Button variant="outline">Vis logg</Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* API Tab */}
        <TabsContent value="api" className="mt-6 space-y-6">
          <Card className="border-slate-200 dark:border-slate-800">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Key className="h-5 w-5" />
                API-nøkler
              </CardTitle>
              <CardDescription>
                Administrer API-nøkler for programmatisk tilgang
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="p-4 bg-slate-50 dark:bg-slate-800 rounded-lg">
                <p className="text-sm text-slate-500 mb-4">
                  API-nøkler gir programmatisk tilgang til Skatt-Flow OS. Behandle dem som passord
                  og del dem aldri offentlig.
                </p>
                <Button
                  onClick={() => toast.info("Funksjon kommer snart")}
                  className="bg-emerald-600 hover:bg-emerald-700"
                >
                  Generer ny API-nøkkel
                </Button>
              </div>

              <div className="text-sm text-slate-500">
                <p className="font-medium mb-2">API-dokumentasjon</p>
                <p>
                  Les vår{" "}
                  <a href="#" className="text-emerald-600 hover:underline">
                    API-dokumentasjon
                  </a>{" "}
                  for å lære hvordan du integrerer med Skatt-Flow OS.
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

function IntegrationItem({
  name,
  description,
  connected,
}: {
  name: string;
  description: string;
  connected: boolean;
}) {
  return (
    <div className="flex items-center justify-between">
      <div>
        <div className="flex items-center gap-2">
          <p className="font-medium">{name}</p>
          {connected ? (
            <Badge className="bg-emerald-100 text-emerald-700">
              <CheckCircle className="h-3 w-3 mr-1" />
              Tilkoblet
            </Badge>
          ) : (
            <Badge variant="outline" className="text-slate-500">
              Ikke tilkoblet
            </Badge>
          )}
        </div>
        <p className="text-sm text-slate-500">{description}</p>
      </div>
      <Button variant="outline" onClick={() => toast.info("Funksjon kommer snart")}>
        {connected ? "Konfigurer" : "Koble til"}
      </Button>
    </div>
  );
}

function NotificationSetting({
  title,
  description,
  defaultChecked,
}: {
  title: string;
  description: string;
  defaultChecked: boolean;
}) {
  const [checked, setChecked] = useState(defaultChecked);

  return (
    <div className="flex items-center justify-between">
      <div>
        <p className="font-medium">{title}</p>
        <p className="text-sm text-slate-500">{description}</p>
      </div>
      <Switch
        checked={checked}
        onCheckedChange={(value) => {
          setChecked(value);
          toast.success(`${title} ${value ? "aktivert" : "deaktivert"}`);
        }}
      />
    </div>
  );
}
