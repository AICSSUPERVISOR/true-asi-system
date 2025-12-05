import { useState } from "react";
import { trpc } from "../lib/trpc";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import { Badge } from "../components/ui/badge";
import { Loader2, Building2, Users, MapPin, Calendar, TrendingUp, AlertCircle, Shield, ExternalLink, DollarSign, Activity } from "lucide-react";
import { toast } from "sonner";
import { useLocation } from "wouter";

export default function CompanyLookup() {
  const [orgnr, setOrgnr] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const [companyData, setCompanyData] = useState<any>(null);
  const [rolesData, setRolesData] = useState<any>(null);
  const [forvaltData, setForvaltData] = useState<any>(null);
  const [isFetchingForvalt, setIsFetchingForvalt] = useState(false);
  const [, setLocation] = useLocation();

  const forvaltQuery = trpc.forvalt.getCreditRating.useQuery(
    { orgNumber: orgnr },
    { enabled: false } // Manual trigger
  );

  const saveCompanyMutation = trpc.brreg.saveCompany.useMutation();
  const saveRolesMutation = trpc.brreg.saveCompanyRoles.useMutation();

  const handleSearch = async () => {
    if (orgnr.length !== 9) {
      toast.error("Organization number must be exactly 9 digits");
      return;
    }

    setIsSearching(true);
    setCompanyData(null);
    setRolesData(null);

    try {
      // Fetch company data from Brreg.no
      const response = await fetch(
        `https://data.brreg.no/enhetsregisteret/api/enheter/${orgnr}`,
        {
          headers: { Accept: "application/json" },
        }
      );

      if (!response.ok) {
        if (response.status === 404) {
          toast.error(`Company with organization number ${orgnr} not found`);
        } else {
          toast.error("Failed to fetch company data");
        }
        setIsSearching(false);
        return;
      }

      const data = await response.json();
      setCompanyData(data);

      // Fetch Forvalt.no credit rating in background
      setIsFetchingForvalt(true);
      try {
        const forvaltResult = await forvaltQuery.refetch();
        if (forvaltResult.data?.success) {
          setForvaltData(forvaltResult.data);
          toast.success("Credit rating loaded from Forvalt.no");
        }
      } catch (error) {
        console.error("Error fetching Forvalt data:", error);
        // Don't show error toast, just log it
      } finally {
        setIsFetchingForvalt(false);
      }

      // Fetch company roles
      const rolesResponse = await fetch(
        `https://data.brreg.no/enhetsregisteret/api/enheter/${orgnr}/roller`,
        {
          headers: { Accept: "application/json" },
        }
      );

      if (rolesResponse.ok) {
        const rolesData = await rolesResponse.json();
        setRolesData(rolesData);
      }

      toast.success(`Found company: ${data.navn}`);
    } catch (error) {
      console.error("Error fetching company:", error);
      toast.error("Failed to fetch company data");
    } finally {
      setIsSearching(false);
    }
  };

  const handleSaveAndAnalyze = async () => {
    if (!companyData) return;

    try {
      // Save company to database
      const saveResult = await saveCompanyMutation.mutateAsync({
        orgnr,
        brregData: companyData,
      });

      // Save roles if available
      if (rolesData && saveResult.companyId) {
        await saveRolesMutation.mutateAsync({
          companyId: saveResult.companyId,
          rolesData,
        });
      }

      toast.success("Company saved! Starting AI analysis...");

      // Navigate to AI recommendations page with real analysis
      setLocation(`/recommendations-ai/${saveResult.companyId}?orgnr=${orgnr}`);
      
      // TODO: Call businessOrchestrator.runCompleteAnalysis in RecommendationsPage
      // This will trigger the 5-step analysis: Brreg → Proff → Website → LinkedIn → AI
    } catch (error) {
      console.error("Error saving company:", error);
      toast.error("Failed to save company");
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-950 p-8">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* Header */}
        <div className="text-center space-y-4">
          <h1 className="text-6xl font-black tracking-tight text-white">
            Company Lookup
          </h1>
          <p className="text-xl text-slate-300 tracking-wider">
            Enter Norwegian organization number to fetch company data and start AI analysis
          </p>
        </div>

        {/* Search Card */}
        <Card className="bg-white/5 backdrop-blur-xl border-white/10 shadow-2xl hover:shadow-blue-500/20 transition-all duration-300">
          <CardHeader>
            <CardTitle className="text-2xl font-bold text-white">Organization Number Search</CardTitle>
            <CardDescription className="text-slate-300">
              Enter a 9-digit Norwegian organization number (e.g., 974760673)
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex gap-4">
              <Input
                type="text"
                placeholder="974760673"
                value={orgnr}
                onChange={(e) => setOrgnr(e.target.value.replace(/\D/g, "").slice(0, 9))}
                onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                className="flex-1 bg-white/10 border-white/20 text-white placeholder:text-slate-400 text-lg"
                maxLength={9}
              />
              <Button
                onClick={handleSearch}
                disabled={isSearching || orgnr.length !== 9}
                className="bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 px-8"
              >
                {isSearching ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Searching...
                  </>
                ) : (
                  "Search"
                )}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Company Data Display */}
        {companyData && (
          <div className="space-y-6">
            {/* Company Overview */}
            <Card className="bg-white/5 backdrop-blur-xl border-white/10 shadow-2xl">
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="space-y-2">
                    <CardTitle className="text-3xl font-black text-white">
                      {companyData.navn}
                    </CardTitle>
                    <div className="flex gap-2 flex-wrap">
                      <Badge className="bg-blue-500/20 text-blue-300 border-blue-500/30">
                        {companyData.organisasjonsform?.beskrivelse || "Unknown"}
                      </Badge>
                      <Badge className="bg-green-500/20 text-green-300 border-green-500/30">
                        Org.nr: {companyData.organisasjonsnummer}
                      </Badge>
                      {companyData.konkurs && (
                        <Badge className="bg-red-500/20 text-red-300 border-red-500/30">
                          <AlertCircle className="w-3 h-3 mr-1" />
                          Bankrupt
                        </Badge>
                      )}
                      {companyData.underAvvikling && (
                        <Badge className="bg-yellow-500/20 text-yellow-300 border-yellow-500/30">
                          <AlertCircle className="w-3 h-3 mr-1" />
                          Under Liquidation
                        </Badge>
                      )}
                    </div>
                  </div>
                  <div className="flex gap-3">
                    {forvaltData && forvaltData.creditRating && (
                      <a
                        href={`https://forvalt.no/ForetaksIndex/Firma/FirmaSide/${orgnr}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-sm text-slate-400 hover:text-white transition-colors flex items-center gap-1"
                      >
                        View Full Forvalt Report
                        <ExternalLink className="w-3 h-3" />
                      </a>
                    )}
                    <Button
                      onClick={handleSaveAndAnalyze}
                      className="bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700"
                    >
                      Save & Analyze with AI
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Forvalt Credit Rating Section */}
                {isFetchingForvalt && (
                  <div className="p-6 bg-gradient-to-r from-blue-500/10 to-cyan-500/10 rounded-xl border border-blue-500/20">
                    <div className="flex items-center gap-3">
                      <Loader2 className="w-5 h-5 text-blue-400 animate-spin" />
                      <span className="text-white font-medium">Loading credit rating from Forvalt.no...</span>
                    </div>
                  </div>
                )}

                {forvaltData && forvaltData.creditRating && (
                  <div className="p-6 bg-gradient-to-r from-blue-500/10 to-cyan-500/10 rounded-xl border border-blue-500/20 space-y-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <Shield className="w-6 h-6 text-blue-400" />
                        <h3 className="text-xl font-bold text-white">Credit Rating & Financial Health</h3>
                      </div>
                      <Badge
                        className={
                          forvaltData.riskLevel === 'very_low'
                            ? 'bg-green-500/20 text-green-300 border-green-500/30 text-lg px-4 py-1'
                            : forvaltData.riskLevel === 'low'
                            ? 'bg-blue-500/20 text-blue-300 border-blue-500/30 text-lg px-4 py-1'
                            : forvaltData.riskLevel === 'moderate'
                            ? 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30 text-lg px-4 py-1'
                            : forvaltData.riskLevel === 'high'
                            ? 'bg-orange-500/20 text-orange-300 border-orange-500/30 text-lg px-4 py-1'
                            : 'bg-red-500/20 text-red-300 border-red-500/30 text-lg px-4 py-1'
                        }
                      >
                        {forvaltData.creditRating}
                      </Badge>
                    </div>

                    <div className="grid md:grid-cols-4 gap-4">
                      {/* Credit Score */}
                      <div className="p-4 bg-white/5 backdrop-blur-xl rounded-lg border border-white/10">
                        <div className="flex items-center gap-2 mb-2">
                          <Activity className="w-4 h-4 text-green-400" />
                          <span className="text-sm text-slate-400">Credit Score</span>
                        </div>
                        <div className="text-3xl font-black text-white">
                          {forvaltData.creditScore || 'N/A'}
                          <span className="text-lg text-slate-400">/100</span>
                        </div>
                      </div>

                      {/* Bankruptcy Probability */}
                      <div className="p-4 bg-white/5 backdrop-blur-xl rounded-lg border border-white/10">
                        <div className="flex items-center gap-2 mb-2">
                          <AlertCircle className="w-4 h-4 text-yellow-400" />
                          <span className="text-sm text-slate-400">Bankruptcy Risk</span>
                        </div>
                        <div className="text-3xl font-black text-white">
                          {forvaltData.bankruptcyProbability !== null
                            ? `${forvaltData.bankruptcyProbability.toFixed(2)}%`
                            : 'N/A'}
                        </div>
                      </div>

                      {/* Credit Limit */}
                      <div className="p-4 bg-white/5 backdrop-blur-xl rounded-lg border border-white/10">
                        <div className="flex items-center gap-2 mb-2">
                          <DollarSign className="w-4 h-4 text-cyan-400" />
                          <span className="text-sm text-slate-400">Credit Limit</span>
                        </div>
                        <div className="text-2xl font-black text-white">
                          {forvaltData.creditLimit
                            ? `${(forvaltData.creditLimit / 1000000).toFixed(1)}M`
                            : 'N/A'}
                          <span className="text-sm text-slate-400"> NOK</span>
                        </div>
                      </div>

                      {/* Risk Level */}
                      <div className="p-4 bg-white/5 backdrop-blur-xl rounded-lg border border-white/10">
                        <div className="flex items-center gap-2 mb-2">
                          <Shield className="w-4 h-4 text-purple-400" />
                          <span className="text-sm text-slate-400">Risk Level</span>
                        </div>
                        <div className="text-lg font-bold text-white capitalize">
                          {forvaltData.riskDescription || forvaltData.riskLevel?.replace('_', ' ') || 'N/A'}
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Company Details Grid */}
                <div className="grid md:grid-cols-2 gap-6">
                {/* Industry */}
                <div className="flex items-start gap-3">
                  <Building2 className="w-5 h-5 text-blue-400 mt-1" />
                  <div>
                    <div className="text-sm text-slate-400">Industry</div>
                    <div className="text-white font-semibold">
                      {companyData.naeringskode1?.beskrivelse || "Not specified"}
                    </div>
                    <div className="text-xs text-slate-500">
                      Code: {companyData.naeringskode1?.kode || "N/A"}
                    </div>
                  </div>
                </div>

                {/* Employees */}
                <div className="flex items-start gap-3">
                  <Users className="w-5 h-5 text-green-400 mt-1" />
                  <div>
                    <div className="text-sm text-slate-400">Employees</div>
                    <div className="text-white font-semibold text-2xl">
                      {companyData.antallAnsatte || "N/A"}
                    </div>
                  </div>
                </div>

                {/* Address */}
                <div className="flex items-start gap-3">
                  <MapPin className="w-5 h-5 text-purple-400 mt-1" />
                  <div>
                    <div className="text-sm text-slate-400">Business Address</div>
                    <div className="text-white">
                      {companyData.forretningsadresse?.adresse?.join(", ") || "Not specified"}
                    </div>
                    <div className="text-sm text-slate-400">
                      {companyData.forretningsadresse?.postnummer}{" "}
                      {companyData.forretningsadresse?.poststed}
                    </div>
                  </div>
                </div>

                {/* Registration Date */}
                <div className="flex items-start gap-3">
                  <Calendar className="w-5 h-5 text-yellow-400 mt-1" />
                  <div>
                    <div className="text-sm text-slate-400">Registration Date</div>
                    <div className="text-white font-semibold">
                      {companyData.registreringsdatoEnhetsregisteret || "Not specified"}
                    </div>
                  </div>
                </div>

                {/* VAT Status */}
                <div className="flex items-start gap-3">
                  <TrendingUp className="w-5 h-5 text-cyan-400 mt-1" />
                  <div>
                    <div className="text-sm text-slate-400">VAT Registered</div>
                    <div className="text-white font-semibold">
                      {companyData.registrertIMvaregisteret ? "Yes" : "No"}
                    </div>
                  </div>
                </div>
                </div>
              </CardContent>
            </Card>

            {/* Company Roles */}
            {rolesData && rolesData.rollegrupper && rolesData.rollegrupper.length > 0 && (
              <Card className="bg-white/5 backdrop-blur-xl border-white/10 shadow-2xl">
                <CardHeader>
                  <CardTitle className="text-2xl font-bold text-white">Company Roles</CardTitle>
                  <CardDescription className="text-slate-300">
                    Board members, CEO, and other key personnel
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {rolesData.rollegrupper.map((gruppe: any, idx: number) => (
                    <div key={idx} className="space-y-2">
                      <h3 className="text-lg font-semibold text-white">
                        {gruppe.type?.beskrivelse || "Unknown Group"}
                      </h3>
                      <div className="grid gap-2">
                        {gruppe.roller?.map((rolle: any, roleIdx: number) => (
                          <div
                            key={roleIdx}
                            className="flex items-center justify-between p-3 bg-white/5 rounded-lg border border-white/10"
                          >
                            <div>
                              <div className="text-white font-medium">
                                {rolle.person?.navn?.fornavn} {rolle.person?.navn?.etternavn}
                                {rolle.enhet?.navn}
                              </div>
                              <div className="text-sm text-slate-400">
                                {rolle.type?.beskrivelse || "Unknown Role"}
                              </div>
                            </div>
                            {rolle.person?.fodselsdato && (
                              <Badge className="bg-slate-700/50 text-slate-300">
                                Born: {rolle.person.fodselsdato}
                              </Badge>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
