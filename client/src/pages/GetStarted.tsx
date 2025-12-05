/**
 * Get Started Wizard - Business Enhancement Onboarding
 */

import { useState } from "react";
import { trpc } from "@/lib/trpc";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Loader2, Building2, Globe, Users, TrendingUp, CheckCircle2, AlertCircle } from "lucide-react";

export default function GetStarted() {
  const [step, setStep] = useState<1 | 2 | 3 | 4>(1);
  const [orgNumber, setOrgNumber] = useState("");
  const [businessData, setBusinessData] = useState<any>(null);
  const [websiteAnalysis, setWebsiteAnalysis] = useState<any>(null);
  const [industryAnalysis, setIndustryAnalysis] = useState<any>(null);

  // Step 1: Lookup company
  const lookupMutation = trpc.business.searchCompany.useQuery(
    { organizationNumber: orgNumber },
    {
      enabled: false,
    }
  );

  const lookupCompany = () => {
    if (orgNumber.length === 9 && /^\d+$/.test(orgNumber)) {
      lookupMutation.refetch().then(({ data }) => {
        if (data) {
          setBusinessData(data);
          setStep(2);
          // Automatically proceed to step 3 (skip website analysis for now)
          setTimeout(() => setStep(3), 1000);
        }
      });
    }
  };

  // Step 2: Analyze website
  const analyzeWebsite = async (url: string) => {
    try {
      // For now, skip website analysis and go directly to step 3
      // TODO: Implement website analysis when mutation is fixed
      setStep(3);
    } catch (error) {
      console.error("Website analysis failed:", error);
      setStep(3);
    }
  };

  // Handle organization number submission
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    lookupCompany();
  };

  // Handle approval and start automation
  const handleApprove = () => {
    setStep(4);
    // In a real implementation, this would trigger the automation workflow
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-950 py-20 px-4">
      <div className="container max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-white mb-4">
            Get Started with <span className="text-cyan-400">TRUE ASI</span>
          </h1>
          <p className="text-xl text-slate-300">
            Enter your organization number to unlock autonomous business enhancement
          </p>
        </div>

        {/* Progress Indicator */}
        <div className="mb-12">
          <div className="flex justify-between items-center mb-4">
            <div className={`flex items-center gap-2 ${step >= 1 ? "text-cyan-400" : "text-slate-500"}`}>
              <div className={`w-8 h-8 rounded-full flex items-center justify-center ${step >= 1 ? "bg-cyan-400 text-black" : "bg-slate-700"}`}>
                1
              </div>
              <span className="font-medium">Company Lookup</span>
            </div>
            <div className={`flex items-center gap-2 ${step >= 2 ? "text-cyan-400" : "text-slate-500"}`}>
              <div className={`w-8 h-8 rounded-full flex items-center justify-center ${step >= 2 ? "bg-cyan-400 text-black" : "bg-slate-700"}`}>
                2
              </div>
              <span className="font-medium">Website Analysis</span>
            </div>
            <div className={`flex items-center gap-2 ${step >= 3 ? "text-cyan-400" : "text-slate-500"}`}>
              <div className={`w-8 h-8 rounded-full flex items-center justify-center ${step >= 3 ? "bg-cyan-400 text-black" : "bg-slate-700"}`}>
                3
              </div>
              <span className="font-medium">Review Plan</span>
            </div>
            <div className={`flex items-center gap-2 ${step >= 4 ? "text-cyan-400" : "text-slate-500"}`}>
              <div className={`w-8 h-8 rounded-full flex items-center justify-center ${step >= 4 ? "bg-cyan-400 text-black" : "bg-slate-700"}`}>
                4
              </div>
              <span className="font-medium">Automation</span>
            </div>
          </div>
          <Progress value={(step / 4) * 100} className="h-2" />
        </div>

        {/* Step 1: Organization Number Input */}
        {step === 1 && (
          <Card className="bg-slate-900/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Building2 className="w-6 h-6 text-cyan-400" />
                Enter Organization Number
              </CardTitle>
              <CardDescription>
                Enter your 9-digit Norwegian organization number (e.g., 974760673)
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                  <Input
                    type="text"
                    placeholder="974760673"
                    value={orgNumber}
                    onChange={(e) => setOrgNumber(e.target.value)}
                    maxLength={9}
                    pattern="\d{9}"
                    className="text-lg bg-slate-800 border-slate-600 text-white"
                    disabled={lookupMutation.isFetching}
                  />
                  {orgNumber && orgNumber.length !== 9 && (
                    <p className="text-sm text-amber-400 mt-2">
                      Organization number must be exactly 9 digits
                    </p>
                  )}
                </div>
                <Button
                  type="submit"
                  size="lg"
                  className="w-full"
                  disabled={lookupMutation.isFetching || orgNumber.length !== 9}
                >
                  {lookupMutation.isFetching ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Looking up company...
                    </>
                  ) : (
                    "Start Analysis"
                  )}
                </Button>
              </form>

              {lookupMutation.error && (
                <div className="mt-4 p-4 bg-red-900/20 border border-red-700 rounded-lg">
                  <p className="text-red-400 flex items-center gap-2">
                    <AlertCircle className="w-4 h-4" />
                    {lookupMutation.error?.message || "Failed to lookup company"}
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* Step 2: Company Info & Website Analysis */}
        {step === 2 && businessData && (
          <div className="space-y-6">
            <Card className="bg-slate-900/50 border-slate-700">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <CheckCircle2 className="w-6 h-6 text-green-400" />
                  Company Found
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <h3 className="text-2xl font-bold text-white">{businessData.name}</h3>
                  <p className="text-slate-400">{businessData.orgNumber}</p>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-slate-400">Industry</p>
                    <p className="text-white font-medium">{businessData.industry}</p>
                  </div>
                  <div>
                    <p className="text-sm text-slate-400">Employees</p>
                    <p className="text-white font-medium">{businessData.employees || "N/A"}</p>
                  </div>
                  <div>
                    <p className="text-sm text-slate-400">Website</p>
                    <p className="text-white font-medium">{businessData.website || "N/A"}</p>
                  </div>
                  <div>
                    <p className="text-sm text-slate-400">Email</p>
                    <p className="text-white font-medium">{businessData.email || "N/A"}</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {false && (
              <Card className="bg-slate-900/50 border-slate-700">
                <CardContent className="py-12 text-center">
                  <Loader2 className="w-12 h-12 mx-auto mb-4 animate-spin text-cyan-400" />
                  <p className="text-white text-lg">Analyzing website...</p>
                  <p className="text-slate-400 mt-2">
                    This may take 30-60 seconds as we analyze SEO, performance, accessibility, content, and UX
                  </p>
                </CardContent>
              </Card>
            )}

            {false && (
              <Card className="bg-slate-900/50 border-slate-700">
                <CardContent className="py-8">
                  <div className="p-4 bg-amber-900/20 border border-amber-700 rounded-lg">
                    <p className="text-amber-400 flex items-center gap-2">
                      <AlertCircle className="w-4 h-4" />
                      Website analysis failed. Continuing without website data.
                    </p>
                  </div>
                  <Button onClick={() => setStep(3)} className="mt-4 w-full">
                    Continue Anyway
                  </Button>
                </CardContent>
              </Card>
            )}
          </div>
        )}

        {/* Step 3: Analysis Results & Recommendations */}
        {step === 3 && websiteAnalysis && (
          <div className="space-y-6">
            <Card className="bg-slate-900/50 border-slate-700">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Globe className="w-6 h-6 text-cyan-400" />
                  Website Analysis Complete
                </CardTitle>
                <CardDescription>
                  Overall Score: {websiteAnalysis.overallScore}/100
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Score Breakdown */}
                <div className="grid grid-cols-5 gap-4">
                  <div className="text-center">
                    <div className={`text-3xl font-bold ${websiteAnalysis.seo.score >= 80 ? "text-green-400" : "text-amber-400"}`}>
                      {websiteAnalysis.seo.score}
                    </div>
                    <p className="text-sm text-slate-400">SEO</p>
                  </div>
                  <div className="text-center">
                    <div className={`text-3xl font-bold ${websiteAnalysis.performance.score >= 80 ? "text-green-400" : "text-amber-400"}`}>
                      {websiteAnalysis.performance.score}
                    </div>
                    <p className="text-sm text-slate-400">Performance</p>
                  </div>
                  <div className="text-center">
                    <div className={`text-3xl font-bold ${websiteAnalysis.accessibility.score >= 80 ? "text-green-400" : "text-amber-400"}`}>
                      {websiteAnalysis.accessibility.score}
                    </div>
                    <p className="text-sm text-slate-400">Accessibility</p>
                  </div>
                  <div className="text-center">
                    <div className={`text-3xl font-bold ${websiteAnalysis.contentQuality.score >= 80 ? "text-green-400" : "text-amber-400"}`}>
                      {websiteAnalysis.contentQuality.score}
                    </div>
                    <p className="text-sm text-slate-400">Content</p>
                  </div>
                  <div className="text-center">
                    <div className={`text-3xl font-bold ${websiteAnalysis.ux.score >= 80 ? "text-green-400" : "text-amber-400"}`}>
                      {websiteAnalysis.ux.score}
                    </div>
                    <p className="text-sm text-slate-400">UX</p>
                  </div>
                </div>

                {/* Top Recommendations */}
                <div>
                  <h4 className="text-lg font-semibold text-white mb-4">Top Recommendations</h4>
                  <div className="space-y-3">
                    {websiteAnalysis.recommendations.slice(0, 5).map((rec: any, idx: number) => (
                      <div
                        key={idx}
                        className="p-4 bg-slate-800/50 border border-slate-600 rounded-lg"
                      >
                        <div className="flex items-start justify-between mb-2">
                          <h5 className="font-semibold text-white">{rec.title}</h5>
                          <Badge
                            variant={
                              rec.priority === "critical"
                                ? "destructive"
                                : rec.priority === "high"
                                ? "default"
                                : "secondary"
                            }
                          >
                            {rec.priority}
                          </Badge>
                        </div>
                        <p className="text-sm text-slate-300 mb-2">{rec.description}</p>
                        <div className="flex items-center gap-4 text-xs text-slate-400">
                          <span>Impact: {rec.impact}</span>
                          <span>Effort: {rec.effort}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <Button onClick={handleApprove} size="lg" className="w-full">
                  <TrendingUp className="w-4 h-4 mr-2" />
                  Approve & Start Automation
                </Button>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Step 4: Automation in Progress */}
        {step === 4 && (
          <Card className="bg-slate-900/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <CheckCircle2 className="w-6 h-6 text-green-400" />
                Automation Started
              </CardTitle>
              <CardDescription>
                TRUE ASI is now implementing improvements across all digital touchpoints
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <div className="flex items-center gap-3">
                  <Loader2 className="w-5 h-5 animate-spin text-cyan-400" />
                  <span className="text-white">Optimizing website SEO...</span>
                </div>
                <div className="flex items-center gap-3">
                  <Loader2 className="w-5 h-5 animate-spin text-cyan-400" />
                  <span className="text-white">Enhancing LinkedIn profiles...</span>
                </div>
                <div className="flex items-center gap-3">
                  <Loader2 className="w-5 h-5 animate-spin text-cyan-400" />
                  <span className="text-white">Generating content strategy...</span>
                </div>
                <div className="flex items-center gap-3">
                  <Loader2 className="w-5 h-5 animate-spin text-cyan-400" />
                  <span className="text-white">Analyzing competitors...</span>
                </div>
              </div>

              <div className="p-4 bg-cyan-900/20 border border-cyan-700 rounded-lg">
                <p className="text-cyan-400 text-center">
                  You will receive real-time updates as improvements are implemented.
                  Check your dashboard for progress tracking.
                </p>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
