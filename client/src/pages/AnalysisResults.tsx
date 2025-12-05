import React from 'react';
import { useParams, useLocation } from 'wouter';
import { trpc } from '@/lib/trpc';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { 
  Building2, Globe, Users, TrendingUp, Star, AlertCircle, 
  CheckCircle2, ArrowRight, Loader2, ExternalLink 
} from 'lucide-react';
import { 
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer 
} from 'recharts';

export default function AnalysisResults() {
  const params = useParams();
  const [, setLocation] = useLocation();
  const navigate = (path: string) => setLocation(path);
  const orgNumber = params.orgNumber as string;

  // Fetch analysis data
  const { mutate: analyzeCompany, data: analysis, isPending, error } = trpc.business.analyzeCompany.useMutation();
  
  // Trigger analysis on mount
  React.useEffect(() => {
    if (orgNumber && !analysis && !isPending) {
      analyzeCompany({ organizationNumber: orgNumber });
    }
  }, [orgNumber, analysis, isPending]);
  
  const isLoading = isPending;

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-950 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-16 h-16 mx-auto mb-4 animate-spin text-cyan-400" />
          <h2 className="text-2xl font-bold text-white mb-2">Analyzing Business...</h2>
          <p className="text-slate-400">This may take 30-60 seconds</p>
        </div>
      </div>
    );
  }

  if (error || !analysis) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-950 flex items-center justify-center p-4">
        <Card className="bg-slate-900/50 border-slate-700 max-w-md">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <AlertCircle className="w-6 h-6 text-red-400" />
              Analysis Failed
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-slate-300 mb-4">
              {error?.message || 'Failed to analyze business. Please try again.'}
            </p>
            <Button onClick={() => navigate('/get-started')}>
              Back to Search
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Prepare chart data
  const scoreData = [
    { name: 'Website', score: analysis.website?.seoScore || 0, color: '#3b82f6' },
    { name: 'LinkedIn', score: Math.round(analysis.linkedin.engagement * 20), color: '#0077b5' },
    { name: 'Social Media', score: Math.round((analysis.socialMedia.facebook.engagement + analysis.socialMedia.instagram.engagement + analysis.socialMedia.twitter.engagement) / 3 * 10), color: '#8b5cf6' },
    { name: 'Reviews', score: Math.round((analysis.reviews.google.rating + analysis.reviews.trustpilot.rating) / 2 * 20), color: '#10b981' },
    { name: 'SEO', score: analysis.website?.seoScore || 0, color: '#f59e0b' }
  ];

  const competitorData = analysis.competitors.map((comp: any, index: number) => ({
    name: comp.name,
    score: 70 - (index * 5) // Mock scores for competitors
  }));

  const COLORS = ['#10b981', '#f59e0b', '#ef4444'];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-950 py-8 px-4">
      <div className="container max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <Button 
            variant="ghost" 
            onClick={() => navigate('/get-started')}
            className="text-slate-400 hover:text-white mb-4"
          >
            ← Back to Search
          </Button>
          <div className="flex items-start justify-between">
            <div>
              <h1 className="text-4xl font-bold text-white mb-2">{analysis.name}</h1>
              <p className="text-slate-400">Org. Number: {analysis.organizationNumber}</p>
              <div className="flex gap-2 mt-2">
                <Badge variant="outline" className="text-cyan-400 border-cyan-400">
                  {analysis.industryName}
                </Badge>
                {analysis.website?.url && (
                  <a 
                    href={analysis.website.url} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="text-slate-400 hover:text-white flex items-center gap-1"
                  >
                    <Globe className="w-4 h-4" />
                    {analysis.website.url}
                    <ExternalLink className="w-3 h-3" />
                  </a>
                )}
              </div>
            </div>
            <div className="text-right">
              <div className="text-6xl font-bold text-cyan-400 mb-1">
                {analysis.digitalMaturityScore}
              </div>
              <p className="text-slate-400">Digital Maturity Score</p>
            </div>
          </div>
        </div>

        {/* Score Breakdown */}
        <div className="grid md:grid-cols-5 gap-4 mb-8">
          {scoreData.map((item) => (
            <Card key={item.name} className="bg-slate-900/50 border-slate-700">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-slate-400">{item.name}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-white mb-2">{item.score}</div>
                <Progress 
                  value={item.score} 
                  className="h-2"
                  style={{ 
                    '--progress-background': item.color 
                  } as React.CSSProperties}
                />
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Charts Row */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          {/* Category Scores Bar Chart */}
          <Card className="bg-slate-900/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Performance by Category</CardTitle>
              <CardDescription>Breakdown of digital presence scores</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={scoreData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="name" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                    labelStyle={{ color: '#e2e8f0' }}
                  />
                  <Bar dataKey="score" fill="#3b82f6" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Competitor Comparison */}
          <Card className="bg-slate-900/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Competitor Comparison</CardTitle>
              <CardDescription>Your position vs. top competitors</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={[
                  { name: analysis.name, score: analysis.digitalMaturityScore, fill: '#06b6d4' },
                  ...competitorData.map((c: any, i: number) => ({ ...c, fill: ['#3b82f6', '#8b5cf6', '#ec4899'][i] }))
                ]}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="name" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                    labelStyle={{ color: '#e2e8f0' }}
                  />
                  <Bar dataKey="score" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>

        {/* Detailed Metrics */}
        <div className="grid md:grid-cols-3 gap-6 mb-8">
          {/* Website Analysis */}
          <Card className="bg-slate-900/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Globe className="w-5 h-5 text-cyan-400" />
                Website Analysis
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex justify-between">
                <span className="text-slate-400">SEO Score</span>
                <span className="text-white font-semibold">{analysis.website?.seoScore || 0}/100</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Performance</span>
                <span className="text-white font-semibold">{analysis.website?.performance || 0}/100</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Mobile Optimized</span>
                <span className="text-white font-semibold">
                  {analysis.website?.mobileOptimized ? 'Yes' : 'No'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Technologies</span>
                <span className="text-white font-semibold text-right">
                  {analysis.website?.technologies?.slice(0, 2).join(', ') || 'N/A'}
                </span>
              </div>
            </CardContent>
          </Card>

          {/* LinkedIn Metrics */}
          <Card className="bg-slate-900/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Users className="w-5 h-5 text-blue-400" />
                LinkedIn Presence
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex justify-between">
                <span className="text-slate-400">Followers</span>
                <span className="text-white font-semibold">{analysis.linkedin.followers.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Employees on LinkedIn</span>
                <span className="text-white font-semibold">{analysis.linkedin.employees}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Engagement Rate</span>
                <span className="text-white font-semibold">{analysis.linkedin.engagement}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Data Completeness</span>
                <span className="text-white font-semibold">{analysis.dataCompleteness}%</span>
              </div>
            </CardContent>
          </Card>

          {/* Social Media & Reviews */}
          <Card className="bg-slate-900/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Star className="w-5 h-5 text-yellow-400" />
                Social & Reviews
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex justify-between">
                <span className="text-slate-400">Google Rating</span>
                <span className="text-white font-semibold flex items-center gap-1">
                  {analysis.reviews.google.rating}
                  <Star className="w-4 h-4 text-yellow-400 fill-yellow-400" />
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Google Reviews</span>
                <span className="text-white font-semibold">{analysis.reviews.google.count}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Trustpilot Rating</span>
                <span className="text-white font-semibold flex items-center gap-1">
                  {analysis.reviews.trustpilot.rating}
                  <Star className="w-4 h-4 text-yellow-400 fill-yellow-400" />
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Social Followers</span>
                <span className="text-white font-semibold">
                  {(analysis.socialMedia.facebook.followers + analysis.socialMedia.instagram.followers + analysis.socialMedia.twitter.followers).toLocaleString()}
                </span>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Action Buttons */}
        <div className="flex gap-4 justify-center">
          <Button
            size="lg"
            onClick={() => navigate(`/recommendations/${orgNumber}`)}
            className="bg-cyan-500 hover:bg-cyan-600"
          >
            View Recommendations
            <ArrowRight className="ml-2 w-5 h-5" />
          </Button>
          <Button
            size="lg"
            variant="outline"
            onClick={() => window.print()}
          >
            Export Report
          </Button>
        </div>
      </div>
    </div>
  );
}
