import React from 'react';
import { useParams, useLocation } from 'wouter';
import { trpc } from '@/lib/trpc';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Checkbox } from '@/components/ui/checkbox';
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import { 
  Loader2, ArrowRight, CheckCircle2, AlertCircle, 
  TrendingUp, DollarSign, Clock, Target, Zap
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

type Category = 'all' | 'website' | 'linkedin' | 'marketing' | 'operations' | 'sales' | 'customer_service';
type Priority = 'all' | 'high' | 'medium' | 'low';

export default function Recommendations() {
  const params = useParams();
  const navigate = useLocation()[1];
  const { toast } = useToast();
  const orgNumber = params.orgNumber as string;

  // State
  const [selectedIds, setSelectedIds] = React.useState<string[]>([]);
  const [categoryFilter, setCategoryFilter] = React.useState<Category>('all');
  const [priorityFilter, setPriorityFilter] = React.useState<Priority>('all');

  // Fetch recommendations
  const { mutate: generateRecs, data: recommendations, isPending, error } = 
    trpc.business.generateRecommendations.useMutation();

  // Execute recommendations
  const { mutate: executeRecs, isPending: isExecuting } = 
    trpc.business.executeRecommendations.useMutation({
      onSuccess: (data) => {
        toast({
          title: 'Execution Started',
          description: `${selectedIds.length} recommendations are being executed.`,
        });
        navigate(`/execution/${data.workflowId}`);
      },
      onError: (error) => {
        toast({
          title: 'Execution Failed',
          description: error.message,
          variant: 'destructive',
        });
      }
    });

  // Trigger generation on mount
  React.useEffect(() => {
    if (orgNumber && !recommendations && !isPending) {
      generateRecs({ organizationNumber: orgNumber });
    }
  }, [orgNumber, recommendations, isPending]);

  // Filter recommendations
  const filteredRecs = React.useMemo(() => {
    if (!recommendations) return [];
    
    return recommendations.recommendations.filter((rec: any) => {
      const matchesCategory = categoryFilter === 'all' || rec.category === categoryFilter;
      const matchesPriority = priorityFilter === 'all' || rec.priority === priorityFilter;
      return matchesCategory && matchesPriority;
    });
  }, [recommendations, categoryFilter, priorityFilter]);

  // Toggle selection
  const toggleSelection = (id: string) => {
    setSelectedIds(prev => 
      prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
    );
  };

  // Select all filtered
  const selectAll = () => {
    setSelectedIds(filteredRecs.map((rec: any) => rec.id));
  };

  // Clear selection
  const clearSelection = () => {
    setSelectedIds([]);
  };

  // Execute selected
  const executeSelected = () => {
    if (selectedIds.length === 0) {
      toast({
        title: 'No Recommendations Selected',
        description: 'Please select at least one recommendation to execute.',
        variant: 'destructive',
      });
      return;
    }

    executeRecs({
      organizationNumber: orgNumber,
      recommendationIds: selectedIds
    });
  };

  // Calculate totals for selected
  const selectedTotals = React.useMemo(() => {
    if (!recommendations) return { cost: 0, roi: 0, time: 0 };
    
    const selected = recommendations.recommendations.filter((rec: any) => 
      selectedIds.includes(rec.id)
    );

    return {
      cost: selected.reduce((sum: number, rec: any) => sum + rec.estimatedCost, 0),
      roi: selected.length > 0 
        ? selected.reduce((sum: number, rec: any) => sum + rec.expectedROI, 0) / selected.length 
        : 0,
      time: selected.reduce((sum: number, rec: any) => {
        const weeks = parseInt(rec.estimatedTime);
        return sum + (isNaN(weeks) ? 0 : weeks);
      }, 0)
    };
  }, [recommendations, selectedIds]);

  if (isPending) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-950 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-16 h-16 mx-auto mb-4 animate-spin text-cyan-400" />
          <h2 className="text-2xl font-bold text-white mb-2">Generating Recommendations...</h2>
          <p className="text-slate-400">Analyzing your business and creating improvement strategies</p>
        </div>
      </div>
    );
  }

  if (error || !recommendations) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-950 flex items-center justify-center p-4">
        <Card className="bg-slate-900/50 border-slate-700 max-w-md">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <AlertCircle className="w-6 h-6 text-red-400" />
              Generation Failed
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-slate-300 mb-4">
              {error?.message || 'Failed to generate recommendations. Please try again.'}
            </p>
            <Button onClick={() => navigate(`/analysis/${orgNumber}`)}>
              Back to Analysis
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high': return 'bg-red-500/20 text-red-400 border-red-500/50';
      case 'medium': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50';
      case 'low': return 'bg-green-500/20 text-green-400 border-green-500/50';
      default: return 'bg-slate-500/20 text-slate-400 border-slate-500/50';
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'website': return '🌐';
      case 'linkedin': return '💼';
      case 'marketing': return '📢';
      case 'operations': return '⚙️';
      case 'sales': return '💰';
      case 'customer_service': return '🎧';
      default: return '📊';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-purple-950 to-slate-950 py-8 px-4 relative overflow-hidden">
      <div className="container max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <Button 
            variant="ghost" 
            onClick={() => navigate(`/analysis/${orgNumber}`)}
            className="text-slate-400 hover:text-white mb-4"
          >
            ← Back to Analysis
          </Button>
          <div className="flex items-start justify-between">
            <div>
              <h1 className="text-5xl font-black text-white mb-2 tracking-tight">Business Enhancement Recommendations</h1>
              <p className="text-slate-400">{recommendations.companyName} • {filteredRecs.length} strategies</p>
            </div>
            <div className="text-right">
              <div className="text-6xl font-bold text-cyan-400 mb-1">
                {recommendations.overallScore}
              </div>
              <p className="text-slate-400">Current Score</p>
            </div>
          </div>
        </div>

        {/* Summary Cards */}
        <div className="grid md:grid-cols-4 gap-4 mb-8">
          <Card className="bg-slate-900/50 border-slate-700">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-slate-400">Total Investment</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-white">
                ${recommendations.totalEstimatedCost.toLocaleString()}
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-900/50 border-slate-700">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-slate-400">Expected ROI</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-400">
                {recommendations.totalExpectedROI}%
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-900/50 border-slate-700">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-slate-400">Timeline</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-white">
                {recommendations.implementationTimeline}
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-900/50 border-slate-700">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-slate-400">Position</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-white capitalize">
                {recommendations.competitivePosition}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Filters and Actions */}
        <div className="flex flex-wrap gap-4 mb-6">
          <Select value={categoryFilter} onValueChange={(v) => setCategoryFilter(v as Category)}>
            <SelectTrigger className="w-48 bg-slate-900/50 border-slate-700 text-white">
              <SelectValue placeholder="Filter by category" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Categories</SelectItem>
              <SelectItem value="website">Website</SelectItem>
              <SelectItem value="linkedin">LinkedIn</SelectItem>
              <SelectItem value="marketing">Marketing</SelectItem>
              <SelectItem value="operations">Operations</SelectItem>
              <SelectItem value="sales">Sales</SelectItem>
              <SelectItem value="customer_service">Customer Service</SelectItem>
            </SelectContent>
          </Select>

          <Select value={priorityFilter} onValueChange={(v) => setPriorityFilter(v as Priority)}>
            <SelectTrigger className="w-48 bg-slate-900/50 border-slate-700 text-white">
              <SelectValue placeholder="Filter by priority" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Priorities</SelectItem>
              <SelectItem value="high">High Priority</SelectItem>
              <SelectItem value="medium">Medium Priority</SelectItem>
              <SelectItem value="low">Low Priority</SelectItem>
            </SelectContent>
          </Select>

          <div className="flex-1" />

          <Button variant="outline" onClick={selectAll} disabled={filteredRecs.length === 0}>
            Select All ({filteredRecs.length})
          </Button>
          <Button variant="outline" onClick={clearSelection} disabled={selectedIds.length === 0}>
            Clear Selection
          </Button>
        </div>

        {/* Selected Summary */}
        {selectedIds.length > 0 && (
          <Card className="bg-cyan-500/10 border-cyan-500/30 mb-6">
            <CardContent className="py-4">
              <div className="flex items-center justify-between">
                <div className="flex gap-8">
                  <div>
                    <div className="text-sm text-slate-400">Selected</div>
                    <div className="text-2xl font-bold text-white">{selectedIds.length}</div>
                  </div>
                  <div>
                    <div className="text-sm text-slate-400">Total Cost</div>
                    <div className="text-2xl font-bold text-white">${selectedTotals.cost.toLocaleString()}</div>
                  </div>
                  <div>
                    <div className="text-sm text-slate-400">Avg ROI</div>
                    <div className="text-2xl font-bold text-green-400">{Math.round(selectedTotals.roi)}%</div>
                  </div>
                  <div>
                    <div className="text-sm text-slate-400">Est. Time</div>
                    <div className="text-2xl font-bold text-white">{selectedTotals.time} weeks</div>
                  </div>
                </div>
                <Button 
                  size="lg" 
                  onClick={executeSelected}
                  disabled={isExecuting}
                  className="bg-cyan-500 hover:bg-cyan-600"
                >
                  {isExecuting ? (
                    <>
                      <Loader2 className="mr-2 w-5 h-5 animate-spin" />
                      Executing...
                    </>
                  ) : (
                    <>
                      Execute Selected
                      <ArrowRight className="ml-2 w-5 h-5" />
                    </>
                  )}
                </Button>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Recommendations List */}
        <div className="space-y-4">
          {filteredRecs.map((rec: any) => (
            <Card key={rec.id} className="bg-slate-900/50 border-slate-700 hover:border-cyan-500/50 transition-colors">
              <CardHeader>
                <div className="flex items-start gap-4">
                  <Checkbox
                    checked={selectedIds.includes(rec.id)}
                    onCheckedChange={() => toggleSelection(rec.id)}
                    className="mt-1"
                  />
                  <div className="flex-1">
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span className="text-2xl">{getCategoryIcon(rec.category)}</span>
                        <CardTitle className="text-white">{rec.title}</CardTitle>
                      </div>
                      <div className="flex gap-2">
                        <Badge className={getPriorityColor(rec.priority)}>
                          {rec.priority}
                        </Badge>
                        <Badge variant="outline" className="text-cyan-400 border-cyan-400">
                          {rec.confidence}% confidence
                        </Badge>
                      </div>
                    </div>
                    <CardDescription className="text-slate-300">
                      {rec.description}
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                {/* Metrics */}
                <div className="grid grid-cols-4 gap-4 mb-4">
                  <div className="flex items-center gap-2">
                    <DollarSign className="w-4 h-4 text-slate-400" />
                    <div>
                      <div className="text-xs text-slate-400">Cost</div>
                      <div className="text-sm font-semibold text-white">${rec.estimatedCost.toLocaleString()}</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <TrendingUp className="w-4 h-4 text-green-400" />
                    <div>
                      <div className="text-xs text-slate-400">ROI</div>
                      <div className="text-sm font-semibold text-green-400">{rec.expectedROI}%</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Clock className="w-4 h-4 text-slate-400" />
                    <div>
                      <div className="text-xs text-slate-400">Time</div>
                      <div className="text-sm font-semibold text-white">{rec.estimatedTime}</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Zap className="w-4 h-4 text-yellow-400" />
                    <div>
                      <div className="text-xs text-slate-400">Automatable</div>
                      <div className="text-sm font-semibold text-white">{rec.automatable ? 'Yes' : 'No'}</div>
                    </div>
                  </div>
                </div>

                {/* Platforms */}
                {rec.automationPlatforms && rec.automationPlatforms.length > 0 && (
                  <div className="mb-4">
                    <div className="text-xs text-slate-400 mb-2">Automation Platforms:</div>
                    <div className="flex flex-wrap gap-2">
                      {rec.automationPlatforms.map((platform: string) => (
                        <Badge key={platform} variant="secondary" className="bg-slate-800 text-slate-300">
                          {platform}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}

                {/* Implementation Steps */}
                <Accordion type="single" collapsible>
                  <AccordionItem value="steps" className="border-slate-700">
                    <AccordionTrigger className="text-white hover:text-cyan-400">
                      View Implementation Steps ({rec.steps.length})
                    </AccordionTrigger>
                    <AccordionContent>
                      <div className="space-y-3 pt-2">
                        {rec.steps.map((step: any) => (
                          <div key={step.stepNumber} className="flex gap-3">
                            <div className="flex-shrink-0 w-6 h-6 rounded-full bg-cyan-500/20 text-cyan-400 flex items-center justify-center text-xs font-bold">
                              {step.stepNumber}
                            </div>
                            <div>
                              <div className="text-white font-medium">{step.action}</div>
                              <div className="text-sm text-slate-400">{step.details}</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </AccordionContent>
                  </AccordionItem>
                </Accordion>

                {/* Expected Impact */}
                {rec.metrics && (
                  <div className="mt-4 p-3 bg-slate-800/50 rounded-lg">
                    <div className="text-xs text-slate-400 mb-2">Expected Impact:</div>
                    <div className="grid grid-cols-3 gap-4 text-sm">
                      {Object.keys(rec.metrics.before).map((key) => (
                        <div key={key}>
                          <div className="text-slate-400 capitalize">{key.replace(/([A-Z])/g, ' $1')}</div>
                          <div className="flex items-center gap-2">
                            <span className="text-slate-500">{rec.metrics.before[key]}</span>
                            <span className="text-slate-600">→</span>
                            <span className="text-green-400 font-semibold">{rec.metrics.after[key]}</span>
                            <span className="text-xs text-green-400">(+{rec.metrics.improvement[key]}%)</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          ))}
        </div>

        {filteredRecs.length === 0 && (
          <Card className="bg-slate-900/50 border-slate-700">
            <CardContent className="py-12 text-center">
              <AlertCircle className="w-12 h-12 mx-auto mb-4 text-slate-500" />
              <p className="text-slate-400">No recommendations match the selected filters.</p>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
