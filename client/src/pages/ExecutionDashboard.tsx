import React from 'react';
import { useParams, useLocation } from 'wouter';
import { trpc } from '@/lib/trpc';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { 
  Loader2, CheckCircle2, AlertCircle, Clock, 
  TrendingUp, Users, DollarSign, Pause, Play, X
} from 'lucide-react';

export default function ExecutionDashboard() {
  const params = useParams();
  const navigate = useLocation()[1];
  const workflowId = params.workflowId as string;

  // Fetch execution status (poll every 2 seconds)
  const { data: status, isLoading, error } = trpc.business.getExecutionStatus.useQuery(
    { workflowId },
    { 
      enabled: !!workflowId,
      refetchInterval: 2000 // Poll every 2 seconds
    }
  );

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-950 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-16 h-16 mx-auto mb-4 animate-spin text-cyan-400" />
          <h2 className="text-2xl font-bold text-white mb-2">Loading Execution Status...</h2>
        </div>
      </div>
    );
  }

  if (error || !status) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-950 flex items-center justify-center p-4">
        <Card className="bg-slate-900/50 border-slate-700 max-w-md">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <AlertCircle className="w-6 h-6 text-red-400" />
              Execution Not Found
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-slate-300 mb-4">
              {error?.message || 'Could not find execution status. The workflow may have been cancelled or completed.'}
            </p>
            <Button onClick={() => navigate('/get-started')}>
              Back to Home
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  const isCompleted = status.status === 'completed';
  const isFailed = false; // Mock data doesn't include failed status
  const isExecuting = status.status === 'executing';

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-950 py-8 px-4">
      <div className="container max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-start justify-between">
            <div>
              <h1 className="text-4xl font-bold text-white mb-2">Automation Execution</h1>
              <p className="text-slate-400">Workflow ID: {workflowId}</p>
              <div className="flex gap-2 mt-2">
                <Badge 
                  variant={isCompleted ? 'default' : isFailed ? 'destructive' : 'secondary'}
                  className={isCompleted ? 'bg-green-500' : isFailed ? 'bg-red-500' : 'bg-cyan-500'}
                >
                  {status.status.toUpperCase()}
                </Badge>
                {isExecuting && (
                  <Badge variant="outline" className="text-cyan-400 border-cyan-400 animate-pulse">
                    In Progress
                  </Badge>
                )}
              </div>
            </div>
            <div className="text-right">
              <div className="text-6xl font-bold text-cyan-400 mb-1">
                {status.progress}%
              </div>
              <p className="text-slate-400">Complete</p>
            </div>
          </div>
        </div>

        {/* Overall Progress */}
        <Card className="bg-slate-900/50 border-slate-700 mb-8">
          <CardHeader>
            <CardTitle className="text-white">Overall Progress</CardTitle>
            <CardDescription>
              {status.completedTasks} of {status.totalTasks} tasks completed
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Progress value={status.progress} className="h-4 mb-4" />
            <div className="grid md:grid-cols-4 gap-4">
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 rounded-full bg-cyan-500/20 flex items-center justify-center">
                  <CheckCircle2 className="w-6 h-6 text-cyan-400" />
                </div>
                <div>
                  <div className="text-sm text-slate-400">Completed</div>
                  <div className="text-2xl font-bold text-white">{status.completedTasks}</div>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 rounded-full bg-yellow-500/20 flex items-center justify-center">
                  <Loader2 className="w-6 h-6 text-yellow-400 animate-spin" />
                </div>
                <div>
                  <div className="text-sm text-slate-400">In Progress</div>
                  <div className="text-2xl font-bold text-white">
                    {isExecuting ? status.totalTasks - status.completedTasks - status.failedTasks : 0}
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 rounded-full bg-red-500/20 flex items-center justify-center">
                  <AlertCircle className="w-6 h-6 text-red-400" />
                </div>
                <div>
                  <div className="text-sm text-slate-400">Failed</div>
                  <div className="text-2xl font-bold text-white">{status.failedTasks}</div>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 rounded-full bg-slate-500/20 flex items-center justify-center">
                  <Clock className="w-6 h-6 text-slate-400" />
                </div>
                <div>
                  <div className="text-sm text-slate-400">Est. Completion</div>
                  <div className="text-sm font-semibold text-white">
                    {new Date(status.estimatedCompletion).toLocaleTimeString()}
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Current Task */}
        {isExecuting && (
          <Card className="bg-cyan-500/10 border-cyan-500/30 mb-8">
            <CardContent className="py-6">
              <div className="flex items-center gap-4">
                <Loader2 className="w-8 h-8 text-cyan-400 animate-spin flex-shrink-0" />
                <div className="flex-1">
                  <div className="text-sm text-slate-400 mb-1">Currently Executing:</div>
                  <div className="text-xl font-semibold text-white">{status.currentTask}</div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Success Message */}
        {isCompleted && (
          <Card className="bg-green-500/10 border-green-500/30 mb-8">
            <CardContent className="py-6">
              <div className="flex items-center gap-4">
                <CheckCircle2 className="w-12 h-12 text-green-400 flex-shrink-0" />
                <div className="flex-1">
                  <div className="text-2xl font-bold text-white mb-2">
                    All Automations Completed Successfully!
                  </div>
                  <div className="text-slate-300">
                    Your business improvements have been implemented. You should see results within the next few weeks.
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Mock Task List */}
        <div className="grid md:grid-cols-2 gap-4 mb-8">
          {[
            { name: 'SEO Optimization', status: status.completedTasks >= 1 ? 'completed' : 'pending', category: 'Website' },
            { name: 'LinkedIn Engagement', status: status.completedTasks >= 2 ? 'completed' : status.completedTasks >= 1 ? 'executing' : 'pending', category: 'LinkedIn' },
            { name: 'Email Marketing Setup', status: status.completedTasks >= 3 ? 'completed' : status.completedTasks >= 2 ? 'executing' : 'pending', category: 'Marketing' },
            { name: 'CRM Implementation', status: status.completedTasks >= 4 ? 'completed' : status.completedTasks >= 3 ? 'executing' : 'pending', category: 'Operations' },
            { name: 'Lead Scoring', status: status.completedTasks >= 5 ? 'completed' : status.completedTasks >= 4 ? 'executing' : 'pending', category: 'Sales' },
            { name: 'Live Chat Support', status: status.completedTasks >= 6 ? 'completed' : status.completedTasks >= 5 ? 'executing' : 'pending', category: 'Customer Service' }
          ].map((task, index) => (
            <Card 
              key={index} 
              className={`bg-slate-900/50 border-slate-700 ${
                task.status === 'executing' ? 'border-cyan-500/50 shadow-lg shadow-cyan-500/20' : ''
              }`}
            >
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-white text-lg">{task.name}</CardTitle>
                  {task.status === 'completed' && (
                    <CheckCircle2 className="w-6 h-6 text-green-400" />
                  )}
                  {task.status === 'executing' && (
                    <Loader2 className="w-6 h-6 text-cyan-400 animate-spin" />
                  )}
                  {task.status === 'pending' && (
                    <Clock className="w-6 h-6 text-slate-500" />
                  )}
                </div>
                <CardDescription>{task.category}</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between text-sm">
                  <span className={`font-semibold ${
                    task.status === 'completed' ? 'text-green-400' :
                    task.status === 'executing' ? 'text-cyan-400' :
                    'text-slate-500'
                  }`}>
                    {task.status === 'completed' ? 'Completed' :
                     task.status === 'executing' ? 'In Progress...' :
                     'Pending'}
                  </span>
                  {task.status === 'executing' && (
                    <span className="text-slate-400">
                      {Math.floor(Math.random() * 40 + 30)}% done
                    </span>
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Action Buttons */}
        <div className="flex gap-4 justify-center">
          {isCompleted && (
            <Button
              size="lg"
              onClick={() => navigate('/get-started')}
              className="bg-cyan-500 hover:bg-cyan-600"
            >
              Start New Analysis
            </Button>
          )}
          {isExecuting && (
            <>
              <Button
                size="lg"
                variant="outline"
                disabled
              >
                <Pause className="mr-2 w-5 h-5" />
                Pause (Coming Soon)
              </Button>
              <Button
                size="lg"
                variant="destructive"
                disabled
              >
                <X className="mr-2 w-5 h-5" />
                Cancel (Coming Soon)
              </Button>
            </>
          )}
          <Button
            size="lg"
            variant="outline"
            onClick={() => window.print()}
          >
            Export Report
          </Button>
        </div>

        {/* Timeline */}
        <Card className="bg-slate-900/50 border-slate-700 mt-8">
          <CardHeader>
            <CardTitle className="text-white">Execution Timeline</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-start gap-4">
                <div className="w-2 h-2 rounded-full bg-green-400 mt-2" />
                <div>
                  <div className="text-white font-semibold">Execution Started</div>
                  <div className="text-sm text-slate-400">
                    {new Date(status.startedAt).toLocaleString()}
                  </div>
                </div>
              </div>
              {status.completedTasks > 0 && (
                <div className="flex items-start gap-4">
                  <div className="w-2 h-2 rounded-full bg-cyan-400 mt-2" />
                  <div>
                    <div className="text-white font-semibold">
                      {status.completedTasks} task{status.completedTasks > 1 ? 's' : ''} completed
                    </div>
                    <div className="text-sm text-slate-400">Just now</div>
                  </div>
                </div>
              )}
              {!isCompleted && (
                <div className="flex items-start gap-4">
                  <div className="w-2 h-2 rounded-full bg-slate-500 mt-2" />
                  <div>
                    <div className="text-white font-semibold">Estimated Completion</div>
                    <div className="text-sm text-slate-400">
                      {new Date(status.estimatedCompletion).toLocaleString()}
                    </div>
                  </div>
                </div>
              )}
              {isCompleted && (
                <div className="flex items-start gap-4">
                  <div className="w-2 h-2 rounded-full bg-green-400 mt-2" />
                  <div>
                    <div className="text-white font-semibold">All Tasks Completed</div>
                    <div className="text-sm text-slate-400">
                      {new Date(status.estimatedCompletion).toLocaleString()}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
