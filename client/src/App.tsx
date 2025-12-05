import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import NotFound from "@/pages/NotFound";
import { Route, Switch } from "wouter";
import ErrorBoundary from "./components/ErrorBoundary";
import { ThemeProvider } from "./contexts/ThemeContext";
import Footer from "./components/Footer";
import { lazy, Suspense } from "react";
import { Activity } from "lucide-react";

// Lazy load pages for better performance
const Home = lazy(() => import("./pages/Home"));
const Dashboard = lazy(() => import("./pages/Dashboard"));
const Agents = lazy(() => import("./pages/Agents"));
const Chat = lazy(() => import("./pages/Chat"));
const KnowledgeGraph = lazy(() => import("./pages/KnowledgeGraph"));
const Analytics = lazy(() => import("./pages/Analytics"));
const Documentation = lazy(() => import("./pages/Documentation"));
const S7Test = lazy(() => import("./pages/S7Test"));
const S7Extended = lazy(() => import("./pages/S7Extended"));
const S7Leaderboard = lazy(() => import("./pages/S7Leaderboard"));
const AgentOrchestrator = lazy(() => import("./pages/AgentOrchestrator"));
const S7Comparison = lazy(() => import("./pages/S7Comparison"));
const AgentAnalytics = lazy(() => import("./pages/AgentAnalytics"));
const S7StudyPath = lazy(() => import("./pages/S7StudyPath"));
const UnifiedAnalytics = lazy(() => import("./pages/UnifiedAnalytics"));
const Login = lazy(() => import("./pages/Login"));
const Terms = lazy(() => import("./pages/Terms"));
const Privacy = lazy(() => import("./pages/Privacy"));
const GetStarted = lazy(() => import("./pages/GetStarted"));
const AnalysisResults = lazy(() => import("./pages/AnalysisResults"));
const Recommendations = lazy(() => import("./pages/Recommendations"));
const ExecutionDashboard = lazy(() => import("./pages/ExecutionDashboard"));
const RevenueTracking = lazy(() => import("./pages/RevenueTracking"));
const AnalysisHistory = lazy(() => import("./pages/AnalysisHistory"));

function LoadingFallback() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-background">
      <div className="text-center">
        <Activity className="w-12 h-12 animate-spin text-primary mx-auto mb-4" />
        <p className="text-muted-foreground">Loading...</p>
      </div>
    </div>
  );
}

function Router() {
  return (
    <Suspense fallback={<LoadingFallback />}>
      <Switch>
        <Route path="/login" component={Login} />
        <Route path="/" component={Home} />
        <Route path="/get-started" component={GetStarted} />
        <Route path="/analysis/:orgNumber" component={AnalysisResults} />
        <Route path="/recommendations/:orgNumber" component={Recommendations} />
        <Route path="/execution-dashboard" component={ExecutionDashboard} />
        <Route path="/revenue-tracking" component={RevenueTracking} />
        <Route path="/analysis-history" component={AnalysisHistory} />
        <Route path="/dashboard" component={Dashboard} />
        <Route path="/agents" component={Agents} />
        <Route path="/chat" component={Chat} />
        <Route path="/knowledge-graph" component={KnowledgeGraph} />
        <Route path="/analytics" component={Analytics} />
      <Route path={"/documentation"} component={Documentation} />
      <Route path={"/s7-test"} component={S7Test} />
      <Route path={"/s7-extended"} component={S7Extended} />
      <Route path={"/s7-leaderboard"} component={S7Leaderboard} />
      <Route path={"/agent-orchestrator"} component={AgentOrchestrator} />
      <Route path={"/s7-comparison"} component={S7Comparison} />
      <Route path={"/agent-analytics"} component={AgentAnalytics} />
      <Route path={"/s7-study-path"} component={S7StudyPath} />
        <Route path="/unified-analytics" component={UnifiedAnalytics} />
        <Route path="/terms" component={Terms} />
        <Route path="/privacy" component={Privacy} />
      <Route path={"/404"} component={NotFound} />
        <Route component={NotFound} />
      </Switch>
    </Suspense>
  );
}

function App() {
  return (
    <ErrorBoundary>
      <ThemeProvider defaultTheme="light">
        <TooltipProvider>
          <div className="flex flex-col min-h-screen">
            <Toaster />
            <Router />
            <Footer />
          </div>
        </TooltipProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;
