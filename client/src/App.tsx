import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import NotFound from "@/pages/NotFound";
import { Route, Switch } from "wouter";
import ErrorBoundary from "./components/ErrorBoundary";
import { ThemeProvider } from "./contexts/ThemeContext";
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
        <Route path="/" component={Home} />
        <Route path="/dashboard" component={Dashboard} />
        <Route path="/agents" component={Agents} />
        <Route path="/chat" component={Chat} />
        <Route path="/knowledge-graph" component={KnowledgeGraph} />
        <Route path="/analytics" component={Analytics} />
      <Route path={"/documentation"} component={Documentation} />
      <Route path={"/s7-test"} component={S7Test} />
      <Route path={"/s7-extended"} component={S7Extended} />
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
          <Toaster />
          <Router />
        </TooltipProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;
