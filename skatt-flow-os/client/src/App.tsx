import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import NotFound from "@/pages/NotFound";
import { Route, Switch } from "wouter";
import ErrorBoundary from "./components/ErrorBoundary";
import { ThemeProvider } from "./contexts/ThemeContext";
import DashboardLayout from "./components/DashboardLayout";

// Pages
import Dashboard from "./pages/Dashboard";
import Companies from "./pages/Companies";
import CompanyDetail from "./pages/CompanyDetail";
import Accounting from "./pages/Accounting";
import Ledger from "./pages/Ledger";
import Filings from "./pages/Filings";
import FilingDetail from "./pages/FilingDetail";
import Documents from "./pages/Documents";
import DocumentDetail from "./pages/DocumentDetail";
import Chat from "./pages/Chat";
import Settings from "./pages/Settings";
import Reports from "./pages/Reports";
import AuditLog from "./pages/AuditLog";
import Reconciliation from "./pages/Reconciliation";
import Help from "./pages/Help";

function Router() {
  return (
    <Switch>
      {/* Dashboard - Main entry point */}
      <Route path="/">
        <DashboardLayout>
          <Dashboard />
        </DashboardLayout>
      </Route>

      {/* Company routes */}
      <Route path="/companies">
        <DashboardLayout>
          <Companies />
        </DashboardLayout>
      </Route>
      <Route path="/companies/:id">
        <DashboardLayout>
          <CompanyDetail />
        </DashboardLayout>
      </Route>
      <Route path="/companies/:id/documents">
        <DashboardLayout>
          <Accounting />
        </DashboardLayout>
      </Route>
      <Route path="/companies/:id/ledger">
        <DashboardLayout>
          <Ledger />
        </DashboardLayout>
      </Route>
      <Route path="/companies/:id/filings">
        <DashboardLayout>
          <Filings />
        </DashboardLayout>
      </Route>
      <Route path="/companies/:id/reconciliation">
        <DashboardLayout>
          <Reconciliation />
        </DashboardLayout>
      </Route>

      {/* Accounting & Documents */}
      <Route path="/accounting">
        <DashboardLayout>
          <Accounting />
        </DashboardLayout>
      </Route>
      <Route path="/documents">
        <DashboardLayout>
          <Documents />
        </DashboardLayout>
      </Route>
      <Route path="/documents/:id">
        <DashboardLayout>
          <DocumentDetail />
        </DashboardLayout>
      </Route>

      {/* Ledger & Reports */}
      <Route path="/ledger">
        <DashboardLayout>
          <Ledger />
        </DashboardLayout>
      </Route>
      <Route path="/reports">
        <DashboardLayout>
          <Reports />
        </DashboardLayout>
      </Route>
      <Route path="/reconciliation">
        <DashboardLayout>
          <Reconciliation />
        </DashboardLayout>
      </Route>

      {/* Filings */}
      <Route path="/filings">
        <DashboardLayout>
          <Filings />
        </DashboardLayout>
      </Route>
      <Route path="/filings/:id">
        <DashboardLayout>
          <FilingDetail />
        </DashboardLayout>
      </Route>

      {/* AI Chat */}
      <Route path="/chat">
        <DashboardLayout>
          <Chat />
        </DashboardLayout>
      </Route>

      {/* Admin & Settings */}
      <Route path="/settings">
        <DashboardLayout>
          <Settings />
        </DashboardLayout>
      </Route>
      <Route path="/audit-log">
        <DashboardLayout>
          <AuditLog />
        </DashboardLayout>
      </Route>

      {/* Help & Support */}
      <Route path="/help">
        <DashboardLayout>
          <Help />
        </DashboardLayout>
      </Route>

      {/* 404 fallback */}
      <Route path="/404" component={NotFound} />
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  return (
    <ErrorBoundary>
      <ThemeProvider defaultTheme="dark" switchable>
        <TooltipProvider>
          <Toaster richColors position="top-right" />
          <Router />
        </TooltipProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;
