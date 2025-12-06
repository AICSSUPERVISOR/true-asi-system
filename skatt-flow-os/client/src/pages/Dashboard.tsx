import { trpc } from "@/lib/trpc";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { 
  FileText, 
  AlertTriangle, 
  Clock, 
  Building2, 
  TrendingUp, 
  TrendingDown,
  ArrowRight,
  Sparkles,
  Calendar,
  CheckCircle2,
  XCircle,
  BarChart3,
  Zap,
  Shield,
  Bot
} from "lucide-react";
import { Link } from "wouter";
import { Button } from "@/components/ui/button";
import { motion } from "framer-motion";

// Animation variants
const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.5,
    },
  },
};

export default function Dashboard() {
  const { data: stats, isLoading } = trpc.dashboard.stats.useQuery();
  const { data: companies } = trpc.company.list.useQuery();

  // Calculate deadline info
  const today = new Date();
  const currentMonth = today.getMonth() + 1;
  const currentTerm = Math.ceil(currentMonth / 2);
  const mvaDeadlineMonth = currentTerm * 2 + 1;
  const mvaDeadline = new Date(today.getFullYear() + (mvaDeadlineMonth > 12 ? 1 : 0), (mvaDeadlineMonth - 1) % 12, 10);
  const daysUntilMVA = Math.ceil((mvaDeadline.getTime() - today.getTime()) / (1000 * 60 * 60 * 24));

  return (
    <motion.div 
      className="space-y-8"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      {/* Premium Header with Gradient */}
      <motion.div variants={itemVariants} className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-emerald-600 via-emerald-500 to-teal-500 p-8 text-white shadow-xl">
        <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxnIGZpbGw9IiNmZmYiIGZpbGwtb3BhY2l0eT0iMC4xIj48cGF0aCBkPSJNMzYgMzRoLTJ2LTRoMnY0em0wLTZoLTJ2LTRoMnY0em0tNiA2aC0ydi00aDJ2NHptMC02aC0ydi00aDJ2NHoiLz48L2c+PC9nPjwvc3ZnPg==')] opacity-30"></div>
        <div className="relative z-10 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold tracking-tight flex items-center gap-3">
              <Sparkles className="h-8 w-8" />
              Skatt-Flow OS
            </h1>
            <p className="mt-2 text-emerald-100 text-lg">
              Autonom regnskaps- og revisjonsplattform
            </p>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-right">
              <p className="text-sm text-emerald-200">Neste MVA-frist</p>
              <p className="text-2xl font-bold">{daysUntilMVA} dager</p>
            </div>
            <div className="h-12 w-px bg-white/20"></div>
            <div className="text-right">
              <p className="text-sm text-emerald-200">Termin {currentTerm}</p>
              <p className="text-lg font-semibold">{mvaDeadline.toLocaleDateString("nb-NO")}</p>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Stats Cards with Glassmorphism */}
      <motion.div variants={itemVariants} className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        <GlassStatsCard
          title="Ubehandlede dokumenter"
          value={stats?.unpostedDocuments ?? 0}
          icon={FileText}
          description="Venter på behandling"
          isLoading={isLoading}
          href="/accounting"
          trend={stats?.unpostedDocuments && stats.unpostedDocuments > 5 ? "up" : "down"}
          trendValue={stats?.unpostedDocuments && stats.unpostedDocuments > 5 ? "+3 fra i går" : "Ingen nye"}
          color="amber"
        />
        <GlassStatsCard
          title="Ventende innleveringer"
          value={stats?.pendingFilings ?? 0}
          icon={Clock}
          description="Utkast klare for innsending"
          isLoading={isLoading}
          href="/filings"
          trend="neutral"
          trendValue="Stabil"
          color="blue"
        />
        <GlassStatsCard
          title="Høyrisiko-selskaper"
          value={stats?.highRiskCompanies ?? 0}
          icon={AlertTriangle}
          description="Krever oppfølging"
          isLoading={isLoading}
          href="/companies"
          trend={stats?.highRiskCompanies && stats.highRiskCompanies > 0 ? "up" : "down"}
          trendValue={stats?.highRiskCompanies && stats.highRiskCompanies > 0 ? "Krever handling" : "Alt OK"}
          color="red"
        />
        <GlassStatsCard
          title="Aktive selskaper"
          value={companies?.length ?? 0}
          icon={Building2}
          description="Registrerte selskaper"
          isLoading={isLoading}
          href="/companies"
          trend="up"
          trendValue="Voksende"
          color="emerald"
        />
      </motion.div>

      {/* Main Content Grid */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Quick Actions - Premium Card */}
        <motion.div variants={itemVariants}>
          <Card className="h-full border-0 bg-gradient-to-br from-slate-50 to-white dark:from-slate-900 dark:to-slate-800 shadow-lg">
            <CardHeader className="pb-4">
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg flex items-center gap-2">
                  <Zap className="h-5 w-5 text-amber-500" />
                  Hurtighandlinger
                </CardTitle>
                <Badge variant="secondary" className="bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400">
                  4 tilgjengelig
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              <QuickActionButton 
                href="/companies" 
                icon={Building2} 
                label="Legg til nytt selskap"
                description="Registrer med Forvalt-berikelse"
                color="emerald"
              />
              <QuickActionButton 
                href="/accounting" 
                icon={FileText} 
                label="Last opp bilag"
                description="AI-klassifisering og kontering"
                color="blue"
              />
              <QuickActionButton 
                href="/filings" 
                icon={Calendar} 
                label="Generer MVA-melding"
                description="Automatisk beregning"
                color="purple"
              />
              <QuickActionButton 
                href="/chat" 
                icon={Bot} 
                label="Spør AI-assistenten"
                description="Regnskapshjelp 24/7"
                color="teal"
              />
            </CardContent>
          </Card>
        </motion.div>

        {/* Upcoming Deadlines */}
        <motion.div variants={itemVariants}>
          <Card className="h-full border-0 bg-gradient-to-br from-slate-50 to-white dark:from-slate-900 dark:to-slate-800 shadow-lg">
            <CardHeader className="pb-4">
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg flex items-center gap-2">
                  <Calendar className="h-5 w-5 text-blue-500" />
                  Kommende frister
                </CardTitle>
                <Badge variant="secondary" className="bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400">
                  3 aktive
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <DeadlineItem 
                title={`MVA-melding termin ${currentTerm}`}
                date={mvaDeadline}
                daysLeft={daysUntilMVA}
                status={daysUntilMVA > 7 ? "ok" : daysUntilMVA > 3 ? "warning" : "urgent"}
              />
              <DeadlineItem 
                title="A-melding"
                date={new Date(today.getFullYear(), today.getMonth() + 1, 5)}
                daysLeft={Math.ceil((new Date(today.getFullYear(), today.getMonth() + 1, 5).getTime() - today.getTime()) / (1000 * 60 * 60 * 24))}
                status="ok"
              />
              <DeadlineItem 
                title="Årsregnskap"
                date={new Date(today.getFullYear(), 5, 30)}
                daysLeft={Math.ceil((new Date(today.getFullYear() + (today.getMonth() > 5 ? 1 : 0), 5, 30).getTime() - today.getTime()) / (1000 * 60 * 60 * 24))}
                status="ok"
              />
            </CardContent>
          </Card>
        </motion.div>

        {/* System Status */}
        <motion.div variants={itemVariants}>
          <Card className="h-full border-0 bg-gradient-to-br from-slate-50 to-white dark:from-slate-900 dark:to-slate-800 shadow-lg">
            <CardHeader className="pb-4">
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg flex items-center gap-2">
                  <Shield className="h-5 w-5 text-emerald-500" />
                  Systemstatus
                </CardTitle>
                <Badge className="bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400">
                  Operativ
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <SystemStatusItem label="AI-agent" status="online" />
              <SystemStatusItem label="Forvalt API" status="online" />
              <SystemStatusItem label="Altinn integrasjon" status="online" />
              <SystemStatusItem label="Regnskapssystem" status="online" />
              <div className="pt-2 border-t border-slate-200 dark:border-slate-700">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-slate-500">Siste synkronisering</span>
                  <span className="font-medium">{new Date().toLocaleTimeString("nb-NO")}</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* Recent Activity & Performance */}
      <motion.div variants={itemVariants} className="grid gap-6 lg:grid-cols-2">
        {/* Recent Documents */}
        <Card className="border-0 bg-gradient-to-br from-slate-50 to-white dark:from-slate-900 dark:to-slate-800 shadow-lg">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-lg flex items-center gap-2">
                <FileText className="h-5 w-5 text-slate-500" />
                Nylige dokumenter
              </CardTitle>
              <Link href="/accounting">
                <Button variant="ghost" size="sm" className="gap-1">
                  Se alle <ArrowRight className="h-4 w-4" />
                </Button>
              </Link>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {isLoading ? (
                <>
                  <Skeleton className="h-12 w-full" />
                  <Skeleton className="h-12 w-full" />
                  <Skeleton className="h-12 w-full" />
                </>
              ) : (
                <>
                  <RecentDocumentItem 
                    filename="Faktura-2024-001.pdf"
                    type="INVOICE_SUPPLIER"
                    status="PROCESSED"
                    date={new Date()}
                  />
                  <RecentDocumentItem 
                    filename="Kvittering-Staples.pdf"
                    type="RECEIPT"
                    status="NEW"
                    date={new Date(Date.now() - 86400000)}
                  />
                  <RecentDocumentItem 
                    filename="Kontrakt-Leverandør.pdf"
                    type="CONTRACT"
                    status="POSTED"
                    date={new Date(Date.now() - 172800000)}
                  />
                </>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Performance Metrics */}
        <Card className="border-0 bg-gradient-to-br from-slate-50 to-white dark:from-slate-900 dark:to-slate-800 shadow-lg">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-lg flex items-center gap-2">
                <BarChart3 className="h-5 w-5 text-slate-500" />
                Ytelsesmålinger
              </CardTitle>
              <Badge variant="outline">Denne måneden</Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-6">
            <PerformanceMetric 
              label="Dokumenter behandlet"
              value={42}
              max={50}
              unit="stk"
            />
            <PerformanceMetric 
              label="AI-klassifiseringsnøyaktighet"
              value={94}
              max={100}
              unit="%"
            />
            <PerformanceMetric 
              label="Gjennomsnittlig behandlingstid"
              value={2.3}
              max={5}
              unit="sek"
              inverted
            />
          </CardContent>
        </Card>
      </motion.div>
    </motion.div>
  );
}

// ============================================================================
// COMPONENT DEFINITIONS
// ============================================================================

interface GlassStatsCardProps {
  title: string;
  value: number;
  icon: React.ComponentType<{ className?: string }>;
  description: string;
  isLoading: boolean;
  href: string;
  trend: "up" | "down" | "neutral";
  trendValue: string;
  color: "emerald" | "blue" | "amber" | "red" | "purple" | "teal";
}

function GlassStatsCard({ title, value, icon: Icon, description, isLoading, href, trend, trendValue, color }: GlassStatsCardProps) {
  const colorClasses = {
    emerald: "from-emerald-500/10 to-emerald-500/5 border-emerald-200 dark:border-emerald-800",
    blue: "from-blue-500/10 to-blue-500/5 border-blue-200 dark:border-blue-800",
    amber: "from-amber-500/10 to-amber-500/5 border-amber-200 dark:border-amber-800",
    red: "from-red-500/10 to-red-500/5 border-red-200 dark:border-red-800",
    purple: "from-purple-500/10 to-purple-500/5 border-purple-200 dark:border-purple-800",
    teal: "from-teal-500/10 to-teal-500/5 border-teal-200 dark:border-teal-800",
  };

  const iconColorClasses = {
    emerald: "text-emerald-600 bg-emerald-100 dark:bg-emerald-900/50",
    blue: "text-blue-600 bg-blue-100 dark:bg-blue-900/50",
    amber: "text-amber-600 bg-amber-100 dark:bg-amber-900/50",
    red: "text-red-600 bg-red-100 dark:bg-red-900/50",
    purple: "text-purple-600 bg-purple-100 dark:bg-purple-900/50",
    teal: "text-teal-600 bg-teal-100 dark:bg-teal-900/50",
  };

  return (
    <Link href={href}>
      <Card className={`group cursor-pointer border bg-gradient-to-br ${colorClasses[color]} backdrop-blur-sm transition-all duration-300 hover:shadow-lg hover:scale-[1.02]`}>
        <CardContent className="p-6">
          <div className="flex items-start justify-between">
            <div className={`rounded-xl p-3 ${iconColorClasses[color]}`}>
              <Icon className="h-6 w-6" />
            </div>
            <div className="flex items-center gap-1 text-sm">
              {trend === "up" && <TrendingUp className="h-4 w-4 text-red-500" />}
              {trend === "down" && <TrendingDown className="h-4 w-4 text-emerald-500" />}
              <span className={trend === "up" ? "text-red-600" : trend === "down" ? "text-emerald-600" : "text-slate-500"}>
                {trendValue}
              </span>
            </div>
          </div>
          <div className="mt-4">
            {isLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <p className="text-3xl font-bold text-slate-900 dark:text-white">{value}</p>
            )}
            <p className="mt-1 text-sm font-medium text-slate-700 dark:text-slate-300">{title}</p>
            <p className="text-xs text-slate-500 dark:text-slate-400">{description}</p>
          </div>
          <div className="mt-4 flex items-center text-sm text-slate-500 group-hover:text-slate-700 dark:group-hover:text-slate-300 transition-colors">
            Se detaljer <ArrowRight className="ml-1 h-4 w-4 transition-transform group-hover:translate-x-1" />
          </div>
        </CardContent>
      </Card>
    </Link>
  );
}

interface QuickActionButtonProps {
  href: string;
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  description: string;
  color: "emerald" | "blue" | "purple" | "teal";
}

function QuickActionButton({ href, icon: Icon, label, description, color }: QuickActionButtonProps) {
  const colorClasses = {
    emerald: "hover:bg-emerald-50 hover:border-emerald-200 dark:hover:bg-emerald-900/20",
    blue: "hover:bg-blue-50 hover:border-blue-200 dark:hover:bg-blue-900/20",
    purple: "hover:bg-purple-50 hover:border-purple-200 dark:hover:bg-purple-900/20",
    teal: "hover:bg-teal-50 hover:border-teal-200 dark:hover:bg-teal-900/20",
  };

  const iconColorClasses = {
    emerald: "text-emerald-600",
    blue: "text-blue-600",
    purple: "text-purple-600",
    teal: "text-teal-600",
  };

  return (
    <Link href={href}>
      <div className={`group flex items-center gap-4 rounded-xl border border-slate-200 dark:border-slate-700 p-4 transition-all duration-200 ${colorClasses[color]} cursor-pointer`}>
        <div className={`rounded-lg bg-slate-100 dark:bg-slate-800 p-2.5 ${iconColorClasses[color]}`}>
          <Icon className="h-5 w-5" />
        </div>
        <div className="flex-1">
          <p className="font-medium text-slate-900 dark:text-white">{label}</p>
          <p className="text-xs text-slate-500">{description}</p>
        </div>
        <ArrowRight className="h-4 w-4 text-slate-400 transition-transform group-hover:translate-x-1" />
      </div>
    </Link>
  );
}

interface DeadlineItemProps {
  title: string;
  date: Date;
  daysLeft: number;
  status: "ok" | "warning" | "urgent";
}

function DeadlineItem({ title, date, daysLeft, status }: DeadlineItemProps) {
  const statusClasses = {
    ok: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400",
    warning: "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400",
    urgent: "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400",
  };

  return (
    <div className="flex items-center justify-between rounded-lg border border-slate-200 dark:border-slate-700 p-3">
      <div>
        <p className="font-medium text-slate-900 dark:text-white">{title}</p>
        <p className="text-sm text-slate-500">{date.toLocaleDateString("nb-NO")}</p>
      </div>
      <Badge className={statusClasses[status]}>
        {daysLeft} dager
      </Badge>
    </div>
  );
}

interface SystemStatusItemProps {
  label: string;
  status: "online" | "offline" | "degraded";
}

function SystemStatusItem({ label, status }: SystemStatusItemProps) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-sm text-slate-600 dark:text-slate-400">{label}</span>
      <div className="flex items-center gap-2">
        {status === "online" && (
          <>
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
            </span>
            <span className="text-sm font-medium text-emerald-600">Online</span>
          </>
        )}
        {status === "offline" && (
          <>
            <XCircle className="h-4 w-4 text-red-500" />
            <span className="text-sm font-medium text-red-600">Offline</span>
          </>
        )}
        {status === "degraded" && (
          <>
            <AlertTriangle className="h-4 w-4 text-amber-500" />
            <span className="text-sm font-medium text-amber-600">Degradert</span>
          </>
        )}
      </div>
    </div>
  );
}

interface RecentDocumentItemProps {
  filename: string;
  type: string;
  status: string;
  date: Date;
}

function RecentDocumentItem({ filename, type, status, date }: RecentDocumentItemProps) {
  const statusIcons = {
    NEW: <Clock className="h-4 w-4 text-slate-400" />,
    PROCESSED: <CheckCircle2 className="h-4 w-4 text-blue-500" />,
    POSTED: <CheckCircle2 className="h-4 w-4 text-emerald-500" />,
    REJECTED: <XCircle className="h-4 w-4 text-red-500" />,
  };

  const typeLabels: Record<string, string> = {
    INVOICE_SUPPLIER: "Leverandørfaktura",
    INVOICE_CUSTOMER: "Kundefaktura",
    RECEIPT: "Kvittering",
    CONTRACT: "Kontrakt",
    OTHER: "Annet",
  };

  return (
    <div className="flex items-center gap-4 rounded-lg border border-slate-200 dark:border-slate-700 p-3">
      <div className="rounded-lg bg-slate-100 dark:bg-slate-800 p-2">
        <FileText className="h-5 w-5 text-slate-500" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="font-medium text-slate-900 dark:text-white truncate">{filename}</p>
        <p className="text-xs text-slate-500">{typeLabels[type] || type}</p>
      </div>
      <div className="flex items-center gap-2">
        {statusIcons[status as keyof typeof statusIcons]}
        <span className="text-xs text-slate-500">{date.toLocaleDateString("nb-NO")}</span>
      </div>
    </div>
  );
}

interface PerformanceMetricProps {
  label: string;
  value: number;
  max: number;
  unit: string;
  inverted?: boolean;
}

function PerformanceMetric({ label, value, max, unit, inverted }: PerformanceMetricProps) {
  const percentage = (value / max) * 100;
  const isGood = inverted ? percentage < 50 : percentage > 70;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm">
        <span className="text-slate-600 dark:text-slate-400">{label}</span>
        <span className="font-medium text-slate-900 dark:text-white">
          {value} {unit}
        </span>
      </div>
      <Progress 
        value={percentage} 
        className={`h-2 ${isGood ? "[&>div]:bg-emerald-500" : "[&>div]:bg-amber-500"}`}
      />
    </div>
  );
}
