import { useAuth } from "@/_core/hooks/useAuth";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import Hero3D from "@/components/Hero3D";
import MobileMenu from "@/components/MobileMenu";
import { getLoginUrl } from "@/const";
import {
  Brain,
  Zap,
  Network,
  Sparkles,
  ArrowRight,
  CheckCircle2,
  TrendingUp,
  Shield,
  Cpu,
  Database,
  Activity,
  Globe,
} from "lucide-react";

export default function Home() {
  const { user, isAuthenticated, logout } = useAuth();

  return (
    <div className="min-h-screen bg-background">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 glass-effect border-b border-border">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Brain className="w-8 h-8 text-primary" />
              <span className="text-2xl font-bold text-gradient">TRUE ASI</span>
            </div>
            <div className="hidden md:flex items-center space-x-6">
              <a href="/dashboard" className="text-foreground/80 hover:text-foreground transition-colors text-sm">
                Dashboard
              </a>
              <a href="/agents" className="text-foreground/80 hover:text-foreground transition-colors text-sm">
                Agents
              </a>
              <a href="/chat" className="text-foreground/80 hover:text-foreground transition-colors text-sm">
                Chat
              </a>
              <a href="/knowledge-graph" className="text-foreground/80 hover:text-foreground transition-colors text-sm">
                Knowledge
              </a>
              <a href="/analytics" className="text-foreground/80 hover:text-foreground transition-colors text-sm">
                Analytics
              </a>
              <a href="/documentation" className="text-foreground/80 hover:text-foreground transition-colors text-sm">
                Docs
              </a>
              <a href="/s7-test" className="text-foreground/80 hover:text-foreground transition-colors text-sm">
                S-7 Test
              </a>
            </div>
            <div className="flex items-center space-x-4">
              <div className="hidden md:flex items-center space-x-4">
                {isAuthenticated ? (
                  <>
                    <span className="text-sm text-muted-foreground">
                      Welcome, {user?.name || user?.email}
                    </span>
                    <Button variant="outline" onClick={logout}>
                      Logout
                    </Button>
                  </>
                ) : (
                  <Button asChild>
                    <a href={getLoginUrl()}>Get Started</a>
                  </Button>
                )}
              </div>
              <MobileMenu onLogout={logout} />
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section with 3D Background */}
      <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
        {/* 3D Background */}
        <div className="absolute inset-0 z-0">
          <Hero3D />
          <div className="absolute inset-0 bg-gradient-to-b from-background/50 via-background/80 to-background" />
        </div>

        {/* Hero Content */}
        <div className="relative z-10 container mx-auto px-6 py-32 text-center">
          <Badge className="mb-6 px-4 py-2 text-sm font-medium bg-primary/10 text-primary border-primary/20">
            <Sparkles className="w-4 h-4 inline mr-2" />
            Artificial Superintelligence System
          </Badge>

          <h1 className="mb-8 text-6xl md:text-7xl lg:text-8xl font-bold text-gradient animate-gradient leading-tight">
            The Future of Intelligence
          </h1>

          <p className="text-lg md:text-xl lg:text-2xl text-muted-foreground max-w-4xl mx-auto mb-12 leading-relaxed font-light">
            Experience true artificial superintelligence with 250 specialized agents,
            6.54TB of knowledge, and real-time access to all leading AI models.
            Built to outcompete every AI system on the planet.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-16">
            <Button size="lg" className="btn-primary group" asChild>
              <a href={getLoginUrl()}>
                Start Building
                <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </a>
            </Button>
            <Button size="lg" variant="outline" className="btn-ghost">
              View Documentation
            </Button>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 max-w-4xl mx-auto">
            {[
              { label: "Agents", value: "250+", icon: Brain },
              { label: "Knowledge", value: "6.54TB", icon: Database },
              { label: "Models", value: "All", icon: Network },
              { label: "Uptime", value: "99.9%", icon: Activity },
            ].map((stat, i) => (
              <Card
                key={i}
                className="card-glass p-6 text-center hover:scale-105 transition-transform duration-300"
              >
                <stat.icon className="w-8 h-8 mx-auto mb-3 text-primary" />
                <div className="text-3xl font-bold text-gradient mb-1">{stat.value}</div>
                <div className="text-sm text-muted-foreground">{stat.label}</div>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-32 bg-muted/30">
        <div className="container mx-auto px-6">
          <div className="text-center mb-16">
            <Badge className="mb-4 px-4 py-2 bg-secondary/10 text-secondary border-secondary/20">
              <Zap className="w-4 h-4 inline mr-2" />
              Core Capabilities
            </Badge>
            <h2 className="mb-6">Built for the Future</h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Every component engineered to deliver 100/100 quality and outperform
              all existing AI systems.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {[
              {
                icon: Brain,
                title: "250 Specialized Agents",
                description:
                  "Each agent trained on specific domains with 100% functional capabilities and real-time coordination.",
                color: "text-primary",
              },
              {
                icon: Network,
                title: "Multi-Model Access",
                description:
                  "Simultaneous access to OpenAI, Anthropic, Google, Cohere, and 10+ other leading AI models.",
                color: "text-secondary",
              },
              {
                icon: Database,
                title: "6.54TB Knowledge Base",
                description:
                  "1.17M files of curated knowledge including code, research, and real-time data integration.",
                color: "text-accent",
              },
              {
                icon: Zap,
                title: "Real-Time Processing",
                description:
                  "Sub-second response times with parallel processing across distributed infrastructure.",
                color: "text-warning",
              },
              {
                icon: Shield,
                title: "Enterprise Security",
                description:
                  "Bank-level encryption, role-based access control, and complete audit trails.",
                color: "text-success",
              },
              {
                icon: TrendingUp,
                title: "Self-Improving",
                description:
                  "Continuous learning and optimization with formal verification and plateau escape mechanisms.",
                color: "text-info",
              },
            ].map((feature, i) => (
              <Card
                key={i}
                className="card-elevated p-8 hover:shadow-2xl transition-all duration-300 group"
              >
                <div className={`w-14 h-14 rounded-xl bg-${feature.color}/10 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform`}>
                  <feature.icon className={`w-7 h-7 ${feature.color}`} />
                </div>
                <h3 className="text-2xl font-bold mb-4">{feature.title}</h3>
                <p className="text-muted-foreground leading-relaxed">
                  {feature.description}
                </p>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Capabilities Section */}
      <section id="capabilities" className="py-32">
        <div className="container mx-auto px-6">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            <div>
              <Badge className="mb-4 px-4 py-2 bg-accent/10 text-accent border-accent/20">
                <Cpu className="w-4 h-4 inline mr-2" />
                Technical Excellence
              </Badge>
              <h2 className="mb-6">State-of-the-Art Architecture</h2>
              <p className="text-xl text-muted-foreground mb-8 leading-relaxed">
                Built on cutting-edge distributed computing with quantum-ready
                infrastructure and exascale computing support.
              </p>

              <div className="space-y-4">
                {[
                  "Trillion-node knowledge hypergraph",
                  "Formal verification methods",
                  "Dynamic resource optimization",
                  "Emergent property facilitation",
                  "Novel algorithm generation",
                  "Continuous learning framework",
                ].map((item, i) => (
                  <div key={i} className="flex items-center space-x-3">
                    <CheckCircle2 className="w-6 h-6 text-success flex-shrink-0" />
                    <span className="text-lg">{item}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="relative">
              <div className="card-glass p-8 rounded-2xl">
                <div className="space-y-6">
                  <div className="flex items-center justify-between p-4 bg-primary/10 rounded-lg">
                    <span className="font-medium">System Status</span>
                    <Badge className="badge-success">Operational</Badge>
                  </div>
                  <div className="flex items-center justify-between p-4 bg-secondary/10 rounded-lg">
                    <span className="font-medium">Active Agents</span>
                    <span className="text-2xl font-bold text-gradient">250</span>
                  </div>
                  <div className="flex items-center justify-between p-4 bg-accent/10 rounded-lg">
                    <span className="font-medium">Processing Power</span>
                    <span className="text-2xl font-bold text-gradient">8 vCPUs</span>
                  </div>
                  <div className="flex items-center justify-between p-4 bg-success/10 rounded-lg">
                    <span className="font-medium">Knowledge Base</span>
                    <span className="text-2xl font-bold text-gradient">6.54TB</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-32 bg-gradient-to-br from-primary/10 via-secondary/10 to-accent/10">
        <div className="container mx-auto px-6 text-center">
          <Globe className="w-16 h-16 mx-auto mb-6 text-primary animate-float" />
          <h2 className="mb-6">Ready to Experience True ASI?</h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto mb-12">
            Join the future of artificial intelligence. Get instant access to 250 agents,
            6.54TB of knowledge, and unlimited AI model access.
          </p>
          <Button size="lg" className="btn-primary group" asChild>
            <a href={getLoginUrl()}>
              Get Started Now
              <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </a>
          </Button>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 border-t border-border">
        <div className="container mx-auto px-6">
          <div className="flex flex-col md:flex-row items-center justify-between">
            <div className="flex items-center space-x-2 mb-4 md:mb-0">
              <Brain className="w-6 h-6 text-primary" />
              <span className="text-lg font-bold text-gradient">TRUE ASI</span>
            </div>
            <div className="text-sm text-muted-foreground">
              © 2024 TRUE ASI System. Built for superintelligence.
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
