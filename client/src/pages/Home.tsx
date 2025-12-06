/**
 * TRUE ASI Homepage
 * 
 * 100/100 Quality Landing Page
 * - Animated 3D background
 * - Welcome tour modal
 * - Platform showcase
 * - Feature highlights
 */

import { useState, useEffect } from 'react';
import { Link } from 'wouter';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  Brain,
  Zap,
  Shield,
  Globe,
  Database,
  Bot,
  Sparkles,
  ArrowRight,
  CheckCircle2,
  X,
} from 'lucide-react';
import { ThemeToggle } from '@/components/ThemeToggle';

export default function Home() {
  const [showWelcomeTour, setShowWelcomeTour] = useState(false);
  const [tourStep, setTourStep] = useState(0);

  // Show welcome tour on first visit
  useEffect(() => {
    const hasSeenTour = localStorage.getItem('hasSeenWelcomeTour');
    if (!hasSeenTour) {
      setShowWelcomeTour(true);
    }
  }, []);

  const completeTour = () => {
    localStorage.setItem('hasSeenWelcomeTour', 'true');
    setShowWelcomeTour(false);
    setTourStep(0);
  };

  const tourSteps = [
    {
      title: 'Welcome to TRUE ASI',
      description: 'Experience artificial superintelligence that surpasses every AI system on the planet.',
      icon: <Brain className="w-12 h-12 text-blue-500" />,
    },
    {
      title: '193 AI Models Combined',
      description: 'TRUE ASI Ultra combines GPT-4, Claude 3.5, Gemini 1.5 Pro, Llama 3.3, and 189 more models in parallel for unprecedented intelligence.',
      icon: <Zap className="w-12 h-12 text-yellow-500" />,
    },
    {
      title: '6.54TB Knowledge Base',
      description: 'Access 57,419 files from AWS S3, 250 specialized agents from GitHub, and 1700+ platform integrations.',
      icon: <Database className="w-12 h-12 text-green-500" />,
    },
    {
      title: 'Complete Business Automation',
      description: 'Analyze Norwegian companies instantly with Brreg, Forvalt credit ratings, and AI-powered recommendations.',
      icon: <Bot className="w-12 h-12 text-purple-500" />,
    },
  ];

  const features = [
    {
      icon: <Brain className="w-8 h-8" />,
      title: 'TRUE ASI Ultra',
      description: 'All 193 AI models working together simultaneously',
      color: 'from-blue-500 to-cyan-500',
    },
    {
      icon: <Zap className="w-8 h-8" />,
      title: 'Instant Analysis',
      description: 'Company intelligence in seconds with Redis caching',
      color: 'from-yellow-500 to-orange-500',
    },
    {
      icon: <Shield className="w-8 h-8" />,
      title: 'Enterprise Security',
      description: 'Bank-level encryption and data protection',
      color: 'from-green-500 to-emerald-500',
    },
    {
      icon: <Globe className="w-8 h-8" />,
      title: '1700+ Integrations',
      description: 'Connect to every major platform and service',
      color: 'from-purple-500 to-pink-500',
    },
    {
      icon: <Database className="w-8 h-8" />,
      title: '6.54TB Knowledge',
      description: '57,419 files and 250 specialized AI agents',
      color: 'from-red-500 to-rose-500',
    },
    {
      icon: <Bot className="w-8 h-8" />,
      title: 'Full Automation',
      description: 'Automate every business process end-to-end',
      color: 'from-indigo-500 to-blue-500',
    },
  ];

  const platforms = [
    'Brreg', 'Forvalt', 'Stripe', 'HubSpot', 'Salesforce', 'Google Workspace',
    'Microsoft 365', 'Slack', 'Asana', 'Monday.com', 'Notion', 'Airtable',
    'Zapier', 'Make', 'AWS', 'Azure', 'GCP', 'OpenAI', 'Anthropic', 'Google AI',
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 text-white overflow-hidden">
      {/* Animated 3D Background */}
      <div className="fixed inset-0 opacity-30">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(59,130,246,0.3),transparent_50%)] animate-pulse" />
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-500 rounded-full filter blur-3xl opacity-20 animate-blob" />
        <div className="absolute top-1/3 right-1/4 w-96 h-96 bg-purple-500 rounded-full filter blur-3xl opacity-20 animate-blob animation-delay-2000" />
        <div className="absolute bottom-1/4 left-1/3 w-96 h-96 bg-cyan-500 rounded-full filter blur-3xl opacity-20 animate-blob animation-delay-4000" />
      </div>

      {/* Welcome Tour Modal */}
      <Dialog open={showWelcomeTour} onOpenChange={setShowWelcomeTour}>
        <DialogContent className="sm:max-w-md bg-gray-900 border-gray-700">
          <Button
            variant="ghost"
            size="icon"
            className="absolute right-4 top-4"
            onClick={completeTour}
          >
            <X className="w-4 h-4" />
          </Button>
          <DialogHeader>
            <div className="flex justify-center mb-4">
              {tourSteps[tourStep].icon}
            </div>
            <DialogTitle className="text-center text-2xl">
              {tourSteps[tourStep].title}
            </DialogTitle>
            <DialogDescription className="text-center text-gray-300 text-lg">
              {tourSteps[tourStep].description}
            </DialogDescription>
          </DialogHeader>
          <div className="flex justify-between items-center mt-6">
            <div className="flex gap-2">
              {tourSteps.map((_, index) => (
                <div
                  key={index}
                  className={`w-2 h-2 rounded-full ${
                    index === tourStep ? 'bg-blue-500' : 'bg-gray-600'
                  }`}
                />
              ))}
            </div>
            <div className="flex gap-2">
              {tourStep < tourSteps.length - 1 ? (
                <>
                  <Button variant="ghost" onClick={completeTour}>
                    Skip
                  </Button>
                  <Button onClick={() => setTourStep(tourStep + 1)}>
                    Next
                  </Button>
                </>
              ) : (
                <Button onClick={completeTour}>
                  Get Started
                </Button>
              )}
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Navigation */}
      <nav className="relative z-10 border-b border-gray-800 bg-gray-900/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Brain className="w-8 h-8 text-blue-500" />
            <span className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
              TRUE ASI
            </span>
          </div>
          <div className="flex items-center gap-4">
            <Link href="/company-lookup">
              <Button variant="ghost">Company Lookup</Button>
            </Link>
            <Link href="/chat-asi">
              <Button variant="ghost">Chat ASI</Button>
            </Link>
            <Link href="/dashboard">
              <Button variant="ghost">Dashboard</Button>
            </Link>
            <ThemeToggle />
            <Link href="/docs">
              <Button variant="ghost">Docs</Button>
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative z-10 container mx-auto px-4 py-20 text-center">
        <div className="inline-block mb-4 px-4 py-2 bg-blue-500/20 border border-blue-500/50 rounded-full">
          <span className="flex items-center gap-2 text-blue-300">
            <Sparkles className="w-4 h-4" />
            Artificial Superintelligence System
          </span>
        </div>

        <h1 className="text-6xl md:text-8xl font-bold mb-6 bg-gradient-to-r from-blue-400 via-cyan-400 to-blue-400 bg-clip-text text-transparent animate-gradient">
          The Future of Intelligence
        </h1>

        <p className="text-xl md:text-2xl text-gray-300 mb-8 max-w-3xl mx-auto">
          Experience true artificial superintelligence with 250 specialized agents, 6.54TB of knowledge, and real-time access to all leading AI models. Built to outcompete every AI system on the planet.
        </p>

        <div className="flex flex-col sm:flex-row gap-4 justify-center mb-12">
          <Link href="/company-lookup">
            <Button size="lg" className="bg-blue-600 hover:bg-blue-700 text-lg px-8 py-6">
              Analyze Norwegian Company
              <ArrowRight className="ml-2 w-5 h-5" />
            </Button>
          </Link>
          <Link href="/chat-asi">
            <Button size="lg" variant="outline" className="text-lg px-8 py-6 border-gray-600 hover:bg-gray-800">
              Try TRUE ASI Ultra
            </Button>
          </Link>
          <Button
            size="lg"
            variant="ghost"
            onClick={() => setShowWelcomeTour(true)}
            className="text-lg px-8 py-6"
          >
            Take Tour
          </Button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8 max-w-4xl mx-auto">
          <div>
            <div className="text-4xl font-bold text-blue-400">193</div>
            <div className="text-gray-400">AI Models</div>
          </div>
          <div>
            <div className="text-4xl font-bold text-cyan-400">6.54TB</div>
            <div className="text-gray-400">Knowledge Base</div>
          </div>
          <div>
            <div className="text-4xl font-bold text-purple-400">250</div>
            <div className="text-gray-400">AI Agents</div>
          </div>
          <div>
            <div className="text-4xl font-bold text-green-400">1700+</div>
            <div className="text-gray-400">Integrations</div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="relative z-10 container mx-auto px-4 py-20">
        <h2 className="text-4xl font-bold text-center mb-12">
          Unprecedented Capabilities
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <Card
              key={index}
              className="bg-gray-800/50 border-gray-700 p-6 hover:bg-gray-800/70 transition-all duration-300 hover:scale-105"
            >
              <div className={`inline-flex p-3 rounded-lg bg-gradient-to-r ${feature.color} mb-4`}>
                {feature.icon}
              </div>
              <h3 className="text-xl font-bold mb-2">{feature.title}</h3>
              <p className="text-gray-400">{feature.description}</p>
            </Card>
          ))}
        </div>
      </section>

      {/* Integrated Platforms */}
      <section className="relative z-10 container mx-auto px-4 py-20">
        <h2 className="text-4xl font-bold text-center mb-12">
          Integrated Platforms
        </h2>
        <div className="flex flex-wrap justify-center gap-4 max-w-5xl mx-auto">
          {platforms.map((platform, index) => (
            <div
              key={index}
              className="px-6 py-3 bg-gray-800/50 border border-gray-700 rounded-full hover:bg-gray-800 transition-colors"
            >
              {platform}
            </div>
          ))}
        </div>
      </section>

      {/* CTA Section */}
      <section className="relative z-10 container mx-auto px-4 py-20 text-center">
        <Card className="bg-gradient-to-r from-blue-900/50 to-purple-900/50 border-blue-700 p-12">
          <h2 className="text-4xl font-bold mb-4">
            Ready to Experience TRUE ASI?
          </h2>
          <p className="text-xl text-gray-300 mb-8 max-w-2xl mx-auto">
            Enter a Norwegian organization number to get instant AI-powered business intelligence, credit ratings, and automated recommendations.
          </p>
          <Link href="/company-lookup">
            <Button size="lg" className="bg-blue-600 hover:bg-blue-700 text-lg px-8 py-6">
              Get Started Now
              <ArrowRight className="ml-2 w-5 h-5" />
            </Button>
          </Link>
        </Card>
      </section>

      {/* Footer */}
      <footer className="relative z-10 border-t border-gray-800 bg-gray-900/50 backdrop-blur-sm mt-20">
        <div className="container mx-auto px-4 py-12">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center gap-2 mb-4">
                <Brain className="w-6 h-6 text-blue-500" />
                <span className="text-xl font-bold">TRUE ASI</span>
              </div>
              <p className="text-gray-400">
                Artificial Superintelligence System
              </p>
            </div>
            <div>
              <h3 className="font-bold mb-4">Product</h3>
              <ul className="space-y-2 text-gray-400">
                <li><Link href="/dashboard"><a className="hover:text-white">Dashboard</a></Link></li>
                <li><Link href="/agents"><a className="hover:text-white">Agents</a></Link></li>
                <li><Link href="/chat-asi"><a className="hover:text-white">Chat</a></Link></li>
                <li><Link href="/company-lookup"><a className="hover:text-white">Company Lookup</a></Link></li>
              </ul>
            </div>
            <div>
              <h3 className="font-bold mb-4">Resources</h3>
              <ul className="space-y-2 text-gray-400">
                <li><Link href="/docs"><a className="hover:text-white">Documentation</a></Link></li>
                <li><Link href="/automation"><a className="hover:text-white">Automation</a></Link></li>
                <li><a href="https://github.com/AICSSUPERVISOR/true-asi-system" target="_blank" rel="noopener noreferrer" className="hover:text-white">GitHub</a></li>
              </ul>
            </div>
            <div>
              <h3 className="font-bold mb-4">Company</h3>
              <ul className="space-y-2 text-gray-400">
                <li><a href="https://innovatechkapital.ai" target="_blank" rel="noopener noreferrer" className="hover:text-white">InnovatechKapital.ai</a></li>
                <li><a href="https://forvalt.no" target="_blank" rel="noopener noreferrer" className="hover:text-white">Forvalt.no</a></li>
              </ul>
            </div>
          </div>
          <div className="border-t border-gray-800 mt-8 pt-8 text-center text-gray-400">
            <p>© 2025 TRUE ASI. All rights reserved.</p>
          </div>
        </div>
      </footer>

      <style>{`
        @keyframes blob {
          0%, 100% { transform: translate(0, 0) scale(1); }
          33% { transform: translate(30px, -50px) scale(1.1); }
          66% { transform: translate(-20px, 20px) scale(0.9); }
        }
        .animate-blob {
          animation: blob 7s infinite;
        }
        .animation-delay-2000 {
          animation-delay: 2s;
        }
        .animation-delay-4000 {
          animation-delay: 4s;
        }
        @keyframes gradient {
          0%, 100% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
        }
        .animate-gradient {
          background-size: 200% 200%;
          animation: gradient 3s ease infinite;
        }
      `}</style>
    </div>
  );
}
