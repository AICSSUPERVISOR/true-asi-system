import { Brain, Github, Twitter, Linkedin, Mail } from "lucide-react";
import { Link } from "wouter";

export default function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-slate-900 border-t border-slate-800 mt-auto">
      <div className="container mx-auto px-4 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-8">
          {/* Brand section */}
          <div className="col-span-1">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-10 h-10 bg-gradient-to-br from-cyan-500 to-purple-600 rounded-lg flex items-center justify-center">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <span className="text-xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                TRUE ASI
              </span>
            </div>
            <p className="text-slate-400 text-sm mb-4">
              Artificial Superintelligence System built to outcompete every AI system on the planet.
            </p>
            <div className="flex gap-3">
              <a
                href="https://github.com/AICSSUPERVISOR"
                target="_blank"
                rel="noopener noreferrer"
                className="w-9 h-9 bg-slate-800 hover:bg-slate-700 rounded-lg flex items-center justify-center transition-colors"
              >
                <Github className="w-5 h-5 text-slate-400" />
              </a>
              <a
                href="https://twitter.com/trueasi"
                target="_blank"
                rel="noopener noreferrer"
                className="w-9 h-9 bg-slate-800 hover:bg-slate-700 rounded-lg flex items-center justify-center transition-colors"
              >
                <Twitter className="w-5 h-5 text-slate-400" />
              </a>
              <a
                href="https://linkedin.com/company/trueasi"
                target="_blank"
                rel="noopener noreferrer"
                className="w-9 h-9 bg-slate-800 hover:bg-slate-700 rounded-lg flex items-center justify-center transition-colors"
              >
                <Linkedin className="w-5 h-5 text-slate-400" />
              </a>
              <a
                href="mailto:contact@trueasi.com"
                className="w-9 h-9 bg-slate-800 hover:bg-slate-700 rounded-lg flex items-center justify-center transition-colors"
              >
                <Mail className="w-5 h-5 text-slate-400" />
              </a>
            </div>
          </div>

          {/* Product links */}
          <div>
            <h3 className="text-white font-semibold mb-4">Product</h3>
            <ul className="space-y-2">
              <li>
                <Link href="/" className="text-slate-400 hover:text-cyan-400 text-sm transition-colors">
                  Dashboard
                </Link>
              </li>
              <li>
                <Link href="/agents" className="text-slate-400 hover:text-cyan-400 text-sm transition-colors">
                  AI Agents
                </Link>
              </li>
              <li>
                <Link href="/knowledge" className="text-slate-400 hover:text-cyan-400 text-sm transition-colors">
                  Knowledge Base
                </Link>
              </li>
              <li>
                <Link href="/s7-test" className="text-slate-400 hover:text-cyan-400 text-sm transition-colors">
                  S-7 Test
                </Link>
              </li>
              <li>
                <Link href="/agent-orchestrator" className="text-slate-400 hover:text-cyan-400 text-sm transition-colors">
                  Agent Orchestrator
                </Link>
              </li>
            </ul>
          </div>

          {/* Resources links */}
          <div>
            <h3 className="text-white font-semibold mb-4">Resources</h3>
            <ul className="space-y-2">
              <li>
                <Link href="/docs" className="text-slate-400 hover:text-cyan-400 text-sm transition-colors">
                  Documentation
                </Link>
              </li>
              <li>
                <Link href="/unified-analytics" className="text-slate-400 hover:text-cyan-400 text-sm transition-colors">
                  Analytics
                </Link>
              </li>
              <li>
                <Link href="/agent-analytics" className="text-slate-400 hover:text-cyan-400 text-sm transition-colors">
                  Agent Performance
                </Link>
              </li>
              <li>
                <Link href="/s7-leaderboard" className="text-slate-400 hover:text-cyan-400 text-sm transition-colors">
                  S-7 Leaderboard
                </Link>
              </li>
              <li>
                <Link href="/s7-study-path" className="text-slate-400 hover:text-cyan-400 text-sm transition-colors">
                  Study Path
                </Link>
              </li>
            </ul>
          </div>

          {/* Legal links */}
          <div>
            <h3 className="text-white font-semibold mb-4">Legal</h3>
            <ul className="space-y-2">
              <li>
                <Link href="/terms" className="text-slate-400 hover:text-cyan-400 text-sm transition-colors">
                  Terms of Service
                </Link>
              </li>
              <li>
                <Link href="/privacy" className="text-slate-400 hover:text-cyan-400 text-sm transition-colors">
                  Privacy Policy
                </Link>
              </li>
              <li>
                <Link href="/cookies" className="text-slate-400 hover:text-cyan-400 text-sm transition-colors">
                  Cookie Policy
                </Link>
              </li>
              <li>
                <Link href="/contact" className="text-slate-400 hover:text-cyan-400 text-sm transition-colors">
                  Contact Us
                </Link>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="pt-8 border-t border-slate-800">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <p className="text-slate-500 text-sm">
              © {currentYear} TRUE ASI. All rights reserved.
            </p>
            <div className="flex gap-6">
              <span className="text-slate-500 text-sm">
                250+ Specialized Agents
              </span>
              <span className="text-slate-500 text-sm">
                6.54TB Knowledge Base
              </span>
              <span className="text-slate-500 text-sm">
                Real-time AI Integration
              </span>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
}
