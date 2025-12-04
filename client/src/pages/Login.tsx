import { useEffect } from "react";
import { useLocation } from "wouter";
import { useAuth } from "@/_core/hooks/useAuth";
import { getLoginUrl } from "@/const";
import { Brain, Sparkles, Zap, Network } from "lucide-react";

export default function Login() {
  const { user, loading } = useAuth();
  const [, setLocation] = useLocation();

  // Redirect if already logged in
  useEffect(() => {
    if (user) {
      setLocation("/");
    }
  }, [user, setLocation]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-cyan-400"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 relative overflow-hidden">
      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-1/2 -left-1/2 w-full h-full bg-gradient-to-br from-cyan-500/10 to-transparent rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute -bottom-1/2 -right-1/2 w-full h-full bg-gradient-to-tl from-purple-500/10 to-transparent rounded-full blur-3xl animate-pulse" style={{ animationDelay: "1s" }}></div>
      </div>

      {/* Login card */}
      <div className="relative z-10 w-full max-w-md mx-4">
        <div className="bg-slate-800/50 backdrop-blur-xl rounded-2xl shadow-2xl border border-slate-700/50 p-8">
          {/* Logo and branding */}
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-cyan-500 to-purple-600 rounded-2xl mb-4 shadow-lg shadow-cyan-500/50">
              <Brain className="w-10 h-10 text-white" />
            </div>
            
            <h1 className="text-4xl font-bold bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400 bg-clip-text text-transparent mb-2">
              TRUE ASI
            </h1>
            
            <p className="text-slate-300 text-lg font-medium">
              Artificial Superintelligence System
            </p>
          </div>

          {/* Welcome message */}
          <div className="mb-8 text-center">
            <h2 className="text-2xl font-semibold text-white mb-3">
              Welcome Back
            </h2>
            <p className="text-slate-400">
              Access the future of intelligence with 250 specialized agents, 6.54TB of knowledge, and real-time AI model integration.
            </p>
          </div>

          {/* Features grid */}
          <div className="grid grid-cols-2 gap-4 mb-8">
            <div className="bg-slate-700/30 rounded-lg p-4 text-center">
              <Network className="w-6 h-6 text-cyan-400 mx-auto mb-2" />
              <p className="text-sm text-slate-300 font-medium">250+ Agents</p>
            </div>
            <div className="bg-slate-700/30 rounded-lg p-4 text-center">
              <Sparkles className="w-6 h-6 text-purple-400 mx-auto mb-2" />
              <p className="text-sm text-slate-300 font-medium">6.54TB Data</p>
            </div>
            <div className="bg-slate-700/30 rounded-lg p-4 text-center">
              <Zap className="w-6 h-6 text-yellow-400 mx-auto mb-2" />
              <p className="text-sm text-slate-300 font-medium">Real-time AI</p>
            </div>
            <div className="bg-slate-700/30 rounded-lg p-4 text-center">
              <Brain className="w-6 h-6 text-blue-400 mx-auto mb-2" />
              <p className="text-sm text-slate-300 font-medium">S-7 Certified</p>
            </div>
          </div>

          {/* Login button */}
          <a
            href={getLoginUrl()}
            className="block w-full bg-gradient-to-r from-cyan-500 to-purple-600 hover:from-cyan-600 hover:to-purple-700 text-white font-semibold py-4 px-6 rounded-xl transition-all duration-200 shadow-lg shadow-cyan-500/30 hover:shadow-xl hover:shadow-cyan-500/40 text-center"
          >
            Sign In to TRUE ASI
          </a>

          {/* Security note */}
          <p className="text-xs text-slate-500 text-center mt-6">
            Secure OAuth authentication • Your data is encrypted and protected
          </p>
        </div>

        {/* Additional info */}
        <div className="mt-6 text-center">
          <p className="text-slate-400 text-sm">
            New to TRUE ASI?{" "}
            <a href={getLoginUrl()} className="text-cyan-400 hover:text-cyan-300 font-medium">
              Create an account
            </a>
          </p>
        </div>
      </div>
    </div>
  );
}
