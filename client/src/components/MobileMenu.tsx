import { useState } from "react";
import { Menu, X, Brain } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/_core/hooks/useAuth";
import { getLoginUrl } from "@/const";

interface MobileMenuProps {
  onLogout?: () => void;
}

export default function MobileMenu({ onLogout }: MobileMenuProps) {
  const [isOpen, setIsOpen] = useState(false);
  const { user, isAuthenticated } = useAuth();

  const toggleMenu = () => setIsOpen(!isOpen);

  const navLinks = [
    { href: "/dashboard", label: "Dashboard" },
    { href: "/agents", label: "Agents" },
    { href: "/chat", label: "Chat" },
    { href: "/knowledge-graph", label: "Knowledge" },
    { href: "/analytics", label: "Analytics" },
    { href: "/documentation", label: "Docs" },
    { href: "/s7-test", label: "S-7 Test" },
  ];

  return (
    <>
      {/* Hamburger Button */}
      <button
        onClick={toggleMenu}
        className="md:hidden p-2 rounded-lg hover:bg-primary/10 transition-colors"
        aria-label="Toggle menu"
      >
        {isOpen ? (
          <X className="w-6 h-6 text-foreground" />
        ) : (
          <Menu className="w-6 h-6 text-foreground" />
        )}
      </button>

      {/* Overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 md:hidden"
          onClick={toggleMenu}
        />
      )}

      {/* Slide-out Drawer */}
      <div
        className={`fixed top-0 right-0 h-full w-80 bg-card border-l border-border z-50 transform transition-transform duration-300 ease-in-out md:hidden ${
          isOpen ? "translate-x-0" : "translate-x-full"
        }`}
      >
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-border">
            <div className="flex items-center space-x-2">
              <Brain className="w-6 h-6 text-primary" />
              <span className="text-xl font-bold text-gradient">TRUE ASI</span>
            </div>
            <button
              onClick={toggleMenu}
              className="p-2 rounded-lg hover:bg-primary/10 transition-colors"
              aria-label="Close menu"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* User Info */}
          {isAuthenticated && user && (
            <div className="p-6 border-b border-border">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 rounded-full bg-primary/20 flex items-center justify-center">
                  <span className="text-primary font-bold">
                    {user.name?.charAt(0) || "U"}
                  </span>
                </div>
                <div>
                  <div className="font-medium">{user.name || "User"}</div>
                  <div className="text-xs text-muted-foreground">
                    {user.email || ""}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Navigation Links */}
          <nav className="flex-1 overflow-y-auto p-6">
            <ul className="space-y-2">
              {navLinks.map((link) => (
                <li key={link.href}>
                  <a
                    href={link.href}
                    onClick={toggleMenu}
                    className="block px-4 py-3 rounded-lg hover:bg-primary/10 transition-colors text-foreground/80 hover:text-foreground font-medium"
                  >
                    {link.label}
                  </a>
                </li>
              ))}
            </ul>
          </nav>

          {/* Footer Actions */}
          <div className="p-6 border-t border-border">
            {isAuthenticated ? (
              <Button
                onClick={() => {
                  onLogout?.();
                  toggleMenu();
                }}
                variant="outline"
                className="w-full"
              >
                Logout
              </Button>
            ) : (
              <Button
                onClick={() => {
                  window.location.href = getLoginUrl();
                }}
                className="w-full btn-primary"
              >
                Login
              </Button>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
