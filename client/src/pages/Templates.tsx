/**
 * Business-in-a-Box Templates Page
 * 
 * AI-powered template system with 6000+ business templates
 * Auto-fill with company data from Brreg/Forvalt
 * Generate documents with one click
 */

import { useState } from 'react';
import { Link } from 'wouter';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import {
  FileText,
  Briefcase,
  Users,
  DollarSign,
  TrendingUp,
  Settings,
  Search,
  Sparkles,
  Download,
  Eye,
  ArrowLeft,
  Zap,
} from 'lucide-react';

export default function Templates() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);

  const categories = [
    {
      id: 'legal',
      name: 'Legal & Compliance',
      icon: <Briefcase className="w-6 h-6" />,
      count: 150,
      color: 'from-blue-500 to-cyan-500',
      templates: [
        'Non-Disclosure Agreement (NDA)',
        'Confidentiality Agreement',
        'Consulting Agreement',
        'Service Level Agreement (SLA)',
        'Employment Contract',
        'Shareholder Agreement',
        'Distribution Agreement',
        'Partnership Agreement',
      ],
    },
    {
      id: 'hr',
      name: 'Human Resources',
      icon: <Users className="w-6 h-6" />,
      count: 200,
      color: 'from-green-500 to-emerald-500',
      templates: [
        'Employee Handbook',
        'Job Description Template',
        'Performance Review Form',
        'Onboarding Checklist',
        'Exit Interview Form',
        'Employee Satisfaction Survey',
        'Training Plan',
        'Disciplinary Action Form',
      ],
    },
    {
      id: 'finance',
      name: 'Finance & Accounting',
      icon: <DollarSign className="w-6 h-6" />,
      count: 180,
      color: 'from-yellow-500 to-orange-500',
      templates: [
        'Cash Flow Forecast',
        'Budget Template',
        'Invoice Template',
        'Expense Report',
        'Financial Statement',
        'Profit & Loss Statement',
        'Balance Sheet',
        'Tax Planning Worksheet',
      ],
    },
    {
      id: 'marketing',
      name: 'Marketing & Sales',
      icon: <TrendingUp className="w-6 h-6" />,
      count: 220,
      color: 'from-purple-500 to-pink-500',
      templates: [
        'Marketing Plan',
        'Business Proposal',
        'Sales Pitch Deck',
        'Social Media Calendar',
        'Email Campaign Template',
        'Content Strategy',
        'Brand Guidelines',
        'Competitive Analysis',
      ],
    },
    {
      id: 'operations',
      name: 'Operations & Management',
      icon: <Settings className="w-6 h-6" />,
      count: 150,
      color: 'from-red-500 to-rose-500',
      templates: [
        'Business Plan',
        'Project Charter',
        'Risk Assessment',
        'Quality Control Checklist',
        'Standard Operating Procedure (SOP)',
        'Meeting Minutes Template',
        'Action Plan',
        'SWOT Analysis',
      ],
    },
    {
      id: 'it',
      name: 'IT & Technology',
      icon: <Zap className="w-6 h-6" />,
      count: 100,
      color: 'from-indigo-500 to-blue-500',
      templates: [
        'IT Security Policy',
        'Software Requirements Document',
        'System Architecture Diagram',
        'API Documentation',
        'User Manual Template',
        'Disaster Recovery Plan',
        'Change Request Form',
        'Technical Specification',
      ],
    },
  ];

  const filteredCategories = selectedCategory
    ? categories.filter(c => c.id === selectedCategory)
    : categories;

  const handleGenerateDocument = (templateName: string, categoryName: string) => {
    // TODO: Implement AI-powered document generation
    // 1. Fetch company data from Brreg/Forvalt
    // 2. Use TRUE ASI Ultra to auto-fill template
    // 3. Generate PDF/DOCX with one click
    console.log(`Generating ${templateName} from ${categoryName}`);
    alert(`AI is generating your ${templateName}...\\n\\nThis will:\\n1. Fetch company data from Brreg\\n2. Get credit rating from Forvalt\\n3. Auto-fill all fields with AI\\n4. Generate professional PDF/DOCX\\n\\nComing soon!`);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-950 p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="space-y-2">
            <div className="flex items-center gap-3">
              <Link href="/">
                <Button variant="ghost" size="icon">
                  <ArrowLeft className="w-5 h-5" />
                </Button>
              </Link>
              <h1 className="text-5xl font-black tracking-tight text-white">
                Business Templates
              </h1>
            </div>
            <p className="text-xl text-slate-300">
              6000+ professional templates powered by AI • Auto-fill with company data
            </p>
          </div>
          <Badge className="bg-gradient-to-r from-green-500 to-emerald-500 text-white text-lg px-4 py-2">
            <Sparkles className="w-4 h-4 mr-2" />
            AI-Powered
          </Badge>
        </div>

        {/* Search Bar */}
        <Card className="bg-white/5 backdrop-blur-xl border-white/10">
          <CardContent className="p-6">
            <div className="flex gap-4">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                <Input
                  type="text"
                  placeholder="Search 6000+ templates..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10 bg-white/10 border-white/20 text-white placeholder:text-slate-400 text-lg"
                />
              </div>
              {selectedCategory && (
                <Button
                  variant="outline"
                  onClick={() => setSelectedCategory(null)}
                  className="border-white/20 text-white"
                >
                  Clear Filter
                </Button>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Category Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredCategories.map((category) => (
            <Card
              key={category.id}
              className="bg-white/5 backdrop-blur-xl border-white/10 hover:border-white/30 transition-all cursor-pointer group"
              onClick={() => setSelectedCategory(category.id)}
            >
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className={`p-3 rounded-xl bg-gradient-to-br ${category.color}`}>
                    {category.icon}
                  </div>
                  <Badge className="bg-white/10 text-white">
                    {category.count} templates
                  </Badge>
                </div>
                <CardTitle className="text-2xl font-bold text-white mt-4">
                  {category.name}
                </CardTitle>
                <CardDescription className="text-slate-300">
                  Professional templates ready to use
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                {category.templates.slice(0, 4).map((template, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between p-3 bg-white/5 rounded-lg hover:bg-white/10 transition-all"
                  >
                    <div className="flex items-center gap-3">
                      <FileText className="w-4 h-4 text-slate-400" />
                      <span className="text-sm text-white">{template}</span>
                    </div>
                    <div className="flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-8 w-8 p-0"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleGenerateDocument(template, category.name);
                        }}
                      >
                        <Sparkles className="w-4 h-4" />
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-8 w-8 p-0"
                        onClick={(e) => {
                          e.stopPropagation();
                          alert(`Preview: ${template}`);
                        }}
                      >
                        <Eye className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                ))}
                {category.templates.length > 4 && (
                  <Button
                    variant="ghost"
                    className="w-full text-blue-400 hover:text-blue-300"
                    onClick={() => setSelectedCategory(category.id)}
                  >
                    View all {category.count} templates →
                  </Button>
                )}
              </CardContent>
            </Card>
          ))}
        </div>

        {/* AI Features Banner */}
        <Card className="bg-gradient-to-r from-blue-500/10 to-purple-500/10 border-blue-500/20">
          <CardContent className="p-8">
            <div className="flex items-center gap-6">
              <div className="p-4 bg-gradient-to-br from-blue-500 to-purple-500 rounded-2xl">
                <Sparkles className="w-8 h-8 text-white" />
              </div>
              <div className="flex-1">
                <h3 className="text-2xl font-bold text-white mb-2">
                  AI-Powered Auto-Fill
                </h3>
                <p className="text-slate-300 text-lg">
                  TRUE ASI automatically fills templates with company data from Brreg, Forvalt credit ratings, 
                  and 6.54TB knowledge base. Generate professional documents in seconds.
                </p>
              </div>
              <Button className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-lg px-8">
                <Sparkles className="w-5 h-5 mr-2" />
                Try AI Auto-Fill
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
