import { useAuth } from "@/_core/hooks/useAuth";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import {
  HelpCircle,
  Search,
  BookOpen,
  Video,
  MessageCircle,
  FileText,
  Building2,
  Receipt,
  Send,
  Shield,
  Sparkles,
  ExternalLink,
  Phone,
  Mail,
} from "lucide-react";

export default function Help() {
  const { user } = useAuth();
  const [searchQuery, setSearchQuery] = useState("");

  const guides = [
    {
      id: "getting-started",
      title: "Kom i gang",
      description: "Grunnleggende oppsett og første steg",
      icon: BookOpen,
      articles: [
        { title: "Registrer ditt første selskap", time: "3 min" },
        { title: "Koble til regnskapssystem", time: "5 min" },
        { title: "Last opp første dokument", time: "2 min" },
        { title: "Forstå dashboardet", time: "4 min" },
      ],
    },
    {
      id: "documents",
      title: "Dokumenthåndtering",
      description: "Laste opp, behandle og bokføre dokumenter",
      icon: Receipt,
      articles: [
        { title: "Støttede filformater", time: "2 min" },
        { title: "AI-drevet dokumentgjenkjenning", time: "4 min" },
        { title: "Godkjenne og bokføre bilag", time: "3 min" },
        { title: "Håndtere avviste dokumenter", time: "3 min" },
      ],
    },
    {
      id: "filings",
      title: "Innleveringer",
      description: "MVA-melding, A-melding og SAF-T",
      icon: Send,
      articles: [
        { title: "Generere MVA-melding", time: "5 min" },
        { title: "Sende til Altinn", time: "4 min" },
        { title: "Eksportere SAF-T", time: "3 min" },
        { title: "Håndtere feilmeldinger", time: "4 min" },
      ],
    },
    {
      id: "ai",
      title: "AI-assistenten",
      description: "Bruke AI for regnskapsspørsmål",
      icon: Sparkles,
      articles: [
        { title: "Stille spørsmål om regnskap", time: "3 min" },
        { title: "Automatisk kontoforslag", time: "4 min" },
        { title: "Risikovurdering", time: "5 min" },
        { title: "Beste praksis for AI-bruk", time: "4 min" },
      ],
    },
  ];

  const faqs = [
    {
      question: "Hvordan kobler jeg til mitt regnskapssystem?",
      answer: "Gå til Innstillinger → Integrasjoner og velg ditt regnskapssystem (Tripletex, PowerOffice, Fiken eller Visma). Følg instruksjonene for å autorisere tilkoblingen med API-nøkkel eller OAuth.",
    },
    {
      question: "Hva skjer hvis AI-en foreslår feil konto?",
      answer: "Du kan alltid overstyre AI-forslaget før bokføring. Systemet lærer av dine korrigeringer og vil gi bedre forslag over tid. Alle endringer logges i revisjonsloggen.",
    },
    {
      question: "Hvordan sender jeg MVA-melding til Altinn?",
      answer: "Gå til Innleveringer → Ny MVA-melding. Velg periode og selskap, og systemet genererer automatisk et utkast basert på bokførte transaksjoner. Gjennomgå og klikk 'Send til Altinn' for å levere.",
    },
    {
      question: "Er dataene mine sikre?",
      answer: "Ja, alle data krypteres både under overføring (TLS 1.3) og lagring (AES-256). Vi følger GDPR og norske personvernregler. Revisjonsloggen sikrer full sporbarhet i henhold til Bokføringsloven.",
    },
    {
      question: "Kan jeg eksportere data til Excel?",
      answer: "Ja, de fleste rapporter kan eksporteres til Excel, CSV eller PDF. Gå til Rapporter-siden og klikk på nedlastingsikonet ved ønsket rapport.",
    },
    {
      question: "Hva er SAF-T og trenger jeg det?",
      answer: "SAF-T (Standard Audit File - Tax) er et standardformat for regnskapsdata som Skatteetaten kan be om ved kontroll. Alle bokføringspliktige må kunne levere SAF-T på forespørsel. Skatt-Flow OS genererer dette automatisk.",
    },
  ];

  const filteredGuides = guides.filter(guide =>
    guide.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    guide.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
    guide.articles.some(a => a.title.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  const filteredFaqs = faqs.filter(faq =>
    faq.question.toLowerCase().includes(searchQuery.toLowerCase()) ||
    faq.answer.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="text-center max-w-2xl mx-auto">
        <h1 className="text-3xl font-bold flex items-center justify-center gap-2">
          <HelpCircle className="h-8 w-8 text-emerald-500" />
          Hjelpesenter
        </h1>
        <p className="text-muted-foreground mt-2">
          Finn svar på spørsmål, lær hvordan du bruker Skatt-Flow OS, og få hjelp når du trenger det.
        </p>
        <div className="relative mt-6">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
          <Input
            placeholder="Søk i hjelpesenter..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-12 h-12 text-lg"
          />
        </div>
      </div>

      {/* Quick Links */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card className="hover:border-emerald-500/50 transition-colors cursor-pointer">
          <CardContent className="pt-6 flex items-center gap-4">
            <div className="p-3 rounded-lg bg-blue-500/10">
              <Video className="h-6 w-6 text-blue-500" />
            </div>
            <div>
              <h3 className="font-semibold">Videoguider</h3>
              <p className="text-sm text-muted-foreground">Se hvordan systemet fungerer</p>
            </div>
          </CardContent>
        </Card>
        <Card className="hover:border-emerald-500/50 transition-colors cursor-pointer">
          <CardContent className="pt-6 flex items-center gap-4">
            <div className="p-3 rounded-lg bg-purple-500/10">
              <FileText className="h-6 w-6 text-purple-500" />
            </div>
            <div>
              <h3 className="font-semibold">API-dokumentasjon</h3>
              <p className="text-sm text-muted-foreground">For utviklere og integrasjoner</p>
            </div>
          </CardContent>
        </Card>
        <Card className="hover:border-emerald-500/50 transition-colors cursor-pointer">
          <CardContent className="pt-6 flex items-center gap-4">
            <div className="p-3 rounded-lg bg-emerald-500/10">
              <MessageCircle className="h-6 w-6 text-emerald-500" />
            </div>
            <div>
              <h3 className="font-semibold">Kontakt support</h3>
              <p className="text-sm text-muted-foreground">Vi hjelper deg gjerne</p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Guides */}
      <div>
        <h2 className="text-2xl font-bold mb-4">Brukerveiledninger</h2>
        <div className="grid gap-4 md:grid-cols-2">
          {filteredGuides.map((guide) => (
            <Card key={guide.id}>
              <CardHeader>
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-muted">
                    <guide.icon className="h-5 w-5 text-emerald-500" />
                  </div>
                  <div>
                    <CardTitle className="text-lg">{guide.title}</CardTitle>
                    <CardDescription>{guide.description}</CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2">
                  {guide.articles.map((article, idx) => (
                    <li key={idx}>
                      <a
                        href="#"
                        className="flex items-center justify-between p-2 rounded-lg hover:bg-muted transition-colors"
                      >
                        <span className="text-sm">{article.title}</span>
                        <Badge variant="secondary" className="text-xs">{article.time}</Badge>
                      </a>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>

      {/* FAQs */}
      <Card>
        <CardHeader>
          <CardTitle>Ofte stilte spørsmål</CardTitle>
          <CardDescription>Svar på de vanligste spørsmålene</CardDescription>
        </CardHeader>
        <CardContent>
          <Accordion type="single" collapsible className="w-full">
            {filteredFaqs.map((faq, idx) => (
              <AccordionItem key={idx} value={`faq-${idx}`}>
                <AccordionTrigger className="text-left">
                  {faq.question}
                </AccordionTrigger>
                <AccordionContent className="text-muted-foreground">
                  {faq.answer}
                </AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
        </CardContent>
      </Card>

      {/* Contact */}
      <Card className="border-emerald-500/50">
        <CardHeader>
          <CardTitle>Trenger du mer hjelp?</CardTitle>
          <CardDescription>Vårt supportteam er klare til å hjelpe deg</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-3">
            <div className="flex items-center gap-3">
              <Mail className="h-5 w-5 text-muted-foreground" />
              <div>
                <p className="text-sm text-muted-foreground">E-post</p>
                <a href="mailto:support@skattflow.no" className="font-medium hover:text-emerald-500">
                  support@skattflow.no
                </a>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <Phone className="h-5 w-5 text-muted-foreground" />
              <div>
                <p className="text-sm text-muted-foreground">Telefon</p>
                <a href="tel:+4722334455" className="font-medium hover:text-emerald-500">
                  +47 22 33 44 55
                </a>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <MessageCircle className="h-5 w-5 text-muted-foreground" />
              <div>
                <p className="text-sm text-muted-foreground">Chat</p>
                <Button variant="link" className="p-0 h-auto font-medium">
                  Start chat
                </Button>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
