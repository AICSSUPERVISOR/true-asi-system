# Skatt-Flow OS - Brukermanual

> Autonom regnskaps- og revisjonsplattform for norske bedrifter

---

## Innholdsfortegnelse

1. [Introduksjon](#introduksjon)
2. [Kom i gang](#kom-i-gang)
3. [Dashboard](#dashboard)
4. [Selskaper](#selskaper)
5. [Regnskapsdokumenter](#regnskapsdokumenter)
6. [Hovedbok](#hovedbok)
7. [Innleveringer](#innleveringer)
8. [Dokumentmaler](#dokumentmaler)
9. [AI-assistent](#ai-assistent)
10. [Innstillinger](#innstillinger)
11. [Vanlige spørsmål](#vanlige-spørsmål)

---

## Introduksjon

Skatt-Flow OS er en komplett, autonom regnskaps- og revisjonsplattform designet spesielt for norske bedrifter. Plattformen automatiserer regnskapsoppgaver ved hjelp av kunstig intelligens og integrerer sømløst med norsk offentlig infrastruktur.

### Hovedfunksjoner

- **Automatisk dokumentbehandling**: Last opp fakturaer og bilag som automatisk klassifiseres og konteres
- **MVA-beregning og innlevering**: Automatisk generering av MVA-meldinger med Altinn-integrasjon
- **SAF-T eksport**: Generer SAF-T filer i henhold til norsk standard
- **Forvalt-integrasjon**: Automatisk kredittsjekk og risikovurdering av kunder og leverandører
- **AI-drevet rådgivning**: Spør AI-assistenten om regnskapsspørsmål

### Brukerroller

| Rolle | Rettigheter |
|-------|-------------|
| **OWNER** | Full tilgang til alle funksjoner, inkludert brukeradministrasjon |
| **ADMIN** | Kan administrere selskaper og godkjenne innleveringer |
| **ACCOUNTANT** | Kan behandle dokumenter og opprette bilag |
| **VIEWER** | Kun lesetilgang |

---

## Kom i gang

### 1. Logg inn

Klikk på "Logg inn" knappen øverst til høyre. Du vil bli videresendt til Manus OAuth for autentisering.

### 2. Legg til ditt første selskap

1. Gå til **Selskaper** i menyen
2. Klikk **Legg til nytt selskap**
3. Skriv inn organisasjonsnummer
4. Systemet henter automatisk data fra Brønnøysundregistrene
5. Klikk **Lagre**

### 3. Koble til regnskapssystem

1. Gå til **Innstillinger** → **Integrasjoner**
2. Velg ditt regnskapssystem (Tripletex, PowerOffice, Fiken, Visma)
3. Følg instruksjonene for å koble til

---

## Dashboard

Dashboardet gir deg en oversikt over alle dine selskaper og oppgaver.

### Statuskort

- **Uposterte dokumenter**: Antall dokumenter som venter på godkjenning
- **Ventende innleveringer**: Innleveringer som må sendes til Altinn
- **Høyrisiko-selskaper**: Selskaper med forhøyet risikovurdering
- **Aktive selskaper**: Totalt antall aktive selskaper

### Kommende frister

Viser de neste MVA-terminene og andre viktige frister med nedtelling.

### Nylige aktiviteter

Logg over de siste handlingene i systemet.

---

## Selskaper

### Legge til selskap

1. Klikk **Legg til nytt selskap**
2. Skriv inn organisasjonsnummer (9 siffer)
3. Systemet henter automatisk:
   - Selskapsnavn
   - Adresse
   - Bransje (NACE-kode)
   - Kredittrating fra Forvalt
4. Legg til bankkontoer ved behov
5. Klikk **Lagre**

### Selskapsoversikt

For hvert selskap kan du se:

- **Grunndata**: Navn, org.nr, adresse, bransje
- **Forvalt-data**: Kredittrating, risikoklasse, nøkkeltall
- **Bankkontoer**: Tilknyttede kontoer
- **Statistikk**: Antall dokumenter, bilag, innleveringer

### Oppdatere Forvalt-data

Klikk på **Oppdater fra Forvalt** for å hente ferske data. Dette gjøres automatisk ukentlig.

---

## Regnskapsdokumenter

### Laste opp dokumenter

1. Gå til **Regnskap** i menyen
2. Klikk **Last opp dokument**
3. Velg fil (PDF, bilde, eller e-faktura)
4. Velg dokumenttype:
   - Leverandørfaktura
   - Kundefaktura
   - Kvittering
   - Kontoutskrift
   - Annet bilag
5. Klikk **Last opp**

### Automatisk behandling

Når et dokument lastes opp:

1. **OCR/Tekstutvinning**: Systemet leser dokumentet
2. **Klassifisering**: AI bestemmer dokumenttype
3. **Feltutvinning**: Beløp, dato, leverandør, etc. ekstraheres
4. **Konteringsforslag**: AI foreslår debet/kredit-kontoer
5. **MVA-beregning**: Korrekt MVA-kode foreslås

### Godkjenne bilag

1. Se gjennom AI-forslaget
2. Gjør eventuelle justeringer
3. Klikk **Godkjenn og poster**

Bilaget blir da lagt inn i hovedboken.

### Dokumentstatus

| Status | Beskrivelse |
|--------|-------------|
| **NY** | Nettopp lastet opp |
| **BEHANDLER** | Under AI-analyse |
| **VENTER_GODKJENNING** | Klar for manuell gjennomgang |
| **GODKJENT** | Godkjent, venter på postering |
| **POSTERT** | Lagt inn i hovedboken |
| **AVVIST** | Avvist av bruker |

---

## Hovedbok

### Visning

Hovedboken viser alle posterte bilag med:

- Bilagsnummer
- Dato
- Beskrivelse
- Debet-konto
- Kredit-konto
- Beløp
- MVA-kode

### Filtrering

Du kan filtrere på:

- Periode (fra/til dato)
- Konto
- Beløpsintervall
- Søketekst

### Eksport

Klikk **Eksporter** for å laste ned hovedboken som:

- Excel (.xlsx)
- CSV
- PDF

---

## Innleveringer

### MVA-melding

#### Opprette ny MVA-melding

1. Gå til **Innleveringer**
2. Klikk **Ny MVA-melding**
3. Velg selskap og termin
4. Systemet beregner automatisk:
   - Utgående MVA
   - Inngående MVA
   - MVA å betale/tilgode
5. Se gjennom beregningen
6. Klikk **Opprett utkast**

#### Sende til Altinn

1. Åpne MVA-meldingen
2. Klikk **Send til Altinn**
3. Bekreft innsending
4. Systemet sender via Altinn API
5. Mottakskvittering vises

### SAF-T eksport

1. Gå til **Innleveringer**
2. Klikk **Ny SAF-T eksport**
3. Velg selskap og periode
4. Klikk **Generer SAF-T**
5. Last ned XML-filen

SAF-T filen valideres automatisk mot norsk standard.

### A-melding

A-meldinger for lønn håndteres tilsvarende MVA-meldinger.

---

## Dokumentmaler

### Tilgjengelige maler

- **Faktura**: Standard fakturamaler
- **Kontrakt**: Avtale- og kontraktsmaler
- **Brev**: Formelle brev
- **Purring**: Betalingspåminnelser
- **Rapport**: Regnskapsrapporter

### Opprette dokument fra mal

1. Gå til **Dokumenter**
2. Velg mal
3. Velg selskap
4. Fyll inn variabler (kundenavn, beløp, etc.)
5. Klikk **Generer**
6. Last ned PDF

### Egne maler

Administratorer kan opprette egne maler med variabler:

```
Kjære {{kundenavn}},

Vi viser til faktura nr. {{fakturanummer}} med forfall {{forfallsdato}}.
Utestående beløp er kr {{beløp}}.

Med vennlig hilsen,
{{selskapsnavn}}
```

---

## AI-assistent

### Starte samtale

1. Gå til **AI Chat** i menyen
2. Velg selskap fra nedtrekksmenyen
3. Skriv ditt spørsmål
4. Trykk Enter eller klikk Send

### Eksempler på spørsmål

- "Hva er MVA-satsen for restauranttjenester?"
- "Klassifiser denne fakturaen for meg"
- "Beregn MVA for denne terminen"
- "Gjør en kredittsjekk på leverandør X"
- "Forklar forskjellen på konto 4000 og 4300"
- "Hva er fristen for neste MVA-innlevering?"

### AI-kapabiliteter

AI-assistenten kan:

- ✅ Klassifisere dokumenter
- ✅ Foreslå kontering
- ✅ Beregne MVA
- ✅ Gjøre kredittsjekk
- ✅ Validere SAF-T
- ✅ Vurdere risiko
- ✅ Generere dokumenter
- ✅ Svare på regnskapsspørsmål

---

## Innstillinger

### Profil

- Endre navn og e-post
- Se påloggingshistorikk

### Integrasjoner

Koble til eksterne systemer:

- **Forvalt/Proff**: Kredittdata (krever API-nøkkel)
- **Altinn**: Innleveringer (krever Maskinporten-sertifikat)
- **Regnskapssystem**: Tripletex, PowerOffice, Fiken, Visma

### Varslinger

Konfigurer varsler for:

- MVA-frister (7, 3, 1 dag før)
- Nye dokumenter
- Høyrisiko-kunder
- Systemoppdateringer

### Brukeradministrasjon (kun OWNER)

- Inviter nye brukere
- Tildel roller
- Fjern tilgang

---

## Vanlige spørsmål

### Hvordan kobler jeg til Altinn?

Du trenger et Maskinporten-sertifikat fra Digitaliseringsdirektoratet. Kontakt din regnskapsfører eller IT-avdeling for hjelp.

### Støtter systemet alle MVA-satser?

Ja, systemet støtter alle norske MVA-satser:
- 25% (standard)
- 15% (mat)
- 12% (transport, overnatting)
- 0% (eksport, fritak)

### Hvordan håndteres GDPR?

- Alle data lagres i EU
- Automatisk sletting etter 10 år (bokføringsloven)
- Eksportfunksjon for persondata
- Revisjonslogg for alle endringer

### Kan jeg importere eksisterende data?

Ja, du kan importere:
- SAF-T filer fra andre systemer
- CSV med bilagsdata
- Excel-filer

### Hva skjer hvis AI-en gjør feil?

AI-forslag må alltid godkjennes manuelt før postering. Systemet logger alle AI-beslutninger for revisjon.

---

## Teknisk støtte

For teknisk støtte, kontakt:

- **E-post**: support@skatt-flow.no
- **Telefon**: +47 22 00 00 00
- **Chat**: Tilgjengelig i appen

---

*Sist oppdatert: Desember 2024*
