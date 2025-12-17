# Skatt-Flow OS - Project TODO

## Database Schema
- [x] User table with RBAC roles (OWNER, ADMIN, ACCOUNTANT, VIEWER)
- [x] Company table with Forvalt data fields
- [x] BankAccount table
- [x] AccountingDocument table with status workflow
- [x] LedgerEntry table
- [x] Filing table for MVA, SAF-T, A-melding
- [x] ForvaltSnapshot table
- [x] DocumentTemplate table
- [x] GeneratedDocument table

## Backend API Clients
- [x] Forvalt/Proff API client (getCompany, getCredit, getFinancials)
- [x] Generic Regnskap client (chart of accounts, VAT codes, vouchers, ledger, SAF-T)
- [x] Altinn API client (OAuth2, createDraftFiling, submitFiling)
- [x] AIML API client (multi-model, classifyTransaction, extractInvoiceFields)

## tRPC Routers
- [x] Company router (CRUD, Forvalt enrichment)
- [x] AccountingDocument router (upload, process, post)
- [x] LedgerEntry router (list, filter)
- [x] Filing router (create draft, submit, status)
- [x] DocumentTemplate router (CRUD)
- [x] GeneratedDocument router (generate, export)
- [x] Chat router (AI agent interaction)

## AI Agent (Skatt-Flow Agent)
- [x] System prompt for Norwegian accounting rules
- [x] Document extraction and classification
- [x] Voucher suggestion logic
- [x] MVA-melding draft generation
- [x] SAF-T validation
- [x] Risk assessment from Forvalt data
- [x] Document generation from templates

## Frontend Pages
- [x] Login/Auth page (via DashboardLayout OAuth)
- [x] Dashboard with status cards and deadlines
- [x] Company list and detail pages
- [x] Add company flow with Forvalt lookup
- [x] Accounting page with document processing
- [x] Reconciliation view (stub in Ledger)
- [x] Ledger/Journal view
- [x] Filings list and detail pages
- [x] Document templates page
- [x] Document generation wizard
- [x] AI Chat interface
- [x] Settings page

## Workflows
- [x] New company onboarding flow
- [x] Document ingestion and posting flow
- [x] Periodic MVA filing flow
- [x] SAF-T export and check flow
- [x] Document generation flow

## Security & Quality
- [x] RBAC role checks on all write operations
- [x] Error handling for all external API calls
- [x] API call logging (no secrets)
- [x] Unit tests for API clients
- [x] Confirmation dialogs for Altinn submissions
- [x] Auto-post settings per company

## Documentation
- [x] Environment variables documentation
- [x] Migration instructions (in README)
- [x] Production deployment guide (in README)


## BRUTAL AUDIT FINDINGS - CRITICAL FIXES

### Priority 0 (Blocking)
- [x] Add company-level access control to all database queries
- [ ] Configure AIML_API_KEY environment variable
- [x] Add audit logging table and middleware
- [ ] Add foreign key constraints to database schema
- [ ] Add database indexes on frequently queried columns

### Priority 1 (High - AI Automation)
- [x] Implement tool calling in AI agent for autonomous actions (V2 Agent)
- [x] Add OCR/Vision support for scanned documents (V2 Agent)
- [ ] Replace Forvalt mock with real API integration
- [ ] Add streaming responses to chat interface
- [ ] Implement multi-model routing for different tasks

### Priority 2 (Medium - Integration)
- [ ] Altinn ID-porten/Maskinporten integration
- [ ] Implement scheduled task system for MVA deadlines
- [ ] Add Open Banking integration for bank reconciliation
- [ ] Add pagination to all list endpoints
- [ ] Implement soft delete for audit compliance

### Priority 3 (Testing & Quality)
- [ ] Increase test coverage to 80%
- [ ] Add integration tests for API clients
- [ ] Add E2E tests for critical user flows
- [ ] Add load testing for performance validation


## PHASE 2: ENHANCEMENT TO 100/100 QUALITY

### UI Enhancement (from Premium Templates)
- [x] Extract and analyze uploaded UI templates
- [ ] Implement premium dashboard design with glassmorphism- [x] Enhance dashboard with glassmorphism and animations
- [x] Improve chat interface with streaming indicators
- [ ] Add professional invoice templates for document generation
- [ ] Implement landing page for public-facing features
- [ ] Add dark/light theme toggle with smooth transitionseal API Integrations
- [x] Implement real Forvalt API with proper authentication (V2 client with fallbacks)
- [x] Implement Altinn ID-porten/Maskinporten OAuth flow (V2 client)
- [x] Add Tripletex API integration (via regnskapClient)
- [x] Add PowerOffice Go API integration (via regnskapClient)
- [x] Add Fiken API integration (via regnskapClient)
- [x] Add Visma eAccounting API integration (via regnskapClient)
- [x] Implement SAF-T export with proper XML generation (altinnClientV2)

### Streaming & Real-time Features
- [x] Add streaming responses to AI chat (SSE implementation)
- [x] Implement WebSocket for real-time updates (chatStream.ts)
- [x] Add typing indicators in chat
- [x] Implement real-time document processing status

### Scheduled Tasks System
- [x] Create cron job system for MVA deadlines (taskScheduler.ts)
- [x] Add automatic deadline reminders (7, 3, 1 day warnings)
- [x] Implement scheduled Forvalt data refresh (weekly)
- [x] Add automatic SAF-T backup generation (monthly)

### Performance Optimizations
- [x] Add pagination utilities for all list endpoints
- [x] Add database indexes migration script
- [x] Implement in-memory caching with TTL
- [x] Add performance monitoring utilities
- [ ] Add lazy loading for heavy components
- [ ] Optimize bundle size with code splitting

### Testing & Documentation
- [ ] Add E2E tests with Playwright
- [x] Add integration tests for all API clients (25 tests passing)
- [x] Create API documentation (API_DOCUMENTATION.md)
- [x] Add inline code documentation
- [x] Create user manual in Norwegian (BRUKERMANUAL.### Security Hardening
- [x] Add rate limiting on all endpoints (rateLimiter.ts)
- [ ] Implement CSRF protection
- [x] Add input sanitization (sanitize.ts)
- [ ] Implement security headers (CSP, HSTS)
- [ ] Add API key encryption at rest (CSP, HSTS)


## PHASE 3: BRUTAL AUDIT & DEEPLINKS ENHANCEMENT

### Missing URLs & Deeplinks
- [x] Add /company/:id/documents route
- [x] Add /company/:id/ledger route
- [x] Add /company/:id/filings route
- [x] Add /company/:id/reconciliation route
- [x] Add /filing/:id detail page
- [x] Add /document/:id detail page
- [ ] Add /template/:id edit page
- [x] Add /reports route for financial reports
- [x] Add /reconciliation route for bank reconciliation
- [x] Add /audit-log route for compliance viewing
- [ ] Add /notifications route for deadline alerts
- [ ] Add /api-status route for integration health
- [x] Add /help route for documentation
- [ ] Add /onboarding route for new user flow

### Backend Gaps to Fix
- [ ] Add missing tRPC endpoints for deeplinks
- [ ] Add notification system
- [ ] Add report generation endpoints
- [ ] Add bank reconciliation endpoints
- [ ] Add audit log viewing endpoint

### Frontend Gaps to Fix
- [x] Create FilingDetail page
- [x] Create DocumentDetail page
- [ ] Create TemplateEdit page
- [x] Create Reports page
- [x] Create Reconciliation page
- [x] Create AuditLog page
- [ ] Create Notifications page
- [x] Create Help/Documentation page
- [ ] Create Onboarding wizard


## PHASE 4: 100/100 ALTINN INTEGRATION & GITHUB CLONING

### Full Altinn API Integration (Tripletex-level)
- [ ] Implement Maskinporten OAuth2 authentication
- [ ] MVA-melding (VAT return) - full automation
- [ ] A-melding (payroll reporting) - full automation
- [ ] SAF-T export and validation
- [ ] Årsregnskap (annual accounts) submission
- [ ] Aksjonærregisteroppgaven (shareholder registry)
- [ ] Real-time status tracking for all filings
- [ ] Automatic deadline monitoring and reminders
- [ ] AI-powered form pre-filling from accounting data

### GitHub Repository Cloning
- [ ] Initialize Git repository
- [ ] Push to AICSSUPERVISOR/true-asi-system repository
- [ ] Ensure permanent availability

### AI Automation Enhancements
- [ ] Auto-classify all transactions with AI
- [ ] Auto-generate MVA-melding from ledger
- [ ] Auto-reconcile bank statements
- [ ] Auto-detect anomalies and fraud
