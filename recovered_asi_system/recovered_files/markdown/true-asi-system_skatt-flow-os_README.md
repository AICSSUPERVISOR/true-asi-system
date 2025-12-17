# Skatt-Flow OS

**Autonomous Accounting & Audit Platform for Norwegian Businesses**

Skatt-Flow OS is a production-ready, AI-powered accounting platform designed specifically for Norwegian businesses. It automates document processing, MVA filing, SAF-T export, and integrates with Norwegian regulatory infrastructure including Altinn, Skatteetaten, and Brønnøysundregistrene.

## Features

### Core Functionality

- **Company Management**: Onboard companies with automatic Forvalt/Proff data enrichment
- **Document Processing**: AI-powered document ingestion, classification, and posting suggestions
- **Ledger Management**: View and filter general ledger entries with full audit trail
- **MVA Filing**: Automated MVA-melding generation and Altinn submission
- **SAF-T Export**: Generate SAF-T files for regulatory compliance
- **Document Generation**: Create contracts, HR documents, and legal templates with AI assistance

### AI Capabilities

- **Intelligent Document Classification**: Automatically categorize invoices, receipts, and contracts
- **Voucher Suggestions**: AI-generated posting recommendations with account and VAT code
- **Risk Assessment**: Analyze company risk using Forvalt credit data
- **Natural Language Chat**: Ask questions about accounting, VAT rules, and company finances

### Integrations

- **Forvalt/Proff**: Company data, credit scores, and risk assessment
- **Altinn**: MVA-melding and A-melding submission
- **Accounting Systems**: Tripletex, PowerOffice, Fiken, Visma eAccounting
- **AI/ML API**: Multi-model AI for document processing

## Technology Stack

- **Frontend**: React 19, TypeScript, Tailwind CSS 4, shadcn/ui
- **Backend**: Node.js, Express, tRPC 11
- **Database**: MySQL/TiDB with Drizzle ORM
- **Authentication**: Manus OAuth with role-based access control
- **AI**: AIML multi-model API with Norwegian accounting expertise

## Getting Started

### Prerequisites

- Node.js 22+
- pnpm 10+
- MySQL/TiDB database

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd skatt-flow-os

# Install dependencies
pnpm install

# Set up environment variables
cp .env.example .env
# Edit .env with your credentials

# Push database schema
pnpm db:push

# Start development server
pnpm dev
```

### Environment Variables

See [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) for complete documentation of all environment variables.

## Project Structure

```
skatt-flow-os/
├── client/                 # Frontend React application
│   ├── src/
│   │   ├── components/     # Reusable UI components
│   │   ├── pages/          # Page components
│   │   ├── contexts/       # React contexts
│   │   ├── hooks/          # Custom hooks
│   │   └── lib/            # Utilities and tRPC client
├── server/                 # Backend Node.js application
│   ├── agents/             # AI agent orchestration
│   ├── clients/            # External API clients
│   ├── _core/              # Framework internals
│   ├── db.ts               # Database queries
│   └── routers.ts          # tRPC routers
├── drizzle/                # Database schema and migrations
├── shared/                 # Shared types and constants
└── docs/                   # Documentation
```

## User Roles

Skatt-Flow OS implements role-based access control (RBAC):

| Role | Description | Permissions |
|------|-------------|-------------|
| **OWNER** | System owner | Full access to all features |
| **ADMIN** | Administrator | Manage companies, users, and settings |
| **ACCOUNTANT** | Accountant | Process documents, create filings |
| **VIEWER** | Read-only | View data without modifications |

## API Documentation

### tRPC Routers

- `company`: Company CRUD and Forvalt enrichment
- `document`: Document upload, processing, and approval
- `filing`: MVA/SAF-T generation and Altinn submission
- `template`: Document template management
- `generatedDoc`: AI-generated document management
- `chat`: AI assistant interaction
- `ledger`: Ledger entry queries
- `dashboard`: Statistics and overview data

### External API Clients

- `forvaltClient`: Company data and credit scoring
- `regnskapClient`: Accounting system integration
- `altinnClient`: Regulatory filing submission
- `aimlClient`: AI document processing

## Security

- All API endpoints require authentication
- Role-based access control on all write operations
- Secure cookie-based sessions with JWT
- API keys stored as environment variables
- Audit logging for all sensitive operations

## Testing

```bash
# Run all tests
pnpm test

# Run tests in watch mode
pnpm test:watch

# Type checking
pnpm check
```

## Deployment

### Manus Platform

1. Create a checkpoint: Click "Save Checkpoint" in the UI
2. Publish: Click "Publish" button in the Management UI header
3. Configure secrets in Settings → Secrets panel

### Self-Hosted

1. Build the application:
   ```bash
   pnpm build
   ```

2. Set production environment variables

3. Start the server:
   ```bash
   pnpm start
   ```

## Norwegian Accounting Standards

Skatt-Flow OS is designed to comply with Norwegian accounting standards:

- **Regnskapsloven**: Norwegian Accounting Act compliance
- **Bokføringsloven**: Bookkeeping Act requirements
- **MVA-loven**: VAT law and reporting
- **SAF-T**: Standard Audit File - Tax format

## License

MIT License - See LICENSE file for details.

## Support

For support and feature requests, please open an issue in the repository.

---

Built with ❤️ for Norwegian businesses
