# Skatt-Flow OS - API Documentation

> Complete API reference for developers

---

## Overview

Skatt-Flow OS uses **tRPC** for type-safe API communication. All endpoints are available under `/api/trpc`.

### Authentication

All protected endpoints require a valid session cookie obtained through Manus OAuth.

```typescript
// Client-side authentication check
const { user, isAuthenticated } = useAuth();
```

### Base URL

```
Production: https://your-domain.com/api/trpc
Development: http://localhost:3000/api/trpc
```

---

## Routers

### Auth Router

#### `auth.me`
Get current authenticated user.

```typescript
// Query
const { data: user } = trpc.auth.me.useQuery();

// Response
{
  id: number;
  openId: string;
  name: string | null;
  email: string | null;
  role: "user" | "admin";
  createdAt: Date;
  updatedAt: Date;
  lastSignedIn: Date;
}
```

#### `auth.logout`
Log out current user.

```typescript
// Mutation
const logout = trpc.auth.logout.useMutation();
await logout.mutateAsync();

// Response
{ success: true }
```

---

### Company Router

#### `company.list`
List all companies accessible to the user.

```typescript
// Query
const { data: companies } = trpc.company.list.useQuery();

// Response
Array<{
  id: number;
  name: string;
  orgNumber: string;
  status: "ACTIVE" | "INACTIVE" | "PENDING";
  forvaltRating: string | null;
  forvaltRiskClass: string | null;
  createdAt: Date;
}>
```

#### `company.get`
Get a single company by ID.

```typescript
// Query
const { data: company } = trpc.company.get.useQuery({ id: 1 });

// Input
{ id: number }

// Response
{
  id: number;
  name: string;
  orgNumber: string;
  address: string | null;
  postalCode: string | null;
  city: string | null;
  country: string;
  industry: string | null;
  naceCode: string | null;
  status: string;
  forvaltRating: string | null;
  forvaltRiskClass: string | null;
  forvaltCreditLimit: number | null;
  forvaltLastUpdated: Date | null;
  regnskapSystemType: string | null;
  regnskapSystemTenantId: string | null;
  autoPostEnabled: boolean;
  createdAt: Date;
  updatedAt: Date;
}
```

#### `company.create`
Create a new company.

```typescript
// Mutation
const create = trpc.company.create.useMutation();
await create.mutateAsync({
  name: "Company AS",
  orgNumber: "123456789",
  address: "Street 1",
  postalCode: "0001",
  city: "Oslo",
});

// Input
{
  name: string;
  orgNumber: string;
  address?: string;
  postalCode?: string;
  city?: string;
  country?: string;
  industry?: string;
  naceCode?: string;
}

// Response
{ id: number }
```

#### `company.update`
Update an existing company.

```typescript
// Mutation
const update = trpc.company.update.useMutation();
await update.mutateAsync({
  id: 1,
  name: "Updated Name AS",
});

// Input
{
  id: number;
  name?: string;
  address?: string;
  postalCode?: string;
  city?: string;
  status?: "ACTIVE" | "INACTIVE" | "PENDING";
  autoPostEnabled?: boolean;
}

// Response
{ success: true }
```

#### `company.enrichFromForvalt`
Fetch and update company data from Forvalt.

```typescript
// Mutation
const enrich = trpc.company.enrichFromForvalt.useMutation();
await enrich.mutateAsync({ id: 1 });

// Input
{ id: number }

// Response
{
  success: boolean;
  rating: string | null;
  riskClass: string | null;
  creditLimit: number | null;
}
```

---

### Accounting Document Router

#### `accountingDocument.list`
List documents for a company.

```typescript
// Query
const { data } = trpc.accountingDocument.list.useQuery({
  companyId: 1,
  status: "NEW",
});

// Input
{
  companyId: number;
  status?: "NEW" | "PROCESSING" | "PENDING_APPROVAL" | "APPROVED" | "POSTED" | "REJECTED";
  sourceType?: string;
}

// Response
Array<{
  id: number;
  companyId: number;
  sourceType: string;
  status: string;
  fileUrl: string;
  originalFileName: string;
  extractedData: object | null;
  suggestedEntry: object | null;
  createdAt: Date;
}>
```

#### `accountingDocument.create`
Upload a new document.

```typescript
// Mutation
const create = trpc.accountingDocument.create.useMutation();
await create.mutateAsync({
  companyId: 1,
  sourceType: "INVOICE_SUPPLIER",
  fileUrl: "https://storage.example.com/doc.pdf",
  originalFileName: "invoice.pdf",
});

// Input
{
  companyId: number;
  sourceType: "INVOICE_SUPPLIER" | "INVOICE_CUSTOMER" | "RECEIPT" | "BANK_STATEMENT" | "OTHER";
  fileUrl: string;
  originalFileName: string;
}

// Response
{ id: number }
```

#### `accountingDocument.process`
Process a document with AI.

```typescript
// Mutation
const process = trpc.accountingDocument.process.useMutation();
await process.mutateAsync({ id: 1 });

// Input
{ id: number }

// Response
{
  success: boolean;
  extractedData: object;
  suggestedEntry: {
    debitAccount: string;
    creditAccount: string;
    amount: number;
    vatCode: string;
    description: string;
  };
}
```

#### `accountingDocument.approve`
Approve and post a document.

```typescript
// Mutation
const approve = trpc.accountingDocument.approve.useMutation();
await approve.mutateAsync({
  id: 1,
  debitAccount: "4000",
  creditAccount: "2400",
  amount: 10000,
  vatCode: "1",
});

// Input
{
  id: number;
  debitAccount: string;
  creditAccount: string;
  amount: number;
  vatCode: string;
  description?: string;
}

// Response
{ success: true; ledgerEntryId: number }
```

---

### Ledger Entry Router

#### `ledgerEntry.list`
List ledger entries for a company.

```typescript
// Query
const { data } = trpc.ledgerEntry.list.useQuery({
  companyId: 1,
  fromDate: new Date("2024-01-01"),
  toDate: new Date("2024-12-31"),
});

// Input
{
  companyId: number;
  fromDate?: Date;
  toDate?: Date;
  account?: string;
}

// Response
Array<{
  id: number;
  companyId: number;
  entryDate: Date;
  voucherNumber: string;
  description: string;
  debitAccount: string;
  creditAccount: string;
  amount: number;
  vatCode: string | null;
  vatAmount: number | null;
  createdAt: Date;
}>
```

---

### Filing Router

#### `filing.list`
List filings for a company.

```typescript
// Query
const { data } = trpc.filing.list.useQuery({ companyId: 1 });

// Input
{
  companyId: number;
  filingType?: "MVA_MELDING" | "SAFT" | "A_MELDING";
  status?: string;
}

// Response
Array<{
  id: number;
  companyId: number;
  filingType: string;
  status: string;
  periodStart: Date;
  periodEnd: Date;
  summaryJson: object | null;
  altinnReferenceId: string | null;
  submittedAt: Date | null;
  createdAt: Date;
}>
```

#### `filing.create`
Create a new filing.

```typescript
// Mutation
const create = trpc.filing.create.useMutation();
await create.mutateAsync({
  companyId: 1,
  filingType: "MVA_MELDING",
  periodStart: new Date("2024-01-01"),
  periodEnd: new Date("2024-02-28"),
});

// Input
{
  companyId: number;
  filingType: "MVA_MELDING" | "SAFT" | "A_MELDING";
  periodStart: Date;
  periodEnd: Date;
}

// Response
{ id: number }
```

#### `filing.submit`
Submit filing to Altinn.

```typescript
// Mutation
const submit = trpc.filing.submit.useMutation();
await submit.mutateAsync({ id: 1 });

// Input
{ id: number }

// Response
{
  success: boolean;
  altinnReferenceId: string;
  submittedAt: Date;
}
```

---

### Chat Router

#### `chat.send`
Send a message to the AI assistant.

```typescript
// Mutation
const send = trpc.chat.send.useMutation();
const response = await send.mutateAsync({
  companyId: 1,
  message: "What is the VAT rate for food?",
  sessionId: "session-123",
});

// Input
{
  companyId: number;
  message: string;
  sessionId: string;
}

// Response
{
  message: string;
  actions: Array<{
    type: string;
    label: string;
    params: object;
    result: unknown;
  }>;
}
```

---

### Dashboard Router

#### `dashboard.stats`
Get dashboard statistics.

```typescript
// Query
const { data } = trpc.dashboard.stats.useQuery();

// Response
{
  unpostedDocuments: number;
  pendingFilings: number;
  highRiskCompanies: number;
  activeCompanies: number;
  recentActivities: Array<{
    type: string;
    description: string;
    timestamp: Date;
  }>;
  upcomingDeadlines: Array<{
    type: string;
    company: string;
    deadline: Date;
    daysRemaining: number;
  }>;
}
```

---

### Document Template Router

#### `documentTemplate.list`
List available templates.

```typescript
// Query
const { data } = trpc.documentTemplate.list.useQuery();

// Response
Array<{
  id: number;
  name: string;
  templateType: string;
  description: string | null;
  isActive: boolean;
  createdAt: Date;
}>
```

#### `documentTemplate.create`
Create a new template.

```typescript
// Mutation
const create = trpc.documentTemplate.create.useMutation();
await create.mutateAsync({
  name: "Invoice Template",
  templateType: "INVOICE",
  content: "# Invoice\n\n{{content}}",
});

// Input
{
  name: string;
  templateType: "INVOICE" | "CONTRACT" | "LETTER" | "REMINDER" | "REPORT";
  content: string;
  description?: string;
}

// Response
{ id: number }
```

---

### Generated Document Router

#### `generatedDocument.generate`
Generate a document from a template.

```typescript
// Mutation
const generate = trpc.generatedDocument.generate.useMutation();
const result = await generate.mutateAsync({
  templateId: 1,
  companyId: 1,
  variables: {
    customerName: "Customer AS",
    amount: "10000",
  },
});

// Input
{
  templateId: number;
  companyId: number;
  variables: Record<string, string>;
}

// Response
{
  id: number;
  fileUrl: string;
}
```

---

## Error Handling

All errors follow the tRPC error format:

```typescript
{
  code: "UNAUTHORIZED" | "FORBIDDEN" | "NOT_FOUND" | "BAD_REQUEST" | "INTERNAL_SERVER_ERROR";
  message: string;
}
```

### Common Error Codes

| Code | Description |
|------|-------------|
| `UNAUTHORIZED` | User not authenticated |
| `FORBIDDEN` | User lacks permission |
| `NOT_FOUND` | Resource not found |
| `BAD_REQUEST` | Invalid input |
| `INTERNAL_SERVER_ERROR` | Server error |

---

## Rate Limiting

- **Standard endpoints**: 100 requests/minute
- **AI endpoints**: 20 requests/minute
- **File upload**: 10 requests/minute

---

## Webhooks (Coming Soon)

Webhook support for:

- Document processed
- Filing submitted
- Risk alert triggered

---

*Last updated: December 2024*
