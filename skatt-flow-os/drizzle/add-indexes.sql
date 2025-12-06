-- ============================================================================
-- DATABASE INDEXES FOR PERFORMANCE OPTIMIZATION
-- Run this migration to add indexes on frequently queried columns
-- ============================================================================

-- Companies table indexes
CREATE INDEX IF NOT EXISTS idx_companies_org_number ON companies(orgNumber);
CREATE INDEX IF NOT EXISTS idx_companies_status ON companies(status);
CREATE INDEX IF NOT EXISTS idx_companies_forvalt_risk ON companies(forvaltRiskClass);
CREATE INDEX IF NOT EXISTS idx_companies_created_at ON companies(createdAt);

-- Accounting documents indexes
CREATE INDEX IF NOT EXISTS idx_accounting_docs_company ON accountingDocuments(companyId);
CREATE INDEX IF NOT EXISTS idx_accounting_docs_status ON accountingDocuments(status);
CREATE INDEX IF NOT EXISTS idx_accounting_docs_source_type ON accountingDocuments(sourceType);
CREATE INDEX IF NOT EXISTS idx_accounting_docs_created_at ON accountingDocuments(createdAt);
CREATE INDEX IF NOT EXISTS idx_accounting_docs_company_status ON accountingDocuments(companyId, status);

-- Ledger entries indexes
CREATE INDEX IF NOT EXISTS idx_ledger_company ON ledgerEntries(companyId);
CREATE INDEX IF NOT EXISTS idx_ledger_entry_date ON ledgerEntries(entryDate);
CREATE INDEX IF NOT EXISTS idx_ledger_debit_account ON ledgerEntries(debitAccount);
CREATE INDEX IF NOT EXISTS idx_ledger_credit_account ON ledgerEntries(creditAccount);
CREATE INDEX IF NOT EXISTS idx_ledger_company_date ON ledgerEntries(companyId, entryDate);

-- Filings indexes
CREATE INDEX IF NOT EXISTS idx_filings_company ON filings(companyId);
CREATE INDEX IF NOT EXISTS idx_filings_type ON filings(filingType);
CREATE INDEX IF NOT EXISTS idx_filings_status ON filings(status);
CREATE INDEX IF NOT EXISTS idx_filings_period ON filings(periodStart, periodEnd);
CREATE INDEX IF NOT EXISTS idx_filings_company_type ON filings(companyId, filingType);

-- Forvalt snapshots indexes
CREATE INDEX IF NOT EXISTS idx_forvalt_company ON forvaltSnapshots(companyId);
CREATE INDEX IF NOT EXISTS idx_forvalt_fetched_at ON forvaltSnapshots(fetchedAt);

-- Document templates indexes
CREATE INDEX IF NOT EXISTS idx_templates_type ON documentTemplates(templateType);
CREATE INDEX IF NOT EXISTS idx_templates_active ON documentTemplates(isActive);

-- Generated documents indexes
CREATE INDEX IF NOT EXISTS idx_generated_docs_company ON generatedDocuments(companyId);
CREATE INDEX IF NOT EXISTS idx_generated_docs_template ON generatedDocuments(templateId);
CREATE INDEX IF NOT EXISTS idx_generated_docs_created_at ON generatedDocuments(createdAt);

-- Chat messages indexes
CREATE INDEX IF NOT EXISTS idx_chat_company ON chatMessages(companyId);
CREATE INDEX IF NOT EXISTS idx_chat_user ON chatMessages(userId);
CREATE INDEX IF NOT EXISTS idx_chat_session ON chatMessages(sessionId);
CREATE INDEX IF NOT EXISTS idx_chat_created_at ON chatMessages(createdAt);

-- API logs indexes
CREATE INDEX IF NOT EXISTS idx_api_logs_endpoint ON apiLogs(endpoint);
CREATE INDEX IF NOT EXISTS idx_api_logs_status ON apiLogs(statusCode);
CREATE INDEX IF NOT EXISTS idx_api_logs_created_at ON apiLogs(createdAt);

-- User company access indexes
CREATE INDEX IF NOT EXISTS idx_user_access_user ON userCompanyAccess(userId);
CREATE INDEX IF NOT EXISTS idx_user_access_company ON userCompanyAccess(companyId);
CREATE INDEX IF NOT EXISTS idx_user_access_role ON userCompanyAccess(role);

-- Bank accounts indexes
CREATE INDEX IF NOT EXISTS idx_bank_accounts_company ON bankAccounts(companyId);
CREATE INDEX IF NOT EXISTS idx_bank_accounts_number ON bankAccounts(accountNumber);

-- Audit logs indexes (if table exists)
-- CREATE INDEX IF NOT EXISTS idx_audit_logs_user ON auditLogs(userId);
-- CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON auditLogs(action);
-- CREATE INDEX IF NOT EXISTS idx_audit_logs_entity ON auditLogs(entityType, entityId);
-- CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON auditLogs(createdAt);
