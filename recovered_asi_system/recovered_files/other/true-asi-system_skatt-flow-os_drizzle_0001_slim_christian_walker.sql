CREATE TABLE `accountingDocuments` (
	`id` int AUTO_INCREMENT NOT NULL,
	`companyId` int NOT NULL,
	`sourceType` enum('INVOICE_SUPPLIER','INVOICE_CUSTOMER','RECEIPT','CONTRACT','OTHER') NOT NULL,
	`originalFileUrl` text,
	`originalFileName` varchar(255),
	`parsedJson` json,
	`status` enum('NEW','PROCESSED','POSTED','REJECTED') NOT NULL DEFAULT 'NEW',
	`suggestedAccount` varchar(20),
	`suggestedVatCode` varchar(10),
	`suggestedDescription` text,
	`suggestedAmount` bigint,
	`postedVoucherId` varchar(100),
	`postedAt` timestamp,
	`postedById` int,
	`rejectedReason` text,
	`rejectedAt` timestamp,
	`rejectedById` int,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `accountingDocuments_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `apiLogs` (
	`id` int AUTO_INCREMENT NOT NULL,
	`companyId` int,
	`userId` int,
	`endpoint` varchar(500) NOT NULL,
	`method` varchar(10) NOT NULL,
	`statusCode` int,
	`correlationId` varchar(64) NOT NULL,
	`durationMs` int,
	`errorMessage` text,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `apiLogs_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `bankAccounts` (
	`id` int AUTO_INCREMENT NOT NULL,
	`companyId` int NOT NULL,
	`bankName` varchar(100) NOT NULL,
	`ibanOrAccountNo` varchar(50) NOT NULL,
	`currency` varchar(3) NOT NULL DEFAULT 'NOK',
	`openBankingProvider` varchar(100),
	`isActive` boolean NOT NULL DEFAULT true,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `bankAccounts_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `chatMessages` (
	`id` int AUTO_INCREMENT NOT NULL,
	`userId` int NOT NULL,
	`companyId` int,
	`sessionId` varchar(64) NOT NULL,
	`role` enum('user','assistant','system') NOT NULL,
	`content` text NOT NULL,
	`attachedDocumentId` int,
	`attachedFilingId` int,
	`metadata` json,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `chatMessages_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `companies` (
	`id` int AUTO_INCREMENT NOT NULL,
	`orgNumber` varchar(20) NOT NULL,
	`name` varchar(255) NOT NULL,
	`country` varchar(2) NOT NULL DEFAULT 'NO',
	`address` text,
	`city` varchar(100),
	`postalCode` varchar(10),
	`industryCode` varchar(20),
	`forvaltRating` varchar(10),
	`forvaltCreditScore` int,
	`forvaltRiskClass` varchar(20),
	`externalRegnskapSystem` enum('TRIPLETEX','POWEROFFICE','FIKEN','VISMA_EACCOUNTING','OTHER'),
	`externalRegnskapCompanyId` varchar(100),
	`autoPostEnabled` boolean NOT NULL DEFAULT false,
	`createdById` int,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `companies_id` PRIMARY KEY(`id`),
	CONSTRAINT `companies_orgNumber_unique` UNIQUE(`orgNumber`)
);
--> statement-breakpoint
CREATE TABLE `documentTemplates` (
	`id` int AUTO_INCREMENT NOT NULL,
	`name` varchar(255) NOT NULL,
	`category` enum('CONTRACT','HR','LEGAL','FINANCIAL','GOVERNANCE','OTHER') NOT NULL,
	`source` enum('BUSINESS_IN_A_BOX','CUSTOM') NOT NULL DEFAULT 'CUSTOM',
	`language` varchar(5) NOT NULL DEFAULT 'no',
	`description` text,
	`bodyMarkdown` text NOT NULL,
	`variablesJson` json,
	`isActive` boolean NOT NULL DEFAULT true,
	`createdById` int,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `documentTemplates_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `filings` (
	`id` int AUTO_INCREMENT NOT NULL,
	`companyId` int NOT NULL,
	`filingType` enum('MVA_MELDING','A_MELDING_SUMMARY','SAF_T','ARSREGNSKAP','OTHER') NOT NULL,
	`periodStart` timestamp NOT NULL,
	`periodEnd` timestamp NOT NULL,
	`status` enum('DRAFT','READY_FOR_REVIEW','SUBMITTED','ERROR') NOT NULL DEFAULT 'DRAFT',
	`altinnServiceCode` varchar(50),
	`altinnReference` varchar(100),
	`altinnDraftId` varchar(100),
	`payloadJson` json,
	`summaryJson` json,
	`errorMessage` text,
	`submittedAt` timestamp,
	`submittedById` int,
	`createdById` int,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `filings_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `forvaltSnapshots` (
	`id` int AUTO_INCREMENT NOT NULL,
	`companyId` int NOT NULL,
	`rawJson` json NOT NULL,
	`rating` varchar(10),
	`creditScore` int,
	`riskClass` varchar(20),
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `forvaltSnapshots_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `generatedDocuments` (
	`id` int AUTO_INCREMENT NOT NULL,
	`companyId` int NOT NULL,
	`templateId` int NOT NULL,
	`title` varchar(255) NOT NULL,
	`filledVariablesJson` json,
	`outputMarkdown` text NOT NULL,
	`outputFileUrl` text,
	`createdById` int,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `generatedDocuments_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `ledgerEntries` (
	`id` int AUTO_INCREMENT NOT NULL,
	`companyId` int NOT NULL,
	`externalSystem` enum('TRIPLETEX','POWEROFFICE','FIKEN','VISMA_EACCOUNTING','OTHER'),
	`externalId` varchar(100),
	`entryDate` timestamp NOT NULL,
	`description` text,
	`debitAccount` varchar(20) NOT NULL,
	`creditAccount` varchar(20) NOT NULL,
	`amount` bigint NOT NULL,
	`currency` varchar(3) NOT NULL DEFAULT 'NOK',
	`vatCode` varchar(10),
	`voucherNumber` varchar(50),
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `ledgerEntries_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `userCompanyAccess` (
	`id` int AUTO_INCREMENT NOT NULL,
	`userId` int NOT NULL,
	`companyId` int NOT NULL,
	`accessRole` enum('OWNER','ADMIN','ACCOUNTANT','VIEWER') NOT NULL DEFAULT 'VIEWER',
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `userCompanyAccess_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
ALTER TABLE `users` ADD `accountingRole` enum('OWNER','ADMIN','ACCOUNTANT','VIEWER') DEFAULT 'VIEWER' NOT NULL;