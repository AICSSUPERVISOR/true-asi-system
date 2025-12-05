CREATE TABLE `companies` (
	`id` varchar(128) NOT NULL,
	`userId` int NOT NULL,
	`orgnr` varchar(9) NOT NULL,
	`name` text NOT NULL,
	`organizationForm` varchar(50),
	`organizationFormDescription` text,
	`registrationDate` varchar(10),
	`industryCode` varchar(10),
	`industryDescription` text,
	`employees` int,
	`businessAddress` text,
	`postalAddress` text,
	`municipality` varchar(100),
	`municipalityNumber` varchar(4),
	`vatRegistered` int,
	`registeredInBusinessRegistry` int,
	`bankrupt` int,
	`underLiquidation` int,
	`rawData` text,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `companies_id` PRIMARY KEY(`id`),
	CONSTRAINT `companies_orgnr_unique` UNIQUE(`orgnr`)
);
--> statement-breakpoint
CREATE TABLE `company_financials` (
	`id` varchar(128) NOT NULL,
	`companyId` varchar(128) NOT NULL,
	`year` int NOT NULL,
	`revenue` int,
	`profit` int,
	`assets` int,
	`liabilities` int,
	`equity` int,
	`creditRating` varchar(10),
	`creditScore` int,
	`riskLevel` varchar(20),
	`rawData` text,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `company_financials_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `company_linkedin` (
	`id` varchar(128) NOT NULL,
	`companyId` varchar(128) NOT NULL,
	`linkedinUrl` text,
	`followerCount` int,
	`employeeCount` int,
	`description` text,
	`specialties` text,
	`website` text,
	`industry` varchar(100),
	`companySize` varchar(50),
	`rawData` text,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `company_linkedin_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `company_roles` (
	`id` varchar(128) NOT NULL,
	`companyId` varchar(128) NOT NULL,
	`roleType` varchar(10) NOT NULL,
	`roleTypeDescription` text,
	`personName` text,
	`personBirthDate` varchar(10),
	`organizationNumber` varchar(9),
	`organizationName` text,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `company_roles_id` PRIMARY KEY(`id`)
);
