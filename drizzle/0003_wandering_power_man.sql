CREATE TABLE `analyses` (
	`id` varchar(128) NOT NULL,
	`userId` int NOT NULL,
	`organizationNumber` varchar(9) NOT NULL,
	`companyName` varchar(255) NOT NULL,
	`digitalMaturityScore` int NOT NULL,
	`dataCompleteness` int NOT NULL,
	`competitivePosition` enum('leader','challenger','follower','niche') NOT NULL,
	`industryCode` varchar(10),
	`industryName` varchar(255),
	`industryCategory` varchar(50),
	`analysisData` text NOT NULL,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `analyses_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `executions` (
	`id` varchar(128) NOT NULL,
	`analysisId` varchar(128) NOT NULL,
	`workflowId` varchar(128) NOT NULL,
	`status` enum('pending','running','completed','failed','cancelled') NOT NULL DEFAULT 'pending',
	`progress` int NOT NULL DEFAULT 0,
	`recommendationIds` text NOT NULL,
	`results` text,
	`metrics` text,
	`errors` text,
	`startedAt` timestamp,
	`completedAt` timestamp,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `executions_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `recommendations` (
	`id` varchar(128) NOT NULL,
	`analysisId` varchar(128) NOT NULL,
	`title` varchar(255) NOT NULL,
	`description` text NOT NULL,
	`category` enum('acquisition','optimization','retention') NOT NULL,
	`priority` enum('high','medium','low') NOT NULL,
	`estimatedCost` int,
	`estimatedROI` int,
	`estimatedTime` int,
	`confidenceScore` int,
	`implementationSteps` text,
	`platforms` text,
	`expectedImpact` text,
	`status` enum('pending','approved','executing','completed','failed') NOT NULL DEFAULT 'pending',
	`approvedAt` timestamp,
	`completedAt` timestamp,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `recommendations_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `revenue_tracking` (
	`id` varchar(128) NOT NULL,
	`analysisId` varchar(128) NOT NULL,
	`executionId` varchar(128),
	`revenue` int,
	`customers` int,
	`newCustomers` int,
	`websiteTraffic` int,
	`websiteConversionRate` int,
	`linkedinFollowers` int,
	`linkedinEngagement` int,
	`socialMediaFollowers` int,
	`socialMediaEngagement` int,
	`averageRating` int,
	`totalReviews` int,
	`periodStart` timestamp NOT NULL,
	`periodEnd` timestamp NOT NULL,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `revenue_tracking_id` PRIMARY KEY(`id`)
);
