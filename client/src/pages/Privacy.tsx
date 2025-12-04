/**
 * Privacy Policy Page
 * GDPR-compliant privacy policy for TRUE ASI System
 */

export default function Privacy() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="container mx-auto px-4 py-16 max-w-4xl">
        <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 shadow-2xl">
          <h1 className="text-4xl font-bold text-white mb-2">Privacy Policy</h1>
          <p className="text-gray-300 mb-8">Last Updated: December 4, 2025</p>

          <div className="space-y-8 text-gray-200">
            {/* Introduction */}
            <section>
              <h2 className="text-2xl font-semibold text-white mb-4">Introduction</h2>
              <p className="mb-4">
                TRUE ASI System ("we," "our," or "us") is committed to protecting your privacy. This Privacy Policy explains
                how we collect, use, disclose, and safeguard your information when you use our artificial superintelligence
                platform and services (the "Service").
              </p>
              <p>
                Please read this privacy policy carefully. If you do not agree with the terms of this privacy policy, please
                do not access the Service.
              </p>
            </section>

            {/* 1. Information We Collect */}
            <section>
              <h2 className="text-2xl font-semibold text-white mb-4">1. Information We Collect</h2>
              
              <h3 className="text-xl font-semibold text-white mb-3 mt-4">1.1 Personal Information</h3>
              <p className="mb-4">We may collect personal information that you voluntarily provide to us when you:</p>
              <ul className="list-disc list-inside space-y-2 ml-4 mb-4">
                <li>Register for an account</li>
                <li>Use the Service</li>
                <li>Contact us for support</li>
                <li>Subscribe to our newsletter</li>
              </ul>
              <p className="mb-4">This information may include:</p>
              <ul className="list-disc list-inside space-y-2 ml-4">
                <li>Name and email address</li>
                <li>Account credentials</li>
                <li>Profile information</li>
                <li>Payment information (processed securely by third-party payment processors)</li>
                <li>Communication preferences</li>
              </ul>

              <h3 className="text-xl font-semibold text-white mb-3 mt-4">1.2 Usage Information</h3>
              <p className="mb-4">We automatically collect certain information when you use the Service, including:</p>
              <ul className="list-disc list-inside space-y-2 ml-4">
                <li>IP address and device information</li>
                <li>Browser type and version</li>
                <li>Pages visited and time spent on pages</li>
                <li>Referring website addresses</li>
                <li>Operating system and platform</li>
                <li>AI model usage and query history</li>
                <li>Performance metrics and error logs</li>
              </ul>

              <h3 className="text-xl font-semibold text-white mb-3 mt-4">1.3 AI Interaction Data</h3>
              <p className="mb-4">
                When you interact with our AI agents and models, we collect:
              </p>
              <ul className="list-disc list-inside space-y-2 ml-4">
                <li>Your prompts and queries</li>
                <li>AI-generated responses</li>
                <li>Feedback and ratings</li>
                <li>Usage patterns and preferences</li>
                <li>S-7 test submissions and scores</li>
              </ul>
            </section>

            {/* 2. How We Use Your Information */}
            <section>
              <h2 className="text-2xl font-semibold text-white mb-4">2. How We Use Your Information</h2>
              <p className="mb-4">We use the information we collect to:</p>
              <ul className="list-disc list-inside space-y-2 ml-4">
                <li>Provide, operate, and maintain the Service</li>
                <li>Improve, personalize, and expand the Service</li>
                <li>Understand and analyze how you use the Service</li>
                <li>Develop new products, services, features, and functionality</li>
                <li>Communicate with you for customer service, updates, and marketing purposes</li>
                <li>Process your transactions and manage your account</li>
                <li>Send you technical notices, updates, and security alerts</li>
                <li>Detect, prevent, and address technical issues and security threats</li>
                <li>Train and improve our AI models (with anonymized data)</li>
                <li>Comply with legal obligations</li>
              </ul>
            </section>

            {/* 3. Data Sharing and Disclosure */}
            <section>
              <h2 className="text-2xl font-semibold text-white mb-4">3. Data Sharing and Disclosure</h2>
              <p className="mb-4">We may share your information in the following situations:</p>
              
              <h3 className="text-xl font-semibold text-white mb-3 mt-4">3.1 Service Providers</h3>
              <p className="mb-4">
                We may share your information with third-party service providers who perform services on our behalf, including:
              </p>
              <ul className="list-disc list-inside space-y-2 ml-4">
                <li>Cloud hosting providers (AWS)</li>
                <li>Payment processors (Stripe)</li>
                <li>Analytics providers</li>
                <li>AI model providers (OpenAI, Anthropic, Google, etc.)</li>
                <li>Email service providers</li>
              </ul>

              <h3 className="text-xl font-semibold text-white mb-3 mt-4">3.2 Legal Requirements</h3>
              <p className="mb-4">
                We may disclose your information if required to do so by law or in response to valid requests by public
                authorities (e.g., a court or government agency).
              </p>

              <h3 className="text-xl font-semibold text-white mb-3 mt-4">3.3 Business Transfers</h3>
              <p className="mb-4">
                If we are involved in a merger, acquisition, or asset sale, your information may be transferred as part of that
                transaction.
              </p>

              <h3 className="text-xl font-semibold text-white mb-3 mt-4">3.4 With Your Consent</h3>
              <p>
                We may disclose your personal information for any other purpose with your consent.
              </p>
            </section>

            {/* 4. Data Security */}
            <section>
              <h2 className="text-2xl font-semibold text-white mb-4">4. Data Security</h2>
              <p className="mb-4">
                We implement appropriate technical and organizational security measures to protect your personal information,
                including:
              </p>
              <ul className="list-disc list-inside space-y-2 ml-4">
                <li>Encryption of data in transit and at rest</li>
                <li>Regular security assessments and audits</li>
                <li>Access controls and authentication mechanisms</li>
                <li>Secure cloud infrastructure (AWS S3, TiDB)</li>
                <li>Regular backups and disaster recovery procedures</li>
                <li>Employee training on data protection</li>
              </ul>
              <p className="mt-4">
                However, no method of transmission over the Internet or electronic storage is 100% secure. While we strive to
                use commercially acceptable means to protect your personal information, we cannot guarantee its absolute
                security.
              </p>
            </section>

            {/* 5. Data Retention */}
            <section>
              <h2 className="text-2xl font-semibold text-white mb-4">5. Data Retention</h2>
              <p className="mb-4">
                We retain your personal information only for as long as necessary to fulfill the purposes outlined in this
                Privacy Policy, unless a longer retention period is required or permitted by law.
              </p>
              <p className="mb-4">Specifically:</p>
              <ul className="list-disc list-inside space-y-2 ml-4">
                <li>Account information: Retained until account deletion</li>
                <li>Usage logs: Retained for 90 days</li>
                <li>AI interaction data: Retained for 1 year (anonymized after 30 days)</li>
                <li>Payment records: Retained for 7 years (legal requirement)</li>
                <li>Support communications: Retained for 3 years</li>
              </ul>
            </section>

            {/* 6. Your Data Rights (GDPR) */}
            <section>
              <h2 className="text-2xl font-semibold text-white mb-4">6. Your Data Rights (GDPR)</h2>
              <p className="mb-4">
                If you are a resident of the European Economic Area (EEA), you have certain data protection rights under the
                General Data Protection Regulation (GDPR):
              </p>
              <ul className="list-disc list-inside space-y-2 ml-4">
                <li><strong>Right to Access:</strong> You can request copies of your personal data</li>
                <li><strong>Right to Rectification:</strong> You can request correction of inaccurate data</li>
                <li><strong>Right to Erasure:</strong> You can request deletion of your personal data</li>
                <li><strong>Right to Restrict Processing:</strong> You can request restriction of processing</li>
                <li><strong>Right to Data Portability:</strong> You can request transfer of your data</li>
                <li><strong>Right to Object:</strong> You can object to processing of your personal data</li>
                <li><strong>Right to Withdraw Consent:</strong> You can withdraw consent at any time</li>
              </ul>
              <p className="mt-4">
                To exercise these rights, please contact us at privacy@trueasi.com. We will respond to your request within 30
                days.
              </p>
            </section>

            {/* 7. Cookies and Tracking */}
            <section>
              <h2 className="text-2xl font-semibold text-white mb-4">7. Cookies and Tracking Technologies</h2>
              <p className="mb-4">
                We use cookies and similar tracking technologies to track activity on our Service and hold certain information.
                Cookies are files with small amounts of data that are sent to your browser from a website and stored on your
                device.
              </p>
              
              <h3 className="text-xl font-semibold text-white mb-3 mt-4">Types of Cookies We Use:</h3>
              <ul className="list-disc list-inside space-y-2 ml-4">
                <li><strong>Essential Cookies:</strong> Required for the Service to function (e.g., authentication)</li>
                <li><strong>Performance Cookies:</strong> Help us understand how visitors use the Service</li>
                <li><strong>Functionality Cookies:</strong> Remember your preferences and settings</li>
              </ul>
              <p className="mt-4">
                You can instruct your browser to refuse all cookies or to indicate when a cookie is being sent. However, if you
                do not accept cookies, you may not be able to use some portions of our Service.
              </p>
            </section>

            {/* 8. Third-Party Links */}
            <section>
              <h2 className="text-2xl font-semibold text-white mb-4">8. Third-Party Links</h2>
              <p>
                Our Service may contain links to third-party websites that are not operated by us. If you click on a third-party
                link, you will be directed to that third party's site. We strongly advise you to review the Privacy Policy of
                every site you visit. We have no control over and assume no responsibility for the content, privacy policies, or
                practices of any third-party sites or services.
              </p>
            </section>

            {/* 9. Children's Privacy */}
            <section>
              <h2 className="text-2xl font-semibold text-white mb-4">9. Children's Privacy</h2>
              <p>
                Our Service is not intended for use by children under the age of 13. We do not knowingly collect personally
                identifiable information from children under 13. If you are a parent or guardian and you are aware that your
                child has provided us with personal information, please contact us. If we become aware that we have collected
                personal information from children without verification of parental consent, we take steps to remove that
                information from our servers.
              </p>
            </section>

            {/* 10. International Data Transfers */}
            <section>
              <h2 className="text-2xl font-semibold text-white mb-4">10. International Data Transfers</h2>
              <p className="mb-4">
                Your information, including personal data, may be transferred to — and maintained on — computers located outside
                of your state, province, country, or other governmental jurisdiction where the data protection laws may differ
                from those of your jurisdiction.
              </p>
              <p>
                If you are located outside the United States and choose to provide information to us, please note that we
                transfer the data, including personal data, to the United States and process it there. Your consent to this
                Privacy Policy followed by your submission of such information represents your agreement to that transfer.
              </p>
            </section>

            {/* 11. Changes to Privacy Policy */}
            <section>
              <h2 className="text-2xl font-semibold text-white mb-4">11. Changes to This Privacy Policy</h2>
              <p className="mb-4">
                We may update our Privacy Policy from time to time. We will notify you of any changes by posting the new Privacy
                Policy on this page and updating the "Last Updated" date.
              </p>
              <p>
                You are advised to review this Privacy Policy periodically for any changes. Changes to this Privacy Policy are
                effective when they are posted on this page.
              </p>
            </section>

            {/* 12. Contact Us */}
            <section>
              <h2 className="text-2xl font-semibold text-white mb-4">12. Contact Us</h2>
              <p className="mb-2">If you have any questions about this Privacy Policy, please contact us:</p>
              <div className="bg-white/5 rounded-lg p-4 mt-4">
                <p className="font-semibold">TRUE ASI System</p>
                <p>Email: privacy@trueasi.com</p>
                <p>Data Protection Officer: dpo@trueasi.com</p>
                <p>Website: https://trueasi.com</p>
              </div>
            </section>
          </div>

          {/* Back to Home */}
          <div className="mt-12 pt-8 border-t border-white/20">
            <a
              href="/"
              className="inline-flex items-center text-purple-400 hover:text-purple-300 transition-colors"
            >
              ← Back to Home
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}
