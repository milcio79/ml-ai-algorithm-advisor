# Security Policy

## Secrets

Do not commit:

- `.env`
- OpenAI API keys
- Vercel tokens
- proxy credentials
- exported project profiles containing sensitive client data

Use `.env.example` as a template and keep real values local.

## Reporting Security Issues

If you find a security issue, please do not open a public issue with secrets or exploit details.

Contact the repository maintainer privately, or open a minimal issue saying that you would like to report a security concern.

## Data Handling

The deterministic recommender runs locally and does not send data to external services.

The ChatGPT ranking feature sends the selected project profile and candidate algorithm list to the OpenAI API only after the user clicks **Rank with ChatGPT**.

Avoid entering confidential, regulated, or client-sensitive data unless your organization allows sending that content to the configured OpenAI account.

