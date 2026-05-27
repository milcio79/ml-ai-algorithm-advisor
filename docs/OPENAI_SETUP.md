# OpenAI Setup

## Environment File

Copy the example file:

```powershell
cd algorithm_advisor
Copy-Item .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-5.5
```

You can also set variables in the current PowerShell session:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
$env:OPENAI_MODEL="gpt-5.5"
```

## Missing Key Behavior

If `OPENAI_API_KEY` is missing:

- deterministic recommendations still work
- **Rank with ChatGPT** shows a clear error
- no OpenAI request is made

## Model

`OPENAI_MODEL` defaults to:

```text
gpt-5.5
```

You can override it with another model available to your OpenAI account.

## Connection Errors

If you see:

```text
Connection error
```

check:

- internet connection
- firewall or antivirus rules
- corporate proxy settings
- whether the configured model is available to your API key

This app configures the OpenAI HTTP client with `trust_env=False` to avoid broken placeholder proxy settings such as `127.0.0.1:9`.

## Security

Never commit `.env`. Use `.env.example` for documentation only.

