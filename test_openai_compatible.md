# OpenAI Compatible Provider Test Configuration

This file provides a configuration example and verification for the OpenAI Compatible provider.

## Example Configuration

Add this to your `settings.json`:

```json
{
  "language_models": {
    "openai_compatible": {
      "api_url": "https://api.x.ai/v1",
      "available_models": [
        {
          "name": "grok-beta",
          "display_name": "X.ai Grok (Beta)",
          "max_tokens": 131072
        },
        {
          "name": "custom-model",
          "display_name": "My Custom Model",
          "max_tokens": 4096,
          "max_output_tokens": 1024
        }
      ]
    }
  }
}
```

## Environment Variable Support

You can also set your API key via environment variable:

```bash
export OPENAI_COMPATIBLE_API_KEY="your-api-key-here"
```

## Features

- ✅ Tool use support
- ✅ Custom model configuration
- ✅ Configurable API endpoint
- ✅ Environment variable API key support
- ✅ Credential storage in keychain
- ✅ Simultaneous use with OpenAI provider

## Provider Details

- **Provider ID**: `openai-compatible`
- **Provider Name**: `OpenAI Compatible`
- **Default API URL**: `https://your-openai-compatible-service.com/v1`
- **Environment Variable**: `OPENAI_COMPATIBLE_API_KEY`

## Configuration through UI

1. Open Zed settings (`agent: open configuration`)
2. Navigate to the "OpenAI Compatible" section
3. First, set your API URL in settings.json
4. Then enter your API key in the UI
5. Configure available models if needed