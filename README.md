# Batch MCP Wrapper - AI Provider Aggregator

A powerful MCP (Model Context Protocol) server that aggregates multiple AI providers from a CSV file into a single unified interface. Each provider's models are exposed as namespaced tools (e.g., `OpenRouter::gpt-4o`, `Pollinations::flux-1`) with automatic capability detection and authentication handling.

## 🚀 Quick Start

1. **Install dependencies:**
```bash
pip install fastapi uvicorn requests python-dotenv pandas
```

2. **Prepare your CSV file** with AI providers (see format below)

3. **Start the MCP server:**
```bash
python aggregator_server.py --csv g4f_providers_example.csv --port 12000
```

4. **Connect your LLM client** to `http://127.0.0.1:12000/api`

## 📋 CSV Format

Your CSV should contain the following columns:

| Name           | Base_URL                     | APIKey         | Models                              | AuthMode        | Template                            | Rate Limit/cost info |
|----------------|------------------------------|----------------|-------------------------------------|-----------------|-------------------------------------|----------------------|
| OpenRouter     | https://openrouter.ai/api/v1 | sk-xxxx        | gpt-4o\|claude-3-sonnet            | bearer          |                                     | 100 RPM              |
| Pollinations   | https://text.pollinations.ai |                | https://pollinations.ai/api/models  | none            |                                     | 10 RPM               |
| CustomAPI      | https://api.custom.com/v1    | api_key_123    | custom-model                        | header:X-Api-Key| custom_adapter.yaml                 | 1000 RPM             |

### Column Details:

- **Name**: Provider identifier (will be used as namespace)
- **Base_URL**: API endpoint URL
- **APIKey**: API key (will be stored in .env file as `{Name}API_Key`)
- **Models**: Can be:
  - Pipe-delimited list: `model1|model2|model3`
  - URL to model list endpoint: `https://api.provider.com/models`
  - Single model name: `gpt-4`
- **AuthMode** (optional, defaults to `bearer`):
  - `bearer` - Bearer token authentication
  - `header:HeaderName` - Custom header (e.g., `header:X-Api-Key`)
  - `query:param` - Query parameter (e.g., `query:api_key`)
  - `none` - No authentication
- **Template** (optional): Path/URL to custom YAML/JSON adapter
- **Rate Limit/cost info** (optional): Human-readable rate limit information

## 🔧 Advanced Configuration

### Authentication Modes

```csv
# Bearer token (default)
OpenAI,https://api.openai.com/v1,sk-xxxx,gpt-4,bearer

# Custom header
Anthropic,https://api.anthropic.com,key123,claude-3,header:x-api-key

# Query parameter
SomeAPI,https://api.example.com,secret,model1,query:apikey

# No authentication
PublicAPI,https://free.api.com,,public-model,none
```

### Custom Templates

For providers requiring special request/response handling, create a YAML template:

```yaml
# custom_adapter.yaml
request:
  text:
    method: POST
    url: "{Base_URL}/chat/completions"
    body:
      model: "{model_id}"
      messages:
        - role: user
          content: "{prompt}"
response:
  text: "$.choices[0].message.content"
```

## 🔌 Connecting to LLM Clients

### Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "batch-mcp-wrapper": {
      "command": "python",
      "args": ["/path/to/aggregator_server.py", "--csv", "/path/to/providers.csv", "--port", "12000"],
      "env": {}
    }
  }
}
```

### Continue.dev

Add to your `config.json`:

```json
{
  "mcpServers": [
    {
      "name": "batch-mcp-wrapper",
      "serverUrl": "http://127.0.0.1:12000/api"
    }
  ]
}
```

### Cline (VSCode Extension)

Add to your MCP settings:

```json
{
  "mcpServers": {
    "batch-mcp-wrapper": {
      "command": "python",
      "args": ["/path/to/aggregator_server.py", "--csv", "/path/to/providers.csv"],
      "cwd": "/path/to/Batch_MCP_Wrapper"
    }
  }
}
```

### Generic MCP Client

Connect to the JSON-RPC endpoint at `http://127.0.0.1:12000/api` and use these methods:

- `tools/list` - Get all available provider::model tools
- `tools/execute` - Execute requests to specific models

## 📡 API Usage Examples

### List Available Tools

```bash
curl -X POST http://127.0.0.1:12000/api \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "1",
    "method": "tools/list",
    "params": {}
  }'
```

### Execute Text Generation

```bash
curl -X POST http://127.0.0.1:12000/api \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "2",
    "method": "tools/execute",
    "params": {
      "id": "OpenRouter::gpt-4o",
      "params": {
        "prompt": "Write a haiku about coding"
      }
    }
  }'
```

### Execute Image Generation

```bash
curl -X POST http://127.0.0.1:12000/api \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "3",
    "method": "tools/execute",
    "params": {
      "id": "Pollinations::flux-1",
      "params": {
        "prompt": "A beautiful sunset over mountains",
        "width": 1024,
        "height": 1024
      }
    }
  }'
```

### Execute Embeddings

```bash
curl -X POST http://127.0.0.1:12000/api \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "4",
    "method": "tools/execute",
    "params": {
      "id": "OpenAI::text-embedding-3-small",
      "params": {
        "text_batch": ["hello world", "how are you"]
      }
    }
  }'
```

## 🎯 Key Features

- **Unified Interface**: Access multiple AI providers through a single MCP server
- **Automatic Capability Detection**: 3-tier system (provider metadata → heuristics → Hugging Face API)
- **Flexible Authentication**: Support for bearer tokens, custom headers, query parameters
- **Environment Management**: Auto-generates `.env` file with API keys
- **Custom Templates**: Full control over request/response handling for complex providers
- **Rate Limit Awareness**: Displays rate limit information for each provider
- **Capability-Specific Schemas**: Different parameter schemas for text, image, embedding, audio, vision
- **Caching**: Model capability results are cached locally for performance

## 🔧 Command Line Options

```bash
python aggregator_server.py [OPTIONS]

Options:
  --csv TEXT              Path to providers CSV file [default: providers.csv]
  --port INTEGER          Port to run server on [default: 12000]
  --dotenv TEXT           Path to .env file [default: .env]
  --hf-token TEXT         Hugging Face token for model lookup [optional]
  --auth-mode TEXT        Default auth mode [default: bearer]
  --template TEXT         Default template path/URL [optional]
```

## 🔍 Troubleshooting

### Server Won't Start
- Check if port is already in use: `lsof -i :12000`
- Verify CSV file format and path
- Check Python dependencies are installed

### Authentication Errors
- Verify API keys are correctly set in `.env` file
- Check `AuthMode` column in CSV matches provider requirements
- Ensure environment variables follow format: `{ProviderName}API_Key`

### Model Not Found
- Check if provider's model list endpoint is accessible
- Verify model names in CSV match provider's actual model IDs
- Check cache file `.model_caps_cache.json` for stale entries

### Connection Refused
- Ensure server is running: `ps aux | grep aggregator_server`
- Check firewall settings
- Verify correct port and host configuration

## 📁 File Structure

```
Batch_MCP_Wrapper/
├── aggregator_server.py      # Main MCP server
├── g4f_providers_example.csv # Example provider configuration
├── README.md                 # This file
├── .env                      # Generated API keys (auto-created)
├── .model_caps_cache.json    # Model capability cache (auto-created)
└── .gitignore               # Git ignore rules
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built on the Model Context Protocol (MCP) standard
- Uses Hugging Face API for model capability detection
- Inspired by the need to unify multiple AI provider APIs