# Aggregator MCP server generator — 3-tier capability detection + capability-specific schemas


* **Normalizes messy CSV** into columns `Name`, `Base_URL`, `APIKey`, `Models`, `AuthMode`, `Template`, `Rate Limit/cost info`.
* **Creates a `.env` file** and sets an env var for each API exactly as you asked: `<Name>API_Key` (and also a sanitized uppercase variant for convenience).
* **Builds a single MCP aggregator server** (FastAPI) that exposes every provider model as a namespaced tool `Provider::Model`.
* **Detects model capabilities** using a 3-tier approach:

  1. **Provider metadata** (model list endpoint if present).
  2. **Local heuristics** based on model name.
  3. **Internet lookup fallback** (Hugging Face model metadata API) for high accuracy on obscure models — results are cached locally.
* **Creates capability-specific parameter schemas** (text, image, audio, vision, embedding) and uses them for each tool.
* **Implements `tools/list` and `tools/execute`**. `tools/execute` forms payloads best-effort based on capability and provider patterns, and tries to intelligently extract the provider response.
* **Supports multiple authentication modes** via `AuthMode` column (bearer, header:NAME, query:NAME, none).
* **Supports provider adapters** via `Template` column (YAML/JSON files that fully control request/response handling).

> Notes:
>
> * The script expects `routers.base_router.MCPBaseRouter` from the `freedanfan/mcp-server` repo to be importable (same as earlier). If you prefer, I can embed a tiny local MCP JSON-RPC handler instead — say so and I’ll adapt.
> * The internet lookup uses Hugging Face's public model metadata endpoint. If many lookups are required, you may want to provide an HF token (optional) to avoid rate limits. Results are cached in `.model_caps_cache.json`.

---

## How to use

1. Put your CSV at `providers.csv` (or pass another path as `--csv`).

## 📝 CSV Format

| Name           | Base_URL                     | APIKey         | Models                              | AuthMode        | Template                            | Rate Limit/cost info |
|----------------|------------------------------|----------------|-------------------------------------|-----------------|-------------------------------------|----------------------|
| OpenAI-Primary | https://api.openai.com/v1    | sk-xxxx        | gpt-4|gpt-3.5-turbo                 | bearer          |                                     |                      |
| Pollinations   | https://text.pollinations.ai |                | https://pollinations.ai/api/models  | bearer          |                                     |         10 rpm       |
| CustomAPI      | https://api.custom.com/v1    | api_key_123    | custom-model                        | header:X-Api-Key| custom_adapter.yaml                 |         100 rpm      |
- **Models** can be:
  - A `|`‑delimited list of model IDs.
  - A URL returning a list of models (JSON).
- **AuthMode** specifies how to attach the API key (optional, defaults to `bearer`):
  - `bearer` - Bearer token authentication
  - `header:HeaderName` - Custom header (e.g., `header:X-Api-Key`)
  - `query:param` - Query parameter (e.g., `query:api_key`)
  - `none` - No authentication
- **Template** specifies a path or URL to a provider adapter (YAML/JSON) that fully controls request/response handling (optional).

### AuthMode Examples:

```csv
# Using custom header authentication
MyAPI,https://api.example.com/v1,my_key,model1|model2,header:Authorization,,"1000 RPM"

# Using query parameter authentication
AnotherAPI,https://api.another.com/v1,secret_key,model3,query:api_key,,"500 RPM"
```

### Template Example:

```csv
# Using a custom template for request/response handling
CustomProvider,https://api.custom.com/v1,custom_key,custom-model,header:X-Api-Key,custom_adapter.yaml,"200 RPM"
```

Template file (`custom_adapter.yaml`) example:
```yaml
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

2. Run:

```bash
pip install fastapi uvicorn requests python-dotenv pandas
# make sure freedanfan mcp-server's `routers` package is on PYTHONPATH (or copy base_router.py next to this script)
python aggregator_server.py --csv providers.csv --port 12000
```

3. The script will write `.env` with env vars like: `MyProviderAPI_Key=the_key_from_csv`. It will also set the same env var in the running process.
4. MCP clients can connect to `http://127.0.0.1:12000/api` and call JSON-RPC methods:

   * `tools/list` → returns `tools` with `id` like `Provider::model-name` and `parameters` (capability schema).
   * `tools/execute` → send `{"id":"Provider::model", "params": { ... capability-specific params ... }}`.

---

## Examples of use (JSON-RPC calls)

1. List tools:

```json
{
  "jsonrpc":"2.0",
  "id":"req-1",
  "method":"tools/list",
  "params": {}
}
```

2. Call a text model (example):

```json
{
  "jsonrpc":"2.0",
  "id":"req-2",
  "method":"tools/execute",
  "params": {
    "id": "OpenAI::gpt-4o",
    "params": {
      "prompt": "Write a 3-line haiku about the ocean."
    }
  }
}
```

3. Call an embedding model:

```json
{
  "jsonrpc":"2.0",
  "id":"req-3",
  "method":"tools/execute",
  "params": {
    "id": "SomeProvider::my-embed-model",
    "params": {
      "text_batch": ["hello world", "how are you"]
    }
  }
}
```

---

## Implementation caveats & recommended improvements

* **Provider-specific request shapes**: The script uses *best-effort* payload shapes (common patterns). For complete control over request/response handling, use the `Template` column with YAML/JSON provider adapters.
* **Authentication modes**: The script supports multiple authentication methods via the `AuthMode` column:
  - `bearer` - Bearer token (default)
  - `header:HeaderName` - Custom header (e.g., `header:X-Api-Key`)
  - `query:param` - Query parameter (e.g., `query:api_key`)
  - `none` - No authentication
* **Template system**: When a `Template` is specified, it fully controls request/response handling. The template supports variable substitution (`{model_id}`, `{Base_URL}`, `{prompt}`, etc.) and JSONPath-like response extraction.
* **Hugging Face rate limits**: the internet lookup uses HF public endpoints. If doing many lookups, supply `--hf-token` or pre-warm the cache.
* **Extensible cache**: it's a simple JSON file. For more robustness use sqlite or Redis.
* **Error handling**: When using templates, ensure they handle edge cases appropriately. The script falls back to default behavior if template loading fails.