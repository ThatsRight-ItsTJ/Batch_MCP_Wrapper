Perfect ✅ — let’s extend the aggregator generator so your CSV can contain two new optional columns:

* **`AuthMode`** → specifies how to attach the API key

  * `bearer` → `Authorization: Bearer <key>`
  * `header:<HeaderName>` → put `<key>` in custom header
  * `query:<param>` → append `?<param>=<key>` to URL
  * `none` → don’t use an API key

* **`Template`** → path or URL to a provider adapter (YAML/JSON) with request/response shapes.

  * If present, overrides the default behavior.
  * If absent, we fall back to generic request builder.

---

## 📝 Updated Aggregator Generator (Core Changes Only)

Here are the relevant modifications you can drop into the script I gave you earlier:

### 1. Update CSV Normalization

```python
def normalize_csv(csv_path: str) -> pd.DataFrame:
    df_raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False).fillna("")
    model_col = guess_models_column(df_raw)

    cleaned = []
    for _, row in df_raw.iterrows():
        name = row.get("Name", "").strip()
        base = row.get("Base_URL", "").strip()
        apikey = row.get("APIKey", "").strip()
        models_field = row.get(model_col, "").strip()
        authmode = row.get("AuthMode", "").strip().lower() or "bearer"
        template = row.get("Template", "").strip()

        # rebuild RateLimit from leftover columns
        other_parts = []
        for c in df_raw.columns:
            if c in ("Name", "Base_URL", "APIKey", model_col, "AuthMode", "Template"):
                continue
            val = str(row.get(c, "")).strip()
            if val:
                other_parts.append(val)
        rate_info = " ".join(other_parts).strip()

        cleaned.append({
            "Name": name,
            "Base_URL": base,
            "APIKey": apikey,
            "Models": models_field,
            "Rate Limit/cost info": rate_info,
            "AuthMode": authmode,
            "Template": template,
        })

    return pd.DataFrame(cleaned)
```

---

### 2. AuthMode Handler

```python
def build_auth_headers_and_url(authmode: str, base_url: str, api_key: Optional[str]) -> (Dict[str, str], str):
    if not api_key or authmode == "none":
        return {}, base_url

    if authmode == "bearer":
        return {"Authorization": f"Bearer {api_key}"}, base_url

    if authmode.startswith("header:"):
        header_name = authmode.split(":", 1)[1]
        return {header_name: api_key}, base_url

    if authmode.startswith("query:"):
        param = authmode.split(":", 1)[1]
        sep = "&" if "?" in base_url else "?"
        return {}, f"{base_url}{sep}{param}={api_key}"

    # default: treat as bearer
    return {"Authorization": f"Bearer {api_key}"}, base_url
```

---

### 3. Template Loader

```python
import yaml

def load_template(path_or_url: str) -> Optional[Dict[str, Any]]:
    if not path_or_url:
        return None
    try:
        if path_or_url.startswith("http"):
            r = requests.get(path_or_url, timeout=10)
            r.raise_for_status()
            text = r.text
        else:
            with open(path_or_url, "r", encoding="utf-8") as f:
                text = f.read()
        if path_or_url.endswith(".json"):
            return json.loads(text)
        return yaml.safe_load(text)
    except Exception as e:
        print(f"[WARN] Could not load template {path_or_url}: {e}")
        return None
```

---

### 4. Integrate in `execute_tool`

Before making a request:

```python
authmode = provider.get("AuthMode", "bearer")
template_path = provider.get("Template", "")
template = load_template(template_path) if template_path else None

headers, url = build_auth_headers_and_url(authmode, base_url, api_key)

if template and primary in template.get("request", {}):
    req_tpl = template["request"][primary]
    method = req_tpl.get("method", "POST")
    req_url = req_tpl.get("url", url).format(Base_URL=base_url, model_id=model_id)
    body = req_tpl.get("body", {})
    # substitute variables
    body_str = json.dumps(body)
    for k, v in payload_params.items():
        body_str = body_str.replace(f"{{{k}}}", str(v))
    body_str = body_str.replace("{model_id}", model_id).replace("{Base_URL}", base_url)
    body = json.loads(body_str)

    r = requests.request(method, req_url, headers=headers, json=body, timeout=60)
    data = r.json()
    # response extraction
    resp_rule = template.get("response", {}).get(primary)
    if resp_rule:
        # simple JSONPath-like extraction
        extracted = data
        for part in resp_rule.strip("$.").split("."):
            if isinstance(extracted, dict):
                extracted = extracted.get(part)
            elif isinstance(extracted, list):
                try:
                    idx = int(part)
                    extracted = extracted[idx]
                except Exception:
                    extracted = None
        return {"raw_response": data, "extracted": extracted}
    else:
        return {"raw_response": data}
```

If no template → fallback to generic behavior (as before).

---

## ✅ What This Gives You

* **CSV columns**:

  ```csv
  Name,Base_URL,APIKey,Models,AuthMode,Template,Rate Limit/cost info
  OpenAI,https://api.openai.com/v1/chat/completions,OPENAI_API_KEY,https://api.openai.com/v1/models,bearer,,
  CustomAPI,https://api.custom.com/v1/run,CUSTOM_KEY,custom-model-id,header:X-Api-Key,custom_adapter.yaml,"500 RPM"
  ```
* If `Template` is set → adapter fully controls request/response.
* If not → script falls back to AuthMode + default heuristics.

---

⚡ Do you want me to also **provide a starter template example** (e.g. for Pollinations or OpenAI), so you can see how to write a JSON/YAML adapter file?

