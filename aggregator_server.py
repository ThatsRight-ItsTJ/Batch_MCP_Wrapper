#!/usr/bin/env python3
"""
aggregator_server.py

Aggregates multiple AI-provider definitions (CSV) into a single MCP-style server
that exposes each provider-model as a namespaced tool: Provider::Model.

Features:
 - Cleans messy CSVs and consolidates rate/cost info
 - Generates .env file with env var <Name>API_Key for each provider that had an APIKey value
 - 3-tier capability detection:
    1) provider model-list endpoint metadata
    2) heuristics on model name
    3) internet lookup fallback via Hugging Face metadata API (cached)
 - Capability-specific parameter schemas for: text, image, embedding, audio, vision
 - tools/list and tools/execute
 - Support for AuthMode column (bearer, header:NAME, query:NAME, none)
 - Support for Template column (YAML/JSON provider adapters)
"""

import argparse
import csv
import json
import os
import re
import time
import yaml
import logging
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import quote_plus
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import pandas as pd
import requests
from dotenv import set_key, load_dotenv
from fastapi import FastAPI
import uvicorn

# NOTE: this import assumes the freedanfan/mcp-server package is available on PYTHONPATH.
# If not, copy routers/base_router.py locally or replace with a minimal JSON-RPC handler.
from routers.base_router import MCPBaseRouter

# Files
DEFAULT_DOTENV = ".env"
CACHE_FILE = ".model_caps_cache.json"
CACHE_TTL = 7 * 24 * 60 * 60  # 7 days in seconds
CONCURRENT_REQUESTS = 10  # Maximum concurrent HTTP requests

# Capability schemas
CAPABILITY_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "text": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Text prompt"}
        },
        "required": ["prompt"],
    },
    "image": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Image generation prompt"},
            "image_url": {"type": "string", "description": "Optional input image URL"},
            "width": {"type": "integer", "description": "Width (optional)"},
            "height": {"type": "integer", "description": "Height (optional)"},
        },
        "required": ["prompt"],
    },
    "embedding": {
        "type": "object",
        "properties": {
            "text_batch": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of texts to embed",
            }
        },
        "required": ["text_batch"],
    },
    "audio": {
        "type": "object",
        "properties": {
            "audio_url": {"type": "string", "description": "Input audio file URL"},
            "task": {"type": "string", "enum": ["transcribe", "translate", "detect_language"]},
        },
        "required": ["audio_url"],
    },
    "vision": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Prompt for multimodal reasoning"},
            "image_url": {"type": "string", "description": "Image input"},
        },
        "required": ["prompt", "image_url"],
    },
}

# Priority order when selecting primary capability
CAPABILITY_PRIORITY = ["embedding", "image", "vision", "audio", "text"]


# -------------------------
# Helpers
# -------------------------
def sanitize_env_name(name: str) -> str:
    """Return two env var names: exact (NameAPI_Key) and sanitized uppercase (NAME_API_KEY)."""
    exact = f"{name}API_Key"
    sanitized = re.sub(r"\W+", "_", name).strip("_").upper() + "_API_KEY"
    return exact, sanitized


@dataclass
class CacheEntry:
    capabilities: List[str]
    last_seen: int
    source: str = "heuristic"  # "heuristic", "internet", "provider"
    
    def is_expired(self) -> bool:
        return time.time() - self.last_seen > CACHE_TTL

def load_cache(cache_path: str = CACHE_FILE) -> Dict[str, Any]:
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_cache(cache: Dict[str, Any], cache_path: str = CACHE_FILE):
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def cleanup_expired_cache(cache: Dict[str, Any]) -> Dict[str, Any]:
    """Remove expired cache entries"""
    current_time = time.time()
    cleaned_cache = {}
    for key, value in cache.items():
        if isinstance(value, dict):
            entry = CacheEntry(
                capabilities=value.get("capabilities", []),
                last_seen=value.get("last_seen", 0),
                source=value.get("source", "heuristic")
            )
            if not entry.is_expired():
                cleaned_cache[key] = value
    return cleaned_cache


def guess_capabilities_from_name(model_name: str) -> List[str]:
    """Enhanced heuristics with improved accuracy to reduce internet lookups"""
    n = model_name.lower()
    caps = []
    
    # Enhanced embedding hints with higher confidence
    strong_embedding_indicators = [
        "embed", "embedding", "vector", "sentence-transformers", "text-embedding",
        "bert", "roberta", "mpnet", "distilbert", "use", "universal-sentence"
    ]
    if any(k in n for k in strong_embedding_indicators):
        caps.append("embedding")
    
    # Enhanced image generation hints with higher confidence
    strong_image_indicators = [
        "dall", "sdxl", "stable", "flux", "diffusion", "sd", "img", "stability",
        "midjourney", "dream", "generate", "create", "paint"
    ]
    if any(k in n for k in strong_image_indicators):
        caps.append("image")
    
    # Vision / multimodal hints
    vision_indicators = [
        "vision", "multimodal", "clip", "img2txt", "ocr", "visual", "vlm", "llava"
    ]
    if any(k in n for k in vision_indicators):
        caps.append("vision")
    
    # Audio hints
    audio_indicators = [
        "audio", "whisper", "speech", "asr", "wav2vec", "speech-to-text",
        "whisper", "piper", "faster-whisper"
    ]
    if any(k in n for k in audio_indicators):
        caps.append("audio")
    
    # Chat/ text fallback - only if no other indicators found
    if not caps:
        # Check for text-specific indicators to avoid unnecessary internet lookups
        text_indicators = ["chat", "completion", "instruct", "llama", "mistral", "gemma", "qwen"]
        if any(k in n for k in text_indicators):
            caps.append("text")
        else:
            # Default to text for unknown models
            caps.append("text")
    
    return caps


def choose_primary_capability(caps: List[str]) -> str:
    for p in CAPABILITY_PRIORITY:
        if p in caps:
            return p
    return caps[0] if caps else "text"


def build_auth_headers_and_url(authmode: str, base_url: str, api_key: Optional[str]) -> tuple[Dict[str, str], str]:
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


async def load_template_async(path_or_url: str, session: Optional[aiohttp.ClientSession] = None) -> Optional[Dict[str, Any]]:
    """Async version of load_template"""
    if not path_or_url:
        return None
    try:
        if path_or_url.startswith("http"):
            async with session.get(path_or_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                response.raise_for_status()
                text = await response.text()
        else:
            with open(path_or_url, "r", encoding="utf-8") as f:
                text = f.read()
        
        if path_or_url.endswith(".json"):
            return json.loads(text)
        return yaml.safe_load(text)
    except Exception as e:
        logger.warning(f"Could not load template {path_or_url}: {e}")
        return None

def load_template(path_or_url: str) -> Optional[Dict[str, Any]]:
    """Synchronous wrapper for load_template"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already in an event loop, run in a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    load_template_async(path_or_url)
                )
                return future.result()
        else:
            return asyncio.run(load_template_async(path_or_url))
    except Exception as e:
        logger.warning(f"Could not load template {path_or_url}: {e}")
        return None


# -------------------------
# Internet lookup (Hugging Face) fallback
# -------------------------
async def internet_lookup_capabilities_async(model_id: str, hf_token: Optional[str] = None, session: Optional[aiohttp.ClientSession] = None) -> List[str]:
    """
    Async version of internet lookup capabilities using aiohttp
    """
    start_time = time.time()
    logger.info(f"Starting Hugging Face lookup for model: {model_id}")
    
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    
    # Try direct model lookup first
    hf_url = f"https://huggingface.co/api/models/{model_id}"
    
    try:
        logger.info(f"Making direct request to: {hf_url}")
        async with session.get(hf_url, headers=headers, timeout=aiohttp.ClientTimeout(total=8)) as response:
            request_time = time.time() - start_time
            logger.info(f"Direct request completed in {request_time:.2f}s, status: {response.status}")
            
            if response.status == 200:
                data = await response.json()
                caps = set()
                pipeline_tag = data.get("pipeline_tag")
                if pipeline_tag:
                    if "text-generation" in pipeline_tag:
                        caps.add("text")
                    if "image-generation" in pipeline_tag:
                        caps.add("image")
                    if pipeline_tag in ("text-embedding", "feature-extraction", "sentence-similarity"):
                        caps.add("embedding")
                    if pipeline_tag in ("automatic-speech-recognition", "speech-to-text"):
                        caps.add("audio")
                    if pipeline_tag in ("image-classification", "image-segmentation", "object-detection"):
                        caps.add("vision")
                
                tags = data.get("tags", []) or []
                tstring = " ".join(tags).lower()
                if any(k in tstring for k in ["embedding", "sentence-transformer", "feature-extraction"]):
                    caps.add("embedding")
                if any(k in tstring for k in ["image", "diffusion", "img2img", "text-to-image"]):
                    caps.add("image")
                if any(k in tstring for k in ["audio", "speech", "asr", "wav2vec"]):
                    caps.add("audio")
                if any(k in tstring for k in ["vision", "multimodal", "clip"]):
                    caps.add("vision")
                
                if caps:
                    total_time = time.time() - start_time
                    logger.info(f"HF lookup successful for {model_id}: {list(caps)} in {total_time:.2f}s")
                    return list(caps)
    except Exception as e:
        logger.warning(f"Direct HF lookup failed for {model_id}: {e}")
        pass

    # If direct lookup fails, try searching
    try:
        s_url = f"https://huggingface.co/api/models?search={quote_plus(model_id)}"
        logger.info(f"Making search request to: {s_url}")
        async with session.get(s_url, headers=headers, timeout=aiohttp.ClientTimeout(total=8)) as response:
            search_time = time.time() - start_time
            logger.info(f"Search request completed in {search_time:.2f}s, status: {response.status}")
            
            if response.status == 200:
                items = await response.json()
                if items:
                    data = items[0]
                    caps = set()
                    pipeline_tag = data.get("pipeline_tag")
                    if pipeline_tag:
                        if "text-generation" in pipeline_tag:
                            caps.add("text")
                        if "image-generation" in pipeline_tag:
                            caps.add("image")
                        if pipeline_tag in ("text-embedding", "feature-extraction"):
                            caps.add("embedding")
                    
                    tags = data.get("tags", []) or []
                    tstring = " ".join(tags).lower()
                    if "embedding" in tstring:
                        caps.add("embedding")
                    if any(k in tstring for k in ["image", "diffusion", "text-to-image", "img2img"]):
                        caps.add("image")
                    if caps:
                        total_time = time.time() - start_time
                        logger.info(f"HF search successful for {model_id}: {list(caps)} in {total_time:.2f}s")
                        return list(caps)
    except Exception as e:
        logger.warning(f"HF search lookup failed for {model_id}: {e}")
        pass

    total_time = time.time() - start_time
    logger.info(f"HF lookup failed for {model_id} in {total_time:.2f}s")
    return []

def internet_lookup_capabilities(model_id: str, hf_token: Optional[str] = None) -> List[str]:
    """
    Synchronous wrapper for internet lookup capabilities
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already in an event loop, run in a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    internet_lookup_capabilities_async(model_id, hf_token)
                )
                return future.result()
        else:
            return asyncio.run(internet_lookup_capabilities_async(model_id, hf_token))
    except Exception as e:
        logger.warning(f"Internet lookup failed for {model_id}: {e}")
        return []


# -------------------------
# CSV normalization + env generation
# -------------------------
def guess_models_column(df: pd.DataFrame) -> str:
    # Try to locate column with model info using a few common names
    cand = None
    for name in df.columns:
        lower = name.lower()
        if "model" in lower or "models" in lower or "model(s)" in lower:
            cand = name
            break
    return cand or df.columns[3]  # fall back to 4th column


def normalize_csv(csv_path: str) -> pd.DataFrame:
    """
    Load CSV and attempt to coalesce trailing columns into 'Rate Limit/cost info'.
    Returns dataframe with columns: Name, Base_URL, APIKey, Models, Rate Limit/cost info, AuthMode, Template
    """
    df_raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False).fillna("")
    # Determine model column heuristically
    model_col = guess_models_column(df_raw)
    # Build cleaned DF
    cleaned = []
    for _, row in df_raw.iterrows():
        name = row.get("Name", "").strip() or row.get("name", "").strip()
        base = row.get("Base_URL", "").strip() or row.get("BaseUrl", "").strip() or row.get("Base Url", "").strip()
        apikey = row.get("APIKey", "").strip() or row.get("API Key", "").strip()
        models_field = row.get(model_col, "").strip()
        authmode = row.get("AuthMode", "").strip().lower() or "bearer"
        template = row.get("Template", "").strip()
        # Rebuild rate/cost info by joining any other columns not used
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
    df_clean = pd.DataFrame(cleaned)
    return df_clean


def generate_env_file_and_set(df_clean: pd.DataFrame, dotenv_path: str = DEFAULT_DOTENV):
    """
    For each provider with non-empty APIKey in the CSV, create env var: <Name>API_Key (exact),
    write into .env, and also set a sanitized uppercase var: NAME_API_KEY.
    """
    # ensure dotenv exists
    if not os.path.exists(dotenv_path):
        with open(dotenv_path, "w", encoding="utf-8"):
            pass
    load_dotenv(dotenv_path)
    for _, r in df_clean.iterrows():
        name = r["Name"].strip()
        key_val = r["APIKey"].strip()
        if not name:
            continue
        if key_val and key_val.lower() != "nan":
            exact, sanitized = sanitize_env_name(name)
            # set both
            set_key(dotenv_path, exact, key_val)
            set_key(dotenv_path, sanitized, key_val)
            os.environ[exact] = key_val
            os.environ[sanitized] = key_val


# -------------------------
# Model parsing + capability discovery
# -------------------------
def parse_models_for_api(row: Dict[str, Any], cache: Dict[str, Any], hf_token: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Given a CSV row dict with fields Name, Base_URL, APIKey, Models, determine the list of model dicts:
        { id, description, capabilities }
    Uses cache and internet lookup fallback.
    """
    models_field = str(row.get("Models", "")).strip()
    provider_name = row.get("Name", "Unknown")
    env_exact, env_sanitized = sanitize_env_name(provider_name)
    api_key = os.environ.get(env_exact) or os.environ.get(env_sanitized) or None
    headers = {}
    if api_key:
        # many providers use Bearer token but not all; we use Bearer by default
        headers["Authorization"] = f"Bearer {api_key}"

    results: List[Dict[str, Any]] = []

    # Helper to determine capabilities for a model id (string)
    def get_caps_for_model(m_id: str) -> List[str]:
        m_id_s = str(m_id)
        if m_id_s in cache and isinstance(cache[m_id_s], dict) and "capabilities" in cache[m_id_s]:
            return cache[m_id_s]["capabilities"]
        # Try heuristics first (quick)
        guessed = guess_capabilities_from_name(m_id_s)
        # If guessed not specific (only 'text') then do internet fallback for high accuracy
        if guessed == ["text"]:
            # Internet lookup
            try:
                internet_caps = internet_lookup_capabilities(m_id_s, hf_token=hf_token)
                if internet_caps:
                    caps = internet_caps
                else:
                    caps = guessed
            except Exception:
                caps = guessed
        else:
            caps = guessed
        # cache
        cache[m_id_s] = {"capabilities": caps, "last_seen": int(time.time())}
        return caps

    # Case 1: models_field is an HTTP URL -> fetch provider metadata
    if models_field.startswith("http://") or models_field.startswith("https://"):
        try:
            logger.info(f"Fetching models from URL: {models_field}")
            url_start_time = time.time()
            r = requests.get(models_field, headers=headers, timeout=10)
            url_time = time.time() - url_start_time
            logger.info(f"URL fetch completed in {url_time:.2f}s, status: {r.status_code}")
            r.raise_for_status()
            data = r.json()
            # Try common shapes: {"data":[...]}, {"models": [...]}, or plain dict mapping
            raw_models = data.get("data") or data.get("models") or data
            # support lists of strings or list of dicts
            if isinstance(raw_models, dict):
                # maybe mapping id->{...}
                raw_models_list = []
                for k, v in raw_models.items():
                    if isinstance(v, dict):
                        v["id"] = v.get("id") or k
                        raw_models_list.append(v)
                    else:
                        raw_models_list.append({"id": k, "description": str(v)})
                raw_models = raw_models_list
            if not isinstance(raw_models, list):
                # if API returned a single model object
                raw_models = [raw_models]
            for m in raw_models:
                if isinstance(m, str):
                    mid = m
                    desc = ""
                    caps = get_caps_for_model(mid)
                elif isinstance(m, dict):
                    mid = m.get("id") or m.get("model") or m.get("name") or str(m)
                    desc = m.get("description", "") or m.get("info", "")
                    caps = m.get("capabilities") or m.get("tags") or []
                    # normalize tags/capabilities if present as list of strings like "text-generation"
                    if isinstance(caps, list):
                        # convert tag signals to our canonical capabilities where possible
                        mapped = []
                        for t in caps:
                            tl = str(t).lower()
                            if "image" in tl or "dall" in tl or "sd" in tl:
                                mapped.append("image")
                            elif "embed" in tl or "vector" in tl or "feature-extraction" in tl:
                                mapped.append("embedding")
                            elif "audio" in tl or "speech" in tl or "asr" in tl:
                                mapped.append("audio")
                            elif "vision" in tl or "multimodal" in tl or "ocr" in tl:
                                mapped.append("vision")
                            elif "text" in tl or "chat" in tl or "generation" in tl:
                                mapped.append("text")
                        if mapped:
                            caps = list(dict.fromkeys(mapped))
                        else:
                            # fallback to guess by name
                            caps = get_caps_for_model(mid)
                    else:
                        caps = get_caps_for_model(mid)
                else:
                    mid = str(m)
                    desc = ""
                    caps = get_caps_for_model(mid)
                results.append({"id": mid, "description": desc, "capabilities": caps})
        except Exception as e:
            # couldn't fetch; treat models_field as literal list fallback
            results.append({"id": f"ERROR_FETCHING_{models_field}", "description": str(e), "capabilities": ["text"]})
            # try fallback: treat URL path's last segment as single model name
            last_seg = models_field.rstrip("/").split("/")[-1]
            if last_seg:
                caps = get_caps_for_model(last_seg)
                results.append({"id": last_seg, "description": "inferred from URL", "capabilities": caps})

    # Case 2: pipe-delimited
    elif "|" in models_field:
        for mid in (m.strip() for m in models_field.split("|") if m.strip()):
            caps = get_caps_for_model(mid)
            results.append({"id": mid, "description": "", "capabilities": caps})

    # Case 3: comma separated (in case)
    elif "," in models_field and "http" not in models_field:
        for mid in (m.strip() for m in models_field.split(",") if m.strip()):
            caps = get_caps_for_model(mid)
            results.append({"id": mid, "description": "", "capabilities": caps})

    # Case 4: single model name or empty
    else:
        if models_field:
            mid = models_field
            caps = get_caps_for_model(mid)
            results.append({"id": mid, "description": "", "capabilities": caps})
        else:
            # No model info: attempt to call a commonly-known "list models" endpoint on the Base_URL (best-effort)
            base_url = str(row.get("Base_URL", "")).rstrip("/")
            if base_url:
                # try common patterns
                for candidate in [base_url + "/models", base_url + "/v1/models", base_url + "/api/models"]:
                    try:
                        logger.info(f"Trying model endpoint: {candidate}")
                        endpoint_start_time = time.time()
                        r = requests.get(candidate, headers=headers, timeout=6)
                        endpoint_time = time.time() - endpoint_start_time
                        logger.info(f"Endpoint {candidate} completed in {endpoint_time:.2f}s, status: {r.status_code}")
                        if r.status_code == 200:
                            data = r.json()
                            raw_models = data.get("data") or data.get("models") or data
                            if isinstance(raw_models, list):
                                for m in raw_models:
                                    if isinstance(m, str):
                                        mid = m
                                        caps = get_caps_for_model(mid)
                                        results.append({"id": mid, "description": "", "capabilities": caps})
                                    elif isinstance(m, dict):
                                        mid = m.get("id") or m.get("name")
                                        caps = m.get("capabilities") or get_caps_for_model(mid)
                                        desc = m.get("description", "")
                                        results.append({"id": mid, "description": desc, "capabilities": caps})
                                break
                    except Exception:
                        continue
            if not results:
                # last resort: add a generic "default" model entry using the provider name
                gen_id = f"{provider_name}-default"
                caps = guess_capabilities_from_name(gen_id)
                results.append({"id": gen_id, "description": "inferred default", "capabilities": caps})

    return results


# -------------------------
# Aggregator server builder
# -------------------------
def build_aggregator_app(csv_path: str, dotenv_path: str = DEFAULT_DOTENV, hf_token: Optional[str] = None):
    start_time = time.time()
    logger.info(f"Starting to build aggregator app from {csv_path}")
    
    df_clean = normalize_csv(csv_path)
    generate_env_file_and_set(df_clean, dotenv_path)

    cache = load_cache()

    apis: List[Dict[str, Any]] = []
    total_rows = len(df_clean)
    logger.info(f"Processing {total_rows} providers from CSV")
    
    for idx, row in df_clean.iterrows():
        row_start_time = time.time()
        rowd = row.to_dict()
        # ensure name exists
        name = rowd.get("Name") or rowd.get("name") or "UnknownProvider"
        rowd["Name"] = name
        
        logger.info(f"Processing provider {idx+1}/{total_rows}: {name}")
        models = parse_models_for_api(rowd, cache=cache, hf_token=hf_token)
        
        row_time = time.time() - row_start_time
        logger.info(f"Provider {name} took {row_time:.2f}s, found {len(models)} models")
        
        # annotate each model with a parameter schema (capability-specific)
        for m in models:
            caps = m.get("capabilities", []) or []
            primary = choose_primary_capability(caps)
            m["primary_capability"] = primary
            m["parameters"] = CAPABILITY_SCHEMAS.get(primary, CAPABILITY_SCHEMAS["text"])
        rowd["models_parsed"] = models
        apis.append(rowd)

    save_cache(cache)
    total_time = time.time() - start_time
    logger.info(f"Completed building aggregator app in {total_time:.2f}s")

    # Build FastAPI + MCP router
    app = FastAPI(title="MCP Aggregator Server (generated)")
    router = MCPBaseRouter()

    # tools/list: returns every model as a tool
    def list_tools(_params=None):
        tools = []
        for api in apis:
            for m in api["models_parsed"]:
                tool_id = f"{api['Name']}::{m['id']}"
                tools.append(
                    {
                        "id": tool_id,
                        "name": tool_id,
                        "description": m.get("description") or f"{api['Name']} model {m['id']}",
                        "parameters": m.get("parameters"),
                        "capabilities": m.get("capabilities"),
                        "rate_limit_info": api.get("Rate Limit/cost info", ""),
                    }
                )
        return {"tools": tools}

    # tools/execute: accepts {"id": "Provider::Model", "params": {...}}
    def execute_tool(params: Dict[str, Any]):
        if not params or "id" not in params or "params" not in params:
            raise ValueError("Invalid params. expected {id: 'Provider::Model', params: {...}}")
        tool_id = params["id"]
        payload_params = params["params"]
        if "::" not in tool_id:
            raise ValueError("Tool id must be namespaced as Provider::Model")
        provider_name, model_id = tool_id.split("::", 1)
        # find provider row
        provider = next((p for p in apis if p["Name"] == provider_name), None)
        if not provider:
            raise ValueError(f"Provider {provider_name} not found")
        # find model metadata
        model_meta = next((m for m in provider["models_parsed"] if m["id"] == model_id), None)
        if not model_meta:
            # tolerate minor mismatches by searching for model_id substring
            model_meta = next((m for m in provider["models_parsed"] if model_id in m["id"]), None)
            if not model_meta:
                raise ValueError(f"Model {model_id} not found for provider {provider_name}")

        primary = model_meta.get("primary_capability", "text")
        base_url = provider.get("Base_URL", "")
        env_exact, env_sanit = sanitize_env_name(provider_name)
        api_key = os.environ.get(env_exact) or os.environ.get(env_sanit) or provider.get("APIKey") or ""
        authmode = provider.get("AuthMode", "bearer")
        template_path = provider.get("Template", "")
        template = load_template(template_path) if template_path else None
        
        headers, url = build_auth_headers_and_url(authmode, base_url, api_key)

        # Build request payload depending on capability (best-effort)
        response_obj: Dict[str, Any] = {"requested_tool": tool_id, "primary_capability": primary, "raw_response": None, "extracted": None}
        
        # Use template if available
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
                response_obj["raw_response"] = data
                response_obj["extracted"] = extracted
                return response_obj
            else:
                response_obj["raw_response"] = data
                return response_obj
        try:
            # Capability-specific mapping
            if primary == "text":
                prompt = payload_params.get("prompt") or payload_params.get("input") or ""
                # try chat-style if base URL looks like a chat endpoint
                if "chat" in base_url or "completions" in base_url or "openai" in base_url.lower():
                    request_payload = {"model": model_id, "messages": [{"role": "user", "content": prompt}]}
                else:
                    request_payload = {"model": model_id, "prompt": prompt}
                r = requests.post(base_url, json=request_payload, headers=headers, timeout=30)
                data = r.json() if r.content else {}
                response_obj["raw_response"] = data
                # extract text
                extracted = None
                if isinstance(data, dict):
                    if "choices" in data and data["choices"]:
                        c0 = data["choices"][0]
                        if isinstance(c0, dict):
                            extracted = c0.get("message", {}).get("content") or c0.get("text") or c0.get("output")
                            # also accept 'content' nestings
                    extracted = extracted or data.get("result") or data.get("output") or data.get("generated_text")
                if extracted is None:
                    # fallback to stringifying raw
                    extracted = str(data)
                response_obj["extracted"] = extracted
                return response_obj

            elif primary == "image":
                prompt = payload_params.get("prompt") or ""
                image_url = payload_params.get("image_url")
                # many image endpoints accept {model, prompt}
                req = {"model": model_id, "prompt": prompt}
                if image_url:
                    req["image"] = image_url
                r = requests.post(base_url, json=req, headers=headers, timeout=60)
                data = r.json() if r.content else {}
                response_obj["raw_response"] = data
                # Try to extract image URLs or base64
                extracted = None
                # Common shapes: {"data":[{"url":...}]} or {"output":"data:..."}
                if isinstance(data, dict):
                    if "data" in data and isinstance(data["data"], list) and data["data"]:
                        # try url or b64
                        d0 = data["data"][0]
                        extracted = d0.get("url") or d0.get("b64_json") or d0.get("b64")
                    extracted = extracted or data.get("output") or data.get("image") or data.get("images")
                if extracted is None:
                    extracted = str(data)
                response_obj["extracted"] = extracted
                return response_obj

            elif primary == "embedding":
                text_batch = payload_params.get("text_batch") or payload_params.get("inputs") or []
                if isinstance(text_batch, str):
                    text_batch = [text_batch]
                req = {"model": model_id, "input": text_batch}
                # some providers expect 'inputs' or 'texts' — we try both if the first doesn't work
                r = requests.post(base_url, json=req, headers=headers, timeout=30)
                data = r.json() if r.content else {}
                response_obj["raw_response"] = data
                # try to extract embeddings
                extracted = None
                if isinstance(data, dict):
                    if "data" in data and isinstance(data["data"], list) and data["data"]:
                        extracted = [item.get("embedding") or item.get("vector") or item.get("embeddings") for item in data["data"]]
                    extracted = extracted or data.get("embeddings") or data.get("embedding") or data.get("result")
                if extracted is None:
                    extracted = str(data)
                response_obj["extracted"] = extracted
                return response_obj

            elif primary in ("audio", "vision"):
                # Generic: pass params through
                req = {"model": model_id}
                req.update(payload_params)
                r = requests.post(base_url, json=req, headers=headers, timeout=60)
                data = r.json() if r.content else {}
                response_obj["raw_response"] = data
                # Best-effort extraction
                extracted = data.get("result") or data.get("transcript") or data.get("output") or str(data)
                response_obj["extracted"] = extracted
                return response_obj

            else:
                # Fallback: POST params with model
                req = {"model": model_id, **payload_params}
                r = requests.post(base_url, json=req, headers=headers, timeout=30)
                data = r.json() if r.content else {}
                response_obj["raw_response"] = data
                response_obj["extracted"] = data.get("result") or data.get("output") or data
                return response_obj

        except Exception as e:
            raise RuntimeError(f"Execution failed for {tool_id}: {e}")

    # Register with MCP router
    router.register_method("tools/list", list_tools)
    router.register_method("tools/execute", execute_tool)

    app.include_router(router.router, prefix="/api")
    return app


# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Aggregator MCP Server generator")
    ap.add_argument("--csv", default="providers.csv", help="Path to providers CSV (possibly messy)")
    ap.add_argument("--port", default=12000, type=int, help="Port to run on")
    ap.add_argument("--dotenv", default=DEFAULT_DOTENV, help="Path to .env file to create/use")
    ap.add_argument("--hf-token", default=None, help="Optional Hugging Face token for internet lookup")
    ap.add_argument("--auth-mode", default="bearer", help="Default authentication mode (bearer, header:NAME, query:NAME, none)")
    ap.add_argument("--template", default=None, help="Path or URL to provider adapter template (YAML/JSON)")
    args = ap.parse_args()

    print(f"Loading CSV: {args.csv}")
    app = build_aggregator_app(args.csv, dotenv_path=args.dotenv, hf_token=args.hf_token)
    print(f"Starting MCP aggregator server on http://127.0.0.1:{args.port}/api")
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
