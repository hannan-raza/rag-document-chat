"""
Provider abstraction for embeddings + generation.
Switch between OpenAI and AWS Bedrock with the LLM_PROVIDER env var.

OpenAI:  text-embedding-3-small (1536 dims) + gpt-4o-mini
Bedrock: Titan Text Embeddings V2 (1024 dims) + Claude Haiku

Note: switching providers changes embedding dimensions, so the documents
table must be recreated at the matching size and re-ingested.
"""
import os
import json
import numpy as np
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_PROFILE = os.getenv("AWS_PROFILE")

# embedding dimensions per provider (used when (re)creating the table)
EMBED_DIMS = {"openai": 1536, "bedrock": 1024}
DIMENSIONS = EMBED_DIMS[PROVIDER]

if PROVIDER == "openai":
    from openai import OpenAI
    _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    GEN_MODEL = os.getenv("OPENAI_GEN_MODEL", "gpt-4o-mini")

    def embed(text):
        resp = _client.embeddings.create(model=EMBED_MODEL, input=text)
        return np.array(resp.data[0].embedding)

    def generate(prompt, max_tokens=400, temperature=0):
        resp = _client.chat.completions.create(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    def generate_stream(prompt, max_tokens=400, temperature=0):
        """Streaming twin of generate(): yields text deltas as they arrive.
        Synchronous generator (the provider layer stays sync); callers that need
        it on an event loop should bridge it through a thread."""
        stream = _client.chat.completions.create(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            # trailing/keepalive chunks can have empty choices or a None delta
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

elif PROVIDER == "bedrock":
    import boto3
    if AWS_PROFILE:
        _session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    else:
        _session = boto3.Session(region_name=AWS_REGION)
    _bedrock = _session.client("bedrock-runtime")
    EMBED_MODEL = os.getenv("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
    GEN_MODEL = os.getenv("BEDROCK_GEN_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")

    def embed(text):
        body = json.dumps({"inputText": text})
        resp = _bedrock.invoke_model(modelId=EMBED_MODEL, body=body)
        return np.array(json.loads(resp["body"].read())["embedding"])

    def generate(prompt, max_tokens=400, temperature=0):
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        })
        resp = _bedrock.invoke_model(modelId=GEN_MODEL, body=body)
        return json.loads(resp["body"].read())["content"][0]["text"]

    def generate_stream(prompt, max_tokens=400, temperature=0):
        """Streaming twin of generate(): yields text deltas as they arrive.
        Synchronous generator (the provider layer stays sync); callers that need
        it on an event loop should bridge it through a thread."""
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        })
        resp = _bedrock.invoke_model_with_response_stream(modelId=GEN_MODEL, body=body)
        for event in resp["body"]:
            chunk = json.loads(event["chunk"]["bytes"])
            if chunk.get("type") == "content_block_delta":
                text = chunk["delta"].get("text")
                if text:
                    yield text

else:
    raise ValueError(f"Unknown LLM_PROVIDER: {PROVIDER}. Use 'openai' or 'bedrock'.")
