"""LLM resolution for Task 3 (RAG pipeline).

Priority (LLM_PROVIDER=auto):
  1. Google Gemini when GEMINI_API_KEY / GOOGLE_API_KEY is set
  2. Local Ollama server (llama3.1 by default)
  3. Any OpenAI-compatible API (OpenAI / Groq / OpenRouter / LM Studio)
     when OPENAI_API_KEY is set (OPENAI_BASE_URL supported)
  4. None -> the pipeline falls back to extractive answers (no LLM).

LLM_PROVIDER=gemini|ollama|openai|none forces a choice; a forced provider
that is unavailable degrades to None instead of crashing the app.
"""
import requests

from rag.config import (GEMINI_API_KEY, GEMINI_MODEL, LLM_PROVIDER,
                        OLLAMA_BASE_URL, OLLAMA_MODEL, OPENAI_API_KEY,
                        OPENAI_BASE_URL, OPENAI_MODEL)


def _ollama_reachable() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def get_llm() -> tuple[object | None, str]:
    """Return (langchain_llm_or_None, human readable label)."""
    provider = LLM_PROVIDER

    def try_ollama():
        if not _ollama_reachable():
            return None, None
        from langchain_ollama import ChatOllama
        return ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1), \
               f"Local LLM: Ollama `{OLLAMA_MODEL}`"

    def try_openai():
        if not OPENAI_API_KEY:
            return None, None
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=OPENAI_MODEL, temperature=0.1,
                          api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL), \
               f"OpenAI-compatible LLM: `{OPENAI_MODEL}`" + (f" @ {OPENAI_BASE_URL}" if OPENAI_BASE_URL else "")

    def try_gemini():
        if not GEMINI_API_KEY:
            return None, None
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except Exception as e:
            return None, f"GEMINI_API_KEY set but `langchain-google-genai` not installed: {e}"
        return ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.1,
                                      google_api_key=GEMINI_API_KEY), \
               f"Google Gemini: `{GEMINI_MODEL}`"

    if provider == "gemini":
        llm, label = try_gemini()
        return (llm, label) if llm else (None, "GEMINI_API_KEY missing - extractive mode")
    if provider == "ollama":
        llm, label = try_ollama()
        return (llm, label) if llm else (None, "Ollama requested but not reachable - extractive mode")
    if provider == "openai":
        llm, label = try_openai()
        return (llm, label) if llm else (None, "OPENAI_API_KEY missing - extractive mode")
    if provider == "none":
        return None, "Extractive mode (LLM disabled via LLM_PROVIDER=none)"

    # auto
    llm, label = try_gemini()
    if llm:
        return llm, label
    if GEMINI_API_KEY:
        # Key is present but Gemini didn't initialize -> tell the user, don't
        # silently fall back (a wrong/missing key is the most common deploy bug).
        return None, ("Extractive mode - GEMINI_API_KEY is set but the Gemini client "
                      "could not be initialized. Check the key at "
                      "aistudio.google.com/apikey and confirm it's added to Streamlit Secrets.")
    llm, label = try_ollama()
    if llm:
        return llm, label
    llm, label = try_openai()
    if llm:
        return llm, label
    return None, ("Extractive mode - no LLM found. "
                  "Set GEMINI_API_KEY / OPENAI_API_KEY in .env, or install Ollama + "
                  "`ollama pull llama3.1`, for generative answers.")
