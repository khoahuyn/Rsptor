from typing import List
from models import SummaryOutput, RaptorParams
from config.llm import get_llm_settings
from .fpt_client import create_fpt_client
from utils import llm_errors, get_llm_cache, set_llm_cache



def _summarize_with_fpt_client(text: str, system_prompt: str, max_tokens: int, temperature: float = None) -> SummaryOutput:
    """Summarize using direct FPT Cloud HTTP client with RAGFlow-style caching"""
    cfg = get_llm_settings()
    
    # Use config default if not provided
    if temperature is None:
        temperature = cfg.summary_temperature
    
    # Prepare chat parameters for caching (RAGFlow approach)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]
    gen_conf = {
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    # Check cache first (RAGFlow approach)
    cached_response = get_llm_cache(
        model=cfg.model,
        system=system_prompt,
        history=messages,
        gen_conf=gen_conf
    )
    
    if cached_response:
        return SummaryOutput(content=cached_response)
    
    # Cache miss - make actual API call
    fpt_client = create_fpt_client(cfg.base_url, cfg.api_key)
    
    response = fpt_client.chat_completions_create(
        model=cfg.model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        max_retries=1
    )
    
    # Get content - let errors bubble up to @llm_errors decorator
    content = response.choices[0].message.content.strip()
    
    # Basic cleanup only
    content = content.replace('```', '').strip()
    
    # Cache the response (RAGFlow approach)
    set_llm_cache(
        model=cfg.model,
        response=content,
        system=system_prompt,
        history=messages,
        gen_conf=gen_conf
    )
    
    return SummaryOutput(content=content)

@llm_errors
def summarize_texts_structured(text: str, system_prompt: str, max_tokens: int, temperature: float = None) -> SummaryOutput:
    """
    Summarize text using FPT Cloud structured output.
    """
    return _summarize_with_fpt_client(text, system_prompt, max_tokens, temperature)


@llm_errors
def summarize_cluster_from_contents(
    member_contents: List[str], 
    member_ids: List[str], 
    p: RaptorParams
) -> str:
    """
    Summarize cluster content from member texts - RAGFlow simple approach.
    Returns only content string, no sources (simplified model).
    """
    
    # Check if summarization is enabled
    if not bool(getattr(p, "enable_summary", False)):
        return f"Cluster summary ({len(member_contents)} nodes)"

    # Get summary settings from LLM config
    llm_settings = get_llm_settings()
    max_tokens = int(getattr(p, "summary_max_tokens", None) or llm_settings.summary_max_tokens)
    
    # Pack content with IDs for LLM context
    lines = [f"[{mid}] {content}" for mid, content in zip(member_ids, member_contents)]
    user_content = "\n".join(lines)
    
    sys_prompt = (getattr(p, "summary_prompt", None) or llm_settings.summary_prompt).replace("{max_tokens}", str(max_tokens))

    temperature = getattr(p, "temperature", None) or llm_settings.summary_temperature
    
    out = summarize_texts_structured(text=user_content, system_prompt=sys_prompt, max_tokens=max_tokens, temperature=temperature)
    
    return out.content
