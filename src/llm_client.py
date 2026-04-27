"""OpenAI LLM client with streaming support."""
import os
from typing import Iterator, Optional
from openai import OpenAI
from dotenv import load_dotenv
from .config import OPENAI_MODEL

load_dotenv()


class LLMClient:
    """Client for OpenAI API with streaming support"""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model or OPENAI_MODEL
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

    def is_configured(self) -> bool:
        """Check if client is properly configured"""
        return bool(self.api_key and self.client)

    def stream_completion(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None
    ) -> Iterator[str]:
        """Stream completion from OpenAI"""

        if not self.is_configured():
            yield "Error: OpenAI API key not configured. Please set OPENAI_API_KEY in your environment."
            return

        # Build messages with system prompt
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                stream=True
            )

            for chunk in response:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            yield f"Error: {str(e)}"

    def complete(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None
    ) -> str:
        """Non-streaming completion"""

        if not self.is_configured():
            return "Error: OpenAI API key not configured. Please set OPENAI_API_KEY in your environment."

        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Error: {str(e)}"

    def structured_completion(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None,
        response_format: Optional[dict] = None
    ) -> str:
        """Completion with structured output (JSON mode)"""

        if not self.is_configured():
            return "Error: OpenAI API key not configured."

        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        selected_response_format = response_format or {"type": "json_object"}

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                response_format=selected_response_format
            )

            return response.choices[0].message.content

        except Exception as e:
            if selected_response_format.get("type") == "json_schema":
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=full_messages,
                        response_format={"type": "json_object"},
                    )
                    return response.choices[0].message.content
                except Exception as fallback_error:
                    return f"Error: {str(fallback_error)}"

            return f"Error: {str(e)}"
