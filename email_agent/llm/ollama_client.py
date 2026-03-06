from __future__ import annotations

import httpx
from typing import Any, Dict, Optional


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.1"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        num_predict: int = 220,          # cap output length (helps a LOT)
        timeout_s: float = 180.0,        # increase timeout for cold start
    ) -> str:
        url = f"{self.base_url}/api/chat"

        payload: Dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        timeout = httpx.Timeout(timeout_s, connect=10.0, read=timeout_s, write=timeout_s)

        with httpx.Client(timeout=timeout) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()

        return (data.get("message") or {}).get("content", "") or ""

    def warmup(self) -> None:
        # tiny request to ensure model is loaded
        _ = self.chat(
            system="Return ONLY JSON: {\"ok\": true}",
            user="Say ok",
            temperature=0.0,
            num_predict=30,
            timeout_s=180.0,
        )
