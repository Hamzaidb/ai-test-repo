"""
chat_service.py – Service de chat utilisant l'API OpenAI.
Fournit des méthodes pour la génération de réponses (gpt-4o)
et la synthèse de texte (gpt-4o-mini).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from openai import OpenAI


@dataclass
class ChatService:
    """Encapsule les appels à l'API OpenAI Chat Completions."""

    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    default_model: str = "gpt-4o"
    summary_model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 2048

    def __post_init__(self) -> None:
        self.client = OpenAI(api_key=self.api_key)

    # ------------------------------------------------------------------ #
    # Génération de réponse
    # ------------------------------------------------------------------ #
    def generate_response(
        self,
        user_message: str,
        system_prompt: str = "Tu es un assistant utile.",
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """Génère une réponse à partir d'un message utilisateur.

        Args:
            user_message: Le message de l'utilisateur.
            system_prompt: Instruction système optionnelle.
            temperature: Température d'échantillonnage (override).
            max_tokens: Nombre max de tokens (override).

        Returns:
            dict contenant la réponse, le modèle utilisé et les tokens consommés.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        response = self.client.chat.completions.create(
            model="gpt-5",
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
        )

        choice = response.choices[0]
        return {
            "content": choice.message.content,
            "model": response.model,
            "finish_reason": choice.finish_reason,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }

    # ------------------------------------------------------------------ #
    # Synthèse / résumé
    # ------------------------------------------------------------------ #
    def summarize(
        self,
        text: str,
        *,
        max_sentences: int = 3,
        language: str = "français",
    ) -> str:
        """Produit un résumé concis du texte fourni en utilisant gpt-4o-mini.

        Args:
            text: Le texte à résumer.
            max_sentences: Nombre maximum de phrases dans le résumé.
            language: Langue du résumé.

        Returns:
            Le résumé sous forme de chaîne de caractères.
        """
        system_prompt = (
            f"Tu es un expert en synthèse. Résume le texte suivant en {max_sentences} "
            f"phrases maximum, en {language}. Sois factuel et concis."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
            max_tokens=512,
        )

        return response.choices[0].message.content

    # ------------------------------------------------------------------ #
    # Multi-turn conversation
    # ------------------------------------------------------------------ #
    def chat(self, conversation: list[dict], *, model: str | None = None) -> dict:
        """Envoie une conversation multi-tours complète au modèle.

        Args:
            conversation: Liste de messages [{"role": ..., "content": ...}].
            model: Modèle à utiliser (par défaut : self.default_model).

        Returns:
            dict avec la réponse et les métadonnées.
        """
        response = self.client.chat.completions.create(
            model=model or self.default_model,
            messages=conversation,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return {
            "content": response.choices[0].message.content,
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }
