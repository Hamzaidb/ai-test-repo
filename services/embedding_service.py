"""
embedding_service.py – Service d'embeddings utilisant l'API OpenAI.
Génère des vecteurs d'embedding via le modèle text-embedding-3-small.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from openai import OpenAI


@dataclass
class EmbeddingService:
    """Encapsule les appels à l'API OpenAI Embeddings."""

    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = "text-embedding-3-small"
    dimensions: int = 1536

    def __post_init__(self) -> None:
        self.client = OpenAI(api_key=self.api_key)

    def embed_text(self, text: str) -> list[float]:
        """Génère un vecteur d'embedding pour un texte donné.

        Args:
            text: Le texte à encoder.

        Returns:
            Liste de floats représentant le vecteur d'embedding.
        """
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Génère des embeddings pour une liste de textes en un seul appel.

        Args:
            texts: Liste de textes à encoder.

        Returns:
            Liste de vecteurs d'embedding (un par texte d'entrée).
        """
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        # Trier par index pour garantir l'ordre
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]

    def cosine_similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        """Calcule la similarité cosinus entre deux vecteurs.

        Args:
            vec_a: Premier vecteur.
            vec_b: Deuxième vecteur.

        Returns:
            Score de similarité entre -1 et 1.
        """
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        magnitude_a = sum(a ** 2 for a in vec_a) ** 0.5
        magnitude_b = sum(b ** 2 for b in vec_b) ** 0.5

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)
