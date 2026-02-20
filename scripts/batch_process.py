"""
batch_process.py – Script standalone de traitement batch.
Lit un fichier CSV, envoie chaque ligne à l'API OpenAI (gpt-4o)
et écrit les résultats dans un fichier CSV de sortie.

Usage :
    python scripts/batch_process.py --input data.csv --output results.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

from openai import OpenAI

# ---------- Configuration ----------
MODEL = "gpt-4o"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2
TEMPERATURE = 0.5
MAX_TOKENS = 1024

SYSTEM_PROMPT = (
    "Tu es un assistant d'analyse de données. "
    "On te fournit une entrée structurée issue d'un fichier CSV. "
    "Analyse-la et fournis une réponse concise et utile."
)


def create_client() -> OpenAI:
    """Crée et retourne un client OpenAI configuré."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Erreur : la variable d'environnement OPENAI_API_KEY n'est pas définie.")
        sys.exit(1)
    return OpenAI(api_key=api_key)


def process_row(client: OpenAI, row: dict, row_index: int) -> str:
    """Traite une ligne du CSV en appelant l'API OpenAI.

    Args:
        client: Instance du client OpenAI.
        row: Dictionnaire représentant une ligne CSV.
        row_index: Numéro de la ligne (pour le logging).

    Returns:
        La réponse générée par le modèle.
    """
    # Formater la ligne CSV en texte lisible
    row_text = "\n".join(f"  {key}: {value}" for key, value in row.items())
    user_message = f"Ligne {row_index} :\n{row_text}\n\nAnalyse cette entrée."

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return response.choices[0].message.content

        except Exception as exc:
            print(f"  ⚠ Tentative {attempt}/{MAX_RETRIES} échouée : {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS * attempt)  # back-off linéaire
            else:
                return f"ERREUR après {MAX_RETRIES} tentatives : {exc}"


def main() -> None:
    """Point d'entrée principal du script batch."""
    parser = argparse.ArgumentParser(
        description="Traitement batch d'un fichier CSV via l'API OpenAI."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Chemin vers le fichier CSV d'entrée.",
    )
    parser.add_argument(
        "--output", "-o",
        default="results.csv",
        help="Chemin vers le fichier CSV de sortie (défaut : results.csv).",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Nombre maximum de lignes à traiter.",
    )
    args = parser.parse_args()

    # Vérification du fichier d'entrée
    if not os.path.isfile(args.input):
        print(f"Erreur : le fichier '{args.input}' n'existe pas.")
        sys.exit(1)

    client = create_client()

    # Lecture du CSV
    with open(args.input, newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not rows:
        print("Le fichier CSV est vide.")
        sys.exit(0)

    if args.limit:
        rows = rows[: args.limit]

    print(f"📄 Fichier : {args.input}")
    print(f"📊 Lignes à traiter : {len(rows)}")
    print(f"🤖 Modèle : {MODEL}")
    print("-" * 50)

    # Traitement ligne par ligne
    results = []
    for idx, row in enumerate(rows, start=1):
        print(f"[{idx}/{len(rows)}] Traitement en cours…")
        ai_response = process_row(client, row, idx)
        result_row = {**row, "ai_response": ai_response}
        results.append(result_row)

    # Écriture du CSV de sortie
    output_fieldnames = list(fieldnames) + ["ai_response"]
    with open(args.output, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("-" * 50)
    print(f"✅ Terminé ! Résultats écrits dans '{args.output}'.")


if __name__ == "__main__":
    main()
