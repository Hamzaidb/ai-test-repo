# AI Test Repo

Repo de test pour vérifier le bon fonctionnement d'un outil de migration automatique de modèles IA.

## Objectif

Ce projet simule une application Python réaliste utilisant l'API OpenAI (SDK v1+).  
Il sert de base pour tester la migration automatique du modèle `gpt-4o` vers `gpt-5` (ou tout autre modèle cible).

## Structure

```
ai-test-repo/
├── README.md
├── requirements.txt
├── app.py                        # API Flask – endpoint /chat
├── services/
│   ├── chat_service.py           # ChatService (gpt-4o, gpt-4o-mini)
│   └── embedding_service.py      # EmbeddingService (text-embedding-3-small)
└── scripts/
    └── batch_process.py          # Traitement batch CSV avec gpt-4o
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
# Lancer le serveur Flask
export OPENAI_API_KEY="sk-..."
python app.py
```

## Modèles utilisés

| Modèle                   | Fichier(s)                              |
| ------------------------ | --------------------------------------- |
| `gpt-4o`                 | app.py, chat_service.py, batch_process.py |
| `gpt-4o-mini`            | chat_service.py                         |
| `text-embedding-3-small` | embedding_service.py                    |
