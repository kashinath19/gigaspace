# embeddings.py
import asyncio
import datetime
import logging
from typing import List

import asyncpg
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Load model once globally
_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


async def _encode_text(text: str) -> List[float]:
    """
    Run blocking model.encode off the event loop.
    Returns a Python list of floats.
    """
    try:
        vec = await asyncio.to_thread(_embedding_model.encode, text)
        return list(vec.tolist()) if hasattr(vec, "tolist") else list(vec)
    except Exception:
        logger.exception("Embedding generation failed")
        raise


def _vec_to_pg_literal(vec: List[float]) -> str:
    """
    Convert python list of floats to Postgres vector literal string:
      [0.123456, -0.234567, ...]
    Floats formatted to 6 decimal places.
    """
    return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"


async def store_creator_knowledge(
    conn: asyncpg.Connection,
    cid: int,
    uid: str,
    creator_text: str,
    model_text: str,
) -> bool:
    """
    Store a creator conversation (creator_text + model_text + embedding) 
    into character_knowledge.
    """
    try:
        embed = await _encode_text(creator_text)
        pg_vec = _vec_to_pg_literal(embed)

        await conn.execute(
            """
            INSERT INTO character_knowledge (cid, uid, creator, model, created_at, knowledge_embedding)
            VALUES ($1, $2, $3, $4, $5, $6::vector)
            """,
            cid,
            uid,
            creator_text,
            model_text,
            datetime.datetime.now(datetime.timezone.utc),
            pg_vec,
        )
        logger.info(f"✨ Creator knowledge stored for character {cid}")
        return True
    except Exception:
        logger.exception("❌ Failed to store creator knowledge")
        return False


async def search_knowledge(
    conn: asyncpg.Connection, cid: int, query: str, limit: int = 5
) -> List[str]:
    """
    Search using vector similarity against character_knowledge.
    Returns formatted snippets of nearest results.
    """
    try:
        qvec = await _encode_text(query)
        pg_qvec = _vec_to_pg_literal(qvec)

        rows = await conn.fetch(
            """
            SELECT id, creator, model, created_at,
                   knowledge_embedding <-> $2::vector AS distance
            FROM character_knowledge
            WHERE cid = $1
            ORDER BY knowledge_embedding <-> $2::vector
            LIMIT $3
            """,
            cid,
            pg_qvec,
            limit,
        )

        results: List[str] = []
        for r in rows:
            created = r["created_at"].isoformat() if r["created_at"] else ""
            distance = float(r["distance"]) if r["distance"] is not None else None
            snippet = (
                f"{r['creator']}\n"
                f"[model_reply: {r['model']}] "
                f"(dist={distance:.6f}, id={r['id']}, at={created})"
            )
            results.append(snippet)

        return results
    except Exception:
        logger.exception("Embedding search failed")
        return []
