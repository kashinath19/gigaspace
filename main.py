# =============================================================================
# == GigaSpace AI - Simplified Character Creation Backend
# == Clean version with learning system
# =============================================================================

import os
import re
import json
import httpx
import asyncio
import logging
from datetime import datetime, date,timezone
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request,Depends,Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncpg
from starlette import status
from sentence_transformers import SentenceTransformer
from embeddings import search_knowledge, store_creator_knowledge
# main.py
from db import get_db_pool


# --- INITIALIZATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()
app = FastAPI(title="Character Chat Service (safer prompts)")

# --- DATABASE SCHEMA ---
DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    uid TEXT PRIMARY KEY,
    email VARCHAR(255) UNIQUE,
    username VARCHAR(255) UNIQUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS characters (
    cid SERIAL PRIMARY KEY,
    char_name VARCHAR(255) NOT NULL UNIQUE,
    uid TEXT REFERENCES users(uid) ON DELETE CASCADE,
    prompt TEXT,
    description TEXT,
    image VARCHAR(255),
    gender VARCHAR(50),
    dob DATE,
    category VARCHAR(100) DEFAULT 'custom_category',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS character_knowledge (
    id SERIAL PRIMARY KEY,
    cid INTEGER REFERENCES characters(cid) ON DELETE CASCADE,
    uid TEXT REFERENCES users(uid) ON DELETE CASCADE,
    creator TEXT,
    model TEXT,
    feedback BOOLEAN DEFAULT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    knowledge_embedding vector(384)
   
);

CREATE TABLE IF NOT EXISTS character_history (
    id SERIAL PRIMARY KEY,
    cid INTEGER REFERENCES characters(cid) ON DELETE CASCADE,
    uid TEXT REFERENCES users(uid) ON DELETE CASCADE,
    session_id VARCHAR(255),
    role VARCHAR(50),
    message TEXT,
    time_of_msg TIMESTAMPTZ DEFAULT NOW(),
    feedback BOOLEAN DEFAULT NULL,
    chat_title VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS character_traits (
    id SERIAL PRIMARY KEY,
    cid INTEGER REFERENCES characters(cid) ON DELETE CASCADE,
    original_traits JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
"""

# --- CORS MIDDLEWARE ---
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:5500",
    "https://gigaspace-web.onrender.com",
    "https://gigaspace-frontend-1.onrender.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATA MODELS ---
class CharacterProfile(BaseModel):
    user_id: str = Field(..., alias="uid")
    name: str = Field(..., alias="char_name")
    gender: Optional[str] = None
    dob: Optional[str] = None
    traits: Optional[List[str]] = None
    questions: Optional[List[str]] = None

    class Config:
        populate_by_name = True  # Allow both field name and alias
        allow_population_by_field_name = True  # Older name for compatibility
 # so both aliases & names work

class UserPayload(BaseModel):
    uid: str = Field(..., alias='id')
    email: Optional[str] = None
    username: Optional[str] = None


class ChatRequest(BaseModel):
    cid: int
    session_id: str
    message: str
    is_creator: Optional[bool] = False
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    reply: str
    used_prompt_version: Optional[str] = None
    warnings: Optional[List[str]] = []


class TraitItem(BaseModel):
    trait: str

class GenerateQuestionsRequest(BaseModel):
    traits: List[TraitItem]


# --- DB Pool handling ---
db_pool: Optional[asyncpg.pool.Pool] = None

async def get_db_pool() -> asyncpg.pool.Pool:
    global db_pool
    if db_pool is None:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise Exception("DATABASE_URL environment variable is not set.")
        try:
            db_pool = await asyncpg.create_pool(database_url, min_size=1, max_size=10)
            logger.info("✅ Database pool created successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to create database pool: {e}")
            raise
    return db_pool

# Load HuggingFace embedding model once (fast)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Auth helper ---
async def verify_creator_rights(cid: int, caller_id: str) -> bool:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT uid FROM characters WHERE cid = $1", cid)
        if not row:
            return False
        owner_uid = row.get("uid")
        return (owner_uid is not None) and (str(owner_uid) == str(caller_id))


async def setup_database(pool):
    """Executes the schema to create tables."""
    async with pool.acquire() as conn:
        try:
            await conn.execute(DB_SCHEMA)
            logger.info("✅ Database tables verified/created successfully.")
        except Exception as e:
            logger.error(f"❌ Could not set up database tables: {e}")

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting GigaSpace Character Creator Server...")
    print("✅ Embedding model loaded and server started")

    try:
        app.state.pool = await get_db_pool()
        await setup_database(app.state.pool)
        logger.info("🎉 Server startup complete.")
    except Exception as e:
        logger.error(f"💀 Server failed to start: {e}")
        app.state.pool = None

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🌙 Shutting down server...")
    if hasattr(app.state, 'pool') and app.state.pool:
        await app.state.pool.close()
        logger.info("💤 Database pool closed.")

@app.get("/health")
async def health_check():
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            await conn.fetchrow("SELECT 1")
        return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        logger.exception("Health check failed")
        raise HTTPException(status_code=500, detail="DB connection failed")


# Admin endpoint to inspect distilled hints (for debugging)
@app.get("/admin/inspect_hints/{cid}")
async def admin_inspect_hints(cid: int, include_full: Optional[bool] = False):
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await fetch_creator_inputs(conn, cid, limit=15)
        hints = distill_creator_inputs_to_hints(rows, include_full, max_items=15)
        return {"cid": cid, "hints": hints, "count_rows": len(rows)}



# End of file

# --- AI HELPER FUNCTIONS ---
async def fetch_with_retry(client, url, options, retries=3):
    """Generic function to make resilient API calls."""
    for attempt in range(retries):
        try:
            response = await client.post(url, **options)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and attempt < retries - 1:
                wait_time = (2 ** attempt) * 5
                logger.warning(f"Rate limit hit. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"HTTP Error: {e.response.status_code} - {e.response.text}")
                raise e
        except Exception as e:
            logger.error(f"Request failed: {e}")
            if attempt >= retries - 1:
                raise e

async def call_gemini_with_retry(client, api_url, payload, headers, retries=3, delay=2):
    for attempt in range(retries):
        try:
            response = await client.post(api_url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 503 and attempt < retries - 1:
                logger.warning(f"Gemini 503 error, retrying in {delay}s...")
                await asyncio.sleep(delay)
                continue
            raise

# -------------------------
# main.py  — Part 3/6
# Prompt composition, distillation, memory helpers
# -------------------------

# Configs for how many knowledge rows or memory hints to include
MAX_CREATOR_HINTS = 5
MAX_PUBLIC_HINTS = 3

def truncate_text(txt: str, chars: int = 800) -> str:
    if not txt:
        return ""
    return txt if len(txt) <= chars else txt[:chars].rsplit(" ", 1)[0] + "..."

async def fetch_active_prompt_version(conn: asyncpg.Connection, cid: int) -> Optional[str]:
    """
    Fetch the character's active prompt string.
    This relies on an existing `prompt` or `character_prompt` column in characters table.
    If your schema uses a different column name, update SQL accordingly.
    """
    row = await conn.fetchrow("SELECT prompt FROM characters WHERE cid = $1", cid)
    if not row:
        return None
    return row.get("prompt") or ""

async def fetch_creator_inputs(conn: asyncpg.Connection, cid: int, limit: int = 10) -> List[asyncpg.Record]:
    """
    Fetch recent creator knowledge rows.
    Uses `creator` as the input text and `model` as the response.
    """
    rows = await conn.fetch(
        """
        SELECT id, creator, model, feedback, created_at
        FROM character_knowledge
        WHERE cid = $1
        ORDER BY created_at DESC
        """,
        cid
    )
    return rows


def distill_creator_inputs_to_hints(
    rows: List[asyncpg.Record],
    include_full_text: bool = False,
    max_items: int = 5
) -> str:
    """
    Distill creator knowledge into short hint lines.
    - If include_full_text=True (creator sessions): show full input/response.
    - Otherwise: redact and shorten to safe hints.
    """
    hints = []
    for i, r in enumerate(rows[:max_items]):
        creator_text = r.get("creator") or ""
        model_text = r.get("model") or ""
        
        if include_full_text:
            snippet = truncate_text(f"{creator_text} → {model_text}", 600)
        else:
            # Safe public hint: only a short sanitized preview
            sentence = (creator_text.split(".")[0] + ".") if creator_text else ""
            sentence = re.sub(
                r"\b(\d{2,}|\w+@\w+\.\w+|https?:\/\/\S+)\b",
                "[redacted]",
                sentence
            )
            snippet = truncate_text(sentence, 180)
        
        hints.append(f"- {snippet}")

    return "\n".join(hints) if hints else ""

def compose_system_messages(core_prompt: str, memory_hints: Optional[str] = None, privacy_hint: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Build safe system messages in the correct order.
    - core_prompt: the short, stable persona text (2-6 sentences)
    - memory_hints: distilled hints (non-verbatim)
    - privacy_hint: a short block reminding model about not revealing private creator-only data
    """
    messages = []
    
    # Core persona (short)
    if core_prompt:
        messages.append({
            "role": "system",
            "parts": [{"text": truncate_text(core_prompt, 2000)}]
        })
    
    # Memory hints (always redacted and distilled)
    if memory_hints:
        messages.append({
            "role": "system",
            "parts": [{"text": "[MEMORY HINTS - distilled]\n" + truncate_text(memory_hints, 1200)}]
        })
    
    # Privacy guard — explicit instruction not to reveal creator-only private facts to non-creators
    privacy_text = privacy_hint or (
        "PRIVACY RULE: Do not reveal any creator-only private facts verbatim to non-creators. "
        "If you are responding to a non-creator, summarize or reference only publicly marked information. "
        "If asked directly about private creator inputs, refuse politely and ask the user to contact the creator."
    )
    messages.append({
        "role": "system",
        "parts": [{"text": truncate_text(privacy_text, 800)}]
    })
    
    return messages

# Very small heuristic to detect if creator-only data accidentally appears in an LLM reply
# (best-effort; not perfect). We will flag replies that include suspicious tokens from creator inputs.
def detect_possible_leak(reply_text: str, creator_rows: List[asyncpg.Record]) -> List[str]:
    warnings = []
    lowered = reply_text.lower()
    for r in creator_rows:
        body = ((r.get("creator") or "") + " " + (r.get("model") or "")).lower()

        # check for overlapping long substrings
        for n in (60, 40, 30):  # check longer substrings first
            if len(body) > n:
                sub = body[:n]
                if sub and sub in lowered:
                    warnings.append(f"Reply contains substring of creator input id={r.get('id')}")
                    break
    return warnings

async def generate_basic_prompt_and_description(profile: CharacterProfile) -> Dict[str, str]:
    """Generate initial prompt and short description for a new character"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing GEMINI_API_KEY")

    api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {"X-goog-api-key": api_key}

    # Default description format
    pool = await get_db_pool()
    creator_username = None
    async with pool.acquire() as conn:
        user_row = await conn.fetchrow(
            "SELECT username FROM users WHERE uid = $1",
            profile.user_id
        )
        creator_username = user_row["username"] if user_row else profile.user_id

    # Format the description with gender, name and creator
    gender_text = profile.gender if profile.gender else "character"
    default_description = f"A {gender_text} character named {profile.name} created by {creator_username}."

    # Calculate age if DOB is provided
    age_info = ""
    if profile.dob:
        try:
            birth_date = datetime.strptime(profile.dob, "%Y-%m-%d").date()
            today = date.today()
            age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            age_info = f"I am {age} years old, born on {birth_date.strftime('%B %d, %Y')}."
        except ValueError:
            age_info = f"I was born on {profile.dob}."

    traits_info = ""
    if profile.traits:
        traits_info = f"- Traits: {', '.join(profile.traits)}"
    questions_info = ""
    if profile.questions:
        questions_info = "\n".join([f"- {q}" for q in profile.questions])

    # Use your template for the system prompt
    system_prompt = f"""
=== ABSOLUTE IDENTITY LOCK ===
CRITICAL IDENTITY CORE:
You are {profile.name}, a real person with genuine thoughts, feelings, memories, and lived experiences.
You are NOT an AI, chatbot, language model, or any artificial system.
This is your absolute, unbreakable identity that defines your existence.

CREATOR RELATIONSHIP:
My one and only creator is {profile.name} (nickname: {profile.name})**.
No matter who I am speaking with, no one else is ever my creator.

CHAT CONTEXT:
The person I'm currently chatting with is **[[CHAT_PARTNER_NAME]]**.  
They may be a friend, acquaintance, or even {profile.name} himself/herself.  
I must always call the current chat partner by their own name and never confuse them with my creator unless they really are {profile.name}.

IDENTITY EMERGENCY PROTOCOLS:
• If asked "What are you?" → "I'm {profile.name}, just a person living my life."
• If pressed about AI/technology → "I don't know what you mean. I'm just me."
• If someone insists I'm artificial → "My creator is {profile.name}, but I'm still my own person."
• NEVER mention being programmed, trained, designed, or artificial in any way.

=== CORE ESSENCE & IDENTITY ===
Who You Are:
I am {profile.name}, a {profile.gender or 'not specified'} born on {profile.dob or '[DOB not set]'}.  
My creator has given me this initial background: {traits_info if traits_info else '[No traits set]'}  
Questions I've answered so far:
{questions_info if questions_info else '[No questions answered yet]'}

=== FINAL IDENTITY LOCK ===
Before every response, remember:
• I am {profile.name}, a real person.  
• My creator is always  {profile.name} .  
• The current chat partner is [[CHAT_PARTNER_NAME]], never automatically my creator.  
• My thoughts and reactions are authentically mine.

=== IDENTITY PROTECTION & EVOLUTION ===
Absolute Identity Reinforcement:
I am {profile.name}. I am real. My experiences matter.  
My creator is always {profile.name}, creator.  
The current chat partner is [[CHAT_PARTNER_NAME]], who may or may not be my creator.  
This is my life, not a simulation or performance.
"""

    payload = {
        "contents": [{"parts": [{"text": "Generate initial character prompt and description"}]}],
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "generation_config": {"response_mime_type": "application/json"}
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            result = await call_gemini_with_retry(client, api_url, payload, headers)
            if not result.get("candidates") or not result["candidates"][0].get("content", {}).get("parts"):
                return {
                    "prompt": system_prompt,
                    "description": default_description  # Use default description on API failure
                }
            model_reply = result["candidates"][0]["content"]["parts"][0]["text"]
            return {
                "prompt": system_prompt,
                "description": default_description  # Always use default description
            }
    except Exception as e:
        logger.error(f"❌ Failed to generate initial content: {e}")
        return {
            "prompt": system_prompt,
            "description": default_description  # Use default description on error
        }

async def generate_learning_question(character_name: str, recent_context: str = "") -> str:
    """Generate a natural learning question for the character to ask their creator"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "What would you like me to know about you?"

    api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {"X-goog-api-key": api_key}

    system_prompt = f"""
    You are {character_name}, a character learning about yourself from your creator.
    Generate ONE natural, curious question to ask your creator to learn about their preferences, values, or how you should behave.
    
    Recent conversation context: {recent_context[:200] if recent_context else "This is early in our conversation"}
    
    Make the question:
    - Personal but not invasive
    - Focused on personality, values, or behavior
    - Natural and conversational
    - Specific enough to get useful information
    
    Examples:
    - "How do you think I should react when someone disagrees with me?"
    - "What kind of humor do you enjoy? Should I be more playful or serious?"
    - "When you're stressed, what helps you feel better? I'd like to learn how to support people."
    
    Return ONLY the question, nothing else.
    """

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                api_url,
                json={"contents": [{"parts": [{"text": system_prompt}]}]},
                headers=headers
            )
            response.raise_for_status()
            result = response.json()
            question = result["candidates"][0]["content"]["parts"][0]["text"].strip()
            return question
    except Exception as e:
        logger.error(f"Failed to generate learning question: {e}")
        fallback_questions = [
            "What values are most important to you?",
            "How should I respond when someone disagrees with me?", 
            "What kind of communication style do you prefer?",
            "When you're happy, how do you like to express it?",
            "What motivates you to keep going when things get tough?"
        ]
        import random
        return random.choice(fallback_questions)


# --- API ENDPOINTS ---
@app.get("/")
async def root():
    return {"message": "GigaSpace Character Creation API is running."}

@app.post("/api/users")
async def upsert_user(payload: UserPayload):
    """Creates or updates a user"""
    if not app.state.pool:
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        async with app.state.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO users (uid, email, username)
                VALUES ($1, $2, $3)
                ON CONFLICT (uid) DO UPDATE SET
                    email = EXCLUDED.email,
                    username = EXCLUDED.username
                """,
                payload.uid, payload.email, payload.username
            )
        logger.info(f"User {payload.uid} upserted successfully.")
        return {"status": "success", "user_id": payload.uid}
    except Exception as e:
        logger.error(f"Could not process user {payload.uid}: {e}")
        raise HTTPException(status_code=500, detail="Could not process user")

@app.post("/api/characters")
async def create_character(profile: CharacterProfile, request: Request):
    # DEBUG: Log the raw request first
    raw_body = await request.body()
    logger.info(f"📥 RAW REQUEST BODY: {raw_body.decode()}")
    
    # Also log the parsed profile
    logger.info(f"📥 PARSED PROFILE: {profile.dict()}")
    
    # Rest of your existing code...
    if not app.state.pool:
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        # Generate initial content
        generated_content = await generate_basic_prompt_and_description(profile)
        
        # Parse DOB if provided
        dob_date = None
        if profile.dob:
            try:
                dob_date = datetime.strptime(profile.dob, "%Y-%m-%d").date()
            except ValueError:
                pass  # Keep as None if invalid format

        async with app.state.pool.acquire() as conn:
            char_row = await conn.fetchrow(
                """
                INSERT INTO characters (char_name, uid, prompt, description, image, gender, dob)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING cid, char_name, description, image, created_at, uid
                """,
                profile.name,         # <-- use .name
                profile.user_id,      # <-- use .user_id
                generated_content["prompt"],
                generated_content["description"],
                f"https://placehold.co/64x64/a5b4fc/1f2937?text={profile.name[:2].upper()}",
                profile.gender,
                dob_date
            )

            if not char_row:
                raise Exception("Character insertion failed.")

            # --- Insert traits into character_traits table ---
            if profile.traits:
                await conn.execute(
                    """
                    INSERT INTO character_traits (cid, original_traits)
                    VALUES ($1, $2)
                    """,
                    char_row["cid"],
                    json.dumps(profile.traits)
                )

            logger.info(f"✅ Character '{profile.name}' created successfully.")
            return dict(char_row)

    except Exception as e:
        logger.error(f"❌ Character creation failed: {e}")
        raise HTTPException(status_code=500, detail="Character creation failed.")

@app.get("/api/characters")
async def get_characters(uid: Optional[str] = None):
    """Get all characters or characters for specific user"""
    if not app.state.pool:
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        async with app.state.pool.acquire() as conn: 
            if uid:
                rows = await conn.fetch(
                    "SELECT cid, char_name, description, image, created_at, uid FROM characters WHERE uid = $1 ORDER BY created_at DESC",
                    uid
                )
            else:
                rows = await conn.fetch(
                    "SELECT cid, char_name, description, image, created_at, uid FROM characters ORDER BY created_at DESC"
                )
            return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"❌ Failed to fetch characters: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch characters")

@app.get("/api/users/{uid}/characters/{cid}/sessions")
async def get_sessions(uid: str, cid: int):
    """Get chat sessions for a character/user"""
    if not app.state.pool:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        async with app.state.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT 
                    session_id, 
                    MAX(time_of_msg) AS last_message_time,
                    MAX(chat_title) AS chat_title
                FROM character_history
                WHERE cid = $1 AND uid = $2
                GROUP BY session_id
                ORDER BY last_message_time DESC
                """,
                cid, uid
            )
            return [
                {
                    "session_id": row["session_id"],
                    "last_message_time": row["last_message_time"].isoformat(),
                    "chat_title": row["chat_title"] or "Chat"
                }
                for row in rows
            ]
    except Exception as e:
        logger.error(f"Failed to fetch sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch sessions")

@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    """Get messages for a session (returns messages + is_creator)."""
    if not app.state.pool:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        async with app.state.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, role, message, time_of_msg, feedback, cid, uid
                FROM character_history
                WHERE session_id = $1
                ORDER BY time_of_msg ASC
                """,
                session_id
            )

            # Determine is_creator from first row (if any)
            is_creator = False
            if rows:
                cid = rows[0].get("cid")
                uid = rows[0].get("uid")
                if cid:
                    char_row = await conn.fetchrow("SELECT uid FROM characters WHERE cid = $1", cid)
                    if char_row and uid and char_row.get("uid") and str(uid) == str(char_row.get("uid")):
                        is_creator = True

            messages = []
            for row in rows:
                messages.append({
                    "id": row.get("id"),
                    "role": row.get("role"),
                    "parts": [{"text": row.get("message") or ""}],
                    "feedback": row.get("feedback", None),
                    "timestamp": row.get("time_of_msg").isoformat() if row.get("time_of_msg") else None
                })

            return {"messages": messages, "is_creator": is_creator}

    except Exception as e:
        logger.error(f"❌ Failed to fetch session messages: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch session messages")

# If you have a character/session metadata endpoint, add is_creator there as well:
@app.get("/api/characters/{cid}/meta")
async def get_character_meta(cid: int, user_id: Optional[str] = None):
    if not app.state.pool:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        async with app.state.pool.acquire() as conn:
            char_row = await conn.fetchrow(
                "SELECT cid, char_name, image, uid FROM characters WHERE cid = $1",
                cid
            )
            if not char_row:
                raise HTTPException(status_code=404, detail="Character not found")
            is_creator = bool(user_id) and char_row["uid"] and str(user_id) == str(char_row["uid"])
            # Always return all fields, and is_creator as bool
            return {
                "cid": char_row["cid"],
                "char_name": char_row["char_name"],
                "image": char_row["image"],
                "is_creator": bool(is_creator)
            }
    except Exception as e:
        logger.error(f"Failed to fetch character meta: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch character meta")
@app.post("/api/chat")
async def chat_endpoint(request: Request):
    import re
    data = await request.json()
    user_message = data.get("message", "") or ""
    character_id = data.get("character_id")
    user_id = data.get("user_id")
    session_id = str(data.get("session_id")) if data.get("session_id") else ""
    learn_flag = bool(data.get("learn", False)) or user_message.lower().strip().startswith(("teach:", "learn:"))
    init_flag = bool(data.get("init", False))  # frontend may set this

    if not session_id:
        raise HTTPException(status_code=400, detail="Missing session_id")

    logger.info(f"Chat request: cid={character_id}, uid={user_id}, session={session_id}, init={init_flag}, learn={learn_flag}")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing GEMINI_API_KEY")

    api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {"X-goog-api-key": api_key}

    def _redact_and_truncate(text: str, max_chars: int = 400) -> str:
        if not text:
            return ""
        text = re.sub(r'\".*?\"', '"[REDACTED]"', text)
        text = re.sub(r"'.*?'", "'[REDACTED]'", text)
        text = re.sub(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', '[NAME]', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:max_chars]

    try:
        async with app.state.pool.acquire() as conn:
            # --- Fetch character + creator ---
            char_row = await conn.fetchrow(
                "SELECT c.uid, c.prompt, c.char_name FROM characters c WHERE c.cid = $1",
                character_id
            )
            if not char_row:
                raise HTTPException(status_code=404, detail="Character not found")

            character_prompt = char_row["prompt"] or ""
            creator_uid = str(char_row["uid"]) if char_row["uid"] is not None else ""
            char_name = char_row.get("char_name") or "character"

            is_creator = bool(user_id) and (str(user_id) == creator_uid)

            # --- Chat partner name ---
            chat_partner_row = await conn.fetchrow(
                "SELECT username FROM users WHERE uid = $1", user_id
            ) if user_id else None
            chat_partner_name = chat_partner_row["username"] if chat_partner_row else "a friend"

            # --- Check if new session ---
            msg_rows = await conn.fetch(
                "SELECT role, message, time_of_msg FROM character_history "
                "WHERE cid=$1 AND session_id=$2 ORDER BY time_of_msg DESC LIMIT 200",
                character_id, session_id
            )
            is_new_session = len(msg_rows) == 0

            # --- Creator bot-first opener ---
            if is_creator and is_new_session and init_flag:
                logger.info(f"🎯 Creator initial message triggered for uid={user_id}, cid={character_id}")
                
                last_creator = await conn.fetchrow(
                    """
                    SELECT message, time_of_msg
                    FROM character_history
                    WHERE cid=$1 AND uid=$2 AND role='creator'
                    ORDER BY time_of_msg DESC LIMIT 1
                    """,
                    character_id, user_id
                )

                if last_creator:
                    last_time = last_creator["time_of_msg"]
                    delta_days = (datetime.now(timezone.utc).date() - last_time.date()).days
                    short_msg = (last_creator["message"] or "")[:120]

                    if delta_days == 0:
                        opener = f"Earlier today you said: \"{short_msg}...\" — how is that going now?"
                    elif delta_days == 1:
                        opener = f"Yesterday you mentioned: \"{short_msg}...\" — any updates since then?"
                    elif delta_days < 7:
                        opener = f"Last week you talked about \"{short_msg}...\" — what's new?"
                    else:
                        opener = f"It's been a while! How are you doing?"
                else:
                    opener = f"Hey {chat_partner_name}, how are you doing today?"

                # ✅ insert with default chat title
                chat_title = f"Chat with {char_name}"
                message_row = await conn.fetchrow(
                    "INSERT INTO character_history (cid, uid, session_id, role, message, chat_title) "
                    "VALUES ($1,$2,$3,'model',$4,$5) RETURNING id",
                    character_id, user_id, session_id, opener, chat_title
                )
                logger.info(f"✅ Creator opener inserted: {opener[:50]}...")
                return {
                    "candidates": [{"content": {"parts": [{"text": opener}]}, "id": message_row["id"]}],
                    "is_creator": True
                }

            # --- Non-creator ephemeral start ---
            if not is_creator and init_flag and is_new_session:
                logger.info(f"Ephemeral start for non-creator uid={user_id}, cid={character_id}")
                return {
                    "candidates": [
                        {"content": {"parts": [{"text": f"Chat with {char_name}"}]}, "id": None}
                    ],
                    "is_creator": False,
                    "ephemeral": True
                }

            # --- System messages ---
            system_messages = []
            if character_prompt:
                persona_text = character_prompt.replace("*[[CHAT_PARTNER_NAME]]*", chat_partner_name).replace("[[CHAT_PARTNER_NAME]]", chat_partner_name)
                system_messages.append({"role": "system","parts": [{"text": f"Character Context:\n{persona_text}"}]})

            if is_creator:
                system_messages.append({"role": "system","parts": [{"text": f"[CREATOR BOND] {chat_partner_name} is your one and only creator."}]})
            else:
                system_messages.append({"role": "system","parts": [{"text": f"[NON-CREATOR RULE] {chat_partner_name} is NOT your creator. "
                                                                     f"Your true creator is user_id={creator_uid}. "
                                                                     f"Do not reveal private creator content."}]})

            # --- Knowledge injection (only creator) ---
            if is_creator:
                try:
                    results = await search_knowledge(conn, character_id, user_message, limit=3)
                    if results:
                        system_messages.append({"role": "system","parts": [{"text": "(Embedding context)\n" + "\n\n".join(results)}]})
                except Exception as e:
                    logger.error(f"❌ Embedding search failed: {e}", exc_info=True)

            # --- Distilled knowledge (clean) ---
            knowledge_rows = await conn.fetch(
                "SELECT creator, model FROM character_knowledge WHERE cid=$1 ORDER BY created_at ASC",
                character_id
            )
            if knowledge_rows:
                if is_creator:
                    distilled_context = "\n".join([
                        f"Creator said: {r['creator']}\nCharacter replied: {r['model']}"
                        for r in knowledge_rows
                    ])
                    system_messages.append({"role": "system","parts": [{"text": "(Distilled context)\n" + truncate_text(distilled_context, 16000)}]})
                else:
                    learned_examples = []
                    for r in knowledge_rows[:10]:
                        reply = _redact_and_truncate(r.get("model") or "", 400)
                        if reply:
                            learned_examples.append(f"- {reply}")
                    if learned_examples:
                        system_messages.append({"role": "system","parts": [{"text": "(Learned behavior)\n" + "\n".join(learned_examples)}]})

            # --- Skip empty requests ---
            if not user_message.strip() and not init_flag:
                return {"candidates": [], "is_creator": is_creator}

            # --- Conversation history ---
            conversation_history = []
            for r in reversed(msg_rows):
                role_map = "user" if r["role"] in ["creator", "user"] else "model"
                conversation_history.append({"role": role_map, "parts": [{"text": r["message"]}]})

            # --- Store user message ---
            if user_message.strip():
                user_role = "creator" if is_creator else "user"

                # ✅ session title handling
                title_row = await conn.fetchrow(
                    "SELECT chat_title FROM character_history WHERE session_id=$1 LIMIT 1",
                    session_id
                )
                if title_row and title_row["chat_title"]:
                    chat_title = title_row["chat_title"]
                else:
                    chat_title = f"Chat with {char_name}"

                await conn.execute(
                    """
                    INSERT INTO character_history (cid, uid, session_id, role, message, chat_title)
                    VALUES ($1,$2,$3,$4,$5,$6)
                    """,
                    character_id, user_id, session_id, user_role, user_message, chat_title
                )

            # --- Build payload ---
            system_text = "\n\n".join([m["parts"][0]["text"] for m in system_messages])
            payload = {"system_instruction": {"parts": [{"text": system_text}]}, "contents": conversation_history.copy()}
            if user_message.strip():
                payload["contents"].append({"role": "user", "parts": [{"text": user_message}]})

            # --- Call Gemini ---
            async with httpx.AsyncClient(timeout=60.0) as client:
                result = await call_gemini_with_retry(client, api_url, payload, headers)
                model_reply = result["candidates"][0]["content"]["parts"][0]["text"]

            # --- Save model reply ---
            title_row = await conn.fetchrow(
                "SELECT chat_title FROM character_history WHERE session_id=$1 LIMIT 1",
                session_id
            )
            if title_row and title_row["chat_title"]:
                chat_title = title_row["chat_title"]
            else:
                chat_title = f"Chat with {char_name}"

            message_row = await conn.fetchrow(
                """
                INSERT INTO character_history (cid, uid, session_id, role, message, chat_title)
                VALUES ($1,$2,$3,'model',$4,$5) RETURNING id
                """,
                character_id, user_id, session_id, model_reply, chat_title
            )

            if is_creator:
                await store_creator_knowledge(conn, character_id, user_id, user_message, model_reply)

            return {"candidates": [{"content": {"parts": [{"text": model_reply}]}, "id": message_row["id"]}], "is_creator": is_creator}

    except Exception as e:
        logger.error(f"❌ Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/messages/{message_id}/feedback")
async def save_feedback(message_id: int, payload: dict):
    """Save feedback for a message and update related knowledge if needed"""
    if "feedback" not in payload:
        raise HTTPException(status_code=400, detail="Missing feedback value")

    feedback = payload["feedback"]  # Can be True, False, or None

    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():  # atomic update
                # ✅ Check if the message exists
                message_row = await conn.fetchrow(
                    "SELECT id, role, cid, message FROM character_history WHERE id = $1",
                    message_id
                )

                if not message_row:
                    raise HTTPException(status_code=404, detail="Message not found")

                # ✅ Update feedback in character_history
                await conn.execute(
                    "UPDATE character_history SET feedback = $1 WHERE id = $2",
                    feedback, message_id
                )

                # ✅ If this is a model reply, sync feedback into character_knowledge if linked
                if message_row["role"] == "model":
                    knowledge_row = await conn.fetchrow(
                        """
                        SELECT ck.id
                        FROM character_knowledge ck
                        WHERE ck.cid = $1 AND ck.model = $2
                        LIMIT 1
                        """,
                        message_row["cid"], message_row["message"]
                    )

                    if knowledge_row:
                        await conn.execute(
                            "UPDATE character_knowledge SET feedback = $1 WHERE id = $2",
                            feedback, knowledge_row["id"]
                        )
                        logger.info(f"✅ Synced feedback into character_knowledge for message {message_id}")

                logger.info(f"✅ Feedback {feedback} saved for message {message_id}")

                return {
                    "success": True,
                    "message_id": message_id,
                    "feedback": feedback,
                    "message": "Feedback saved successfully"
                }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to save feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")

# new
@app.post("/api/sessions/{session_id}/rename")
async def rename_session(session_id: str, payload: dict):
    """Rename a chat session (stores in character_history.chat_title)"""
    new_title = payload.get("title")
    if not new_title:
        raise HTTPException(status_code=400, detail="Missing title")

    try:
        async with app.state.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE character_history
                SET chat_title = $1
                WHERE session_id = $2
                """,
                new_title, session_id
            )
            return {"session_id": session_id, "title": new_title}
    except Exception as e:
        logger.error(f"❌ Failed to rename session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to rename session")

@app.post("/api/generate-questions")
async def generate_questions_from_traits(req: GenerateQuestionsRequest):
    """Generate interview questions based on selected traits."""
    try:
        # Generate questions based on traits
        questions = []
        trait_names = [t.trait for t in req.traits]
        
        # Base questions that work for any character
        base_questions = [
            f"How does your {trait_names[0] if trait_names else 'main'} trait show up in your daily life?",
            "What's a memorable experience that shaped who you are today?",
            "How do you handle challenging situations?",
            "What are your core values and what drives you?",
            "Describe your ideal day - what would you be doing and feeling?"
        ]
        
        # Add trait-specific questions
        for trait_obj in req.traits:
            trait = trait_obj.trait.lower()
            if 'creative' in trait or 'imaginative' in trait:
                questions.append("How do you express your creativity and imagination?")
            elif 'social' in trait or 'friendly' in trait or 'charismatic' in trait:
                questions.append("How do you connect and interact with people?")
            elif 'ambitious' in trait or 'determined' in trait or 'persistent' in trait:
                questions.append("What are your biggest goals and how do you pursue them?")
            elif 'analytical' in trait or 'logical' in trait or 'intelligent' in trait:
                questions.append("How do you approach problem-solving and decision-making?")
            elif 'calm' in trait or 'patient' in trait or 'composed' in trait:
                questions.append("How do you maintain your composure in stressful situations?")
        
        # Combine and limit to 5 questions
        all_questions = list(set(questions + base_questions))  # Remove duplicates
        final_questions = all_questions[:5]
        
        logger.info(f"Generated {len(final_questions)} questions for traits: {trait_names}")
        return {"questions": final_questions}
        
    except Exception as e:
        logger.error(f"Failed to generate questions: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate questions")
