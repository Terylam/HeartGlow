import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
import json
from datetime import datetime, timedelta, timezone
from fastapi.responses import FileResponse
from pydantic import BaseModel
from gtts import gTTS
import uuid
import os
import time
import uvicorn
from rag_system_new import rag_generate
from typing import Dict, Any, Union
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from fastapi import UploadFile, File

# Hong Kong timezone
hkt = timezone(timedelta(hours=8))

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection (Docker)
#mongo_url = os.getenv("MONGODB_URL", "mongodb://mongodb:27017/myapp") 
#mongo_url = os.getenv("MONGODB_URL", "mongodb://mongodb:27017/")

# MongoDB connection (Local)
mongo_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017/")
try:
    client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
    # Test the connection
    client.admin.command('ping')
    print("✓ MongoDB connected successfully")
    db = client["chatbot_db"]
    messages = db["messages"]
except Exception as e:
    print(f"⚠ MongoDB connection failed: {e}")
    print("  Chat functionality will be limited - messages won't persist")
    # Create a simple in-memory fallback (for testing without MongoDB)
    class MockCollection:
        def __init__(self):
            self.data = []
        def insert_one(self, doc):
            # Use the outer datetime class directly (imported from datetime module)
            doc['_id'] = len(self.data)
            # Store time as timestamp if it's a datetime object
            if 'time' in doc:
                if hasattr(doc['time'], 'timestamp'):
                    doc['_id'] = doc['time'].timestamp()
                elif isinstance(doc['time'], str):
                    # Try to parse ISO format string
                    try:
                        ts = datetime.fromisoformat(doc['time'].replace("Z", "+00:00"))
                        doc['_id'] = ts.timestamp()
                    except:
                        doc['_id'] = datetime.now(hkt).timestamp()
            else:
                doc['_id'] = datetime.now(hkt).timestamp()
            self.data.append(doc)
            return type('obj', (object,), {'inserted_id': doc['_id']})
        def find(self, query=None):
            if query is None:
                return iter(self.data)
            # Simple query filtering
            result = self.data
            for key, value in query.items():
                result = [d for d in result if d.get(key) == value]
            return iter(result)
        def sort(self, field, direction=1):
            return sorted(self.data, key=lambda x: x.get(field, ''), reverse=(direction == -1))
        def limit(self, n):
            return iter(list(self.data)[:n])
    
    messages = MockCollection()

# RAG system initialization
RAG_PERSIST_DIR = os.getenv("RAG_PERSIST_DIR", "./shenlab_db")

# Music directory for storing user-uploaded music files
MUSIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "music")

os.makedirs(MUSIC_DIR, exist_ok=True)

def _format_msg(m: dict) -> str:
    role = m.get("role", "unknown")
    text = m.get("text", "")
    
    # Get the stored timestamp from the database
    ts = m.get("time") or m.get("timestamp")
    if ts is None:
        time_str = "00:00"
    elif isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            time_str = ts.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            time_str = "00:00"
    elif isinstance(ts, datetime):
        time_str = ts.strftime("%Y-%m-%d %H:%M")
    else:
        time_str = "00:00"
    
    return f"{role}: {text} [{time_str}]"


@app.get("/")
async def root():
    return {"status": "success", "message": "HeartGlow Backend is connected via ngrok!"}

@app.websocket("/chat")
async def chat_ws(websocket: WebSocket):
    await websocket.accept()

    user_name: str | None = None
    reply_counter = 0
    time_lapsed = 0.0  # Total time elapsed across all bot responses

    # ------------------------------------------------------------------
    # 1. Send welcome message immediately
    # ------------------------------------------------------------------
    welcome = (
        #"Hi! How can I assist you today? ------Please start by telling me your name.------"
    )
    #await websocket.send_text(json.dumps({"type": "message", "text": welcome}))
    # messages.insert_one({"role": "bot", "text": welcome, "name": None})  # system message

    try:
        while True:
            raw = await websocket.receive_text()
            user_input = raw.strip()
   # -------------------------------------------------
            # Inside the WebSocket – after we know the username
            # -------------------------------------------------
            if user_name is None:
                user_name = user_input.replace(" ", "_")
                datetime_hkt = datetime.now(hkt)
                print("Date & Time in UTC : ",
                    datetime_hkt.strftime('%Y:%m:%d %H:%M:%S %Z %z'))

                # Load user's history
                user_history = list(messages.find({"name": user_name}))
                user_history.sort(key=lambda x: x.get("_id", 0))
                print("Sending conversation history (one line at a time):")
                print("-" * 60)
                await websocket.send_text(json.dumps({"type": "message", "text": "Sending conversation history (one line at a time)..."}))
                
                # --- Send each message line-by-line (preserving pairing) ---
                i = 0
                while i < len(user_history):
                    msg = user_history[i]
                    # Format message without using _format_msg()
                    msg_text = msg.get("text", "")
                    msg_role = msg.get("role", "unknown")
                    
                    # Get timestamp if available
                    msg_time = msg.get("time") or msg.get("timestamp")
                    time_str = ""
                    if msg_time:
                        if isinstance(msg_time, str):
                            try:
                                ts = datetime.fromisoformat(msg_time.replace("Z", "+00:00"))
                                time_str = ts.strftime("%Y-%m-%d %H:%M")
                            except ValueError:
                                time_str = "00:00"
                        elif isinstance(msg_time, datetime):
                            time_str = msg_time.strftime("%Y-%m-%d %H:%M")
                    
                        #formatted = f"{msg_role}: {msg_text} [{time_str}]"
                        formatted = f"{msg_role}: {msg_text}"

                    # Get timestamp for the payload
                    current_msg_time = msg.get("time") or msg.get("timestamp")
                    time_str = ""
                    if current_msg_time:
                        if isinstance(current_msg_time, str):
                            try:
                                ts = datetime.fromisoformat(current_msg_time.replace("Z", "+00:00"))
                                time_str = ts.strftime("%Y-%m-%d %H:%M")
                            except ValueError:
                                time_str = "00:00"
                        elif isinstance(current_msg_time, datetime):
                            time_str = current_msg_time.strftime("%Y-%m-%d %H:%M")
                    
                    # Send current message (user or bot)
                    payload = {
                        "type": "message",
                        "text": msg_text,
                        "time": time_str
                    }
                    await websocket.send_text(json.dumps(payload))
                    print(f"  → {formatted}")

                    # If user message → also send the next bot reply (if exists)
                    if (msg["role"] == "user"
                            and i + 1 < len(user_history)
                            and user_history[i + 1]["role"] == "bot"):
                        bot_msg = user_history[i + 1]
                        bot_text = bot_msg.get("text", "")
                        bot_role = bot_msg.get("role", "unknown")
                        bot_time = bot_msg.get("time") or bot_msg.get("timestamp")
                        bot_time_str = ""
                        if bot_time:
                            if isinstance(bot_time, str):
                                try:
                                    ts = datetime.fromisoformat(bot_time.replace("Z", "+00:00"))
                                    bot_time_str = ts.strftime("%Y-%m-%d %H:%M")
                                except ValueError:
                                    bot_time_str = "00:00"
                            elif isinstance(bot_time, datetime):
                                bot_time_str = bot_time.strftime("%Y-%m-%d %H:%M")
                        
                            bot_formatted = f"{bot_role}: {bot_text}"
                        
                        bot_payload = {
                            "type": "message",
                            "text": bot_formatted,
                            "time": bot_time_str
                        }
                        await websocket.send_text(json.dumps(bot_payload))
                        print(f"  → {bot_formatted}")
                        i += 1  # skip bot on next loop

                    i += 1

                # Send end of history message
                formatted = "End of history"
                payload = {
                        "type": "message",
                        "text": formatted,
                        "time": datetime.now(hkt).strftime("%Y-%m-%d %H:%M")
                    }
                await websocket.send_text(json.dumps(payload))

                print("-" * 60 + "\n")

            # --------------------------------------------------------------
            # 3. Normal chat (after username is known)
            # --------------------------------------------------------------
            start_time = time.perf_counter()  # Start timing when user message is received
            
            # Store user message with name
            messages.insert_one({
                "role": "user",
                "name": user_name,
                "text": user_input,
                "time": datetime.now(hkt)
            })

            # Generate bot response using RAG
            # Build conversation history for RAG
            user_history = list(messages.find({"name": user_name})) if user_name else []
            user_history.sort(key=lambda x: x.get("_id", 0))
            history = [{"role": msg["role"], "content": msg["text"]} for msg in user_history]
            if reply_counter == 0:
                bot_reply = rag_generate(query="Greet the user in a friendly way only in a few sentences", history=history, persist_dir=RAG_PERSIST_DIR)
            else:
                bot_reply = rag_generate(query=user_input, history=history, persist_dir=RAG_PERSIST_DIR)

            reply_counter += 1
            if reply_counter % 3 == 0:
                bot_reply += (
                    "\n\nIf you're finding this helpful, please share it with friends "
                    "or anyone in need! Also, explore our website's health regulation "
                    "section for tips on heartbeat monitoring, sleeping habits, and more."
                )
            
            # Store bot reply (full text) in database
            messages.insert_one({
                "role": "bot",
                "text": bot_reply,
                "name": user_name,
                "time": datetime.now(hkt)
            })
            
            end_time = time.perf_counter()  # End timing when bot response is sent
            response_time = end_time - start_time
            time_lapsed += response_time
            
            # Format the bot response with timestamp for the frontend
            current_time = datetime.now(hkt)
            time_str = current_time.strftime("%Y-%m-%d %H:%M")
            
            # Split bot reply into paragraphs and send one at a time
            paragraphs = [p.strip() for p in bot_reply.split('\n\n') if p.strip()]
            
            if not paragraphs:
                paragraphs = [bot_reply.strip()]
            
            # Send each paragraph as a separate message with time
            for i, paragraph in enumerate(paragraphs):
                payload = {
                    "type": "message",
                    "text": paragraph,
                    "time": time_str
                }
                await websocket.send_text(json.dumps(payload))
                
                # Small delay between paragraphs for natural spacing
                if i < len(paragraphs) - 1:
                    await asyncio.sleep(0.5)
            
            print(f"[{user_name}] User: {user_input}")
            print(f"[{user_name}] Bot: {bot_reply}")
            print(f"Response time: {response_time:.2f}s")
            
    except WebSocketDisconnect:
        print(f"User disconnected: {user_name or 'unknown'}")
        # Clean up user-specific music files
        if user_name:
            import glob
            user_music_pattern = os.path.join(MUSIC_DIR, f"{user_name}_*")
            user_music_files = glob.glob(user_music_pattern)
            for f in user_music_files:
                try:
                    os.remove(f)
                    print(f"Cleaned up user music file: {f}")
                except Exception as e:
                    print(f"Error cleaning up {f}: {e}")
        print(f"Total time elapsed: {time_lapsed:.2f}s\n")

# --- TTS endpoint ---
class TTSRequest(BaseModel):
    text: str



@app.get("/check-user/{username}")
async def check_user(username: str):
    """Check if a user exists in the database by looking for their messages."""
    user_name = username.replace(" ", "_")
    existing_messages = list(messages.find({"name": user_name}).limit(1))
    return {"exists": len(existing_messages) > 0}


# --- Music endpoints ---
class MusicRequest(BaseModel):
    filename: str

@app.post("/music/upload")
async def upload_music(file: UploadFile = File(...)):
    """Upload a music file to the server"""
    # Validate file type
    content_type = file.content_type
    if not content_type or not content_type.startswith("audio"):
        return {"error": "Only audio files are allowed", "valid": False}
    
    # Generate unique filename
    filename = f"{uuid.uuid4()}_{file.filename}"
    filepath = os.path.join(MUSIC_DIR, filename)
    
    # Save file
    with open(filepath, "wb") as f:
        f.write(await file.read())
    
    return {
        "filename": filename,
        "original_name": file.filename,
        "valid": True,
        "message": "Music uploaded successfully"
    }

@app.get("/music/list")
async def list_music():
    """List all available music files on the server"""
    try:
        files = []
        for f in os.listdir(MUSIC_DIR):
            # Skip system files like .DS_Store
            if f.startswith('.'):
                continue
            filepath = os.path.join(MUSIC_DIR, f)
            if os.path.isfile(filepath):
                files.append({
                    "filename": f,
                    "name": f.split("_", 1)[1] if "_" in f else f,
                    "size": os.path.getsize(filepath),
                    "path": f"/music/{f}",
                    "modified_time": os.path.getmtime(filepath)
                })
        # Sort by modified_time (upload time) - newest first
        files.sort(key=lambda x: x["modified_time"], reverse=True)
        return {"music_files": files, "valid": True}
    except Exception as e:
        return {"error": str(e), "music_files": [], "valid": False}

@app.get("/music/{filename}")
async def get_music(filename: str):
    """Stream a music file from the server"""
    filepath = os.path.join(MUSIC_DIR, filename)
    if not os.path.exists(filepath):
        return {"error": "File not found", "valid": False}
    
    return FileResponse(filepath, media_type="audio/mpeg", filename=filename)

@app.post("/tts/")
@app.post("/tts/")
async def text_to_speech(req: TTSRequest):
    '''global switch
    print("Received:", req.text)   # confirm input
    if req.text == "audio_off":
        switch = False
        return
    if req.text == "audio_on":
        switch = True
        return
    if switch == True:'''
    filename = f"{uuid.uuid4()}.mp3"
    filepath = os.path.join("tmp", filename)
    os.makedirs("tmp", exist_ok=True)

    # Convert text to speech
    tts = gTTS(text=req.text, lang="en")
    tts.save(filepath)

    # Read file into memory
    #with open(filepath, "rb") as f:
    #    audio_bytes = f.read()

    # Send audio file back to React
    return FileResponse(filepath, media_type="audio/mpeg", filename=filename)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
         