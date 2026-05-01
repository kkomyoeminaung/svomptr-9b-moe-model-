# svomptr/memory/long_term.py
import os
import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class LongTermMemory:
    """Persistent storage using FAISS and SQLite"""
    def __init__(self, db_path="data/database/memory.db", index_path="data/database/faiss.index"):
        self.db_path = db_path
        self.index_path = index_path
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()
        
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = self._init_index()
        print(f"✅ RAG/SQLite Memory Initialized (Total: {self.index.ntotal} vectors)")

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def _init_index(self):
        if os.path.exists(self.index_path):
            try:
                return faiss.read_index(self.index_path)
            except Exception as e:
                print(f"⚠️ Failed to load FAISS index: {e}. Creating new one.")
        base_index = faiss.IndexFlatL2(384)
        return faiss.IndexIDMap(base_index)

    def add_memory(self, text):
        embedding = self.encoder.encode([text]).astype('float32')
        
        # Store in SQLite first to get row ID
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO memory (text) VALUES (?)", (text,))
        db_id = cursor.lastrowid
        self.conn.commit()
        
        # Add to FAISS with db_id
        self.index.add_with_ids(embedding, np.array([db_id], dtype=np.int64))
        
        # Persist index
        faiss.write_index(self.index, self.index_path)

    def get_relevant_context(self, query: str, top_k: int = 3) -> str:
        """FAISS vector search to retrieve relevant memories"""
        if self.index.ntotal == 0:
            return ""
            
        embedding = self.encoder.encode([query]).astype('float32')
        distances, indices = self.index.search(embedding, top_k)
        
        results = []
        cursor = self.conn.cursor()
        for idx in indices[0]:
            if idx == -1: continue
            # Retrieve by exact db_id
            cursor.execute("SELECT text FROM memory WHERE id = ?", (int(idx),))
            row = cursor.fetchone()
            if row:
                results.append(row[0])
        
        return "\n".join(results)

    def get_all_memories(self):
        cursor = self.conn.execute("SELECT text FROM memory")
        return [row[0] for row in cursor.fetchall()]
