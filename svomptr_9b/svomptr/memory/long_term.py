# svomptr/memory/long_term.py
import os
import sqlite3
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

import threading

class LongTermMemory:
    """Persistent storage using FAISS and SQLite"""
    def __init__(self, db_path="data/database/memory.db", index_path="data/database/faiss.index"):
        self.db_path = db_path
        self.index_path = index_path
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._init_db()
        
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = self._init_index()
        self._write_counter = 0
        print(f"✅ RAG/SQLite Memory Initialized (Total: {self.index.ntotal} vectors)")

    def _init_db(self):
        with self._lock:
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

    def add_grammar_rule(self, english: str, correct_burmese: str, slots: dict):
        """Specifically stores a grammatical correction as a priority rule."""
        rule_text = f"RULE_ENG: {english} | RULE_MYA: {correct_burmese} | SLOTS: {json.dumps(slots)}"
        self.store_memory(rule_text, metadata="grammar_correction")

    def get_relevant_grammar(self, query: str) -> list:
        """Retrieves specific grammar rules that might apply to the current query."""
        if self.index.ntotal == 0:
            return []
            
        embedding = self.encoder.encode([query]).astype('float32')
        distances, indices = self.index.search(embedding, 5)
        
        rules = []
        with self._lock:
            cursor = self.conn.cursor()
            for idx in indices[0]:
                if idx == -1: continue
                cursor.execute("SELECT text FROM memory WHERE id = ?", (int(idx),))
                row = cursor.fetchone()
                if row and "RULE_ENG" in row[0]:
                    rules.append(row[0])
        
        return rules

    def store_memory(self, text, metadata=""):
        """Stores text with optional metadata for RAG."""
        combined = f"{text}\nMetadata: {metadata}" if metadata else text
        embedding = self.encoder.encode([combined]).astype('float32')
        
        # Store in SQLite first to get row ID
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("INSERT INTO memory (text) VALUES (?)", (combined,))
            db_id = cursor.lastrowid
            self.conn.commit()
        
        # Add to FAISS with db_id
        self.index.add_with_ids(embedding, np.array([db_id], dtype=np.int64))
        
        # Persist index every 1000 writes to save disk I/O
        self._write_counter += 1
        if self._write_counter % 1000 == 0:
            faiss.write_index(self.index, self.index_path)

    def close(self):
        """Safe connection termination"""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
            self.conn = None

    def __del__(self):
        self.close()

    def flush(self):
        """Call at end of pipeline to ensure final save"""
        try:
            faiss.write_index(self.index, self.index_path)
            print("✅ FAISS index flushed to disk.")
        except Exception as e:
            print(f"⚠️ Error flushing FAISS index: {e}")

    def get_relevant_context(self, query: str, top_k: int = 3) -> str:
        """FAISS vector search to retrieve relevant memories"""
        if self.index.ntotal == 0:
            return ""
            
        embedding = self.encoder.encode([query]).astype('float32')
        distances, indices = self.index.search(embedding, top_k)
        
        results = []
        with self._lock:
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
        with self._lock:
            cursor = self.conn.execute("SELECT text FROM memory")
            return [row[0] for row in cursor.fetchall()]

    def synthesize_rules(self, batch_size=10):
        """
        Unsupervised Synthesis: Automatically attempts to find patterns in memories.
        """
        memories = self.get_all_memories()
        if len(memories) < 5: return []
        
        # In a real setup, we would run a clustering or NLP summarization here.
        # For this prototype, we'll flag any text that has consistent structural components.
        synthesized = []
        for m in memories:
            if "RULE_" not in m and "Translation:" in m:
                # Potential candidate for rule extraction
                synthesized.append(m)
        return synthesized
