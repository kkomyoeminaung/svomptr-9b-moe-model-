# svomptr/ingestion/auto_learner.py
from duckduckgo_search import DDGS
from .loaders import FileLoader
from ..memory.long_term import LongTermMemory
import os
import json

class AutoLearner:
    def __init__(self, drive_base_path=None, memory=None):
        self.memory = memory if memory else LongTermMemory()
        self.ddgs = DDGS()
        
        # Bug #25 Fix: Cross-platform brain path
        if drive_base_path is None:
            drive_base_path = os.environ.get('SVOMPTR_BRAIN_PATH', os.path.join(os.getcwd(), 'svomptr_brain'))
            
        self.drive_path = drive_base_path
        self.progress_file = os.path.join(self.drive_path, 'learning_progress.json')
        self.subjects_file = os.path.join(self.drive_path, 'subjects_config.json')
        
        os.makedirs(self.drive_path, exist_ok=True)
        self.subjects = self._load_subjects()

    def _load_subjects(self):
        if os.path.exists(self.subjects_file):
            with open(self.subjects_file, 'r') as f:
                return json.load(f)
        # Default subjects
        return ["Mathematics", "Computer Science", "Artificial Intelligence"]

    def save_subjects(self, subjects):
        with open(self.subjects_file, 'w') as f:
            json.dump(subjects, f)
        self.subjects = subjects

    def _load_progress(self):
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {"completed": []}

    def _save_progress(self, subject):
        prog = self._load_progress()
        prog["completed"].append(subject)
        with open(self.progress_file, 'w') as f:
            json.dump(prog, f)

    def learn_all(self):
        """Sequential learning with resume support."""
        print(f"🚀 Starting Auto-Learning sequence for {len(self.subjects)} subjects.")
        for subject in self.subjects:
            self.learn_from_subject(subject)
        print("🎉 All subjects processed.")

    def add_subject(self, subject: str):
        """Dynamically add a new subject to the queue."""
        if subject not in self.subjects:
            self.subjects.append(subject)
            self.save_subjects(self.subjects)
            print(f"➕ Added subject: {subject}")

    def learn_from_subject(self, subject):
        prog = self._load_progress()
        if subject in prog["completed"]:
            print(f"⏩ Skipping {subject} (Already learned)")
            return
            
        print(f"🕵️ Learning about: {subject}")
        results = list(self.ddgs.text(subject, max_results=5))
        for res in results:
            url = res['href']
            try:
                text = FileLoader.load_html(url)
                self.memory.add_memory(text[:5000])
                print(f"✅ Learned from: {url}")
            except Exception as e:
                print(f"❌ Failed {url}: {e}")
        self._save_progress(subject)

    def ingest_uploads(self):
        upload_dir = os.path.join(os.getcwd(), "data", "uploads")
        if not os.path.exists(upload_dir):
            return
            
        print("📥 Checking for new uploads...")
        for filename in os.listdir(upload_dir):
            path = os.path.join(upload_dir, filename)
            try:
                if filename.endswith(".txt"):
                    text = FileLoader.load_txt(path)
                elif filename.endswith(".pdf"):
                    text = FileLoader.load_pdf(path)
                elif filename.endswith(".docx"):
                    text = FileLoader.load_docx(path)
                elif filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    text = FileLoader.load_image(path)
                else:
                    continue
                
                self.memory.add_memory(text)
                print(f"✅ Ingested: {filename}")
                # Move to processed or delete
                os.remove(path)
            except Exception as e:
                print(f"❌ Failed to ingest {filename}: {e}")
