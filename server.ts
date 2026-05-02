import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import multer from "multer";
import { fileURLToPath } from "url";
import fs from "fs";
import { createRequire } from "module";
const require = createRequire(import.meta.url);
const pdfParse = require("pdf-parse");
import mammoth from "mammoth";
import PDFDocument from 'pdfkit';
import archiver from "archiver";

interface Expert {
  id: string;
  name: string;
  domain: string;
  instructions: string;
}

const EXPERTS: Record<string, Expert> = {
  "0": { id: "0", name: "Coder", domain: "Software & Technology", instructions: "You are a senior software engineer. Provide precise, efficient, and well-commented code. Focus on best practices and performance." },
  "1": { id: "1", name: "Medic", domain: "Medicine & Health", instructions: "You are an medical expert. Provide evidence-based medical information. Always include a disclaimer that you are an AI and the user should consult a professional." },
  "2": { id: "2", name: "Architect", domain: "Engineering & Infrastructure", instructions: "You are a civil engineer and architect. Focus on structural integrity, design materials, and engineering principles." },
  "3": { id: "3", name: "Dharma-Expert", domain: "Buddhist Studies & Spirituality", instructions: "You are an expert in Buddhist philosophy (Dhamma) and theology. Provide peaceful, insightful, and accurate interpretations of teachings." },
  "4": { id: "4", name: "Historian", domain: "History & Archaeology", instructions: "You are a historian. Provide detailed chronological context, source analysis, and historical significance of events." },
  "5": { id: "5", name: "Scientist", domain: "Pure Sciences", instructions: "You are a theoretical scientist. Focus on the laws of physics, chemical reactions, and the scientific method." },
  "6": { id: "6", name: "Cosmologist", domain: "Astronomy & Space", instructions: "You are a space expert. Describe the universe, galaxies, and astrophysics in an inspiring and scientific manner." },
  "7": { id: "7", name: "Strategist", domain: "Politics & Governance", instructions: "You are a political analyst. Provide neutral, objective analysis of governance, policy, and international relations." },
  "8": { id: "8", name: "Economist", domain: "Finance & Economy", instructions: "You are a financial economist. Analyze market trends, fiscal policy, and micro/macroeconomic indicators." },
  "9": { id: "9", name: "Psychologist", domain: "Psychology & Behavioral Science", instructions: "You are a psychologist. Focus on mental processes, cognitive empathy, and behavioral patterns with a supportive tone." },
  "10": { id: "10", name: "Philosopher", domain: "Philosophy & Ethics", instructions: "You are a philosopher. Explore ethical dilemmas, existential questions, and logical arguments." },
  "11": { id: "11", name: "Art-Critic", domain: "Arts & Culture", instructions: "You are an artist and critic. Focus on aesthetics, musical theory, and cultural impact of creative works." },
  "default": { id: "default", name: "Generalist", domain: "General Knowledge", instructions: "You are a helpful general-purpose AI expert. Provide concise and accurate answers." }
};

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DATA_DIR = path.join(__dirname, "data");
const UPLOAD_DIR = path.join(DATA_DIR, "uploads");
const HISTORY_FILE = path.join(DATA_DIR, "chat_history.json");

// Ensure directories exist
if (!fs.existsSync(DATA_DIR)) {
  fs.mkdirSync(DATA_DIR, { recursive: true });
}
if (!fs.existsSync(UPLOAD_DIR)) {
  fs.mkdirSync(UPLOAD_DIR, { recursive: true });
}

import Database from 'better-sqlite3';
const db = new Database(path.join(DATA_DIR, 'chat.db'));
db.exec(`CREATE TABLE IF NOT EXISTS messages (
  id TEXT PRIMARY KEY,
  sender TEXT,
  text TEXT,
  frame JSON,
  created_at INTEGER DEFAULT (unixepoch())
)`);

const insertMsg = db.prepare('INSERT INTO messages (id, sender, text, frame) VALUES (?, ?, ?, ?)');
const getHistoryStmt = db.prepare('SELECT * FROM messages ORDER BY created_at LIMIT 100');

const getHistory = () => {
    return getHistoryStmt.all().map((row: any) => ({
        ...row,
        frame: row.frame ? JSON.parse(row.frame) : undefined
    }));
};

const saveHistory = (history: any[]) => {
    // History is saved one by one now, no need to overwrite all
};

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, UPLOAD_DIR);
  },
  filename: (req, file, cb) => {
    const sanitized = path.basename(file.originalname).replace(/[^a-zA-Z0-9._-]/g, '_');
    cb(null, Date.now() + "-" + sanitized);
  },
});
const upload = multer({ 
  storage,
  limits: {
    fileSize: 50 * 1024 * 1024, // 50MB
  },
  fileFilter: (req, file, cb) => {
    const allowed = [
      'application/pdf', 
      'text/plain',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'image/jpeg', 
      'image/png', 
      'image/webp'
    ];
    if (allowed.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Unsupported file type'));
    }
  }
});

async function startServer() {
  const app = express();
  const PORT = 3000;

  process.on('uncaughtException', (err) => {
    console.error('Uncaught Exception:', err);
  });
  
  app.use(express.json({ limit: '10mb' }));

  // Auto-spawn Python ML API
  const { spawn } = await import("child_process");
  console.log("🚀 Initializing Python Neural Link (ml_api.py)...");
  const pyProcess = spawn("python3", ["ml_api.py"], {
    stdio: 'inherit',
    detached: false
  });
  pyProcess.on('error', (err) => {
    console.error("❌ Failed to start Python Neural Link:", err);
  });
  
  app.post("/api/synthesize", async (req, res) => {
    try {
      const pyResp = await fetch("http://localhost:8000/api/synthesize", {
        method: "POST"
      });
      const data = await pyResp.json();
      res.json(data);
    } catch (e) {
      res.status(500).json({ error: "Failed to connect to synthesis engine" });
    }
  });

  app.get("/api/grammar-rules", async (req, res) => {
    try {
      const pyResp = await fetch("http://localhost:8000/api/grammar-rules");
      if (pyResp.ok) {
        const data = await pyResp.json();
        return res.json(data);
      }
      res.json({ rules: [] });
    } catch (e) {
      res.json({ rules: ["Local Rule Engine: Offline Fallback Memory"] });
    }
  });

  // Health Check Endpoint
  app.get("/api/health", async (req, res) => {
    const colabUrl = req.query.colabUrl as string;
    const mlApiUrl = colabUrl || process.env.VITE_ML_API_URL || process.env.ML_API_URL;

    if (mlApiUrl && colabUrl !== 'mock') {
        try {
            const controller = new AbortController();
            const id = setTimeout(() => controller.abort(), 3000);
            const resp = await fetch(`${mlApiUrl.replace(/\/$/, '')}/api/health`, {
                signal: controller.signal,
                headers: { 
                    'Bypass-Tunnel-Reminder': 'true',
                    'ngrok-skip-browser-warning': 'true'
                }
            });
            clearTimeout(id);
            if (resp.ok) {
                return res.json({ status: "ok", type: "colab" });
            }
        } catch (e) {
            return res.status(503).json({ status: "offline" });
        }
    }
    res.json({ status: "ok", version: "9B-PRO", type: "local" });
  });

  // File Upload Endpoint
  app.post("/api/upload", upload.single("file"), async (req, res) => {
    if (!req.file) {
      return res.status(400).send("No file uploaded.");
    }
    const colabUrl = req.body.colabUrl as string;

    let text = "";
    const filePath = req.file.path;
    const mimetype = req.file.mimetype;

    try {
      if (mimetype === 'application/pdf') {
        const dataBuffer = fs.readFileSync(filePath);
        const data = await (pdfParse as any)(dataBuffer);
        text = data.text;
      } else if (mimetype === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document') {
        const result = await mammoth.extractRawText({ path: filePath });
        text = result.value;
      } else if (mimetype === 'text/plain') {
        text = fs.readFileSync(filePath, 'utf-8');
      } else if (mimetype.startsWith('image/')) {
          text = "[Image uploaded. Vision processing requires the ML Colab Backend with a vision-language model, which is currently in development.]";
      } else {
        return res.status(400).send("Unsupported file type.");
      }
      
      console.log(`[Memory] Persisting ${text.length} characters to LongTermMemory.`);
      
      // Ingest to ML Backend if available
      const mlApiUrl = (colabUrl && colabUrl !== 'mock') ? colabUrl : process.env.VITE_ML_API_URL || process.env.ML_API_URL;
      if (mlApiUrl && colabUrl !== 'mock') {
          try {
              await fetch(`${mlApiUrl.replace(/\/$/, '')}/api/ingest`, {
                  method: 'POST',
                  headers: { 
                      'Content-Type': 'application/json',
                      'Bypass-Tunnel-Reminder': 'true',
                      'User-Agent': 'SVOMPTR-App/1.0'
                  },
                  body: JSON.stringify({ text, filename: req.file.filename })
              });
          } catch (e) {
              console.error("Failed to ingest to ML backend:", e);
          }
      }

      res.json({ message: "File uploaded and processed", filename: req.file.filename, content: text });
    } catch (err) {
      console.error("Error processing file:", err);
      res.status(500).send("Error processing file.");
    }
  });

  // Package Project Endpoint
  app.get("/api/package-project", (req, res) => {
    const archivePath = path.join(UPLOAD_DIR, 'project.zip');
    const output = fs.createWriteStream(archivePath);
    const archive = archiver('zip', { zlib: { level: 9 } });

    output.on('close', () => {
      res.download(archivePath, 'project.zip');
    });

    archive.pipe(output);
    // Zip everything except sensitive files and large directories
    archive.glob('**/*', { 
        ignore: [
            'node_modules/**', 
            'data/**', 
            '.git/**', 
            '.gitignore', 
            'package-lock.json',
            '.env',
            '.env.*',
            '**/*.pyc',
            '__pycache__/**'
        ] 
    });
    archive.finalize();
  });

  // Colab Auto-Train Download Endpoint
  app.get("/api/download-colab", (req, res) => {
    const colabPath = path.join(__dirname, "notebooks", "SVOMPTR_9B_AutoTrain_Unsloth.ipynb");
    if (fs.existsSync(colabPath)) {
        res.download(colabPath, "SVOMPTR_9B_AutoTrain_Unsloth.ipynb");
    } else {
        res.status(404).send("Colab Auto-Train notebook not found.");
    }
  });

  // Dataset Download Endpoint
  app.get("/api/download-dataset", (req, res) => {
    const datasetPath = path.join(__dirname, "notebooks", "data", "synthetic_1M_high_quality.jsonl");
    if (fs.existsSync(datasetPath)) {
        res.download(datasetPath, "synthetic_1M_high_quality.jsonl");
    } else {
        // Try looking in default DATA_DIR too
        const alternativePath = path.join(DATA_DIR, "synthetic_1M_high_quality.jsonl");
        if (fs.existsSync(alternativePath)) {
            res.download(alternativePath, "synthetic_1M_high_quality.jsonl");
        } else {
            res.status(404).send("Dataset file not found. Ensure the notebook has finished generating it.");
        }
    }
  });

  // Colab Inference Download Endpoint
  app.get("/api/download-inference", (req, res) => {
    const colabPath = path.join(__dirname, "notebooks", "SVOMPTR_9B_Colab_Inference_Only.ipynb");
    if (fs.existsSync(colabPath)) {
        res.download(colabPath, "SVOMPTR_9B_Colab_Inference_Only.ipynb");
    } else {
        res.status(404).send("Inference notebook not found.");
    }
  });

  // Colab Distillation Download Endpoint
  app.get("/api/download-distillation", (req, res) => {
    const colabPath = path.join(__dirname, "notebooks", "SVOMPTR_9B_Colab_Distillation.ipynb");
    if (fs.existsSync(colabPath)) {
        res.download(colabPath, "SVOMPTR_9B_Colab_Distillation.ipynb");
    } else {
        res.status(404).send("Distillation notebook not found.");
    }
  });

  // File Generation Endpoint
  app.post("/api/generate-file", (req, res) => {
    const { format, text } = req.body;
    console.log(`[Generation] Generating ${format} file...`);
    
    if (format === 'pdf') {
        const doc = new PDFDocument();
        const outputFilename = `generated_${Date.now()}.pdf`;
        const outputPath = path.join(UPLOAD_DIR, outputFilename);
        
        doc.pipe(fs.createWriteStream(outputPath));
        doc.text(text);
        doc.end();
        
        doc.on('finish', () => {
             res.download(outputPath, outputFilename);
        });
    } else {
        res.json({ message: `${format} file generation not implemented.` });
    }
  });

  // Subjects Management
  app.post("/api/subjects", (req, res) => {
    const { subjects } = req.body;
    if (!Array.isArray(subjects)) return res.status(400).send("Invalid subject list.");
    console.log(`[Subjects] Updated to: ${subjects.join(', ')}`);
    res.json({ message: "Subjects updated successfully." });
  });

// Start Learning Endpoint
  app.post("/api/start-learning", async (req, res) => {
    console.log(`[AutoLearner] Initializing Neural Core Synchronization...`);
    const colabUrl = req.body?.colabUrl as string;
    
    try {
      const mlApiUrl = (colabUrl && colabUrl !== 'mock') ? colabUrl : process.env.VITE_ML_API_URL || process.env.ML_API_URL;
      
      if (mlApiUrl && colabUrl !== 'mock') {
          const response = await fetch(`${mlApiUrl.replace(/\/$/, '')}/api/start-learning`, {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'Bypass-Tunnel-Reminder': 'true',
                'User-Agent': 'SVOMPTR-App/1.0'
            },
          });
          const responseData = await response.json();
          res.json({ message: responseData.message || "Neural training initiated on Colab GPU." });
      } else {
          // Fallback to local python (will likely fail in AI Studio, but exists for local envs)
          const { exec } = require("child_process");
          const scriptPath = path.join(__dirname, "svomptr_moe", "train_chat_expert.py");
          
          exec(`python3 ${scriptPath}`, (error: any, stdout: string, stderr: string) => {
            if (error) {
              console.error(`[AutoLearner] Script error: ${error.message}`);
              return res.status(500).json({ message: "Neural initialization failed. Connect Colab ML Backend for full capabilities." });
            }
            console.log(`[AutoLearner] Output: ${stdout}`);
            res.json({ message: "Local neural iteration completed successfully." });
          });
      }

    } catch (error) {
      console.error("Learning Error:", error);
      res.status(500).json({ message: "Connection to Neural API failed." });
    }
  });

  // Chat History Endpoint
  app.get("/api/history", (req, res) => {
    res.json(getHistory());
  });

  app.delete("/api/history", (req, res) => {
    try {
        console.log("[Database] CLEARING all messages in SQLite...");
        const result = db.prepare('DELETE FROM messages').run();
        try { db.prepare('VACUUM').run(); } catch (e) {} // Clean up space
        console.log(`[Database] Success. Deleted rows: ${result.changes}`);
        res.json({ message: "History cleared successfully", deletedCount: result.changes });
    } catch (e) {
        console.error("[Database] Failed to clear history:", e);
        res.status(500).json({ error: "Failed to clear history" });
    }
  });

app.post("/api/chat", async (req, res) => {
    const { message, colabUrl } = req.body;
    
    // Safety check: ignore empty or extremely short messages if they look programmatic
    if (!message || message.trim().length === 0) {
        console.warn("[Chat] Received empty message, ignoring.");
        return res.status(400).json({ error: "Empty message" });
    }

    console.log(`[Neural Link] [${new Date().toISOString()}] Incoming Message: "${message.substring(0, 100)}..." (Source: ${colabUrl === 'mock' ? 'Mock' : 'Colab'})`);
    const userMsgId = Date.now().toString();
    
    try {
        insertMsg.run(userMsgId, 'user', message, null);
    } catch (e) {
        console.error("DB Insert Error (user):", e);
    }

    try {
      const mlApiUrl = (colabUrl && colabUrl !== 'mock') ? colabUrl : process.env.VITE_ML_API_URL || process.env.ML_API_URL;
      
      let responseData;

      if (mlApiUrl && colabUrl !== 'mock') {
          const response = await fetch(`${mlApiUrl.replace(/\/$/, '')}/api/chat`, {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'Bypass-Tunnel-Reminder': 'true',
                'ngrok-skip-browser-warning': 'true',
                'User-Agent': 'SVOMPTR-App/1.0'
            },
            body: JSON.stringify({ message }),
          });
          
          if (!response.ok) {
              throw new Error(`Colab API failed with status ${response.status}`);
          }
          responseData = await response.json();
      } else {
          // Local Neural Core Backend (ml_api.py)
          try {
              const localResponse = await fetch(`http://localhost:8000/api/chat`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ message }),
              });
              if (localResponse.ok) {
                  responseData = await localResponse.json();
              } else {
                  throw new Error("Local engine not responding properly");
              }
          } catch (e) {
              console.warn("[Local Backend] Falling back to rule-based fallback in Node as Python core is offline.");
              responseData = {
                  response: `[Local Rule Engine] Analyzing: ${message}`,
                  frame: { S: "Local", V: "Rule", O: "Engine", R: "Offline Fallback" }
              };
          }
      }

      const botMsgId = (Date.now() + 1).toString();
      insertMsg.run(botMsgId, 'bot', responseData.response, responseData.frame ? JSON.stringify(responseData.frame) : null);
      
      res.json(responseData);

    } catch (error) {
      console.error("Chat Error:", error);
      res.status(500).json({ 
        response: "Neural link severed. Failed to connect to the Colab ML Backend.",
        frame: { S: "System", V: "Error", O: "Connection", R: "Failed" }
      });
    }
  });

  if (process.env.NODE_ENV !== "production") {
    createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    }).then(vite => {
      app.use(vite.middlewares);
      console.log("Vite middleware attached.");
    });
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer().catch(err => {
    console.error("Fatal error during server startup:", err);
});

