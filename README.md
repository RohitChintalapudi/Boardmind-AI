# BoardMind AI — From Whiteboard Scribbles to Smart Notes 🎓✨

**BoardMind AI** is an educational computer vision and artificial intelligence web application designed to convert classroom whiteboard photographs into clean, readable, organized study notes.

---

## 🚀 Live Local Application Links

The application is running locally:

| Service | Local URL | Description |
| :--- | :--- | :--- |
| **Frontend Web App** | [http://127.0.0.1:5174](http://127.0.0.1:5174) | Interactive React + TypeScript + Tailwind UI |
| **Backend API Docs** | [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs) | Interactive Swagger UI API playground |
| **API ReDoc** | [http://127.0.0.1:8001/redoc](http://127.0.0.1:8001/redoc) | Clean schema & endpoint specification |
| **System Health** | [http://127.0.0.1:8001/api/health](http://127.0.0.1:8001/api/health) | Real-time ML model & hardware diagnostics |

---

## 🏛️ System Architecture & Workflow

```
[ Classroom Whiteboard Photo ]
               │
               ▼
   [ FastAPI REST Backend ]
               │
       ┌───────┴───────────────────────────┐
       ▼                                   ▼
 [ 1. YOLOv8n Detector ]       [ 2. Preprocessor ]
 • 99.4% Precision             • High-contrast CLAHE
 • 100% Recall                 • Bilateral stroke filter
 • Perspective isolation       • Background wall rejection
       │                                   │
       └───────┬───────────────────────────┘
               ▼
   [ 3. Modular HTR Engine ]
   • Horizontal ink projection profiling
   • Stroke density calculation
   • Transparent status reporting
               │
               ▼
   [ 4. Smart Note Structuring Engine ]
   • Extracts Main Title & Headings
   • Formats Mathematical Formulas & Key Terms
   • Numbered Procedural Steps
   • Bulleted Core Concepts
   • Clean GitHub-Flavored Markdown
               │
               ▼
   [ 5. React Web Application ]
   • Side-by-side interactive inspection
   • Toggle YOLO Bounding Box vs Cropped Board
   • Live Editable Scribble Canvas
   • 1-Click Copy & Download (.MD / .TXT)
```

---

## 📁 Project Structure

```
BoardMindAI/
├── backend/
│   ├── app/
│   │   ├── api/                  # REST API routes
│   │   │   ├── health.py         # GET /api/health
│   │   │   ├── detection.py      # POST /api/detect
│   │   │   ├── recognition.py    # POST /api/extract-text
│   │   │   ├── notes.py          # POST /api/generate-notes
│   │   │   └── process.py        # POST /api/process (Full Pipeline)
│   │   ├── detection/
│   │   │   └── whiteboard_detector.py # YOLOv8n singleton service on CPU
│   │   ├── recognition/
│   │   │   └── handwriting_recognizer.py # Modular HTR interface & line segmenter
│   │   ├── processing/
│   │   │   ├── image_preprocessor.py  # CLAHE & contrast cropping
│   │   │   └── text_formatter.py      # Rule-based Smart Note structuring
│   │   ├── schemas/
│   │   │   └── models.py         # Pydantic request & response schemas
│   │   └── main.py               # FastAPI entrypoint, CORS, lifespan loader
│   ├── requirements.txt
│   └── test_backend.py           # Automated test suite
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.tsx             # Navbar & system status pill
│   │   │   ├── SystemInfoModal.tsx    # ML hardware diagnostics modal
│   │   │   ├── ImageUploader.tsx      # Drag & drop upload + sample loader
│   │   │   ├── ProcessingTimeline.tsx # 5-stage progress indicator
│   │   │   └── ResultsViewer.tsx      # Side-by-side vision & note canvas
│   │   ├── services/
│   │   │   └── api.ts            # Typed REST API client
│   │   ├── types/
│   │   │   └── index.ts          # TypeScript interfaces
│   │   ├── App.tsx               # Root application coordinator
│   │   ├── main.tsx              # React DOM bootstrap
│   │   └── index.css             # Tailwind v4 styles & typography
│   ├── package.json
│   └── vite.config.ts
│
├── models/
│   └── whiteboard_detector_best.pt # Champion YOLOv8n model checkpoint
│
├── notebooks/                    # Research & training notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_yolo_annotation_check.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_evaluation.ipynb
│   ├── 05_whiteboard_model_experiments.ipynb
│   ├── 06_whiteboard_augmentation_experiments.ipynb
│   ├── 07_whiteboard_error_analysis.ipynb
│   └── 08_whiteboard_final_evaluation.ipynb
│
├── datasets/
│   ├── whiteboard/               # Whiteboard bounding box dataset
│   └── handwriting/              # IAM line handwriting dataset (Parquet)
│
└── runs/                         # Model training outputs & checkpoints
```

---

## 🛠️ How to Start the App Manually

### 1. Backend (FastAPI + Uvicorn)

```powershell
# Activate Python virtual environment
.\.venv\Scripts\activate

# Start Uvicorn server
uvicorn backend.app.main:app --host 127.0.0.1 --port 8001 --reload
```
API Documentation will be accessible at: `http://127.0.0.1:8001/docs`

### 2. Frontend (React + Vite)

```powershell
# Navigate to frontend folder
cd frontend

# Start Vite dev server
npm run dev
```
Frontend Web App will be accessible at: `http://127.0.0.1:5173`

---

## 🧠 Machine Learning Engine Details

### Whiteboard Detection (YOLOv8n)
- **Model Checkpoint:** `models/whiteboard_detector_best.pt`
- **Architecture:** Ultralytics YOLOv8n (3.14M parameters, 8.7 GFLOPs)
- **Execution Target:** CPU inference (`device='cpu'`)
- **Latency:** ~29 ms per frame (~34 FPS on Intel Core i9-13900H)
- **Metrics on Validation Set:**
  - Precision: **99.40%**
  - Recall: **100.0%**
  - F1 Score: **0.9970**
  - mAP@50: **99.49%**
  - mAP@50-95: **91.61%**

### Handwriting Text Recognition (HTR)
- **Status:** Modular baseline interface operational.
- **Dataset Audited:** IAM handwriting lines in `datasets/handwriting/` (6,482 train, 976 val, 2,915 test line crops).
- **Extensibility:** Implements `BaseHandwritingRecognizer` in `src/recognition/handwriting_recognizer.py`. To plug in a deep learning HTR checkpoint (such as HuggingFace `microsoft/trocr-base-handwritten` or a custom CRNN PyTorch model), simply drop the model weights into `models/handwriting_recognizer.pt`.

