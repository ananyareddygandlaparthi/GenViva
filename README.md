# 🎓 GenAI Viva Coach

An AI-powered, multimodal capstone presentation coach that evaluates your delivery using **facial emotion analysis**, **speech metrics**, and **LLM-driven viva questioning** — with a full iterative feedback loop.

---

## 📌 Overview

GenAI Viva Coach is a Streamlit web application designed to help students prepare for capstone project demos and viva evaluations. It processes your **presentation slides** (PPTX/PDF) and **video recording**, computes a holistic confidence score, generates personalized viva questions using Gemini AI, and measures your improvement after a second attempt.

---

## ✨ Features

- **Multimodal Analysis** — Combines facial emotion recognition (DeepFace) + audio transcription (Whisper) + speech metrics (librosa)
- **RAG-based Viva Question Generation** — Uses FAISS vector search + Sentence Transformers to retrieve relevant slide context before querying Gemini
- **Intelligent Answer Evaluation** — Gemini 2.5 Flash grades your viva answers on Relevance, Depth, and Clarity
- **Iterative Feedback Loop** — Upload a second improved video and measure score improvement across two rounds
- **Weak Area Detection** — Automatically identifies nervousness, fluency issues, content gaps, and retrieval mismatches
- **Structured JSON Output** — Uses Pydantic schemas with Gemini's structured output for reliable parsing

---

## 🔄 Application Flow

```
Phase 1 → Upload slides + video → Initial Analysis
Phase 2 → Answer AI-generated viva questions
Phase 3 → View feedback → Upload improved video (optional)
Phase 4 → Answer new targeted viva questions
Phase 5 → Final score comparison & comprehensive report
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **UI Framework** | Streamlit |
| **Speech Transcription** | OpenAI Whisper (`base` model) |
| **Facial Analysis** | DeepFace |
| **Audio Analysis** | librosa |
| **Video Processing** | OpenCV, MoviePy, FFmpeg (via imageio-ffmpeg) |
| **Embeddings / RAG** | Sentence Transformers (`all-MiniLM-L6-v2`) + FAISS |
| **LLM (Questions & Evaluation)** | Google Gemini 2.5 Flash (`google-genai`) |
| **Slide Parsing** | python-pptx, PyMuPDF |
| **Data Validation** | Pydantic |

---

## 🚀 Setup & Installation

### Prerequisites

- Python 3.10+
- A valid [Google Gemini API Key](https://aistudio.google.com/app/apikey)

### 1. Clone the repository

```bash
git clone https://github.com/ananyareddygandlaparthi/GenViva.git
cd GenViva
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `openai-whisper` requires `ffmpeg`. This project auto-bundles it via `imageio-ffmpeg` — no separate system install needed.

### 4. Configure your API keys

Open `app.py` and replace the placeholder values:

```python
GEMINI_API_KEY_1 = "YOUR_FIRST_GEMINI_API_KEY"
GEMINI_API_KEY_2 = "YOUR_SECOND_GEMINI_API_KEY"
```

> You can use the same key for both, or use two separate keys to avoid rate limits.

### 5. Run the app

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
Mini Project/
├── app.py                  # Main Streamlit application (all logic)
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── system_architecture.png # Architecture diagram
```

---

## 📊 Scoring Methodology

### Initial Confidence Score
| Component | Weight |
|---|---|
| Facial Confidence (DeepFace emotion) | 35% |
| Vocal Confidence (RMS energy) | 30% |
| Fluency Score (WPM + fillers + pauses) | 20% |
| Content Delivery (slide keyword coverage) | 15% |

### Final Confidence Score
| Component | Weight |
|---|---|
| Initial Score carry-forward | 60% |
| Viva Answer Quality (Gemini-graded) | 25% |
| Improvement Bonus | 15% |

---

## ⚠️ Known Limitations

- Whisper `base` model may struggle with heavy accents or noisy audio
- DeepFace requires a visible, well-lit face in the video
- Gemini Free Tier has rate limits — slide text is truncated to 8000 characters to manage token usage
- Processing time scales with video length (~2–5 min for a 10-min video)

---

