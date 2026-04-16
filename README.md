# GenViva: Presentation and Viva Coach

GenViva is a full-stack application designed to evaluate and improve student capstone presentations. It performs multimodal analysis on presentation slides (PPT/PDF) and video recordings to provide detailed feedback, identify weak areas, and conduct an interactive AI-driven viva session.

## Features

- **Multimodal Input Parsing:** Extracts text from PowerPoint presentations (PPTX) and PDFs.
- **Audio & Speech Analysis:** Utilizes `OpenAI Whisper` for accurate speech-to-text transcription and `librosa` for gathering speech metrics such as words per minute, energy levels, and pause detection.
- **Visual Emotion Tracking:** Analyzes facial expressions using `DeepFace` to detect emotions and compute a facial confidence score.
- **RAG-based Context Retrieval:** Uses `faiss` and `sentence-transformers` on lecture slide texts and video transcripts to build a retrieval foundation for the viva questions.
- **Automated Viva Questions:** Dynamically identifies potential knowledge gaps and behavioral weaknesses to trigger specific, targeted follow-up questions.
- **Interactive Evaluation:** Evaluates user responses against the extracted slide context to measure continuous improvement and boost overall presentation confidence. 
- **Continuous Feedback Loop:** Allows the user to provide an initial video, take the viva, receive feedback, upload a second improved video, and ultimately tracks their score growth over cycles.

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd <your-repository-directory>
   ```

2. **Set up a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   *Note on `ffmpeg`: The app uses `imageio_ffmpeg` and automatically handles ffmpeg binaries, so a system-level ffmpeg installation is not strictly required.*

## Usage

1. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **App Workflow:**
   - **Phase 1:** Upload your presentation slides (`.pptx` or `.pdf`) and your presentation recording (`.mp4`, `.mov`, etc.). Upon clicking "Run Initial Analysis", the application processes your audio, video, and text content.
   - **Phase 2:** Answer the dynamically generated AI viva questions, which test you on areas where nervousness, fluency dips, or content retrieval gaps were detected.
   - **Phase 3:** Receive comprehensive feedback with an Initial Confidence Score, your strengths, and detailed areas for improvement.
   - **Phase 4 & 5:** Optionally, upload a second, improved video presentation to re-evaluate your confidence score and validate your progress.

## Technologies Used

- **UI Framework:** Streamlit
- **Media Processing:** OpenCV, MoviePy, Librosa, FFmpeg (via imageio-ffmpeg)
- **AI Models:** DeepFace, SentenceTransformers, OpenAI Whisper
- **Vector Search:** FAISS
