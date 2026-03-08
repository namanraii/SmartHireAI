# ⚡ SmartHire AI

**SmartHire AI** is an intelligent, multi-model hiring decision support system designed to revolutionize the recruitment process. By leveraging advanced NLP, Machine Learning, and Explainable AI, it automates resume screening, predicts candidate suitability, and provides deep insights into hiring decisions.

---

## 🚀 Key Features

### 🧠 Intelligent Analysis
- **Resume Parsing**: Extracts text from PDF and DOCX files with precision.
- **Semantic Matching**: Uses **BERT** (Sentence-Transformers) to understand the *meaning* of resumes beyond keyword matching.
- **Skill Gap Analysis**: Identifies missing critical skills and recommends learning paths.
- **Bias Detection**: Scans for potential bias indicators (age, gender, origin) to ensure fair hiring.

### 🤖 Predictive Engine
- **Multi-Model Pipeline**: Ensembles **Logistic Regression, SVM, Random Forest, and XGBoost** for robust predictions.
- **Explainable AI (XAI)**: Uses **SHAP (SHapley Additive exPlanations)** to visualize *why* a candidate was selected or rejected.
- **Scoring System**: Weighted evaluation based on Skills (40%), Experience (30%), Education (20%), and Projects (10%).

### 📊 Interactive Dashboard
- **10-Tab Interface**: A comprehensive Streamlit UI covering everything from analysis to reporting.
- **Live Leaderboard**: Ranks candidates in real-time based on AI scores.
- **HR Analytics**: Visualizes hiring funnels and model performance metrics.
- **PDF Dossiers**: Generates detailed, downloadable hiring reports for each candidate.

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/SmartHireAI.git
   cd SmartHireAI
   ```

2. **Create a virtual environment (Optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLP Models:**
   The system requires specific spaCy and NLTK models.
   ```bash
   python -m spacy download en_core_web_sm
   python -m nltk.downloader punkt stopwords wordnet
   ```

---

## ⚡ Usage

1. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

2. **Navigate the UI:**
   - **Input Tab**: Upload a resume (PDF/DOCX) and paste the Job Description.
   - **Click "Initialize Analysis Sequence"**: The system will process the data.
   - **Explore Results**: Switch between Analysis, Prediction, Explain, and other tabs to view insights.

---

## 📂 Project Structure

```text
SmartHireAI/
├── app.py                 # Main Streamlit Application Entry Point
├── requirements.txt       # Python Dependencies
├── utils/                 # Core Utility Modules
│   ├── resume_parser.py   # Text Extraction
│   ├── preprocessor.py    # NLP Cleaning & Tokenization
│   ├── ner_extractor.py   # Entity Extraction (Skills, Org, etc.)
│   ├── skill_gap.py       # Skill Matching Logic
│   ├── similarity_engine.py # BERT & TF-IDF Similarity
│   ├── experience_scorer.py # Weighted Scoring System
│   └── bias_detector.py   # Fairness Checks
├── ml/                    # Machine Learning Pipeline
│   ├── train_model.py     # Model Training Script
│   ├── predict.py         # Inference Engine
│   └── shap_explainer.py  # XAI Visualizations
├── features/              # Advanced Features
│   ├── recommender.py     # Learning Path Generator
│   ├── interview_qgen.py  # AI Interview Question Generator
│   ├── ranker.py          # Candidate Leaderboard
│   └── report_generator.py # PDF Report Creator
└── assets/                # Static Assets (Logos, styles)
```

---

## 🏗️ Technologies Used

- **Frontend**: Streamlit (Custom CSS)
- **NLP**: spaCy, NLTK, Sentence-Transformers (BERT)
- **Machine Learning**: Scikit-Learn, XGBoost
- **Explainability**: SHAP
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Utilities**: PyMuPDF (Fitz), Python-Docx, FPDF2

---

## 📜 License

This project is licensed under the MIT License.
