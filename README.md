# Vintell

**Vintell** is an AI-powered fashion assistant that helps users discover, search, and visualize clothing items.  
It combines **Retrieval-Augmented Generation (RAG)** with an interactive web interface to provide smart and personalized fashion exploration.

---

## 🚀 Getting Started (Setup & Run)

Follow these steps to run the project locally.

### 1. Clone the repository
```bash
git clone https://github.com/zilan-turunc/vintell.git
cd vintell
```

### 2. Create & activate virtual environment
```bash
python -m venv .venv
```

Activate it:

- **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- **Mac/Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the project (single command)
From the project root:
```bash
python main.py
```

- Backend will start (RAG services)  
- Frontend will start (FastAPI + HTML templates)  
- Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser  

---

## 📦 Requirements

All dependencies are pinned for reliability:

```
fastapi
uvicorn
pandas
scikit-learn
python-dotenv
torch==2.3.1
transformers==4.44.2
sentence-transformers==2.7.0
```

---

## ✨ Features

- **🛍️ Natural Language Search**  
  Describe items like *“beige cropped blazer for office”* and get AI-powered matches.

- **🎨 Mood Board Generator**  
  Build and visualize outfit ideas or seasonal looks in an interactive mood board.

- **🔍 Image-Based Search**  
  Upload or provide an image, and Vintell finds visually similar items.

- **📊 Structured Dataset Integration**  
  Items are stored in CSV/JSON, making it possible to add personalization like *“avoid red clothes”* in recommendations.

---

## 🔮 Future Development

- **Personalized Fashion Profiles** – tailor results to individual style and dislikes.  
- **E-commerce Integration** – pull live products, availability, and pricing from real stores.  
- **Advanced Mood Boards** – collaborative boards, drag-and-drop, AI-generated styling tips.  
- **Analytics Dashboard** – track trends, top searched styles, and engagement.  
- **Mobile/Desktop Apps** – beyond browser, seamless multi-platform experience.

---

## 👩‍💻 Authors

Developed by **Zilan Turunc** for Up School AI First Developer Capstone Graduation project.
