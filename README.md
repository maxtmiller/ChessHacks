# **AlphaPoisson**

AlphaPoisson is a collection of self-trained AI chess engines developed during **ChessHacks 2025** by our team *Échec et mat du poisson*. Our goal was to explore how far lightweight, custom-trained models can go using a mix of self-play, handcrafted evaluation heuristics, and iterative refinement.

---

## 🚀 Overview

AlphaPoisson features multiple chess engines trained under different configurations, each with its own style and strengths. You can play against them directly on our website or explore the source code to see how each engine was built.

Key ideas behind the project include:

- **Self-play training loops**
- **Custom evaluation functions**
- **Iterative parameter tuning**
- **Search techniques (minimax, alpha-beta pruning, etc.)**
- **Lightweight model experimentation within hackathon constraints**

---

## ♟️ Features

- Multiple AI chess engines with different difficulty levels  
- Deterministic and stochastic evaluation variants  
- Modular design — easy to tweak evaluation weights or search depth  
- Web interface for live gameplay  
- Clear engine structure for students, hobbyists, and researchers  

---

## 📍 Play Locally

Ensure you have Python installed (preferably Python 3.12 or later). You also need to have `pip` installed.

### Setup Instructions
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd AlphaPoisson
2. Set up Python environment:
   ```bash
     python -m venv .venv
     source .venv/bin/activate  # On Windows: .venv\Scripts\activate
     pip install -r requirements.txt
3. Install Node.js dependencies:
   ```bash
     cd devtools
     npm install
4. Configure environment variables:

   Create a .env.local file in the root directory and /devtools with the following:
   ```bash
     PYTHON_EXECUTABLE=path/to/python/environment
     SERVE_PORT=5058
     PORT=3000
5. Run the dev server:

   From the root directory, run:
   ```bash
     python3 serve.py
   ```
   
   From the devtools directory, run:
   ```bash
     npm run dev
   ```
     
7. Open the application in your browser using the terminal local host link. You're all set!

---

## 🌐 Play Online

Play against the AlphaPoisson engines in your browser:

👉 *(Will update with live website)*

---
