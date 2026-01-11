"BDH Narrative Consistency Engine"

Team sim-sim
Team Members:
Rishit Mohanty (Team Leader)
Smruti Ranjan Nayak
Soham Das
Prasmit Prayansu

"Project Description"-

This project is an AI-powered narrative consistency checker that verifies whether a given back-story about a character is consistent with the original novel.
It is based on two powerful ideas:

1.BDH (Belief-Driven Hypothesis Model)-
A neural model trained on story event sequences that learns how a character’s beliefs and identity evolve.

2.Semantic Similarity (MiniLM)-
A transformer-based language model that compares the meaning of the user’s back-story with the original novel.

The system combines semantic similarity with belief drift to determine if a story matches the original narrative.


"What It Does"-
For a given character back-story, the system:
1.Compares it semantically with the original novel
2.Measures belief drift using BDH
3.Produces a verdict:
    Consistent ✅
    Contradict ❌

This allows us to detect:
  1.Wrong origins
  2.False relationships
  3.Incorrect events
  4.Fabricated character histories


"Example"-
Input (Correct):
Edmond Dantès was a young sailor from Marseille, loyal to his father and engaged to Mercédès…

Output:
✅ Consistent

Input (Wrong):
Edmond Dantès was a noble prince raised in Paris…

Output:
❌ Contradict

"Web Interface"-
The system is deployed using FastAPI + Jinja2.
Users can enter a back-story and instantly see whether it matches the novel.

Run the website:
uvicorn app:app --reload

Open in browser:
http://127.0.0.1:8000

"Project Structure"-
BDH_PROJECT/
│
├── app.py                  # FastAPI web app
├── consistency_engine.py   # Core logic (BDH + Semantic)
├── tokenizer.py            # Narrative tokenizer
├── bdh.py                  # BDH neural model
├── train_bdh.py            # Train BDH
├── train_classifier.py     # Train consistency classifier
├── cache_novels.py         # Cache novel embeddings
├── calibrate.py            # Threshold calibration
│
├── data/
│   ├── novels/
│   │   ├── monte_cristo.txt
│   │   └── castaways.txt
│   ├── train.csv
│   └── test.csv
│
├── models/
│   ├── bdh.pt
│   ├── tokenizer.km
│   ├── monte_tokens.km
│   └── cast_tokens.km
│
├── templates/
│   └── index.html
├── static/
│   └── style.css
└── requirements.txt


"Installation"-
Create environment:
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt


"Training Pipeline"-
python train_tokenizer.py
python train_bdh.py
python cache_novels.py
python train_classifier.py
python calibrate.py

"Why This Matters"-
This system can be used in:
    1.AI story validation
    2.Fiction plagiarism detection
    3.Game character consistency
    4.Narrative AI agents
It ensures that AI doesn’t hallucinate false character histories.


"Credits"-
Built by Team sim-sim as an advanced AI + Narrative Intelligence project.


"License"-
For academic and educational use.
