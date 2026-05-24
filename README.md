# DecodeLabs AI Internship  -  Muhammad Subhan

AI Intern (Virtual) | May-June 2026

Three projects built during the DecodeLabs 4-week internship program. Each one
builds on the previous: from rule-based logic, to learned pattern recognition,
to personalized ranking  -  mirroring how AI systems evolve from deterministic
rules toward data-driven intelligence.

## Projects

| Task | Project | Core Technique |
|------|---------|----------------|
| [Task 1](task1-chatbot/) | ARIA  -  Rule-Based Chatbot | spaCy NLP, regex, Streamlit |
| [Task 2](task2-classification/) | Heart Disease Classifier | 5-model ML pipeline, SHAP, cross-validation |
| [Task 3](task3-recommendation/) | Movie Recommendation Engine | TF-IDF, cosine similarity, hybrid scoring |

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Navigate into any task folder and follow its README.

## What I built

Starting with a rule-based chatbot showed me pretty quickly how brittle if-else logic is once
real users start typing. Adding spaCy on top helped a lot  -  "mchine lernign" gets handled
correctly now, which it wouldn't with plain regex.

The classification project was the one I spent the most time on. Comparing 5 models against a
dummy baseline made it obvious which models were actually learning something vs. just guessing
the majority class. SHAP was new to me  -  seeing how each feature pushed individual predictions
was more useful than a feature importance bar chart.

The recommendation system was trickier than I expected. Getting the cold-start case right
(user hasn't seen any movies from the list) took a few tries. Precision@5 came out at 0.960
which I'm happy with.
