# ARIA  -  Rule-Based AI Chatbot

A conversational assistant that uses NLP preprocessing and regex pattern matching to understand
and respond to natural language input across multiple topic domains.

## How to run

```bash
cd task1-chatbot
pip install -r ../requirements.txt
python -m spacy download en_core_web_sm

# Web interface (recommended)
streamlit run app.py

# Terminal interface
python -c "from aria import Chatbot; Chatbot('Your Name').run()"
```

## Technical highlights

- **spaCy NLP preprocessing**: user input is lemmatized before matching  -  "machines that LEARN"
  and "machine learning" resolve to the same tokens, so typos and paraphrases are handled gracefully
- **Multi-intent detection**: a message like "tell me a joke about deep learning" matches both
  the `jokes` and `ai_concepts` domains simultaneously, producing a combined response
- **Fuzzy fallback**: when both regex and keyword matching fail, `difflib.SequenceMatcher` compares
  word-by-word against keyword lists  -  "pythn" and "mchine lernign" still reach the correct domain
- **Streamlit UI**: browser-based chat interface with live session stats in the sidebar
  and a conversation download button

## Domains

`greeting` | `ai_concepts` | `python_questions` | `jokes` | `motivation` | `chatbot_meta` | `farewell` | `help`

Each domain has 4-5 varied response options selected randomly to avoid repetition.

## Sample output

```
ARIA: Good afternoon, Subhan! I'm ARIA  -  your AI assistant.

Subhan: what is machine learning

ARIA: Machine learning is a subset of AI where systems learn patterns from data
rather than following hand-coded rules. Instead of programming every decision,
you feed a model thousands of examples and let it figure out the boundaries itself.

Subhan: mchine lernign

ARIA: The core distinction: AI is the goal (machines that think), machine learning is
the technique (learn from data), deep learning is a specific method (neural networks).
```

## Dependencies

`spacy` | `streamlit` | `en_core_web_sm` (spaCy model)
