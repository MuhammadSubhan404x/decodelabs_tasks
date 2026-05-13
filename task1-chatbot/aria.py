import re
import random
import time
import datetime
from difflib import SequenceMatcher
from pathlib import Path

try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    _SPACY_OK = True
except Exception:
    _nlp = None
    _SPACY_OK = False

from responses import DOMAINS, get_response


def _compile_patterns():
    compiled = []
    for domain, data in DOMAINS.items():
        if domain == "unknown":
            continue
        for pat in data["patterns"]:
            compiled.append((re.compile(pat, re.IGNORECASE), domain))
    return compiled


_PATTERNS = _compile_patterns()


class Chatbot:
    def __init__(self, name="there"):
        self.name = name.strip() or "there"
        self.start_time = datetime.datetime.now()
        self.message_count = 0
        self.topics_seen = set()
        self.last_topic = None
        self.history = []

        if not _SPACY_OK:
            print("Note: spaCy model not found  -  running in regex-only mode.\n")

    def _time_greeting(self):
        hour = datetime.datetime.now().hour
        if hour < 12:
            return "Good morning"
        elif hour < 17:
            return "Good afternoon"
        return "Good evening"

    def _nlp_preprocess(self, text):
        if not _SPACY_OK:
            return text.lower()
        doc = _nlp(text)
        tokens = [t.lemma_.lower() for t in doc if not t.is_stop and not t.is_punct and t.lemma_.strip()]
        return " ".join(tokens) if tokens else text.lower()

    def _detect_intents(self, raw_text, lemmatized):
        matched = []

        for pattern, domain in _PATTERNS:
            if pattern.search(raw_text):
                if domain not in matched:
                    matched.append(domain)

        for domain, data in DOMAINS.items():
            if domain == "unknown":
                continue
            if domain in matched:
                continue
            for kw in data["keywords"]:
                kw_lemma = " ".join(t.lemma_.lower() for t in _nlp(kw)) if _SPACY_OK else kw.lower()
                kw_words = kw_lemma.split()
                if len(kw_words) == 1:
                    found = bool(re.search(r"\b" + re.escape(kw_lemma) + r"\b", lemmatized))
                else:
                    found = kw_lemma in lemmatized
                if found:
                    matched.append(domain)
                    break

        return matched

    def _fuzzy_fallback(self, lemmatized):
        best_domain = None
        best_score = 0.0
        tokens = [t for t in lemmatized.split() if len(t) >= 3]

        for domain, data in DOMAINS.items():
            if domain == "unknown":
                continue
            for kw in data["keywords"]:
                kw_words = [w for w in kw.lower().split() if len(w) >= 3]
                for kw_word in kw_words:
                    for token in tokens:
                        score = SequenceMatcher(None, token, kw_word).ratio()
                        if score > best_score:
                            best_score = score
                            best_domain = domain

        if best_score >= 0.72:
            return best_domain, best_score
        return None, 0.0

    def respond(self, user_input):
        raw = user_input.strip()
        if not raw:
            return ""

        lemmatized = self._nlp_preprocess(raw)
        intents = self._detect_intents(raw, lemmatized)

        if not intents:
            domain, confidence = self._fuzzy_fallback(lemmatized)
            if domain:
                resp = get_response(domain, self.name)
                if confidence < 0.85:
                    resp = f"I think you're asking about {domain.replace('_', ' ')}  -  {resp}"
                intents = [domain]
            else:
                resp = get_response("unknown", self.name)
                intents = ["unknown"]
        elif len(intents) >= 2:
            primary = intents[0]
            secondary = intents[1]
            r1 = get_response(primary, self.name)
            r2 = get_response(secondary, self.name)
            resp = f"{r1}\n\nAlso on {secondary.replace('_', ' ')}: {r2}"
        else:
            resp = get_response(intents[0], self.name)

        for intent in intents:
            if intent != "unknown":
                self.topics_seen.add(intent)
        self.last_topic = intents[0] if intents else None
        self.message_count += 1
        self.history.append({"role": "user", "content": raw})
        self.history.append({"role": "aria", "content": resp})

        return resp

    def get_session_stats(self):
        elapsed = int((datetime.datetime.now() - self.start_time).total_seconds())
        minutes, seconds = divmod(elapsed, 60)
        return {
            "duration": f"{minutes}m {seconds}s",
            "message_count": self.message_count,
            "topics": sorted(self.topics_seen),
        }

    def save_log(self):
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        log_path = Path(__file__).parent / f"chat_log_{date_str}.txt"
        lines = [f"ARIA Chat Log  -  {date_str}\n", "=" * 40 + "\n"]
        for entry in self.history:
            speaker = self.name if entry["role"] == "user" else "ARIA"
            lines.append(f"{speaker}: {entry['content']}\n\n")
        log_path.write_text("".join(lines), encoding="utf-8")
        return str(log_path)

    def welcome_message(self):
        greeting = self._time_greeting()
        return (
            f"{greeting}, {self.name}! I'm ARIA  -  your AI assistant.\n"
            "I can talk about machine learning, Python, tell jokes, offer motivation, and more.\n"
            "Type 'help' to see topics, or just ask away."
        )

    def run(self):
        print("\n" + "=" * 50)
        print(self.welcome_message())
        print("=" * 50 + "\n")

        while True:
            try:
                user_input = input(f"{self.name}: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue

            time.sleep(0.4)
            response = self.respond(user_input)
            print(f"\nARIA: {response}\n")

            if self.last_topic == "farewell":
                break

        stats = self.get_session_stats()
        print("\n" + "-" * 40)
        print(f"Session summary  -  {stats['duration']} | {stats['message_count']} messages")
        if stats["topics"]:
            print(f"Topics covered: {', '.join(stats['topics'])}")
        log_path = self.save_log()
        print(f"Conversation saved to: {log_path}")
        print("-" * 40 + "\n")
