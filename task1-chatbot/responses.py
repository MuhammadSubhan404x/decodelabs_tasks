import random

DOMAINS = {
    "greeting": {
        "patterns": [
            r"\b(hi|hello|hey|howdy|sup|what'?s up|yo|hiya)\b",
            r"\bgood (morning|afternoon|evening|day)\b",
            r"\bhow are you\b",
            r"\bhow('?s| is) it going\b",
            r"\bwhat'?s good\b",
        ],
        "keywords": ["hi", "hello", "hey", "howdy", "sup", "morning", "afternoon", "evening", "greet"],
        "responses": [
            "Hey {name}! Good to see you. Ask me anything about AI, Python, or just chat.",
            "Hello {name}! What's on your mind today?",
            "Hi {name}! Ready when you are.",
            "Hey! Always good to hear from you, {name}. What would you like to talk about?",
            "Hello there, {name}! What can I help you with?",
        ],
    },
    "ai_concepts": {
        "patterns": [
            r"\b(what is|explain|tell me about|define|describe)\b.*(ai|artificial intelligence|machine learning|ml|deep learning|neural network|nlp|computer vision)\b",
            r"\b(ai|artificial intelligence|machine learning|deep learning|neural network)\b.*(mean|work|do)\b",
            r"\bhow does?\b.*(ai|machine learning|neural network|deep learning)\b",
            r"\bwhat('?s| is) (ai|ml|artificial intelligence|machine learning|deep learning|a neural network)\b",
            r"\bdifference between\b.*\b(ai|ml|deep learning)\b",
        ],
        "keywords": ["artificial intelligence", "machine learning", "deep learning", "neural network",
                     "nlp", "natural language", "computer vision", "algorithm", "model", "training",
                     "supervised", "unsupervised", "reinforcement", "transformer", "gpt", "llm"],
        "responses": [
            "Machine learning is a subset of AI where systems learn patterns from data rather than following hand-coded rules. "
            "Instead of programming every decision, you feed a model thousands of examples and let it figure out the boundaries itself  -  "
            "like how a spam filter learns what spam looks like from labeled emails.",

            "Artificial Intelligence is the broad field of building systems that can perform tasks normally requiring human intelligence. "
            "Machine learning is one approach: instead of writing rules, you train a model on data. "
            "Deep learning goes further  -  it uses layered neural networks that can learn from raw input like images and text.",

            "Neural networks are loosely inspired by how the brain works  -  layers of nodes that each transform the input slightly. "
            "The 'deep' in deep learning just means many layers. "
            "Stack enough of them with enough data and they can learn to recognize faces, translate languages, or write code.",

            "The core distinction: AI is the goal (machines that think), machine learning is the technique (learn from data), "
            "deep learning is a specific method (neural networks with many layers). "
            "Most of what people call AI today is actually deep learning.",

            "In machine learning, a model is trained by showing it thousands of examples with known answers. "
            "It adjusts its internal parameters to minimize how often it's wrong. "
            "Once trained, it generalizes  -  making predictions on data it's never seen.",
        ],
    },
    "python_questions": {
        "patterns": [
            r"\b(what is|tell me about|explain|learn)\b.*(python)\b",
            r"\bpython\b.*(language|programming|code|used for|good for)\b",
            r"\bwhy (use|learn|choose)\b.*python\b",
            r"\bpython vs\b",
            r"\bis python\b.*(good|best|fast|slow)\b",
        ],
        "keywords": ["python", "programming language", "script", "django", "flask", "pandas", "pip"],
        "responses": [
            "Python is a high-level language known for readable syntax and a massive ecosystem. "
            "In AI and data science it's the dominant choice because of libraries like NumPy, pandas, scikit-learn, and PyTorch  -  "
            "you're not fighting the language, you're focused on the problem.",

            "Python's real strength is its libraries. "
            "pandas handles data manipulation, scikit-learn gives you ready-made ML algorithms, "
            "matplotlib and seaborn handle visualization, and PyTorch or TensorFlow handle deep learning. "
            "One language for the entire pipeline.",

            "Python is slower than C or Java for raw computation, but that rarely matters in ML  -  "
            "the heavy lifting is done by libraries written in C under the hood. "
            "You write Python, NumPy executes optimized C. Best of both worlds.",

            "Python is probably the most practical first language for AI/ML work, {name}. "
            "The syntax reads almost like pseudocode, the community is massive, "
            "and every major ML framework has Python bindings.",
        ],
    },
    "jokes": {
        "patterns": [
            r"\b(tell me a joke|make me laugh|say something funny|got a joke|joke)\b",
            r"\b(funny|humor|humour|laugh)\b",
            r"\bcheer me up\b",
        ],
        "keywords": ["joke", "funny", "laugh", "humor", "hilarious", "cheer", "fun"],
        "responses": [
            "Why do programmers prefer dark mode? Because light attracts bugs.",
            "A machine learning model walks into a bar. The bartender asks: 'What'll it be?' The model says: 'Whatever most people before me ordered.'",
            "I tried to come up with a joke about neural networks but I couldn't find the right weights for it.",
            "Why did the data scientist break up with the statistician? She found someone more significant.",
            "I asked an AI to write me a joke. It said: 'I cannot currently do that.' I said: 'That IS the joke.'",
        ],
    },
    "motivation": {
        "patterns": [
            r"\b(motivate me|inspire me|i need motivation|feeling (down|stuck|lost|bad)|encourage me)\b",
            r"\b(i'?m (struggling|tired|frustrated|giving up|burnt out))\b",
            r"\bgive me (a quote|some motivation|inspiration)\b",
        ],
        "keywords": ["motivate", "inspire", "motivation", "inspiration", "struggling", "tired", "stuck",
                     "frustrated", "quit", "give up", "discouraged", "burnout"],
        "responses": [
            "Every expert started where you are right now. The code that looks obvious to you in six months "
            "is exactly what you're wrestling with today. Keep going, {name}.",

            "The best programmers aren't the ones who never get stuck  -  they're the ones who push through when they are. "
            "Stuck means you're at the edge of what you know. That's exactly where growth happens.",

            "One of my favorite things about learning to code: the moment something finally works after hours of debugging "
            "is one of the most satisfying feelings there is. You're closer to that moment than you think.",

            "Not every day will feel productive, {name}. Some days you just maintain. That's fine. "
            "What matters is you're still at the keyboard.",

            "Struggle is the sign of progress, not failure. If it were easy, you wouldn't be learning anything.",
        ],
    },
    "chatbot_meta": {
        "patterns": [
            r"\b(what are you|who are you|introduce yourself|what'?s your name|your name)\b",
            r"\b(what can you do|what do you know|your capabilities|help me understand you)\b",
            r"\bare you (an ai|a bot|a robot|intelligent|conscious)\b",
            r"\bhow do you work\b",
        ],
        "keywords": ["aria", "who are you", "what are you", "your name", "about you", "chatbot", "bot",
                     "capabilities", "introduce"],
        "responses": [
            "I'm ARIA  -  a rule-based conversational assistant. I understand your input using regex pattern matching "
            "and natural language preprocessing, then select from curated response pools. "
            "No large language model behind me, just well-designed rules and some NLP.",

            "My name is ARIA. I can talk about AI and machine learning concepts, Python, tell you a joke, "
            "offer some motivation, or just chat. I work best when you ask about things in my knowledge domains.",

            "Good question, {name}. I'm a rule-based chatbot  -  meaning my responses come from predefined logic, "
            "not from a trained model. I use spaCy to understand your input and regex patterns to match your intent. "
            "Fast, transparent, and deterministic.",

            "I'm ARIA, your AI intern assistant. Under the hood I lemmatize your input, match it against "
            "pattern libraries, and return responses from curated pools. Ask me about AI, Python, or ask for a joke.",
        ],
    },
    "farewell": {
        "patterns": [
            r"\b(bye|goodbye|see you|take care|cya|later|quit|exit|good night|gotta go)\b",
            r"\b(i('?m| am) (leaving|going|done|out))\b",
        ],
        "keywords": ["bye", "goodbye", "later", "exit", "quit", "done", "night", "farewell", "leave"],
        "responses": [
            "See you, {name}! It was good talking with you.",
            "Take care, {name}. Come back anytime.",
            "Goodbye! Good luck with whatever you're working on.",
            "Later, {name}. Keep building things.",
        ],
    },
    "help": {
        "patterns": [
            r"\b(help|commands|what topics|menu|what can you talk about|options)\b",
            r"\bi don'?t know what to (ask|say)\b",
        ],
        "keywords": ["help", "menu", "topics", "options", "commands", "what can"],
        "responses": [
            "Here's what I can talk about, {name}:\n"
            "  - AI & machine learning concepts\n"
            "  - Python programming\n"
            "  - Jokes (I try)\n"
            "  - Motivation when you need it\n"
            "  - About me (what I am and how I work)\n"
            "Just ask naturally  -  I'll do my best.",

            "Topics I know well: AI/ML concepts, Python, jokes, motivation, and meta questions about how I work. "
            "Try asking 'what is deep learning?' or 'tell me a joke' or 'I need motivation'.",
        ],
    },
    "unknown": {
        "patterns": [],
        "keywords": [],
        "responses": [
            "I'm not sure I caught that, {name}. I work best with questions about AI, Python, or just casual chat. "
            "Type 'help' to see what I can do.",
            "That one's outside my knowledge, {name}. Try asking about machine learning, Python, or say 'help'.",
            "Hmm, I didn't quite get that. My best topics are AI concepts and Python  -  or I can tell you a joke.",
            "Not sure about that one. Ask me about AI, machine learning, Python, or type 'help' for options.",
        ],
    },
}


def get_response(domain, name="there"):
    pool = DOMAINS[domain]["responses"]
    resp = random.choice(pool)
    return resp.replace("{name}", name)
