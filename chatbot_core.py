"""Chatbot core: answers questions about Salesforce opportunities.

Pipeline for each message:
    greeting?  ->  find opportunity name(s)  ->  classify intent  ->  build answer

Intent classification is rule-based by default (fast, no downloads). If you set
USE_ZERO_SHOT=1 and `transformers` + `torch` are installed, a zero-shot model
(facebook/bart-large-mnli) is tried first and the rules act as the fallback,
which is how the original project worked.
"""
import logging
import os
import random
import re
from difflib import SequenceMatcher

import pandas as pd

log = logging.getLogger("chatbot")

# ---------------------------------------------------------------------------
# Greetings
# ---------------------------------------------------------------------------

GREET_SINGLE = {
    "hello", "hi", "hey", "sup", "whatsup", "wassup", "morning", "afternoon",
    "evening", "bonjour", "salut", "hallo",
}
GREET_PHRASES = {
    "what's up", "whats up", "good morning", "good afternoon", "good evening",
}
GREET_RESPONSES = [
    "😃 Hi there! What can I do for you today?",
    "😃 Hey! How can I assist?",
    "👋 Hello! What do you need?",
    "👋 Nice to see you! How can I help you?",
    "😃 Hi! Let's get started.",
    "👋 Welcome! Ask me anything.",
]

HELP_TEXT = (
    "Ask me about an opportunity by name, for example:\n"
    "• Who owns AM003?\n"
    "• What stage is BMWGEN6 in?\n"
    "• When does VW426 start / end?\n"
    "• Give me a summary of AM003\n"
    "I can tell you the owner, stage, account, SOP/EOP/DOS dates, sales, volume, "
    "priority, predecessor carline and creation / modification dates."
)


def greet(sentence):
    """Return a greeting if the message is just a greeting, otherwise None."""
    cleaned = re.sub(r"[^\w\s']", "", sentence.lower()).strip()
    words = cleaned.split()
    if not words or len(words) > 4:
        return None
    if cleaned in GREET_PHRASES or words[0] in GREET_SINGLE:
        return random.choice(GREET_RESPONSES)
    return None


# ---------------------------------------------------------------------------
# Chatbot
# ---------------------------------------------------------------------------

class Chatbot:
    def __init__(self, dataframe, use_zero_shot=None):
        self.df = dataframe.reset_index(drop=True)
        self.opportunity_names = self.df["Opportunity Name"].dropna().tolist()
        self.company_names = self.df["Account Name"].dropna().unique().tolist()

        # lower-case name -> row index, and "normalised" name (no trailing "__")
        self._row_by_name = {n.lower(): i for i, n in enumerate(self.df["Opportunity Name"]) if isinstance(n, str)}
        self._norm = {n: n.lower().rstrip("_") for n in self.opportunity_names}

        self.classifier = None
        if use_zero_shot is None:
            use_zero_shot = os.environ.get("USE_ZERO_SHOT", "0") == "1"
        if use_zero_shot:
            self._init_zero_shot()

        self._build_intent_keywords()
        self.create_getters()
        self.memory = {"last_opportunity": None, "last_intent": None}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _init_zero_shot(self):
        try:
            from transformers import pipeline
            self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
            log.info("Zero-shot model loaded")
        except Exception as e:  # missing package, no network, ...
            print(f"[chatbot] Zero-shot model unavailable ({e}). Using rule-based intents.")
            self.classifier = None

    def _build_intent_keywords(self):
        self.intent_keywords = {
            "owner": ["owner", "owns", "who owns", "responsible", "who is responsible",
                      "responsible for", "assigned to", "in charge", "who manages", "managing",
                      "manager"],
            "stage": ["stage", "phase", "status", "current stage"],
            "sop": ["sop", "start", "begin", "starting", "start date", "start of production"],
            "eop": ["eop", "end", "finish", "ending", "end date", "end of production"],
            "dos": ["dos", "decision", "decision date", "expected dos"],
            "sales": ["sales", "total sales", "revenue", "amount", "money", "cost", "worth"],
            "volume": ["volume", "total volume", "quantity", "units"],
            "account": ["account", "company", "client", "customer", "oem", "brand"],
            "priority": ["priority", "importance", "urgent"],
            "predecessor": ["predecessor", "predecessor carline", "previous carline",
                            "previous car", "replaces", "successor of"],
            "created date": ["created date", "creation date", "created",
                             "when created", "creation"],
            "last modified date": ["last modified date", "last modified", "modified date",
                                   "last change", "modification", "when modified",
                                   "last updated", "updated", "modified"],
            "summary": ["summary", "summarize", "summarise", "overview", "show all",
                        "full info", "all details", "details", "tell me about", "describe",
                        "info"],
        }

    def create_getters(self):
        """Create get_owner(), get_stage(), ... (kept from the original API)."""
        columns = {
            "get_owner": "Owner Full Name",
            "get_stage": "Stage",
            "get_sop": "SOP",
            "get_eop": "EOP",
            "get_dos": "DOS",
            "get_total_sales": "Total Sales",
            "get_total_volume": "Total Volume",
            "get_account_name": "Account Name",
            "get_actual_priority": "Actual Priority",
            "get_created_date": "Created Date",
            "get_last_modified_date": "Last Modified Date",
            "get_predecessor_carline": "Predecessor Carline",
        }
        for method, column in columns.items():
            setattr(self, method, self._create_getter(column))

    def _create_getter(self, column_name):
        def getter(opportunity_name):
            idx = self._row_by_name.get(str(opportunity_name).lower())
            if idx is None:
                return "Opportunity not found"
            value = self.df.at[idx, column_name]
            if pd.isna(value) or str(value).strip() == "":
                return "not available" if column_name != "Predecessor Carline" else "none recorded"
            if isinstance(value, pd.Timestamp):
                return value.strftime("%d.%m.%Y")
            if column_name in ("Total Sales", "Total Volume"):
                try:
                    return f"{float(value):,.0f}"
                except (TypeError, ValueError):
                    return value
            return value
        return getter

    # ------------------------------------------------------------------
    # Intent classification
    # ------------------------------------------------------------------

    def classify_intent_with_llm(self, text):
        """Zero-shot classification if available, rules otherwise."""
        if self.classifier is None:
            return self.classify_intent_rules(text)
        labels = {
            "asking about the owner": "owner",
            "asking about the stage": "stage",
            "asking about the start of production date": "sop",
            "asking about the end of production date": "eop",
            "asking about the decision date": "dos",
            "asking about total sales": "sales",
            "asking about total volume": "volume",
            "asking about the account": "account",
            "asking about the priority": "priority",
            "asking about the creation date": "created date",
            "asking about the last modified date": "last modified date",
            "asking about the predecessor carline": "predecessor",
            "asking for a summary": "summary",
        }
        try:
            normalized = re.sub(r"\bsop\b", "start of production date", text, flags=re.I)
            normalized = re.sub(r"\beop\b", "end of production date", normalized, flags=re.I)
            normalized = re.sub(r"\bdos\b", "decision date", normalized, flags=re.I)
            result = self.classifier(normalized, list(labels))
            if result["scores"][0] > 0.6:
                return labels[result["labels"][0]]
        except Exception as e:
            log.warning("Zero-shot failed: %s", e)
        return self.classify_intent_rules(text)

    def _intent_scores(self, text):
        """Whole-word keyword hits per intent; longer phrases count more."""
        text_lower = text.lower()
        scores = {}
        for intent, keywords in self.intent_keywords.items():
            score = 0
            for kw in keywords:
                if re.search(rf"(?<![a-z0-9]){re.escape(kw)}(?![a-z0-9])", text_lower):
                    score += len(kw.split())
            if score:
                scores[intent] = score
        return scores

    def classify_intent_rules(self, text):
        """Best single intent ("unknown" if nothing matches). Ties keep dict order."""
        scores = self._intent_scores(text)
        if not scores:
            return "unknown"
        return max(scores, key=lambda i: scores[i])  # max() keeps the first of equal scores

    def classify_intents(self, text):
        """All intents asked in one message, e.g. "when does X end and who owns it"."""
        if self.classifier is not None:
            intent = self.classify_intent_with_llm(text)
            return [] if intent == "unknown" else [intent]
        intents = list(self._intent_scores(text))
        if len(intents) > 1 and "summary" in intents:
            intents.remove("summary")  # "tell me about the owner of X" -> just the owner
        return intents

    # ------------------------------------------------------------------
    # Entity extraction (opportunity names)
    # ------------------------------------------------------------------

    def find_opportunities(self, text):
        """Opportunity names written in the text (the trailing "__" is optional)."""
        text_lower = text.lower()
        spans = []
        for name, norm in self._norm.items():
            if not norm:
                continue
            for m in re.finditer(rf"(?<![a-z0-9]){re.escape(norm)}(?![a-z0-9])", text_lower):
                spans.append((m.start(), m.end(), name))
        # drop matches that sit inside a longer match ("vw326" inside "vw326/7")
        keep = [s for s in spans
                if not any(o is not s and o[0] <= s[0] and s[1] <= o[1] and (o[1] - o[0]) > (s[1] - s[0])
                           for o in spans)]
        keep.sort()
        found = []
        for _, _, name in keep:
            if name not in found:
                found.append(name)
        return found

    def suggest_opportunities(self, text, limit=3):
        """Close (not exact) matches for tokens that look like carline codes."""
        scored = {}
        for token in re.findall(r"[a-z]{1,8}\d[a-z0-9/]*", text.lower()):
            for name, norm in self._norm.items():
                if len(token) >= 4 and norm.startswith(token):
                    score = 0.9
                else:
                    score = SequenceMatcher(None, token, norm).ratio()
                if score >= 0.85:
                    scored[name] = max(scored.get(name, 0), score)
        return [n for n, _ in sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]]

    def extract_entities(self, text):
        """Kept for API compatibility with the original project."""
        return {
            "Opportunities": self.find_opportunities(text),
            "Companies": [c for c in self.company_names if c.lower() in text.lower()],
        }

    # ------------------------------------------------------------------
    # Response generation
    # ------------------------------------------------------------------

    def generate_response(self, intent, opportunity_name):
        n = opportunity_name
        handlers = {
            "owner":              lambda: f"👤 Owner: {self.get_owner(n)}",
            "stage":              lambda: f"📊 Stage: {self.get_stage(n)}",
            "sop":                lambda: f"📅 Start of production (SOP): {self.get_sop(n)}",
            "eop":                lambda: f"🏁 End of production (EOP): {self.get_eop(n)}",
            "dos":                lambda: f"⚖️ Decision date (DOS): {self.get_dos(n)}",
            "sales":              lambda: f"💰 Total Sales: {self.get_total_sales(n)}",
            "volume":             lambda: f"📦 Total Volume: {self.get_total_volume(n)}",
            "account":            lambda: f"🏢 Account: {self.get_account_name(n)}",
            "priority":           lambda: f"⭐ Priority: {self.get_actual_priority(n)}",
            "created date":       lambda: f"📅 Created: {self.get_created_date(n)}",
            "last modified date": lambda: f"🕒 Last modified: {self.get_last_modified_date(n)}",
            "predecessor":        lambda: f"🔁 Predecessor carline: {self.get_predecessor_carline(n)}",
            "summary":            lambda: self.get_opportunity_summary(n),
        }
        handler = handlers.get(intent)
        return handler() if handler else HELP_TEXT

    def get_opportunity_summary(self, opportunity_name):
        n = opportunity_name
        return (
            f"📋 Summary for {n}\n"
            f"👤 Owner: {self.get_owner(n)}\n"
            f"📊 Stage: {self.get_stage(n)}\n"
            f"🏢 Account: {self.get_account_name(n)}\n"
            f"⭐ Priority: {self.get_actual_priority(n)}\n"
            f"⚖️ Decision date (DOS): {self.get_dos(n)}\n"
            f"📅 Start of production (SOP): {self.get_sop(n)}\n"
            f"🏁 End of production (EOP): {self.get_eop(n)}\n"
            f"💰 Total Sales: {self.get_total_sales(n)}\n"
            f"📦 Total Volume: {self.get_total_volume(n)}\n"
            f"🔁 Predecessor carline: {self.get_predecessor_carline(n)}"
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def respond(self, user_input):
        if not user_input or not user_input.strip():
            return HELP_TEXT

        greeting = greet(user_input)
        if greeting:
            return greeting

        opportunities = self.find_opportunities(user_input)
        intents = self.classify_intents(user_input)
        log.debug("input=%r opportunities=%s intents=%s", user_input, opportunities, intents)

        note = ""
        if not opportunities:
            suggestions = self.suggest_opportunities(user_input)
            codes = re.findall(r"[a-z]{1,8}\d[a-z0-9/]*", user_input.lower())
            if len(suggestions) == 1:
                opportunities = suggestions
                note = f"(Assuming you mean {suggestions[0]}.)\n"
            elif len(suggestions) > 1:
                return "🤔 I found several close matches. Which one do you mean?\n• " + "\n• ".join(suggestions)
            elif codes:  # the user named something, but it isn't in the data: don't guess
                return f"❌ I couldn't find an opportunity called {codes[0].upper()}."
            elif self.memory["last_opportunity"] and intents:
                opportunities = [self.memory["last_opportunity"]]  # follow-up question
            elif not intents:
                return HELP_TEXT
            else:
                return "❌ I couldn't find an opportunity name in your message. Please mention one, e.g. AM003."

        if not intents:
            intents = ["summary"]  # only a name was given: show everything

        opportunities = opportunities[:3]
        self.memory["last_opportunity"] = opportunities[-1]
        self.memory["last_intent"] = intents[-1]

        blocks = []
        for opp in opportunities:
            lines = [self.generate_response(intent, opp) for intent in intents]
            body = "\n".join(lines)
            blocks.append(body if len(opportunities) == 1 or intents == ["summary"] else f"{opp}\n{body}")
        return note + "\n\n".join(blocks)
