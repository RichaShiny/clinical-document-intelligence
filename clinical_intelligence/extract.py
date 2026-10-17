"""Two deliberately small, auditable dictionary NLP baselines."""
import re

TERMS = {
    "condition": {"hypertension": ["hypertension", "HTN"],
                  "type 2 diabetes": ["type 2 diabetes", "T2DM"],
                  "asthma": ["asthma"], "pneumonia": ["pneumonia"],
                  "heart failure": ["heart failure", "CHF"]},
    "medication": {"metformin": ["metformin"], "lisinopril": ["lisinopril"],
                   "albuterol": ["albuterol"], "amoxicillin": ["amoxicillin"]},
    "procedure": {"chest x-ray": ["chest x-ray", "CXR"],
                  "hemoglobin A1c": ["hemoglobin A1c", "HbA1c"]},
}
STATUSES = ("present", "negated", "uncertain", "historical", "family", "planned")


def sentence_spans(text):
    for m in re.finditer(r"[^.!?\n]+(?:[.!?]|$)", text):
        if m.group().strip():
            start = m.start() + len(m.group()) - len(m.group().lstrip())
            yield start, m.end(), text[start:m.end()]


def assertion(prefix, sentence):
    # Clause-local, limited scope. This is a heuristic, not a clinical assertion model.
    prefix = re.split(r"\b(?:but|however)\b|;", prefix, flags=re.I)[-1].lower()
    if re.search(r"\b(?:mother|father|sister|brother|family history)\b", prefix):
        return "family"
    if re.search(r"\b(?:no|denies|without|negative for|ruled out)\b", prefix):
        return "negated"
    if re.search(r"\b(?:possible|suspected|may have|rule out|concern for)\b", prefix):
        return "uncertain"
    if re.search(r"\b(?:history of|previous|resolved|stopped|discontinued)\b", prefix):
        return "historical"
    if re.search(r"\b(?:plan|ordered|schedule|will obtain)\b", prefix):
        return "planned"
    return "present"


def extract(text, contextual=True):
    entities = []
    for start, end, sentence in sentence_spans(text):
        for kind, concepts in TERMS.items():
            for canonical, aliases in concepts.items():
                for alias in aliases:
                    for match in re.finditer(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", sentence, re.I):
                        left, right = start + match.start(), start + match.end()
                        entities.append(dict(kind=kind, value=canonical,
                            status=assertion(sentence[:match.start()], sentence) if contextual else "present",
                            start=left, end=right, evidence=text[left:right]))
        for match in re.finditer(r"\bfollow[- ]up\b[^.!?\n]*", sentence, re.I):
            left, right = start + match.start(), start + match.end()
            entities.append(dict(kind="follow_up", value=text[left:right].lower(),
                status="planned", start=left, end=right, evidence=text[left:right]))
    return sorted(entities, key=lambda e: (e["start"], e["kind"]))


def summarize(text, entities):
    """Extractive evidence cards; no newly generated medical claims."""
    cards, seen = [], set()
    for start, end, sentence in sentence_spans(text):
        relevant = [e for e in entities if start <= e["start"] < end]
        if relevant and sentence not in seen:
            cards.append(dict(text=sentence, start=start, end=end,
                              statuses=sorted({e["status"] for e in relevant})))
            seen.add(sentence)
    return cards
