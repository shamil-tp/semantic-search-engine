"""Intent classification module using TF-IDF and Linear SVM (LinearSVC)."""

import json
import logging
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)

# Default training data for domain-specific query intents
DEFAULT_TRAINING_DATA = [
    # informational: finding concept definitions, explanations, overviews
    {"text": "what is semantic search", "intent": "informational"},
    {"text": "how do vector embeddings work", "intent": "informational"},
    {"text": "explain contextual understanding in NLP", "intent": "informational"},
    {"text": "overview of hybrid retrieval", "intent": "informational"},
    {"text": "difference between lexical and dense search", "intent": "informational"},
    {"text": "definition of reciprocal rank", "intent": "informational"},
    {"text": "what are dense vector spaces", "intent": "informational"},
    # technical_spec: APIs, schemas, configurations, data parameters
    {"text": "document_id schema specification", "intent": "technical_spec"},
    {"text": "chunk size and overlap parameters", "intent": "technical_spec"},
    {"text": "metadata dictionary structure", "intent": "technical_spec"},
    {"text": "jsonl schema fields start_char end_char", "intent": "technical_spec"},
    {"text": "cli flags and arguments", "intent": "technical_spec"},
    {"text": "max_features and ngram_range settings", "intent": "technical_spec"},
    {"text": "api contract for health and search endpoints", "intent": "technical_spec"},
    # troubleshooting: errors, failures, malformed data, debugging
    {"text": "handle UnicodeDecodeError on utf-8 file", "intent": "troubleshooting"},
    {"text": "fix division by zero in ndcg calculation", "intent": "troubleshooting"},
    {"text": "corrupt index metadata file error", "intent": "troubleshooting"},
    {"text": "why are chunks empty after cleaning", "intent": "troubleshooting"},
    {"text": "resolve missing required index artifact", "intent": "troubleshooting"},
    {"text": "debug duplicate document detection hash collision", "intent": "troubleshooting"},
    {"text": "malformed encoding crashing loader", "intent": "troubleshooting"},
]


@dataclass(frozen=True)
class IntentPrediction:
    """Represents predicted query intent and confidence."""

    intent: str
    confidence: float
    decision_scores: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class IntentClassifier:
    """
    Lightweight query intent classifier.
    Uses TfidfVectorizer + LinearSVC with calibrated decision margins for confidence estimation.
    """

    def __init__(
        self,
        c_param: float = 1.0,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 1,
    ):
        self.c_param = c_param
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            sublinear_tf=True,
            norm="l2",
        )
        self.classifier = LinearSVC(C=self.c_param, random_state=42)
        self.classes_: List[str] = []
        self._is_fitted = False

    def train(self, samples: Sequence[Dict[str, str]]) -> "IntentClassifier":
        """Train classifier on labeled queries."""
        texts = [s["text"] for s in samples]
        labels = [s["intent"] for s in samples]

        x_mat = self.vectorizer.fit_transform(texts)
        self.classifier.fit(x_mat, labels)
        self.classes_ = list(self.classifier.classes_)
        self._is_fitted = True
        logger.info(
            f"Trained IntentClassifier on {len(samples)} samples. Classes: {self.classes_}"
        )
        return self

    def predict(self, query: str) -> IntentPrediction:
        """
        Predict intent for a given query text.
        Returns 'unknown' with 0.0 confidence if unfitted or query is blank.
        """
        if not self._is_fitted or not query or not query.strip():
            return IntentPrediction(
                intent="unknown",
                confidence=0.0,
                decision_scores={},
            )

        x_vec = self.vectorizer.transform([query])
        if x_vec.nnz == 0:
            return IntentPrediction(
                intent="unknown",
                confidence=0.0,
                decision_scores={c: 0.0 for c in self.classes_},
            )

        # Decision function values
        decision = self.classifier.decision_function(x_vec)
        if decision.ndim == 1:
            # Binary classification case
            if len(self.classes_) == 2:
                score_val = float(decision[0])
                scores_dict = {
                    self.classes_[0]: -score_val,
                    self.classes_[1]: score_val,
                }
            else:
                scores_dict = {self.classes_[0]: float(decision[0])}
        else:
            scores_dict = {
                cls_name: float(score)
                for cls_name, score in zip(self.classes_, decision[0])
            }

        # Softmax / normalized confidence approximation over decision scores
        scores_array = np.array(list(scores_dict.values()))
        exp_scores = np.exp(scores_array - np.max(scores_array))
        probs = exp_scores / exp_scores.sum()

        best_idx = int(np.argmax(scores_array))
        predicted_class = list(scores_dict.keys())[best_idx]
        confidence = float(probs[best_idx])

        return IntentPrediction(
            intent=predicted_class,
            confidence=round(confidence, 4),
            decision_scores={k: round(v, 4) for k, v in scores_dict.items()},
        )

    def save(self, model_dir: str | Path) -> None:
        """Persist classifier and vectorizer to directory."""
        if not self._is_fitted:
            raise ValueError("Cannot save an unfitted IntentClassifier.")
        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)

        payload = {
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
            "classes_": self.classes_,
            "config": {
                "c_param": self.c_param,
                "ngram_range": list(self.ngram_range),
                "min_df": self.min_df,
            },
        }
        with open(path / "intent_model.pkl", "wb") as f:
            pickle.dump(payload, f)

        with open(path / "intent_metadata.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "classes": self.classes_,
                    "num_classes": len(self.classes_),
                    "ngram_range": list(self.ngram_range),
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, model_dir: str | Path) -> "IntentClassifier":
        """Load persisted classifier from directory."""
        path = Path(model_dir)
        model_file = path / "intent_model.pkl"
        if not model_file.is_file():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        with open(model_file, "rb") as f:
            data = pickle.load(f)

        cfg = data.get("config", {})
        instance = cls(
            c_param=cfg.get("c_param", 1.0),
            ngram_range=tuple(cfg.get("ngram_range", (1, 2))),
            min_df=cfg.get("min_df", 1),
        )
        instance.vectorizer = data["vectorizer"]
        instance.classifier = data["classifier"]
        instance.classes_ = data["classes_"]
        instance._is_fitted = True
        return instance
