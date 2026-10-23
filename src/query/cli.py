"""CLI for training and testing query understanding modules."""

import argparse
import json
import logging
from pathlib import Path

from src.query.expansion import QueryExpander
from src.query.intent import DEFAULT_TRAINING_DATA, IntentClassifier
from src.query.ner import EntityRecognizer
from src.query.pipeline import QueryPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("query_cli")


def train_intent_command(args):
    """Train intent classifier and save to directory."""
    if args.data_file:
        with open(args.data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        logger.info("Using default domain intent training data.")
        data = DEFAULT_TRAINING_DATA

    classifier = IntentClassifier(c_param=args.c_param)
    classifier.train(data)
    classifier.save(args.output_dir)
    print(f"\nSuccessfully trained and saved intent classifier to: {args.output_dir}")


def analyze_query_command(args):
    """Analyze a single query with intent, NER, and expansion."""
    intent_clf = None
    if args.intent_model:
        intent_clf = IntentClassifier.load(args.intent_model)

    pipeline = QueryPipeline(
        intent_classifier=intent_clf,
        enable_intent=bool(intent_clf),
        enable_ner=True,
        enable_expansion=True,
    )

    processed = pipeline.process_query(args.query)
    print("\n=== Query Analysis Report ===")
    print(json.dumps(processed.to_dict(), indent=2))


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Query Understanding CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train intent
    train_parser = subparsers.add_parser("train-intent", help="Train intent classifier")
    train_parser.add_argument("--data-file", help="Path to JSON file with training queries")
    train_parser.add_argument("--output-dir", "-o", required=True, help="Output directory to save model")
    train_parser.add_argument("--c-param", type=float, default=1.0, help="SVM regularization parameter C")
    train_parser.set_defaults(func=train_intent_command)

    # Analyze query
    analyze_parser = subparsers.add_parser("analyze", help="Analyze query understanding stages")
    analyze_parser.add_argument("--query", "-q", required=True, help="Input query to analyze")
    analyze_parser.add_argument("--intent-model", help="Path to saved intent model directory")
    analyze_parser.set_defaults(func=analyze_query_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
