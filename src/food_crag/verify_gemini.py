"""Explicit live check; unit tests never invoke this paid-service command."""
import argparse
import json

from .config import Settings
from .pipeline import RecommendationPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file")
    args = parser.parse_args()
    settings = Settings.from_env(args.env_file)
    result = RecommendationPipeline(settings=settings).handle("live-check", "想吃麵，不要火鍋")
    evaluation = result.evaluation
    print(json.dumps({"model": settings.gemini_model, "model_called": evaluation.model_called,
                      "decision": evaluation.decision, "evidence_state": evaluation.evidence_state,
                      "accepted_ids": evaluation.accepted_ids, "attempts": len(result.attempts),
                      "reasons": evaluation.reasons}, ensure_ascii=False, indent=2))
    return 0 if evaluation.model_called and evaluation.decision == "accept" else 1


if __name__ == "__main__":
    raise SystemExit(main())
