# Author: Naveen Muppana
"""Weekly self-improvement job: propose prompt changes from failures, then gate them on evals.

    python -m evals.run_evals            # propose + evaluate
    python -m evals.run_evals --no-propose
"""
import json
import logging
import sys

from assistant.evals import evaluate_candidates
from assistant.orchestrator import Assistant

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def main() -> None:
    a = Assistant()
    if "--no-propose" not in sys.argv:
        cid = a.learner.propose_prompt_improvements("planner")
        print("proposed:", cid or "nothing (not enough failures)")
    print(json.dumps(evaluate_candidates(a), indent=2))
    print(json.dumps(a.learner.digest(), indent=2))


if __name__ == "__main__":
    main()
