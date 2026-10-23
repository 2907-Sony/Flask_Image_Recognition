"""Quality gate for the project's code.

Runs Pylint and exits with an error if the score is under THRESHOLD,
so the CI pipeline fails when code quality drops.
"""
import os
import sys

from pylint import lint

THRESHOLD = 9.0
FILES_TO_CHECK = ["app.py", "model.py", "lint.py", "tests"]

# Only lint paths that exist (the tests folder is added later in the project)
existing = [path for path in FILES_TO_CHECK if os.path.exists(path)]
run = lint.Run(existing, exit=False)
score = run.linter.stats.global_note

if score < THRESHOLD:
    print(f"Linter failed: score {score:.2f} is below the threshold of {THRESHOLD}")
    sys.exit(1)

print(f"Linter passed: score {score:.2f} meets the threshold of {THRESHOLD}")
sys.exit(0)
