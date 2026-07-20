from pathlib import Path


PROMPTS_DIR = Path(__file__).resolve().parent


def load_prompt(filename: str, /, **variables: str) -> str:
    """Load and format a UTF-8 prompt from the prompts package."""
    prompt_path = (PROMPTS_DIR / filename).resolve()
    if prompt_path.parent != PROMPTS_DIR:
        raise ValueError(f"Prompt must be stored in {PROMPTS_DIR}")

    template = prompt_path.read_text(encoding="utf-8").strip()
    try:
        return template.format(**variables)
    except KeyError as error:
        missing_variable = error.args[0]
        raise ValueError(
            f"Missing variable {missing_variable!r} for prompt {filename!r}"
        ) from error
