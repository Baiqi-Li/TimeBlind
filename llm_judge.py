#!/usr/bin/env python3
"""
Extract answers from model_output using GPT, and overwrite model_output with the raw extracted string.

Usage:
    python extract_answers.py \
        --result_file results/glm_v_timeblind.json \
        --benchmark_file TimeBlind/data.jsonl \
        --output_file results/glm_v_timeblind.json

Arguments:
    --result_file: Path to the result file containing model_output
    --benchmark_file: Path to the benchmark data file (JSONL format)
    --output_file: Path to the output file (optional; default: append _extracted)
    --api_key: OpenAI API key (optional; defaults to env var OPENAI_API_KEY)
    --model: GPT model to use
    --max_workers: Number of parallel worker threads (default: 5)
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm


def load_benchmark_data(benchmark_file: str) -> dict[int, dict]:
    """Load benchmark data and build a mapping from index -> data."""
    index_to_data: dict[int, dict] = {}
    with open(benchmark_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            index_to_data[data["index"]] = data
    return index_to_data


def load_result_data(result_file: str) -> list[dict]:
    """Load result data."""
    with open(result_file, "r", encoding="utf-8") as f:
        return json.load(f)


def create_extraction_prompt(model_output: str, question_type: str) -> str:
    """Create a strict extraction prompt based on the question type."""
    if question_type == "yes_no":
        return f"""You must extract the final answer from the model output.

STRICT OUTPUT FORMAT (REQUIRED):
- Output exactly one token: yes OR no
- Use lowercase only.
- Do NOT output anything else (no words, no punctuation, no quotes, no explanation).
- If you are uncertain, choose the most likely option.

Model output:
{model_output}
"""
    elif question_type == "multiple_choice":
        return f"""You must extract the final answer from the model output.

STRICT OUTPUT FORMAT (REQUIRED):
- Output exactly one character: A OR B
- Use uppercase only.
- Do NOT output anything else (no words, no punctuation, no quotes, no explanation).
- If you are uncertain, choose the most likely option.

Model output:
{model_output}
"""
    else:
        raise ValueError(f"Unknown question type: {question_type}")


def extract_answer_with_gpt(
    client: OpenAI,
    model_output: str,
    question_type: str,
    model: str = "gpt-5",
    max_retries: int = 4,
) -> str:
    """
    Extract the answer using GPT.
    Returns the raw model text (stripped) or '' on failure.
    """
    prompt = create_extraction_prompt(model_output, question_type)
    system_msg = "You are a strict answer-extraction engine. Follow the output format rules exactly."

    last_text = ""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=10,
                temperature=0,
            )
            last_text = (response.choices[0].message.content or "").strip()
            if last_text:
                return last_text
        except Exception:
            pass

        time.sleep(min(8.0, 0.5 * (2**attempt)))

    return ""


def process_single_item(
    client: OpenAI,
    item: dict,
    benchmark_data: dict[int, dict],
    model: str,
) -> dict:
    """
    Process a single item and overwrite item["model_output"] with GPT's raw extracted string.
    No extra fields are added.
    """
    index = item.get("index")
    original_output = item.get("model_output", "")

    # If index not in benchmark data, cannot determine type -> set empty
    if index not in benchmark_data:
        item["model_output"] = ""
        return item

    question_type = benchmark_data[index].get("type", "")
    if question_type not in {"yes_no", "multiple_choice"}:
        item["model_output"] = ""
        return item

    if not original_output:
        item["model_output"] = ""
        return item

    extracted_raw = extract_answer_with_gpt(client, original_output, question_type, model)
    item["model_output"] = extracted_raw
    return item


def main():
    parser = argparse.ArgumentParser(
        description="Extract answers from model_output using GPT (overwrite model_output with raw extracted text)"
    )
    parser.add_argument(
        "--result_file",
        type=str,
        required=True,
        help="Path to the result file containing model_output",
    )
    parser.add_argument(
        "--benchmark_file",
        type=str,
        required=True,
        help="Path to the benchmark data file (JSONL format)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to the output file (default: append _extracted to the input filename)",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="GPT model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=5,
        help="Number of parallel worker threads (default: 5)",
    )

    args = parser.parse_args()

    # Set output filename
    if args.output_file is None:
        base, ext = os.path.splitext(args.result_file)
        args.output_file = f"{base}_extracted{ext}"

    # Initialize OpenAI client
    client = OpenAI(api_key=args.api_key) if args.api_key else OpenAI()

    # Load data
    print(f"Loading benchmark data: {args.benchmark_file}")
    benchmark_data = load_benchmark_data(args.benchmark_file)
    print(f"Loaded {len(benchmark_data)} benchmark items")

    print(f"Loading result data: {args.result_file}")
    result_data = load_result_data(args.result_file)
    print(f"Loaded {len(result_data)} result items")

    # Parallel processing
    print(f"Starting answer extraction (using {args.max_workers} parallel threads)...")
    processed_data: list[tuple[int, dict]] = []

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {}
        for i, item in enumerate(result_data):
            futures[executor.submit(process_single_item, client, item, benchmark_data, args.model)] = i

        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting answers"):
            i = futures[future]
            try:
                result = future.result()
            except Exception:
                # Keep structure; just blank out model_output on unexpected errors
                result = dict(result_data[i]) if isinstance(result_data[i], dict) else {"index": None}
                result["model_output"] = ""
            processed_data.append((i, result))

    # Sort back to original order
    processed_data.sort(key=lambda x: x[0])
    processed_items = [item for _, item in processed_data]

    # Save results
    print(f"Saving results to: {args.output_file}")
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(processed_items, f, ensure_ascii=False, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
