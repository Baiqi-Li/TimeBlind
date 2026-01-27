import os
import re
import json
import unicodedata
import argparse
from typing import Dict, Any


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_answer(output_string: str, task_type: str = "yes_no") -> int:
    """
    Extract answer from model output.
    Returns: 1 (yes/A), 0 (no/B), -1 (invalid)
    """
    if not output_string or not str(output_string).strip():
        return -1

    if task_type not in ("yes_no", "multiple_choice"):
        raise ValueError("task_type must be 'yes_no' or 'multiple_choice'")

    text = _normalize(output_string)

    if task_type == "yes_no":
        patterns = [
            r"(?i)(?:final(?: answer)?|answer|prediction)\s*[:：]\s*(yes|no)\b",
            r"(?i)\b(yes|no)\b(?=[\s\.\,\!\?\)]|$)",
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                return 1 if m.group(1).lower() == "yes" else 0
        return -1

    else:  # multiple_choice
        patterns = [
            r"(?i)(?:final(?: answer)?|answer|prediction)\s*[:：]\s*([AB])\b",
            r"(?i)(?:option|choice)\s*[:：]?\s*([AB])\b",
            r"(?i)[\(\[\{]\s*([AB])\s*[\)\]\}]",
            r"(?i)\b([AB])\s*[\.\)]\b",
            r"(?i)(?<![A-Za-z0-9/])([AB])(?![A-Za-z0-9/])",
        ]
        for i, pat in enumerate(patterns):
            if i < len(patterns) - 1:
                m = re.search(pat, text)
                if m:
                    return 1 if m.group(1).upper() == "A" else 0
            else:
                anchor = re.search(r"(?i)(?:final(?: answer)?|answer|prediction)\s*[:：]\s*([AB])\b", text)
                if anchor:
                    return 1 if anchor.group(1).upper() == "A" else 0
                
                answer_keyword = re.search(r"(?i)\b(final(?: answer)?|answer|prediction)\b", text)
                if answer_keyword:
                    before = text[:answer_keyword.start()]
                    m_before = re.search(pat, before)
                    if m_before:
                        return 1 if m_before.group(1).upper() == "A" else 0
                    
                    after = text[answer_keyword.end():]
                    m_after = re.search(pat, after)
                    if m_after:
                        return 1 if m_after.group(1).upper() == "A" else 0
                
                m = re.search(pat, text)
                if m:
                    return 1 if m.group(1).upper() == "A" else 0
        return -1


def add_question_suffix(question: str, task_type: str) -> str:
    """Add answer format suffix based on task type."""
    if task_type == "yes_no":
        return question + " Please output Yes or No."
    elif task_type == "multiple_choice":
        return question + " Please output A or B."
    return question


def _load_json_list(path: str) -> list:
    """Load JSON array or JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        first_char = ""
        for ch in f.read():
            if not ch.isspace():
                first_char = ch
                break
        f.seek(0)
        
        if not first_char:
            data = []
        elif first_char == "[":
            data = json.load(f)
        else:
            data = [json.loads(line) for line in f if line.strip()]
    
    return [
        {
            "index": item.get("index"),
            "video_path": item.get("video_path"),
            "question": item.get("question"),
            "answer": item.get("answer"),
            "type": item.get("type"),
        }
        for item in data
    ]


def read_json_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_scores(scores):
    """
    Calculate Q_Acc, V_Acc, Acc, I_Acc from answer results.
    
    Args:
        scores: dict or list with q0_i0, q0_i1, q1_i0, q1_i1 values (1 or 0)
    """
    Q_Acc = V_Acc = Acc = I_Acc = 0.0
    num_samples = len(scores)

    def calc_video_score(r):
        score = 0
        if isinstance(r, dict):
            if r["q0_i0"] == 1.0 and r["q1_i0"] == 0.0:
                score += 1
            if r["q1_i1"] == 1.0 and r["q0_i1"] == 0.0:
                score += 1
        else:
            if r[0] == 1.0 and r[2] == 0.0:
                score += 1
            if r[3] == 1.0 and r[1] == 0.0:
                score += 1
        return score
    
    def calc_question_score(r):
        score = 0
        if isinstance(r, dict):
            if r["q0_i0"] == 1.0 and r["q0_i1"] == 0.0:
                score += 1
            if r["q1_i1"] == 1.0 and r["q1_i0"] == 0.0:
                score += 1
        else:
            if r[0] == 1.0 and r[1] == 0.0:
                score += 1
            if r[3] == 1.0 and r[2] == 0.0:
                score += 1
        return score

    def calc_binary_score(r):
        if isinstance(r, dict):
            return sum([
                r["q0_i0"] == 1.0,
                r["q0_i1"] == 0.0,
                r["q1_i0"] == 0.0,
                r["q1_i1"] == 1.0,
            ])
        return sum([r[0] == 1.0, r[1] == 0.0, r[2] == 0.0, r[3] == 1.0])

    def calc_instance_score(r):
        return 1 if calc_question_score(r) == 2 and calc_video_score(r) == 2 else 0
    
    results = scores.values() if isinstance(scores, dict) else scores
    for r in results:
        Q_Acc += calc_question_score(r)
        V_Acc += calc_video_score(r)
        Acc += calc_binary_score(r)
        I_Acc += calc_instance_score(r)

    return {
        'Q_Acc': Q_Acc / (num_samples * 2),
        'V_Acc': V_Acc / (num_samples * 2),
        'Acc': Acc / (num_samples * 4),
        'I_Acc': I_Acc / num_samples
    }


def build_answers(predictions, dataset):
    """Build answers dict from predictions and dataset.
    
    Correctly handles shuffled predictions and partial data.
    """
    answers: Dict[str, Dict[str, float]] = {}
    
    # Group predictions by sample_id (index // 4)
    # role: 0=q0_i0, 1=q0_i1, 2=q1_i0, 3=q1_i1
    sample_groups: Dict[int, Dict[int, dict]] = {}
    for pred in predictions:
        idx = pred.get('index')
        if idx is None:
            continue
        sample_id = idx // 4
        role = idx % 4
        
        if sample_id not in sample_groups:
            sample_groups[sample_id] = {}
        sample_groups[sample_id][role] = pred
    
    max_idx = len(dataset) - 1
    
    for sample_id, group in sample_groups.items():
        try:
            # Check if we have all 4 roles
            if len(group) != 4 or not all(r in group for r in [0, 1, 2, 3]):
                raise ValueError(f"Sample {sample_id} is incomplete")
            
            # Get predictions by role (not by array position)
            pred_q0_i0 = group[0]
            pred_q0_i1 = group[1]
            pred_q1_i0 = group[2]
            pred_q1_i1 = group[3]
            
            idx_0 = pred_q0_i0.get('index')
            idx_1 = pred_q0_i1.get('index')
            idx_2 = pred_q1_i0.get('index')
            idx_3 = pred_q1_i1.get('index')
            
            if not all(0 <= idx <= max_idx for idx in [idx_0, idx_1, idx_2, idx_3]):
                raise IndexError("Index out of bounds")
            
            q0_i0 = extract_answer(pred_q0_i0.get('model_output'), dataset[idx_0]["type"])
            q0_i1 = extract_answer(pred_q0_i1.get('model_output'), dataset[idx_1]["type"])
            q1_i0 = extract_answer(pred_q1_i0.get('model_output'), dataset[idx_2]["type"])
            q1_i1 = extract_answer(pred_q1_i1.get('model_output'), dataset[idx_3]["type"])
        except Exception:
            q0_i0 = q0_i1 = q1_i0 = q1_i1 = -1

        answers[str(sample_id)] = {
            "q0_i0": float(q0_i0),
            "q0_i1": float(q0_i1),
            "q1_i0": float(q1_i0),
            "q1_i1": float(q1_i1),
        }
    
    return answers


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate scores for TimeBlind")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--results_dir", type=str, default=os.path.join(base_dir, "results"))
    parser.add_argument("--dataset_path", type=str, default=os.path.join(base_dir, "TimeBlind", "data.jsonl"))
    args = parser.parse_args()

    dataset = _load_json_list(args.dataset_path)
    if not dataset:
        raise RuntimeError(f"Failed to load dataset: {args.dataset_path}")

    if os.path.isfile(args.results_dir) and args.results_dir.endswith(".json"):
        predictions = read_json_file(args.results_dir)
        if not predictions:
            raise RuntimeError(f"Invalid predictions file: {args.results_dir}")
        
        answers = build_answers(predictions, dataset)
        scores = get_scores(answers)
        result = {**scores, "num_instances": len(answers)}
        
        file_dir = os.path.dirname(os.path.abspath(args.results_dir))
        file_name = os.path.splitext(os.path.basename(args.results_dir))[0]
        output_path = os.path.join(file_dir, f"score_{file_name}.json")
        
        os.makedirs(file_dir, exist_ok=True)
        save_json_file(output_path, result)
        print(f"Saved scores to: {output_path}")
    else:
        output_path = os.path.join(args.results_dir, "scores/scores.json")
        aggregated: Dict[str, Any] = {}
        
        for fname in sorted(os.listdir(args.results_dir)):
            fpath = os.path.join(args.results_dir, fname)
            if not os.path.isfile(fpath) or not fname.endswith(".json") or fname == "scores.json":
                continue
            if os.path.getsize(fpath) == 0:
                continue
            try:
                predictions = read_json_file(fpath)
            except (json.JSONDecodeError, ValueError):
                continue
            
            if not isinstance(predictions, list) or not predictions:
                continue

            answers = build_answers(predictions, dataset)
            scores = get_scores(answers)
            aggregated[os.path.splitext(fname)[0]] = {**scores, "num_samples": len(predictions) // 4}

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_json_file(output_path, aggregated)
        print(f"Saved scores to: {output_path}")


if __name__ == "__main__":
    main()
