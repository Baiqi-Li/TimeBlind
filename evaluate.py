import json
from utils import _load_json_list, build_answers, get_scores, get_tag_scores, add_question_suffix

data = _load_json_list("TimeBlind/data.jsonl")
predictions = []

for sample in data:
    video_path = sample["video_path"]  
    question = add_question_suffix(sample["question"], sample["type"])
    
    # TODO: Replace with your model inference
    # model_output = your_model(video_path, question)
    model_output = "Yes"  
    
    predictions.append({
        "index": sample["index"],
        "video_path": video_path,
        "question": question,
        "model_output": model_output,
    })

json.dump(predictions, open("results/predictions.json", "w"), indent=2)

answers = build_answers(predictions, data)
# llm_judge.py: for long outputs (e.g., thinking models), use LLM-as-a-judge to extract answers before scoring.
scores = get_scores(answers)
print(scores)

# Per-tag scores: tags.json maps each tag (11 fine-grained + 3 coarse-grained) to dataset indices.
tag_scores = get_tag_scores(predictions, data, "tags.json")
print(json.dumps(tag_scores, indent=2))
