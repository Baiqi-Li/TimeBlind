import json
from utils import _load_json_list, build_answers, get_scores, add_question_suffix

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
#llm_judge.py: If using a thinking model makes the output too complex, it's better to use an LLM for matching before scoring.
scores = get_scores(answers)
print(scores)
