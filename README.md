# TimeBlind Benchmark

TimeBlind: A video VQA benchmark for evaluating temporal understanding in vision-language models.

<div align="center">
| [🏠**Home Page**(coming soon)]() | [&#129303;**HuggingFace**](https://huggingface.co/datasets/BaiqiL/TimeBlind) | [**📖Paper**(coming soon)]() | [🖥️ **Code**](https://github.com/Baiqi-Li/TimeBlind) |
</div>

## Setup

```bash
git clone https://github.com/Baiqi-Li/TimeBlind.git
cd TimeBlind
git clone https://huggingface.co/datasets/BaiqiL/TimeBlind
```

## Data Format

Each sample in `TimeBlind/data.jsonl` contains:
- `video_path`: path to video file (e.g., `TimeBlind/videos/vid_00000_0.mp4`)
- `question`: the question
- `answer`: the grounding answer
- `type`: `"yes_no"` or `"multiple_choice"`

## Evaluation

see evaluate.py for more details!

```python
import json
from utils import _load_json_list, build_answers, get_scores, add_question_suffix

data = _load_json_list("TimeBlind/data.jsonl")
predictions = []

for sample in data:
    video_path = sample["video_path"]  
    question = add_question_suffix(sample["question"], sample["type"])
    
    # Replace with your model inference
    model_output = your_model(video_path, question)
    
    predictions.append({
        "index": sample["index"],
        "video_path": video_path,
        "question": question,
        "model_output": model_output,
    })

json.dump(predictions, open("predictions.json", "w"), indent=2)

answers = build_answers(predictions, data)
scores = get_scores(answers)
print(scores)  # {'Q_Acc': ..., 'V_Acc': ..., 'Acc': ..., 'I_Acc': ...}
```

## Metrics

- **Acc**: Binary VQA accuracy
- **Q_Acc**: Question accuracy
- **V_Acc**: Video accuracy
- **I_Acc**: Instance accuracy

