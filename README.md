<div align="center">
  <h2>TimeBlind: A Spatio-Temporal Compositionality Benchmark for Video LLMs</h2>
</div>

<div align="center"> Baiqi Li<sup>1</sup>, Kangyi Zhao<sup>2</sup>, Ce Zhang<sup>1</sup>, Chancharik Mitra<sup>3</sup>, Jean de Dieu Nyandwi<sup>3</sup>, Gedas Bertasius<sup>1</sup> </div> <div align="center"> <sup>1</sup>University of North Carolina at Chapel Hill&nbsp;&nbsp; <sup>2</sup>University of Pittsburgh&nbsp;&nbsp; <sup>3</sup>Carnegie Mellon University </div>


<div align="center">

[🏠**Home Page**](https://baiqi-li.github.io/timeblind_project/) | [🤗**HuggingFace**](https://huggingface.co/datasets/BaiqiL/TimeBlind) | [**📖Paper**(coming soon)]() | [🖥️ **Code**](https://github.com/Baiqi-Li/TimeBlind)

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
I-Acc serves as our primary metric.

- **Acc**: Binary VQA accuracy
- **Q_Acc**: Question accuracy
- **V_Acc**: Video accuracy
- **I_Acc**: **Instance accuracy**

