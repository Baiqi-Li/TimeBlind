<div align="center">
  <h2>[EMNLP 2026] TimeBlind: A Spatio-Temporal Compositionality Benchmark for Video LLMs</h2>
</div>

<div align="center">
  <b>Baiqi Li</b><sup>1</sup>&nbsp;&nbsp;
  <b>Kangyi Zhao</b><sup>2</sup>&nbsp;&nbsp;
  <b>Ce Zhang</b><sup>1</sup>&nbsp;&nbsp;
  <b>Chancharik Mitra</b><sup>3</sup>&nbsp;&nbsp;
  <b>Jean de Dieu Nyandwi</b><sup>3</sup>&nbsp;&nbsp;
  <b>Gedas Bertasius</b><sup>1</sup>
  <br><br>
  <sup>1</sup>University of North Carolina at Chapel Hill&nbsp;&nbsp;&nbsp;
  <sup>2</sup>University of Pittsburgh&nbsp;&nbsp;&nbsp;
  <sup>3</sup>Carnegie Mellon University
  <br><br>
  ✉️ Corresponding author:
  <a href="mailto:baiqili@cs.unc.edu">baiqili@cs.unc.edu</a> ·
  <a href="mailto:libaiqi123@gmail.com">libaiqi123@gmail.com</a>
</div>

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
from utils import _load_json_list, build_answers, get_scores, get_tag_scores, add_question_suffix

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

# Per-tag scores (11 fine-grained + 3 coarse-grained categories):
tag_scores = get_tag_scores(predictions, data, "tags.json")
print(tag_scores)  # {tag: {'Q_Acc': ..., 'V_Acc': ..., 'Acc': ..., 'I_Acc': ..., 'num_instances': ...}}
```

> **Note:** For long model outputs (e.g., reasoning/thinking models that produce lengthy chains of thought), the built-in regex extraction may fail to locate the final answer. In that case, please use LLM-as-a-judge (`llm_judge.py`) to extract answers before scoring.

## Metrics
I-Acc serves as our primary metric.

- **Acc**: Binary VQA accuracy
- **Q_Acc**: Question accuracy
- **V_Acc**: Video accuracy
- **I_Acc**: **Instance accuracy**

## Tags

`tags.json` assigns every instance to one of 11 fine-grained categories, grouped into 3 coarse-grained categories:

| Coarse-grained | Fine-grained |
|---|---|
| Event | State_Transition, Fine-Grained_Actions |
| Event_Attribute | Direction, Speed, Force, Duration, Magnitude, Repetition |
| Structural Event Logic | Temporal_Relation, Cross-Event_Comparison, Causal_Contingency |

Each tag maps to a list of dataset indices. Use `get_tag_scores` (see Evaluation above) to compute per-category results.

## Limitation:
While TimeBlind offers a rigorous diagnostic for compositional spatio-temporal understanding, it has several limitations. Some sub-categories contain few samples, and some Event Attribute sub-categories (e.g., Force, Magnitude) involve inherently subjective judgments. Despite substantial inter-annotator agreement (κ = 0.84 for Event Attributes), some ambiguity remains intrinsic to these concepts. We view these as natural extensions of TimeBlind rather than challenges to its conclusions.