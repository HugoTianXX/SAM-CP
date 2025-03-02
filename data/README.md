# Dataset Template for Fine-Tuning Tasks

This repository provides a JSON-formatted dataset template designed for fine-tuning tasks. The dataset includes original text and a set of candidate replacement words for specific positions in the text. Below, you will find a detailed explanation of the data format, an example, and instructions on how to use the dataset.

---

## Data Format

Each entry in the dataset is a dictionary with the following structure:

```json
{
  "text": "The original text.",
  "replace": {
    "position_id": ["candidate_word_1", "candidate_word_2", ...]
  }
}
```

### Fields Description

- **`text`**: The original text string.
- **`replace`**: A dictionary where:
  - The key is a `position_id` (string) indicating the position of a word in the text (starting from 0).
  - The value is a list of candidate replacement words for the word at the specified position.

## Example Dataset

Below is an example of the dataset format:

```json
[
  {
    "text": "and made even more so by the largely depressingly tragic nature of the film...",
    "replace": {
      "1": ["made", "had", "set", "did", "played"],
      "19": ["judge", "criticize", "evaluate", "weigh"],
      "20": ["some", "approximately", "roughly"]
    }
  },
  {
    "text": "after all these years i still consider this series the finest example of world war ii documentary film making...",
    "replace": {
      "3": ["years", "ages", "centuries"],
      "5": ["still", "even", "but", "also"],
      "6": ["consider", "regard", "value", "think"]
    }
  }
]
```

### Explanation of the Example

- In the first entry:
  - The word at position `1` in the text (`"made"`) can be replaced with any of the candidates: `"made"`, `"had"`, `"set"`, `"did"`, or `"played"`.
  - The word at position `19` (`"comment"`) can be replaced with `"judge"`, `"criticize"`, `"evaluate"`, or `"weigh"`.
- In the second entry:
  - The word at position `3` (`"years"`) can be replaced with `"years"`, `"ages"`, or `"centuries"`.

## How to Use

1. **Prepare Your Data**:
   - Replace the placeholders in the JSON template with your own data.
     - `Template_Finetune_data_with_gpt_ranking.json`
   - Ensure the structure adheres to the provided format.
2. **Load the Dataset**:
   - Use the dataset in conjunction with the code provided in this repository.
3. **Example File**:
   - A complete example file is provided in `Template_Finetune_data_with_gpt_ranking.json`. Refer to it for a detailed example of the dataset structure.

The code for data preparation will be organized into a pipeline and uploaded soon.
