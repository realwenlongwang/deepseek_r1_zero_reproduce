# Dataset Validation Tests

This directory contains tests that validate the `src/data/dataset.py` implementation against **real datasets** downloaded from Hugging Face.

## Test Files

### `test_real_dataset_exact_pattern.py`
- **Purpose**: Implements the exact testing pattern requested by the user
- **Pattern**: 
  ```python
  @pytest.fixture(scope="session")
  def dataset():
      return load_from_disk("data/numina_math_tir")

  @pytest.mark.parametrize("split", ["train", "test"])
  def test_dataset_structure_complete(dataset, split):
      for idx, sample in enumerate(dataset[split]):
          res = validate_dataset_format(sample)
          assert res["has_required_fields"], f"row {idx} missing: {res['missing_fields']}"
          assert res["correct_prompt_format"], f"row {idx} bad prompt"
  ```
- **Real Data**: Downloads actual "AI-MO/NuminaMath-TIR" dataset (72,441 train + 99 test samples)
- **Validation**: Tests every sample for required fields (`problem`, `prompt`) and correct conversation format

### `test_real_dataset_download.py`
- **Purpose**: Comprehensive testing of real dataset download and processing
- **Features**:
  - Tests actual dataset loading from Hugging Face
  - Validates dataset structure and content quality
  - Analyzes real sample content
  - Tests dataset processing pipeline
- **Real Data**: Uses actual downloaded datasets without mocking

## Usage

Run the exact pattern test:
```bash
uv run pytest tests/test_real_dataset_exact_pattern.py::test_dataset_structure_complete -v
```

Run comprehensive dataset tests:
```bash
uv run pytest tests/test_real_dataset_download.py -v
```

Run all dataset tests:
```bash
uv run pytest tests/test_real_dataset* -v
```

## Validation Results

✅ **All tests pass** with real downloaded datasets:
- Required fields: `problem` and `prompt` are present in all samples
- Prompt format: Correct conversation structure (system + user messages)
- Dataset size: 72,441 train samples + 99 test samples validated
- Content quality: Real mathematical problems correctly processed

## Notes

- These tests download real datasets on first run (may take time)
- Datasets are cached locally after first download
- Tests validate the actual implementation against real data
- No mocking is used - pure real dataset validation