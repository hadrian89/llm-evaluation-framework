
from src.datasets.builder import EvalDataset
from src.datasets.golden_set import GoldenSetManager
from src.datasets.synthetic import SyntheticGenerator


def test_dataset_from_records():
    records = [
        {"question": "Q1", "answer": "A1", "contexts": ["C1"], "extra_field": "x"},
        {"question": "Q2", "answer": "A2"},
    ]
    dataset = EvalDataset.from_records(records)
    assert len(dataset) == 2
    assert dataset[0].contexts == ["C1"]
    assert dataset[0].metadata["extra_field"] == "x"


def test_dataset_json_roundtrip(tmp_path):
    dataset = EvalDataset.from_records([{"question": "Q", "answer": "A", "contexts": ["C"]}])
    path = tmp_path / "cases.json"
    dataset.to_json(path)

    loaded = EvalDataset.from_json(path)
    assert len(loaded) == 1
    assert loaded[0].question == "Q"


def test_dataset_from_csv(tmp_path):
    csv_path = tmp_path / "cases.csv"
    csv_path.write_text("question,answer,contexts\nQ1,A1,C1|C2\n", encoding="utf-8")
    dataset = EvalDataset.from_csv(csv_path)
    assert len(dataset) == 1
    assert dataset[0].contexts == ["C1", "C2"]


def test_dataset_filter_and_sample():
    dataset = EvalDataset.from_records([{"question": f"Q{i}", "answer": "A"} for i in range(10)])
    filtered = dataset.filter(lambda c: c.question in {"Q1", "Q2"})
    assert len(filtered) == 2

    sampled = dataset.sample(3, seed=42)
    assert len(sampled) == 3


def test_golden_set_add_and_lookup():
    manager = GoldenSetManager()
    entry = manager.add("Q", "GT", contexts=["C"], tags=["rag"])
    assert len(manager) == 1
    assert manager.get(entry.entry_id) is entry
    assert manager.by_tag("rag") == [entry]


def test_golden_set_persistence_roundtrip(tmp_path):
    manager = GoldenSetManager()
    manager.add("What is 2+2?", "4", contexts=["Math basics."])
    path = tmp_path / "golden.json"
    manager.save(path)

    loaded = GoldenSetManager.load(path)
    assert len(loaded) == 1
    assert loaded.entries[0].question == "What is 2+2?"


def test_golden_set_load_missing_file(tmp_path):
    manager = GoldenSetManager.load(tmp_path / "does_not_exist.json")
    assert len(manager) == 0


def test_golden_set_to_eval_cases():
    manager = GoldenSetManager()
    entry = manager.add("Q", "GT", contexts=["C"])
    cases = manager.to_eval_cases({entry.entry_id: "the actual answer"})
    assert len(cases) == 1
    assert cases[0].answer == "the actual answer"
    assert cases[0].ground_truth == "GT"


def test_golden_set_to_eval_cases_skips_missing_answers():
    manager = GoldenSetManager()
    manager.add("Q", "GT")
    cases = manager.to_eval_cases({})
    assert cases == []


def test_synthetic_generator_offline():
    generator = SyntheticGenerator()
    manager = generator.generate(["Paris is the capital of France."])
    assert len(manager) == 1
    assert manager.entries[0].ground_truth == "Paris is the capital of France."
    assert "synthetic" in manager.entries[0].tags


def test_synthetic_generator_from_document():
    generator = SyntheticGenerator()
    document = "Sentence one. Sentence two. Sentence three. Sentence four."
    manager = generator.generate_from_document(document, chunk_size=2)
    assert len(manager) == 2
