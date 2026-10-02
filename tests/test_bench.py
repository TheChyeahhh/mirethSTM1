"""Benchmark harness (bench/): metrics on hand-computed cases, the first-token tie rule, sampling,
JevBench item mapping, report rebuilds from saved records, and one tiny run on the test model."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

# The benchmark needs the bench extra (pip install -e ".[dev,bench]"); without it these tests skip.
# datasets fixes its cache path at import, before the home fixture swaps HOME.
pytest.importorskip("datasets")
pytest.importorskip("matplotlib")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repository root holds the bench package

from bench import arms, data, latency, metrics, multifield, public_items, report, run_bench  # noqa: E402
from bench.common import slug  # noqa: E402
from mirethstm import schema as sch  # noqa: E402
from mirethstm.calibration import ece  # noqa: E402
from mirethstm.engine import branch_ids, encode, shared_ids  # noqa: E402

from conftest import MODEL_ID, max_diff, tolerance  # noqa: E402


# --- metrics -------------------------------------------------------------------------------

PROBS = [np.array([0.7, 0.2, 0.1]), np.array([0.1, 0.6, 0.3]), np.array([0.5, 0.5, 0.0])]
GOLD = [0, 2, 1]


def test_rows_by_hand():
    r = metrics.rows(PROBS, GOLD)
    assert r["pred"].tolist() == [0, 1, 0]  # the 0.5 / 0.5 tie goes to the earlier label
    assert r["correct"].tolist() == [True, False, False]
    assert r["conf"].tolist() == pytest.approx([0.7, 0.6, 0.5])
    assert r["nll"].tolist() == pytest.approx([-math.log(0.7), -math.log(0.3), -math.log(0.5)])
    # (0.3^2 + 0.2^2 + 0.1^2), (0.1^2 + 0.6^2 + 0.7^2), (0.5^2 + 0.5^2 + 0)
    assert r["brier"].tolist() == pytest.approx([0.14, 0.86, 0.5])


def test_nll_clips_a_zero_probability():
    r = metrics.rows([np.array([1.0, 0.0])], [1])
    assert r["nll"][0] == pytest.approx(-math.log(1e-12))


def test_scored_means_by_hand():
    m = metrics.scored(PROBS, GOLD, n_resamples=0)
    assert m["accuracy"] == (pytest.approx(1 / 3), None, None)
    assert m["nll"][0] == pytest.approx((-math.log(0.7) - math.log(0.3) - math.log(0.5)) / 3)
    assert m["brier"][0] == pytest.approx(0.5)
    assert m["ece15"][0] == pytest.approx(ece([0.7, 0.6, 0.5], [1, 0, 0]))


def test_macro_f1_by_hand():
    # class 0: TP 1, support 2, predicted 1 -> 2/3; class 1: TP 1, support 2, predicted 2 -> 1/2;
    # class 2: 1. The invalid answer (-1) is wrong and predicts no class.
    assert metrics.macro_f1([0, 1, 1, -1, 2], [0, 0, 1, 1, 2]) == pytest.approx((2 / 3 + 1 / 2 + 1) / 3)
    # A class that is predicted but never gold counts with F1 = 0.
    assert metrics.macro_f1([0, 1], [0, 0]) == pytest.approx((2 / 3 + 0) / 2)
    assert metrics.macro_f1([-1, -1], [0, 1]) == 0.0
    assert metrics.macro_f1([2, 0, 1], [2, 0, 1]) == 1.0


def test_reliability_bins_sum_to_ece():
    rng = np.random.default_rng(1)
    conf = np.concatenate([rng.uniform(0, 1, 500), [0.0, 1.0, 1 / 15, 2 / 15, 0.1]])
    correct = rng.uniform(0, 1, conf.size) < conf
    for n_bins in (15, 10):
        bins = metrics.reliability_bins(conf, correct, n_bins)
        assert sum(b["n"] for b in bins) == conf.size
        total = sum(b["n"] / conf.size * abs(b["acc"] - b["conf"]) for b in bins if b["n"])
        assert total == pytest.approx(ece(conf, correct, n_bins=n_bins))
    edge = metrics.reliability_bins([1 / 15], [1], 15)
    assert edge[0]["n"] == 1  # bins are ((m-1)/M, m/M], as calibration.ece counts them


def test_bootstrap_ci():
    values = np.array([0, 1] * 50, dtype=float)
    stat = lambda i: values[i].mean()  # noqa: E731
    lo, hi = metrics.bootstrap_ci(stat, values.size, 500, seed=3)
    assert lo < 0.5 < hi
    assert (lo, hi) == metrics.bootstrap_ci(stat, values.size, 500, seed=3)
    assert metrics.bootstrap_ci(lambda i: 7.0, 10, 100) == (7.0, 7.0)


def test_ordinal_by_hand():
    out = metrics.ordinal([0, 2, 4], [1, 2, 2])  # errors -1, 0, 2
    assert out["mae"] == pytest.approx(1.0)
    assert out["rmse"] == pytest.approx(math.sqrt(5 / 3))
    assert metrics.expected_level(np.array([0.1, 0.2, 0.7])) == pytest.approx(1.6)


def test_accuracy_and_f1_do_not_depend_on_temperature():
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 3, (200, 5))
    logits[:20, 1] = logits[:20, 3]  # exact ties, as the first-token arm makes them
    gold = rng.integers(0, 5, 200)
    results = [metrics.scored([metrics.softmax(z, t) for z in logits], gold, n_resamples=0)
               for t in (0.25, 1.0, 4.0)]
    assert len({r["accuracy"] for r in results}) == 1
    assert len({r["macro_f1"] for r in results}) == 1
    assert len({round(r["nll"][0], 6) for r in results}) == 3  # NLL does move


# --- the first-token arm -------------------------------------------------------------------

QUESTIONS = {
    "topic": {"type": "choice", "instructions": "Which topic?",
              "criteria": {"Sci-Tech": None, "Sci/Tech": None, "Sports": None}},
    "refund": {"type": "noul", "instructions": "Refund?"},
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["Low", "Some", "High", "Now"]},
    "only": {"type": "choice", "instructions": "One option", "criteria": {"single": None}},
}


def test_first_token_suffix_ends_where_the_label_starts(tokenizer):
    plan = arms.first_token_plan(tokenizer, QUESTIONS)
    assert set(plan) == {"topic", "refund", "urgency"}  # one label: not scored, as in the engine
    for qid, q in QUESTIONS.items():
        if qid not in plan:
            continue
        suffix, first = plan[qid]
        assert suffix[:len(encode(tokenizer, sch.SUFFIX))] == encode(tokenizer, sch.SUFFIX)
        for label, cand, token in zip(sch.labels(q), sch.candidates(q), first):
            # Suffix plus the scored token is how the engine's own continuation begins ...
            assert (sch.SUFFIX + cand).startswith(tokenizer.decode(suffix + [token]))
            # ... and the scored token is the start of the label itself, not JSON syntax.
            start = tokenizer.decode([token]).lstrip(' "')
            assert start and label.startswith(start)
    assert tokenizer.decode(plan["topic"][0]).endswith('{"q1": "')


def test_first_token_ties(tokenizer):
    plan = arms.first_token_plan(tokenizer, QUESTIONS)
    labels = sch.labels(QUESTIONS["topic"])
    assert arms.tie_groups(labels, plan["topic"][1]) == [["Sci-Tech", "Sci/Tech"]]  # both start with " Sci"
    assert arms.tie_groups(["true", "false"], plan["refund"][1]) == []
    assert arms.tie_groups(sch.labels(QUESTIONS["urgency"]), plan["urgency"][1]) == []


def test_salvage_takes_the_first_allowed_value_of_its_own_key():
    # Output cut off in the middle (not valid JSON); q2 and q3 first hold a value that is not allowed.
    text = '{"q1": "Sports", "q2": 0, "q3": 7, "q2": true, "q3": 3, "q1": "Sci'
    topic, refund, urgency = (QUESTIONS[qid] for qid in ("topic", "refund", "urgency"))
    assert arms.salvage(text, 1, topic) == "Sports"
    assert arms.salvage(text, 2, refund) == "true"  # 0 is not an answer to a yes/no question
    assert arms.salvage(text, 3, urgency) == "3"    # 7 is not a level
    assert arms.salvage(text, 2, topic) is None and arms.salvage(text, 4, refund) is None


def test_tie_rule_and_its_count_in_the_report():
    tied = {"Sci-Tech": -1.0, "Sci/Tech": -1.0, "Sports": -2.0}
    records = [{"scores": tied, "gold": "Sci/Tech"},  # decided by the tie rule: the earlier option, wrong
               {"scores": {"Sci-Tech": -3.0, "Sci/Tech": -3.0, "Sports": -0.5}, "gold": "Sports"}]
    probs, gold = report.vectors(records)
    assert metrics.rows(probs, gold)["pred"].tolist() == [0, 2]
    assert probs[0][0] == probs[0][1]  # the tied options split their probability
    lines = report.tie_section({("m", "ag_news", "eval", "first_token"): records}, ["m"], ["ag_news"])
    assert "| m | ag_news | 2 | 2 | 2.0 | 1 | 0.000 |" in lines


# --- datasets ------------------------------------------------------------------------------


def test_stratified_by_hand():
    gold = [0] * 10 + [1] * 2 + [2] * 5
    picked = data.stratified(gold, 9, seed=5)
    counts = np.bincount(np.array(gold)[picked], minlength=3).tolist()
    assert counts == [4, 2, 3]  # turns in label order; class 1 runs out after two
    assert len(set(picked)) == 9
    assert picked == data.stratified(gold, 9, seed=5)
    assert sorted(data.stratified(gold, 100)) == list(range(17))
    even = data.stratified(gold, 6, keep=lambda i: i % 2 == 0)
    assert len(even) == 6 and all(i % 2 == 0 for i in even)


def test_clip_keeps_the_head_at_a_word_boundary():
    assert data.clip("short") == "short"
    text = "word " * 1000
    assert len(data.clip(text)) <= data.MAX_CHARS and data.clip(text).endswith("word")
    assert text.startswith(data.clip(text))


def _cached(name):
    try:
        return data.samples(name, 20, 20)
    except (ConnectionError, FileNotFoundError) as e:  # not in the local Hugging Face cache (tests run offline)
        pytest.skip(f"{name} is not cached: {e}")


def test_sst2_samples_are_fixed_balanced_and_disjoint():
    ev, cal = _cached("sst2")
    ev2, cal2 = data.samples("sst2", 20, 20)
    assert [(s.index, s.text, s.gold) for s in ev + cal] == [(s.index, s.text, s.gold) for s in ev2 + cal2]
    assert np.bincount([s.gold for s in ev]).tolist() == [10, 10]
    assert np.bincount([s.gold for s in cal]).tolist() == [10, 10]
    assert all(len(s.text.split()) >= 8 for s in cal)  # train is phrase level; short phrases are skipped
    assert not {s.text.strip() for s in cal} & {s.text.strip() for s in ev}
    # The yes/no and the choice form ask about the same sentences.
    assert [s.index for s in data.samples("sst2_choice", 20, 20)[0]] == [s.index for s in ev]


class _Split:
    """Just enough of a datasets.Dataset: a column by name, a row by index."""

    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, key):
        return [r[key] for r in self.rows] if isinstance(key, str) else self.rows[key]


def test_calibration_rows_never_repeat_evaluation_texts(monkeypatch):
    long = "one two three four five six seven eight {}"
    ev = _Split([{"sentence": long.format(i), "label": i % 2} for i in range(10)])
    cal = [{"sentence": f" {long.format(i)} ", "label": i % 2} for i in range(10)]  # every evaluation text again
    cal += [{"sentence": long.format(f"new{i}"), "label": i % 2} for i in range(6)]
    cal += [{"sentence": "too short", "label": i % 2} for i in range(4)]  # under SST-2's 8 words
    splits = {"validation": ev, "train": _Split(cal)}
    monkeypatch.setattr(data, "load_split", lambda spec, split: splits[split])
    _, picked = data.samples("sst2", 4, 100)
    assert sorted(s.text for s in picked) == sorted(long.format(f"new{i}") for i in range(6))


def test_banking77_label_names_and_strata():
    _cached("banking77")
    names = data.label_names("banking77")
    assert len(names) == 77
    assert names[0] == "activate_my_card" and names[-1] == "wrong_exchange_rate_for_cash_withdrawal"
    assert "refund_not_showing_up" in names and "reverted_card_payment" in names
    ev, _ = data.samples("banking77", 154, 0)
    assert np.bincount([s.gold for s in ev]).tolist() == [2] * 77


def test_questions_validate():
    for name in ("ag_news", "sst2", "sst2_choice", "yelp"):
        question, labels = data.question(name)
        sch.validate("text", {data.QID: question})
        assert sorted(labels) == sorted(sch.labels(question))
    assert data.question("sst2")[1] == ["false", "true"]  # by gold index: SST-2 label 1 is positive


# --- JevBench items and the latency schemas -------------------------------------------------

ITEMS = [
    {"id": "a", "family": "policy", "state": "s", "expected": "yes", "labels": ["no", "yes"],
     "question": {"type": "noul", "instructions": "Allowed?", "criteria": {"true": "ok", "false": "no"}}},
    {"id": "b", "family": "ordinal", "state": {"k": 1}, "expected": 2, "labels": ["0", "1", "2"],
     "question": {"type": "score", "instructions": "How bad?", "criteria": ["a", "b", "c"]}},
    {"id": "c", "family": "intent", "state": "s", "expected": "x", "labels": ["y", "x"],
     "question": {"type": "choice", "instructions": "Which?", "criteria": {"x": "X", "y": None}}},
    {"id": "d", "family": "fact", "state": "s", "expected": "no", "labels": ["no", "yes"],
     "question": {"type": "noul", "instructions": "True?", "criteria": None}},
]


def test_public_items_become_tasks():
    tasks = [public_items.to_task(i, item) for i, item in enumerate(ITEMS)]
    assert [t.gold for t in tasks] == ["true", "2", "x", "false"]
    assert tasks[1].state == {"k": 1}
    assert "criteria" not in tasks[3].questions["decision"]
    assert [t.extra["n_labels"] for t in tasks] == [2, 3, 2, 2]
    for t in tasks:
        sch.validate(t.state, t.questions)
        assert t.gold in sch.labels(t.questions["decision"])


def test_public_items_over_http(tmp_path):
    """Items go to POST /v1/<route> as TypeSafe requests; the returned probabilities are recorded."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    seen = []

    class Stub(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, body))
            q = body["questions"]["decision"]
            answer = ({"type": "noul", "noul": 0.73} if q["type"] == "noul" else
                      {"type": q["type"], "probabilities": {k: 1 / len(sch.labels(q)) for k in sch.labels(q)}})
            out = json.dumps({"model": "org/served", "answers": {"decision": answer}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        tasks = [public_items.to_task(i, item) for i, item in enumerate(ITEMS)]
        public_items.over_http(f"http://127.0.0.1:{server.server_port}/", "systemone", "easy", tasks, tmp_path)
    finally:
        server.shutdown()
    assert [path for path, _ in seen] == ["/v1/systemone"] * 4
    assert seen[1][1]["state"] == {"k": 1} and seen[1][1]["questions"] == tasks[1].questions
    records = [json.loads(line) for line in
               (tmp_path / "org--served" / "jevbench_easy.eval.systemone.jsonl").read_text(encoding="utf-8").splitlines()]
    assert records[0]["probs"] == {"true": 0.73, "false": pytest.approx(0.27)}
    assert records[1]["probs"] == {"0": pytest.approx(1 / 3), "1": pytest.approx(1 / 3), "2": pytest.approx(1 / 3)}
    assert [r["arm"] for r in records] == ["systemone"] * 4 and records[2]["n_labels"] == 2


def test_public_item_files_are_hash_checked(tmp_path):
    (tmp_path / "easy.jsonl").write_bytes(b'{"changed": true}\r\n')
    with pytest.raises(SystemExit, match="SHA-256"):
        public_items.fetch("easy", tmp_path)


def test_latency_schemas_and_rules():
    topic, names = data.question("ag_news")
    for n in latency.FIELD_COUNTS:
        questions = latency.field_schema(n, topic)
        sch.validate("text", questions)
        assert len(questions) == n
    text = 'Stocks rose 3% in 2004, said the "bank" (AP).'
    truth = latency.truth(latency.field_schema(20, topic), text, "Business")
    assert truth["topic"] == "Business"
    assert truth["check_01"] and not truth["check_02"] and truth["check_04"]  # digit, no "?", a quote
    with pytest.raises(ValueError):
        latency.field_schema(len(latency.SYNTHETIC) + 2, topic)


# --- many questions in one call ----------------------------------------------------------------


def test_multifield_modes_place_the_checked_questions():
    checked = list(multifield.CHECKED)
    alone, first, last, spread = (multifield.MODES[m][1] for m in ("alone", "first", "last", "spread"))
    assert [list(call) for call in alone] == [[qid] for qid in checked]
    for (call,), places in ((first, [1, 2, 3, 4, 5]), (last, [16, 17, 18, 19, 20]), (spread, [4, 8, 12, 16, 20])):
        sch.validate("text", call)
        assert len(call) == 20
        assert [list(call).index(qid) + 1 for qid in checked] == places
    assert multifield.truth("Sports") == {"topic": "Sports", "is_sports": "true", "is_business": "false",
                                          "is_scitech": "false", "is_world": "false"}
    assert sorted(multifield.TOPICS) == sorted(data.DATASETS["ag_news"].names)


def _multifield_records(mode, shift):
    """Two Sports articles: every answer right, except that `shift` > 1 turns is_world to yes in the second."""
    records = []
    for index in (0, 1):
        scores = {"topic": {"World": -3.0, "Sports": -0.1 - shift, "Business": -4.0, "Sci/Tech": -5.0}}
        for qid, name in multifield.IS_TOPIC.items():
            scores[qid] = {"true": -0.2, "false": -2.0} if name == "Sports" else {"true": -2.0, "false": -0.5}
        if index == 1:
            scores["is_world"] = {"true": -2.0 + shift, "false": -0.5}
        records.append({"model": "org/fake-model", "mode": mode, "index": index,
                        "truth": multifield.truth("Sports"), "scores": scores, "latency_ms": 50.0})
    return records


def test_multifield_table_by_hand():
    by_mode = {"alone": _multifield_records("alone", 0.0), "last": _multifield_records("last", 2.0)}
    header, rows = multifield.table(by_mode)
    assert header[:2] == ["Mode", "topic"] and [row[0] for row in rows] == [multifield.MODES[m][0] for m in by_mode]
    #                  topic    is_sports      is_business    is_scitech     is_world       mean
    assert rows[0][1:] == ["1.000", "1.000 [1.00]", "1.000 [0.00]", "1.000 [0.00]", "1.000 [0.00]", "1.000", "n/a", "n/a"]
    # One of two is_world answers flips to yes: 9 of 10 answers right and equal to the alone mode's;
    # the largest score difference is the 2.0 added to that score and taken from the two topic scores.
    assert rows[1][1:] == ["1.000", "1.000 [1.00]", "1.000 [0.00]", "1.000 [0.00]", "0.500 [0.50]", "0.900", "0.900", "2"]


# --- report from saved records (no model) -----------------------------------------------------


def _write(run, model, stem, records):
    from bench.common import slug, write_jsonl

    write_jsonl(run / slug(model) / f"{stem}.jsonl", records)


def test_report_rebuilds_from_records(tmp_path):
    rng = np.random.default_rng(0)
    run, model = tmp_path / "fake", "org/fake-model"
    by_split = {}
    labels = ["World", "Sports", "Business", "Sci/Tech"]
    for split, n in (("cal", 40), ("eval", 30)):
        records = by_split[split] = []
        for i in range(n):
            gold = int(rng.integers(0, 4))
            scores = rng.normal(0, 1, 4)
            scores[gold] += 2.0
            records.append({"model": model, "dataset": "ag_news", "split": split, "index": i, "gold": labels[gold],
                            "arm": "mireth", "scores": dict(zip(labels, scores.tolist())), "latency_ms": 10.0})
        _write(run, model, f"ag_news.{split}.mireth", records)
    _write(run, model, "ag_news.eval.baseline", [
        {"model": model, "dataset": "ag_news", "split": "eval", "index": i, "gold": "World", "arm": "baseline",
         "pred": [None, "World", "Sports"][i % 3], "salvaged": ["Business", "World", None][i % 3],
         "cap_hit": i % 3 == 0, "latency_ms": 500.0}
        for i in range(30)])
    _write(run, model, "latency.f05", [
        {"model": model, "setting": "5 fields", "scenario": None, "arm": arm, "run": i, "fields": 5,
         "latency_ms": base + i, "correct": 4, "peak_mem_mb": None,
         **({"input_tokens": 300} if arm == "mireth" else
            {"output_tokens": 40, "valid_json": True, "hallucinated": 0, "missing": 1})}
        for i in range(9) for arm, base in (("mireth", 20.0), ("baseline", 400.0))])
    for mode, shift in (("alone", 0.0), ("last", 2.0)):
        _write(run, model, f"multifield.{mode}", _multifield_records(mode, shift))
    # A second scoring arm, and the run "fake-fp32": the first 20 evaluation rows in another dtype, 0.8 added
    # to the second-highest score in ten of them, the calibration rows unchanged, and the many-questions run
    # without the flip.
    first = [{**r, "arm": "first_token", "scores": dict(zip(labels, np.random.default_rng(i).normal(0, 1, 4).tolist()))}
             for i, r in enumerate(by_split["eval"])]
    _write(run, model, "ag_news.eval.first_token", first)
    exact = []
    for r in by_split["eval"][:20]:
        second = sorted(r["scores"], key=r["scores"].get)[-2]
        exact.append({**r, "scores": {**r["scores"], second: r["scores"][second] + (0.8 if r["index"] < 10 else 0.0)}})
    _write(tmp_path / "fake-fp32", model, "ag_news.eval.mireth", exact)
    _write(tmp_path / "fake-fp32", model, "ag_news.cal.mireth", by_split["cal"])
    for mode in ("alone", "last"):
        _write(tmp_path / "fake-fp32", model, f"multifield.{mode}", _multifield_records(mode, 0.0))
    md = report.build(run, run / "benchmark.md", resamples=50)
    text = md.read_text(encoding="utf-8")
    assert "## Many questions in one call (SPEC 3.1)" in text and "### org/fake-model (AG News test, n = 2)" in text
    assert "### org/fake-model, fp32 (AG News test, n = 2)" in text and "| 1.000 [0.00] | 1.000 | 1.000 | 0 |" in text
    assert "| One call of 20, checked questions last (16 to 20) | 1.000 | 1.000 [1.00] |" in text
    assert "| 0.500 [0.50] | 0.900 | 0.900 | 2 |" in text
    assert "### ag_news (fancyzhx/ag_news test, 4 labels, n = 30)" in text
    assert "| org/fake-model | Normal generation | 30 | 0.333 [" in text  # 10 of 30 right, invalid counted wrong
    assert "| 0.667 | 0.333 | 0.333 |" in text  # valid; salvaged: 10 right, 10 wrong, 10 unreadable; token cap
    assert "## Calibration" in text and "[png](report/org--fake-model.ag_news.mireth.png)" in text
    assert (run / "report" / "org--fake-model.ag_news.mireth.png").stat().st_size > 1000
    bins = json.loads((run / "report" / "org--fake-model.ag_news.mireth.json").read_text(encoding="utf-8"))
    assert list(bins)[0] == "Before: T = 1" and len(bins) == 2
    assert "| 5 fields | 5 | 9 | 24.0 [" in text and "16.8x" in text  # p50 404 / 24 = 16.83
    assert "| 27.6 [" in text  # p95 of 20..28, with its CI
    assert "0.80 / 0.80" in text
    # The calibration table's T = 1 columns are the evaluation split's, computed here by hand.
    p = [np.exp(s - max(s)) / np.exp(s - max(s)).sum()
         for s in (np.array(list(r["scores"].values())) for r in by_split["eval"])]
    gold = [labels.index(r["gold"]) for r in by_split["eval"]]
    nll = float(np.mean([-math.log(pi[g]) for pi, g in zip(p, gold)]))
    ece15 = ece([pi.max() for pi in p], [int(np.argmax(pi)) == g for pi, g in zip(p, gold)], n_bins=15)
    row = next(line for line in text.splitlines() if line.startswith(f"| {model} | MirethSTM1 (full label) | ag_news |"))
    cells = [c.strip() for c in row.split("|")[1:-1]]
    assert cells[5].startswith(f"{ece15:.3f} [") and cells[6] == f"{nll:.3f}"
    # The header counts the rows from the records; the summary row is the same numbers, one dataset.
    assert ("- org/fake-model: dtype not recorded; scoring arms on 30 evaluation rows per dataset and 40 calibration "
            "rows; normal generation on 30 evaluation rows; latency over 9 timed runs per field count.") in text
    hits = [int(np.argmax(pi)) == g for pi, g in zip(p, gold)]
    hits_first = [max(r["scores"], key=r["scores"].get) == r["gold"] for r in first]
    assert (f"| {model} | 30 / 30 | {np.mean(hits):.3f} | {np.mean(hits_first):.3f} | 0.333 (0.667) | "
            f"{np.mean(hits):.3f} | {ece15:.3f} |") in text
    assert f"| {model} | 24.0 | n/a |" in text and f"| {model} | 404 | 99.0 |" in text  # 360 tokens in 3.636 s
    mass = [float(np.exp(list(r["scores"].values())).sum()) for r in by_split["eval"]]  # exp(score) over the labels
    assert f"| {model} | {np.median(mass):.3f} [{np.mean(np.array(mass) < 0.01):.3f}] |" in text
    # Precision: the 20 shared rows against the fp32 run; 10 of 80 scores are 0.8 apart, which changes some
    # answers; T fits the same 40 rows.
    assert "## Precision: the same rows in other dtypes" in text
    assert f"calibration rows per dataset: bf16 {cells[4]}, fp32 {cells[4]}." in text and cells[3] == cells[4]
    same = np.mean([max(a["scores"], key=a["scores"].get) == max(b["scores"], key=b["scores"].get)
                    for a, b in zip(by_split["eval"], exact)])
    assert 0.5 <= same < 1
    assert f"| ag_news | 20 | {np.mean(hits[:20]):.3f} / " in text
    assert f"| {same:.3f} | 0.100 (0.80) | 10.0 / 10.0 |" in text
    # A report written elsewhere links to the diagrams kept next to it: the MirethSTM1 arm only, no bin tables.
    elsewhere = report.build(run, tmp_path / "docs" / "benchmark.md", resamples=0, fig_dir=tmp_path / "docs" / "img")
    public = elsewhere.read_text(encoding="utf-8")
    assert "[png](img/org--fake-model.ag_news.mireth.png)" in public and "| not published |" in public
    assert [f.name for f in (tmp_path / "docs" / "img").iterdir()] == ["org--fake-model.ag_news.mireth.png"]


# --- the arms on the test model (CPU) -----------------------------------------------------------


@pytest.mark.model
def test_first_token_arm_equals_uncached_forward(engine):
    questions = {k: QUESTIONS[k] for k in ("topic", "refund", "urgency")}
    state = "The match ended 3 to 1 after extra time."
    got = arms.first_token(engine, state, questions)
    plan = arms.first_token_plan(engine.tokenizer, questions)
    for qid, (suffix, first) in plan.items():
        # Every question is asked alone, with the engine's own prompt for it (SPEC 3.1).
        prefix = shared_ids(engine.tokenizer, state) + branch_ids(engine.tokenizer, questions[qid])
        with torch.inference_mode():
            ids = torch.tensor([prefix + suffix], device=engine.model.device)
            logits = engine.model(input_ids=ids).logits[0, -1].float()
        want = logits.log_softmax(-1)[first].tolist()
        assert list(got[qid]["scores"].values()) == pytest.approx(want, abs=tolerance(engine))


@pytest.mark.model
def test_questions_first_arm_equals_uncached_forward(engine):
    questions = {k: QUESTIONS[k] for k in ("topic", "refund")}
    state = {"ticket": "Please refund me", "lang": "en"}
    got = arms.questions_first(engine, state, questions)
    suffix = encode(engine.tokenizer, sch.SUFFIX)
    for qid, q in questions.items():
        prefix = arms.questions_first_ids(engine.tokenizer, state, q)
        text = engine.tokenizer.decode(prefix)
        assert text.index("Question:") < text.index("State:")
        for label, cand in zip(sch.labels(q), sch.candidates(q)):
            c = encode(engine.tokenizer, cand)
            with torch.inference_mode():
                ids = torch.tensor([prefix + suffix + c], device=engine.model.device)
                logp = engine.model(input_ids=ids).logits[0].float().log_softmax(-1)
            start = len(prefix) + len(suffix) - 1
            want = sum(logp[start + t, tok].item() for t, tok in enumerate(c))
            assert got[qid]["scores"][label] == pytest.approx(want, abs=tolerance(engine))


@pytest.mark.model
def test_tiny_run_all_three_arms(engine, tmp_path):
    question, labels = data.question("sst2")
    try:
        ev, cal = data.samples("sst2", 3, 3)
    except (ConnectionError, FileNotFoundError) as e:
        pytest.skip(f"sst2 is not cached: {e}")
    jobs = [(f"sst2.{split}", arm, run_bench.dataset_tasks(question, labels, rows), {"dataset": "sst2", "split": split})
            for split, rows in (("cal", cal), ("eval", ev)) for arm in run_bench.DEFAULT_ARMS
            if split == "eval" or arm in arms.SCORING]
    run = tmp_path / "tiny"
    run_bench.run_files(engine, MODEL_ID, jobs, run)
    folder = run / slug(MODEL_ID)
    assert sorted(p.name for p in folder.iterdir()) == [
        "sst2.cal.first_token.jsonl", "sst2.cal.mireth.jsonl", "sst2.eval.baseline.jsonl",
        "sst2.eval.first_token.jsonl", "sst2.eval.mireth.jsonl"]
    for arm in run_bench.DEFAULT_ARMS:
        records = [json.loads(line) for line in (folder / f"sst2.eval.{arm}.jsonl").read_text(encoding="utf-8").splitlines()]
        assert [r["index"] for r in records] == [s.index for s in ev]
        for r in records:
            assert {"model", "dataset", "split", "index", "gold", "arm", "latency_ms"} <= set(r)
            assert r["gold"] in ("true", "false") and r["latency_ms"] > 0
            if arm == "baseline":
                assert r["pred"] in ("true", "false", None) and isinstance(r["text"], str)
                assert isinstance(r["cap_hit"], bool) and r["salvaged"] in ("true", "false", None)
            else:
                assert list(r["scores"]) == ["true", "false"]
    mireth = [json.loads(line) for line in (folder / "sst2.eval.mireth.jsonl").read_text(encoding="utf-8").splitlines()]
    want = engine.score(ev[0].text, {data.QID: question})[data.QID]
    assert mireth[0]["scores"] == pytest.approx(want, abs=1e-4)
    run_bench.run_files(engine, MODEL_ID, jobs, run)  # every file exists: nothing reruns
    text = report.build(run, run / "benchmark.md", resamples=20).read_text(encoding="utf-8")
    for name in ("MirethSTM1 (full label)", "Normal generation", "First token (original demo's method)"):
        assert f"| {MODEL_ID} | {name} |" in text


@pytest.mark.model
def test_multifield_scores_do_not_depend_on_the_other_questions(engine):
    try:
        rows, _ = data.samples("ag_news", 2, 0)
    except (ConnectionError, FileNotFoundError) as e:
        pytest.skip(f"ag_news is not cached: {e}")
    by_mode = {mode: multifield.run_mode(engine, mode, rows, {"model": MODEL_ID}) for mode in ("alone", "spread")}
    for row, alone, spread in zip(rows, by_mode["alone"], by_mode["spread"]):
        assert alone["index"] == spread["index"] == row.index and list(alone["scores"]) == list(multifield.CHECKED)
        assert alone["truth"] == multifield.truth(multifield.TOPICS[row.gold])
        assert max_diff(spread["scores"], alone["scores"], engine) < tolerance(engine)
    header, body = multifield.table(by_mode)
    assert len(body) == 2 and len(body[0]) == len(header)
