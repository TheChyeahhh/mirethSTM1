"""Benchmark datasets: one TypeSafe question each, gold labels, fixed stratified samples.

Facts and split choices: docs/research/07-benchmark-design.md section 1. The data is
downloaded by id at run time (Hugging Face cache) and never copied into the repository.

Each dataset has an evaluation split and a disjoint calibration split drawn from train:
a calibration row whose text also appears anywhere in the evaluation split is skipped.
"""

from dataclasses import dataclass

import numpy as np

SEED = 0
QID = "answer"
# Texts are cut to this many characters (at a word boundary, keeping the head) so every
# model and every arm reads the same text. Only long Yelp reviews reach it.
MAX_CHARS = 2000


@dataclass(frozen=True)
class Spec:
    hf_id: str
    eval_split: str
    cal_split: str
    text: str
    type: str
    instructions: str
    names: tuple = ()  # label names in dataset index order; empty: read from the data (Banking77)
    criteria: tuple = ()  # noul: (false description, true description); score: level descriptions
    min_words: int = 0  # calibration rows shorter than this are skipped (SST-2 train is phrase level)


DATASETS = {
    "ag_news": Spec("fancyzhx/ag_news", "test", "train", "text", "choice",
                    "Which topic is this news article about?",
                    names=("World", "Sports", "Business", "Sci/Tech")),
    "banking77": Spec("mteb/banking77", "test", "train", "text", "choice",
                      "Which intent does this banking customer message express?"),
    # Test labels are hidden (all -1), so validation (872 rows) is the evaluation split.
    "sst2": Spec("stanfordnlp/sst2", "validation", "train", "sentence", "noul",
                 "Is the sentiment of this text positive?",
                 criteria=("The sentiment is negative", "The sentiment is positive"), min_words=8),
    # The same rows as a two-option choice, so a yes-bias of the yes/no form shows up as a gap.
    "sst2_choice": Spec("stanfordnlp/sst2", "validation", "train", "sentence", "choice",
                        "What is the sentiment of this text?", names=("negative", "positive"), min_words=8),
    # Level descriptions spell the stars in words, so a level number (0 to 4) is never
    # confused with a star count (1 to 5).
    "yelp": Spec("Yelp/yelp_review_full", "test", "train", "text", "score",
                 "How many stars does this review give?",
                 criteria=("One star: terrible", "Two stars: poor", "Three stars: average",
                           "Four stars: good", "Five stars: excellent")),
}


def load_split(spec, split):
    from datasets import load_dataset

    return load_dataset(spec.hf_id, split=split)


def banking77_names(train):
    """Intent names in label order from the train split's label_text column.

    The raw names hold two oddities, "Refund_not_showing_up" and "reverted_card_payment?";
    names are lowercased and a trailing "?" is dropped.
    """
    pairs = sorted(set(zip(train["label"], train["label_text"])))
    if [label for label, _ in pairs] != list(range(len(pairs))):
        raise ValueError("banking77: label ids and label_text do not map one to one")
    names = [name.lower().rstrip("?") for _, name in pairs]
    if len(set(names)) != len(names):
        raise ValueError("banking77: two intents share a name after normalizing")
    return names


def label_names(name):
    spec = DATASETS[name]
    if spec.names:
        return list(spec.names)
    return banking77_names(load_split(spec, spec.cal_split))


def question(name, names=None):
    """The dataset's TypeSafe question and the engine label of each gold index."""
    spec = DATASETS[name]
    if spec.type == "choice":
        names = names or label_names(name)
        return {"type": "choice", "instructions": spec.instructions, "criteria": dict.fromkeys(names)}, list(names)
    if spec.type == "noul":  # gold index 1 is the "true" answer
        criteria = {"true": spec.criteria[1], "false": spec.criteria[0]}
        return {"type": "noul", "instructions": spec.instructions, "criteria": criteria}, ["false", "true"]
    return ({"type": "score", "instructions": spec.instructions, "criteria": list(spec.criteria)},
            [str(i) for i in range(len(spec.criteria))])


def clip(text, max_chars=MAX_CHARS):
    """The head of the text, at most max_chars characters, cut at a word boundary."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(None, 1)[0]  # drops the word the cut went through


def stratified(gold, n, seed=SEED, keep=None):
    """Up to n row indices drawn with a fixed seed, as even across classes as the data allows.

    Classes take turns in label order, each from its own seeded shuffle, so a class short of
    rows leaves its turns to the others. `keep(index)` can reject rows. Returns the picks in
    a seeded random order.
    """
    rng = np.random.default_rng(seed)
    gold = np.asarray(gold)
    pools = [iter(rng.permutation(np.flatnonzero(gold == c)).tolist()) for c in np.unique(gold)]
    picked = []
    while pools and len(picked) < n:
        for pool in list(pools):
            if len(picked) == n:
                break
            index = next((i for i in pool if keep is None or keep(i)), None)
            if index is None:
                pools.remove(pool)
            else:
                picked.append(index)
    return [picked[i] for i in rng.permutation(len(picked))]


@dataclass
class Sample:
    index: int  # row in the split
    text: str
    gold: int  # gold label index


def samples(name, n_eval, n_cal, seed=SEED):
    """(eval samples, calibration samples) for a dataset; the same lists on every run."""
    spec = DATASETS[name]
    ev, cal = load_split(spec, spec.eval_split), load_split(spec, spec.cal_split)
    eval_rows = stratified(ev["label"], n_eval, seed)
    eval_texts = {t.strip() for t in ev[spec.text]}

    def keep(i):
        text = cal[i][spec.text]
        return text.strip() not in eval_texts and len(text.split()) >= spec.min_words

    cal_rows = stratified(cal["label"], n_cal, seed, keep=keep) if n_cal else []

    def pick(split, rows):
        return [Sample(i, clip(split[i][spec.text]), int(split[i]["label"])) for i in rows]

    return pick(ev, eval_rows), pick(cal, cal_rows)
