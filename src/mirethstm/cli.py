"""Command line: mirethstm decide (SPEC 5)."""

import argparse
import json
import sys

from .engine import Engine
from .schema import validate

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def _parser():
    parser = argparse.ArgumentParser(prog="mirethstm", description="Typed decisions from a local model.")
    commands = parser.add_subparsers(dest="command", required=True)
    decide = commands.add_parser("decide", help="answer a questions map about the state read from stdin")
    decide.add_argument("--schema", required=True, help="JSON file holding a TypeSafe questions map")
    decide.add_argument("--model", default=DEFAULT_MODEL, help=f"Hugging Face model id (default {DEFAULT_MODEL})")
    decide.add_argument("--temperature", type=float, help="softmax temperature (default: the model's shipped value, else 1.0)")
    decide.add_argument("--device", help='"cuda" or "cpu" (default: cuda when available)')
    decide.add_argument("--chunk-size", type=int, default=8, help="sequences per cached batch (default 8)")
    decide.add_argument("--log", help="append the JSONL events of this call to this file")
    decide.add_argument("--state-json", action="store_true", help="parse stdin as JSON instead of a plain string")
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if args.chunk_size < 1:
        parser.error("--chunk-size must be at least 1")
    if args.temperature is not None and not args.temperature > 0:
        parser.error("--temperature must be greater than 0")
    try:
        with open(args.schema, encoding="utf-8-sig") as f:
            questions = json.load(f)
        state = sys.stdin.buffer.read().decode("utf-8-sig")
        if args.state_json:
            state = json.loads(state)
        # Validate before loading the model, so a bad schema fails fast.
        validate(state, questions)
    except (OSError, ValueError) as e:  # ValueError covers SchemaError, bad JSON and bad UTF-8
        print(f"mirethstm: {e}", file=sys.stderr)
        return 2
    engine = Engine.load(args.model, device=args.device, chunk_size=args.chunk_size,
                         temperature=args.temperature, event_log=args.log)
    print(json.dumps(engine.decide(state, questions), indent=2))
    return 0
