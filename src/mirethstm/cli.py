"""Command line: mirethstm decide and mirethstm console (SPEC 5)."""

import argparse
import json
import sys

from . import console
from .engine import Engine
from .schema import validate

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def _parser():
    parser = argparse.ArgumentParser(prog="mirethstm", description="Typed decisions from a local model.")
    commands = parser.add_subparsers(dest="command", required=True)
    decide = commands.add_parser("decide", help="answer a questions map about the state read from stdin")
    decide.add_argument("--schema", required=True, help="JSON file holding a TypeSafe questions map")
    decide.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Hugging Face model id (default {DEFAULT_MODEL}); ids off the approved list are unvetted")
    decide.add_argument("--temperature", type=float, help="softmax temperature (default: the model's shipped value, else 1.0)")
    decide.add_argument("--device", help='"cuda" or "cpu" (default: cuda when available)')
    decide.add_argument("--batch-tokens", type=int, default=2048,
                        help="tree nodes per scoring pass; the first pass also reads the prompt (default 2048)")
    decide.add_argument("--log", help="append the JSONL events of this call to this file")
    decide.add_argument("--state-json", action="store_true", help="parse stdin as JSON instead of a plain string")
    decide.add_argument("--no-tarnlight", action="store_true", help="do not feed Tarnlight's drop folder")
    race = commands.add_parser("console", help="race view in the browser plus the local API (POST /v1/systemone)")
    race.add_argument("--model", default=console.DEFAULT_MODEL,
                      help=f"Hugging Face model id to load first (default {console.DEFAULT_MODEL}); "
                           "the picker offers the approved models")
    race.add_argument("--device", help='"cuda" or "cpu" (default: cuda when available)')
    race.add_argument("--host", default="127.0.0.1", help="address to listen on (default 127.0.0.1)")
    race.add_argument("--port", type=int, default=8766, help="port to listen on (default 8766)")
    race.add_argument("--no-tarnlight", action="store_true", help="do not feed Tarnlight's drop folder")
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "console":
        console.serve(model=args.model, device=args.device, host=args.host, port=args.port,
                      tarnlight=not args.no_tarnlight)
        return 0
    if args.batch_tokens < 1:
        parser.error("--batch-tokens must be at least 1")
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
    engine = Engine.load(args.model, device=args.device, batch_tokens=args.batch_tokens,
                         temperature=args.temperature, event_log=args.log, tarnlight=not args.no_tarnlight)
    print(json.dumps(engine.decide(state, questions), indent=2))
    return 0
