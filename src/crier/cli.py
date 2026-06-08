"""``crier`` command-line entry point.

Two subcommands:

* ``crier doctor`` — probe every backend and print a status table.
* ``crier generate "prompt"`` — quick one-shot generation for sanity
  checks.

Both subcommands respect ``--accelerator`` for explicit selection.
"""

from __future__ import annotations

import argparse
import sys

from .diagnostics import probe
from .errors import CrierError
from .llm import LLM
from .types import GenerationConfig, Message


def _cmd_doctor(_: argparse.Namespace) -> int:
    results = probe()
    print(f"{'backend':>10}  {'status':>16}  detail")
    print(f"{'-' * 10:>10}  {'-' * 16:>16}  {'-' * 40}")
    for r in results:
        print(r)
        if r.install_hint:
            print(f"{'':>10}  {'':>16}  hint: {r.install_hint}")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    try:
        llm = LLM.load(model=args.model, accelerator=args.accelerator)
    except CrierError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"# backend: {llm.info.name}  ({llm.info.execution_provider} on {llm.info.device})")
    if llm.info.fallback_reason:
        print(f"# fallback: {llm.info.fallback_reason}", file=sys.stderr)

    config = GenerationConfig(max_tokens=args.max_tokens, temperature=args.temperature)
    messages = [Message(role="user", content=args.prompt)]

    try:
        if args.stream:
            for chunk in llm.stream(messages, config):
                sys.stdout.write(chunk.text)
                sys.stdout.flush()
            sys.stdout.write("\n")
        else:
            reply = llm.generate(messages, config)
            print(reply.text)
    except CrierError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    finally:
        llm.close()

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="crier", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_doctor = subparsers.add_parser("doctor", help="Probe every backend.")
    p_doctor.set_defaults(func=_cmd_doctor)

    p_gen = subparsers.add_parser("generate", help="Run a one-shot generation.")
    p_gen.add_argument("prompt", help="User prompt.")
    p_gen.add_argument(
        "--model", default="phi-3.5-mini-instruct", help="Preset name or local path."
    )
    p_gen.add_argument(
        "--accelerator",
        default="auto",
        choices=["auto", "cpu", "directml", "coreml", "ryzenai", "openvino", "qnn", "cuda"],
    )
    p_gen.add_argument("--max-tokens", type=int, default=256)
    p_gen.add_argument("--temperature", type=float, default=0.7)
    p_gen.add_argument("--stream", action="store_true", help="Stream tokens to stdout.")
    p_gen.set_defaults(func=_cmd_generate)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
