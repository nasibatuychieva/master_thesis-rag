from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# =============================================================================
# CONFIG: Define your sequential steps here
# =============================================================================

@dataclass
class Step:
    key: str                      # short name for --only
    module: str                   # python -m <module>
    title: str = ""
    env: Dict[str, str] = field(default_factory=dict)   # optional step-specific env overrides


STEPS: List[Step] = [
    # --- Global community retrievers ---
    Step(
        key="global_simplekg",
        module="main.knowledge_graph.build_communities.global_communities.simplekg.04_global_community_retriever_simplekg",
        title="Global community retriever (simplekg)",
    ),
    Step(
        key="global_llmgraph",
        module="main.knowledge_graph.build_communities.global_communities.llmgraph.04_global_community_retriever_llmgraph",
        title="Global community retriever (llmgraph)",
    ),
    Step(
        key="global_llmaindex",
        module="main.knowledge_graph.build_communities.global_communities.llmaindex.04_global_community_retriever_llmaindex",
        title="Global community retriever (llmaindex)",
    ),

    # --- Retrieve graph pipelines ---
    Step(
        key="llmgraphtransformer",
        module="main.knowledge_graph.retrieve_graph.retrieve_kg_LLMGraphTransformer",
        title="Retrieve KG (LLMGraphTransformer)",
    ),
    Step(
        key="llmgraphtransformer_hyb_rer",
        module="main.knowledge_graph.retrieve_graph.retrieve_kg_LLMGraphTransformer_Hyb_Rer",
        title="Retrieve KG (LLMGraphTransformer Hybrid + Rerank)",
    ),
    Step(
        key="llmgraphtransformer_hyb",
        module="main.knowledge_graph.retrieve_graph.retrieve_kg_LLMGraphTransformer_Hyb",
        title="Retrieve KG (LLMGraphTransformer Hybrid)",
    ),
    Step(
        key="llamaindex",
        module="main.knowledge_graph.retrieve_graph.retrieve_kg_LlamaIndex",
        title="Retrieve KG (LlamaIndex)",
    ),
    Step(
        key="llamaindex_rerank_old",
        module="main.knowledge_graph.retrieve_graph.retrieve_kg_LlamaIndex_rerank_old",
        title="Retrieve KG (LlamaIndex rerank old)",
    ),
    Step(
        key="llamaindex_hybr_rer",
        module="main.knowledge_graph.retrieve_graph.retrieve_kg_LlamaIndex_Hybr_Rer",
        title="Retrieve KG (LlamaIndex Hybrid + Rerank)",
    ),
    Step(
        key="simplekg",
        module="main.knowledge_graph.retrieve_graph.retrieve_kg_SimpleKGPipeline",
        title="Retrieve KG (SimpleKG pipeline)",
    ),
    Step(
        key="simplekg_hybrid",
        module="main.knowledge_graph.retrieve_graph.retrieve_kg_SimpleKGPipeline_Hybrid",
        title="Retrieve KG (SimpleKG Hybrid)",
    ),
    Step(
        key="simplekg_hyb_rer",
        module="main.knowledge_graph.retrieve_graph.retrieve_kg_SimpleKGPipeline_Hyb_Rer",
        title="Retrieve KG (SimpleKG Hybrid + Rerank)",
    ),
]


# =============================================================================
# Runner
# =============================================================================

def run_step(step: Step, *, base_env: Dict[str, str], dry_run: bool = False) -> int:
    title = step.title or step.module
    cmd = [sys.executable, "-m", step.module]

    env = dict(base_env)
    env.update(step.env or {})

    print("\n" + "=" * 100)
    print(f"[RUN] {step.key} | {title}")
    print(f"[CMD] {' '.join(cmd)}")
    if step.env:
        print(f"[ENV OVERRIDES] {step.env}")
    print("=" * 100)

    if dry_run:
        print("[DRY RUN] skipping execution.")
        return 0

    t0 = time.time()
    proc = subprocess.run(cmd, env=env)
    dt = time.time() - t0

    code = int(proc.returncode)
    if code == 0:
        print(f"[OK] {step.key} finished in {dt:.1f}s")
    else:
        print(f"[FAIL] {step.key} exited with code={code} after {dt:.1f}s")

    return code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all KG/community scripts sequentially (python -m ...), with optional filtering."
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running next steps even if a step fails.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Run only selected step keys (space-separated). Example: --only global_simplekg llamaindex_hybr_rer",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands but do not execute.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available step keys and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list:
        print("\nAvailable steps:")
        for s in STEPS:
            print(f"  - {s.key:25s}  {s.module}")
        print()
        return 0

    selected: List[Step] = STEPS
    if args.only is not None:
        wanted = set(args.only)
        selected = [s for s in STEPS if s.key in wanted]
        missing = [k for k in args.only if k not in {s.key for s in STEPS}]
        if missing:
            print(f"[ERROR] Unknown step keys: {missing}")
            print("Use --list to see valid keys.")
            return 2
        if not selected:
            print("[ERROR] No steps selected.")
            return 2

    base_env = dict(os.environ)

    print("\n" + "#" * 100)
    print("[INFO] Central sequential runner started")
    print(f"[INFO] Steps total={len(STEPS)} | selected={len(selected)}")
    print(f"[INFO] continue_on_error={args.continue_on_error} | dry_run={args.dry_run}")
    print("#" * 100)

    failed: List[str] = []

    for idx, step in enumerate(selected, start=1):
        print(f"\n[INFO] Step {idx}/{len(selected)} -> {step.key}")
        code = run_step(step, base_env=base_env, dry_run=args.dry_run)
        if code != 0:
            failed.append(step.key)
            if not args.continue_on_error:
                print("\n[STOP] Aborting because a step failed (no --continue-on-error).")
                break

    print("\n" + "#" * 100)
    if failed:
        print(f"[DONE] Finished with failures: {failed}")
        print("#" * 100)
        return 1
    else:
        print("[DONE] All selected steps finished successfully.")
        print("#" * 100)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
