from __future__ import annotations

import argparse
import json
import logging

from pathlib import Path
from artifacts.save import save_training_outputs
from training.runner import RunnerConfig, run_training_pipeline, runner_config_to_dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train BookRecs recommendation pipeline")
    parser.add_argument("--data-root", default="data/data06")
    parser.add_argument("--split-name", default="local_v1")
    parser.add_argument("--sample-users-n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output-dir", default="pipeline/output")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--content-candidate-top-n", type=int, default=200)
    parser.add_argument("--item2item-candidate-top-n", type=int, default=200)
    parser.add_argument("--hybrid-cf-weight", type=float, default=0.5)
    parser.add_argument("--hybrid-content-weight", type=float, default=0.35)
    parser.add_argument("--hybrid-pop-weight", type=float, default=0.15)
    parser.add_argument("--hybrid-cf-top-n", type=int, default=300)
    parser.add_argument("--hybrid-content-top-n", type=int, default=300)
    parser.add_argument("--hybrid-pop-top-n", type=int, default=300)
    return parser


def args_to_config(args: argparse.Namespace) -> RunnerConfig:
    return RunnerConfig(
        data_root=args.data_root,
        split_name=args.split_name,
        sample_users_n=args.sample_users_n,
        seed=args.seed,
        k=args.k,
        content_candidate_top_n=args.content_candidate_top_n,
        item2item_candidate_top_n=args.item2item_candidate_top_n,
        hybrid_cf_weight=args.hybrid_cf_weight,
        hybrid_content_weight=args.hybrid_content_weight,
        hybrid_pop_weight=args.hybrid_pop_weight,
        hybrid_cf_top_n=args.hybrid_cf_top_n,
        hybrid_content_top_n=args.hybrid_content_top_n,
        hybrid_pop_top_n=args.hybrid_pop_top_n,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = build_parser()
    args = parser.parse_args()
    config = args_to_config(args)
    metrics_df, bundle, models = run_training_pipeline(config)

    saved = save_training_outputs(
        metrics_df=metrics_df,
        models=models,
        config=runner_config_to_dict(config),
        bundle=bundle,
        base_dir=args.output_dir,
        run_name=args.run_name,
    )

    print(metrics_df.to_string(index=False))
    print()
    print(json.dumps(saved, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
