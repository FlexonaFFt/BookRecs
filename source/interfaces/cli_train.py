from __future__ import annotations

import argparse

from source.application.use_cases.training import TrainPipelineCommand, TrainPipelineUseCase


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run training pipeline")
    parser.add_argument("--dataset-dir", default="artifacts/tmp_preprocessed/goodreads_ya")
    parser.add_argument("--output-root", default="artifacts/runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--eval-users-limit", type=int, default=2000)
    parser.add_argument("--candidate-pool-size", type=int, default=1000)
    parser.add_argument("--candidate-per-source-limit", type=int, default=300)
    parser.add_argument("--pre-top-m", type=int, default=300)
    parser.add_argument("--final-top-k", type=int, default=10)
    parser.add_argument("--cf-max-neighbors", type=int, default=120)
    parser.add_argument("--content-max-neighbors", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def run_train(args: argparse.Namespace) -> None:
    use_case = TrainPipelineUseCase()
    result = use_case.execute(
        TrainPipelineCommand(
            dataset_dir=args.dataset_dir,
            output_root=args.output_root,
            run_name=args.run_name,
            eval_users_limit=args.eval_users_limit,
            candidate_pool_size=args.candidate_pool_size,
            candidate_per_source_limit=args.candidate_per_source_limit,
            pre_top_m=args.pre_top_m,
            final_top_k=args.final_top_k,
            cf_max_neighbors=args.cf_max_neighbors,
            content_max_neighbors=args.content_max_neighbors,
            seed=args.seed,
        )
    )
    print("train completed")
    print(f"run_id={result.run_id}")
    print(f"run_dir={result.run_dir}")
    print(f"manifest={result.manifest_path}")
    print(f"metrics={result.metrics_path}")
    print(f"timings={result.timings_path}")
    print(f"log={result.log_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_train(args)


if __name__ == "__main__":
    main()
