from __future__ import annotations

from source.application.use_cases.training import TrainPipelineCommand, TrainPipelineUseCase
from source.infrastructure.config import load_pipeline_settings


def run_train_from_env() -> None:
    settings = load_pipeline_settings()
    train_use_case = TrainPipelineUseCase()
    train_result = train_use_case.execute(
        TrainPipelineCommand(
            dataset_dir=settings.dataset_dir,
            output_root=settings.output_root,
            run_name=settings.run_name,
            eval_users_limit=settings.eval_users_limit,
            candidate_pool_size=settings.candidate_pool_size,
            candidate_per_source_limit=settings.candidate_per_source_limit,
            pre_top_m=settings.pre_top_m,
            final_top_k=settings.final_top_k,
            cf_mode=settings.cf_mode,
            cf_max_neighbors=settings.cf_max_neighbors,
            cf_max_items_per_user=settings.cf_max_items_per_user,
            content_max_neighbors=settings.content_max_neighbors,
            seed=settings.seed,
        )
    )
    print(f"[train] completed run_id={train_result.run_id}")
    print(f"[train] run_dir={train_result.run_dir}")
    print(f"[train] metrics_path={train_result.metrics_path}")


def main() -> None:
    run_train_from_env()


if __name__ == "__main__":
    main()
