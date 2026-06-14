from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from tqdm import tqdm

import tomotopy as tp

from .data import load_dataframe, prepare_corpus, summarize_corpus


TERM_WEIGHT_MAP = {
    "ONE": tp.TermWeight.ONE,
    "IDF": tp.TermWeight.IDF,
    "PMI": tp.TermWeight.PMI,
}


def make_run_dir(output_dir: str | Path, prefix: str = "run") -> Path:
    kst = ZoneInfo("Asia/Seoul")
    timestamp = datetime.now(kst).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"{prefix}_{timestamp}"
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "results").mkdir(parents=True, exist_ok=True)
    return run_dir


def build_dmr_model(
    df: pd.DataFrame,
    k: int,
    min_df: int,
    rm_top: int,
    seed: int,
    term_weight: str = "PMI",
) -> tp.DMRModel:
    """DataFrame의 token/category를 tomotopy DMRModel에 추가한다."""
    if term_weight not in TERM_WEIGHT_MAP:
        raise ValueError(f"term_weight는 {list(TERM_WEIGHT_MAP)} 중 하나여야 합니다.")

    model = tp.DMRModel(
        k=k,
        min_df=min_df,
        rm_top=rm_top,
        seed=seed,
        tw=TERM_WEIGHT_MAP[term_weight],
    )

    for _, row in tqdm(df.iterrows(), total=len(df), desc="add documents"):
        model.add_doc(row["token"], metadata=str(row["category"]))

    return model


def train_model(model: tp.DMRModel, iterations: int = 500, step: int = 20) -> pd.DataFrame:
    """모델을 학습하고 log-likelihood 기록을 반환한다."""
    logs: list[dict[str, float | int]] = []

    # tomotopy는 train(0)을 먼저 호출하면 vocabulary 등 초기 상태 확인이 쉽다.
    model.train(0)
    print(
        f"Num docs={len(model.docs)}, Vocab={model.num_vocabs}, "
        f"Num words={model.num_words}, Removed top words={model.removed_top_words}"
    )

    for iteration in tqdm(range(step, iterations + 1, step), desc="train"):
        model.train(step)
        logs.append({"iteration": iteration, "ll_per_word": float(model.ll_per_word)})
        print(f"Iteration={iteration}\tLL/token={model.ll_per_word:.6f}")

    return pd.DataFrame(logs)


def extract_topic_words(model: tp.DMRModel, top_n: int = 30) -> pd.DataFrame:
    """각 토픽의 상위 단어와 확률을 long-format DataFrame으로 반환한다."""
    rows = []
    for topic_id in range(model.k):
        for rank, (word, prob) in enumerate(model.get_topic_words(topic_id, top_n=top_n), start=1):
            rows.append(
                {
                    "topic_id": topic_id,
                    "rank": rank,
                    "word": word,
                    "probability": float(prob),
                }
            )
    return pd.DataFrame(rows)


def save_training_outputs(
    model: tp.DMRModel,
    run_dir: Path,
    model_name: str,
    training_log: pd.DataFrame,
    topic_words: pd.DataFrame,
    corpus_summary: pd.DataFrame,
    config: dict,
) -> None:
    model_path = run_dir / "models" / model_name
    model.save(str(model_path), full=True)

    results_dir = run_dir / "results"
    training_log.to_csv(results_dir / "training_log.csv", index=False, encoding="utf-8-sig")
    topic_words.to_csv(results_dir / "topic_words.csv", index=False, encoding="utf-8-sig")
    corpus_summary.to_csv(results_dir / "corpus_summary.csv", index=False, encoding="utf-8-sig")

    with (run_dir / "config_used.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"모델 저장: {model_path}")
    print(f"결과 저장: {results_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-Qin corpus DMR topic modeling")
    parser.add_argument("--data-path", required=True, help="전처리 완료 DataFrame 파일 경로")
    parser.add_argument("--output-dir", default="outputs", help="결과 저장 루트 폴더")
    parser.add_argument("--model-name", default="dmr_model.bin", help="저장할 모델 파일명")

    parser.add_argument("--token-col", default="token", help="토큰 리스트 컬럼명")
    parser.add_argument("--text-col", default="sentence", help="문장/문단 텍스트 컬럼명")
    parser.add_argument("--metadata-col", default="category", help="DMR metadata 컬럼명")
    parser.add_argument("--era-col", default="Era", help="시대 필터 컬럼명. 필터 미사용은 빈 문자열")
    parser.add_argument("--era-value", default="先秦", help="시대 필터 값. 필터 미사용은 빈 문자열")
    parser.add_argument("--min-tokens", type=int, default=1)

    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--min-df", type=int, default=30)
    parser.add_argument("--rm-top", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--term-weight", default="PMI", choices=list(TERM_WEIGHT_MAP))
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--top-n-words", type=int, default=30)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = vars(args)

    raw_df = load_dataframe(args.data_path)
    df = prepare_corpus(
        raw_df,
        token_col=args.token_col,
        text_col=args.text_col,
        metadata_col=args.metadata_col,
        era_col=args.era_col or None,
        era_value=args.era_value or None,
        min_tokens=args.min_tokens,
    )

    run_dir = make_run_dir(args.output_dir)
    corpus_summary = summarize_corpus(df)

    model = build_dmr_model(
        df,
        k=args.k,
        min_df=args.min_df,
        rm_top=args.rm_top,
        seed=args.seed,
        term_weight=args.term_weight,
    )
    training_log = train_model(model, iterations=args.iterations, step=args.step)
    topic_words = extract_topic_words(model, top_n=args.top_n_words)

    save_training_outputs(
        model=model,
        run_dir=run_dir,
        model_name=args.model_name,
        training_log=training_log,
        topic_words=topic_words,
        corpus_summary=corpus_summary,
        config=config,
    )


if __name__ == "__main__":
    main()
