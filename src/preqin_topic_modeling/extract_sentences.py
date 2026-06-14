from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from tqdm import tqdm

import tomotopy as tp

from .data import load_dataframe, prepare_corpus


def make_output_dir(output_dir: str | Path, prefix: str = "extraction") -> Path:
    kst = ZoneInfo("Asia/Seoul")
    timestamp = datetime.now(kst).strftime("%Y%m%d_%H%M%S")
    out = Path(output_dir) / f"{prefix}_{timestamp}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def get_topic_words(model: tp.DMRModel, topic_id: int, top_n: int = 30) -> list[tuple[str, float]]:
    return [(word, float(prob)) for word, prob in model.get_topic_words(int(topic_id), top_n=top_n)]


def get_hybrid_keywords(model: tp.DMRModel, topic_id: int, min_words: int = 5, max_words: int = 50) -> list[str]:
    """기존 노트북의 Hybrid 토픽어 선택 로직을 정리한 함수."""
    candidates = [(w, float(p)) for w, p in model.get_topic_words(int(topic_id), top_n=100) if p >= 0.001]
    if len(candidates) < min_words:
        return [w for w, _ in model.get_topic_words(int(topic_id), top_n=min_words)]

    probs = np.array([p for _, p in candidates])
    probs_norm = probs / probs.sum()
    cumsum = np.cumsum(probs_norm)

    cutoff_50 = int(np.argmax(cumsum >= 0.5))
    cutoff_70 = int(np.argmax(cumsum >= 0.7))
    top_prob = candidates[0][1]
    cutoff_relative = sum(1 for _, p in candidates if p >= top_prob * 0.05)

    final_cutoff = int(np.median([cutoff_50, cutoff_70, cutoff_relative]))
    final_cutoff = max(min_words, min(final_cutoff, max_words))
    return [w for w, _ in candidates[:final_cutoff]]


def top_topics_by_alpha(model: tp.DMRModel, top_n: int = 10) -> list[int]:
    """DMR alpha 평균 기준으로 상위 토픽을 반환한다."""
    alpha_matrix = np.array(model.alpha)
    topic_mean_alpha = alpha_matrix.mean(axis=1)
    return np.argsort(topic_mean_alpha)[::-1][:top_n].astype(int).tolist()


def find_topics_by_keyword(
    model: tp.DMRModel,
    keyword: str,
    top_n_rank: int = 30,
) -> pd.DataFrame:
    """상위 토픽어 안에 특정 키워드가 포함된 토픽을 찾는다."""
    rows = []
    for topic_id in range(model.k):
        words = get_topic_words(model, topic_id, top_n=top_n_rank)
        for rank, (word, prob) in enumerate(words, start=1):
            if word == keyword:
                rows.append(
                    {
                        "keyword": keyword,
                        "topic_id": topic_id,
                        "keyword_rank": rank,
                        "keyword_probability": prob,
                        "top_words": ", ".join(w for w, _ in words[:10]),
                    }
                )
                break
    return pd.DataFrame(rows).sort_values(
        ["keyword_rank", "keyword_probability"], ascending=[True, False]
    )


def align_df_to_model(df: pd.DataFrame, model: tp.DMRModel) -> pd.DataFrame:
    """모델 문서 수와 DataFrame 행 수가 다르면 앞쪽 기준으로 정렬한다."""
    if len(df) == len(model.docs):
        return df.copy()

    min_len = min(len(df), len(model.docs))
    print(f"[WARN] DataFrame 행 수({len(df)})와 모델 문서 수({len(model.docs)})가 달라 {min_len}개로 맞춥니다.")
    return df.iloc[:min_len].copy()


def extract_by_topic_probability(
    df: pd.DataFrame,
    model: tp.DMRModel,
    topic_id: int,
    top_n: int = 50,
) -> pd.DataFrame:
    """문서별 topic distribution에서 해당 topic 확률이 높은 문장을 추출한다."""
    topic_id = int(topic_id)
    work = align_df_to_model(df, model)

    probs = []
    for doc in tqdm(model.docs[: len(work)], desc=f"topic {topic_id} probabilities"):
        probs.append(float(doc.get_topic_dist()[topic_id]))

    work["topic_probability"] = probs
    return (
        work.drop_duplicates(subset="sentence")
        .sort_values("topic_probability", ascending=False)
        .head(top_n)
    )


def extract_by_weighted_entropy(
    df: pd.DataFrame,
    topic_keywords: list[str],
    top_n: int = 50,
    min_length: int = 10,
) -> pd.DataFrame:
    """기존 노트북의 Weighted Shannon Entropy 방식 대표문장 추출.

    토픽어가 실제 문장 안에 함께 분포하는 정도를 보되,
    앞순위 토픽어에 더 큰 가중치를 부여한다.
    """
    work = df.copy()
    work["sentence_length"] = work["token"].apply(len)

    for word in topic_keywords:
        work[f"{word}_count"] = work["token"].apply(lambda tokens: tokens.count(word))

    count_cols = [f"{word}_count" for word in topic_keywords]
    work["sum_count"] = work[count_cols].sum(axis=1)
    work = work[(work["sum_count"] > 0) & (work["sentence_length"] >= min_length)].copy()

    if len(work) == 0:
        return work.head(0)

    weights = {word: len(topic_keywords) - idx for idx, word in enumerate(topic_keywords)}

    def weighted_entropy(row: pd.Series) -> float:
        entropy = 0.0
        for word in topic_keywords:
            p = row[f"{word}_count"] / row["sum_count"] if row["sum_count"] else 0
            if p > 0:
                entropy -= weights[word] * p * math.log(p)
        return entropy

    work["entropy"] = work.apply(weighted_entropy, axis=1)
    work["entropy_adj"] = work["entropy"] / np.log(work["sentence_length"] + 1)

    return (
        work.drop_duplicates(subset="sentence")
        .sort_values("entropy_adj", ascending=False)
        .head(top_n)
    )


def save_keyword_extraction(
    output_path: Path,
    summary_df: pd.DataFrame,
    sentence_results: dict[int, pd.DataFrame],
) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        for topic_id, rows in sentence_results.items():
            if len(rows) == 0:
                continue
            cols = [
                "sentence",
                "category",
                "book_title",
                "chapter_title",
                "entropy_adj",
                "topic_probability",
            ]
            available = [col for col in cols if col in rows.columns]
            rows[available].to_excel(writer, sheet_name=f"Topic{topic_id}"[:31], index=False)


def parse_topic_ids(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_keywords(value: str | None) -> list[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract topic keywords and representative sentences")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default="outputs")

    parser.add_argument("--token-col", default="token")
    parser.add_argument("--text-col", default="sentence")
    parser.add_argument("--metadata-col", default="category")
    parser.add_argument("--era-col", default="Era")
    parser.add_argument("--era-value", default="先秦")

    parser.add_argument("--topic-ids", default="", help="예: 0,5,12")
    parser.add_argument("--target-keywords", default="", help="예: 心,禮,法")
    parser.add_argument("--top-topics", type=int, default=0, help="alpha 기준 상위 N개 토픽")
    parser.add_argument("--top-n-rank", type=int, default=30)
    parser.add_argument("--sentences-per-topic", type=int, default=50)
    parser.add_argument("--keyword-method", choices=["fixed", "hybrid"], default="hybrid")
    parser.add_argument("--sentence-method", choices=["entropy", "topic_prob"], default="entropy")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_df = load_dataframe(args.data_path)
    df = prepare_corpus(
        raw_df,
        token_col=args.token_col,
        text_col=args.text_col,
        metadata_col=args.metadata_col,
        era_col=args.era_col or None,
        era_value=args.era_value or None,
    )

    model = tp.DMRModel.load(args.model_path)
    out_dir = make_output_dir(args.output_dir)

    requested_topic_ids = parse_topic_ids(args.topic_ids)
    if args.top_topics:
        requested_topic_ids.extend(top_topics_by_alpha(model, top_n=args.top_topics))

    summary_rows = []
    sentence_results: dict[int, pd.DataFrame] = {}

    for keyword in parse_keywords(args.target_keywords):
        topics_df = find_topics_by_keyword(model, keyword, top_n_rank=args.top_n_rank)
        topics_df.to_csv(out_dir / f"topics_containing_{keyword}.csv", index=False, encoding="utf-8-sig")
        requested_topic_ids.extend(topics_df["topic_id"].astype(int).tolist())

    # 중복 제거
    topic_ids = list(dict.fromkeys(int(t) for t in requested_topic_ids))
    if not topic_ids:
        raise ValueError("--topic-ids, --target-keywords, --top-topics 중 하나 이상을 지정하세요.")

    for topic_id in topic_ids:
        keywords = (
            get_hybrid_keywords(model, topic_id)
            if args.keyword_method == "hybrid"
            else [w for w, _ in get_topic_words(model, topic_id, top_n=args.top_n_rank)]
        )

        if args.sentence_method == "entropy":
            sentences = extract_by_weighted_entropy(
                df,
                keywords,
                top_n=args.sentences_per_topic,
                min_length=10,
            )
        else:
            sentences = extract_by_topic_probability(
                df,
                model,
                topic_id,
                top_n=args.sentences_per_topic,
            )

        sentence_results[topic_id] = sentences
        summary_rows.append(
            {
                "topic_id": topic_id,
                "num_keywords": len(keywords),
                "top_10_keywords": ", ".join(keywords[:10]),
                "num_sentences": len(sentences),
                "sample_sentence": sentences.iloc[0]["sentence"][:80] if len(sentences) else "",
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "extraction_summary.csv", index=False, encoding="utf-8-sig")
    save_keyword_extraction(out_dir / "representative_sentences.xlsx", summary_df, sentence_results)
    print(f"결과 저장: {out_dir}")


if __name__ == "__main__":
    main()
