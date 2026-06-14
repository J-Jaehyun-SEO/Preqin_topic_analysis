from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_BASE_COLUMNS = ("category", "book_title", "chapter_title")


def safe_literal_eval(value: Any) -> Any:
    """문자열로 저장된 리스트를 실제 리스트로 변환한다.

    Pickle/Parquet에는 token 컬럼이 리스트로 남아 있을 수 있고,
    CSV에는 "['禮', '義']"와 같은 문자열로 저장될 수 있다.
    """
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                return ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                return value
    return value


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """확장자에 따라 DataFrame을 읽는다.

    지원 형식: .pkl/.pickle, .parquet, .csv, .xlsx
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {path}")

    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError(f"지원하지 않는 파일 형식입니다: {suffix}")


def normalize_token_column(df: pd.DataFrame, token_col: str = "token") -> pd.DataFrame:
    """token 컬럼을 리스트형으로 정규화한다."""
    if token_col not in df.columns:
        raise KeyError(f"token 컬럼이 없습니다: {token_col}")

    out = df.copy()
    out[token_col] = out[token_col].apply(safe_literal_eval)

    bad_rows = out[~out[token_col].apply(lambda x: isinstance(x, list))]
    if len(bad_rows) > 0:
        raise ValueError(
            f"{token_col} 컬럼에 리스트가 아닌 값이 있습니다. "
            f"예시 index: {bad_rows.index[:5].tolist()}"
        )
    return out


def prepare_corpus(
    df: pd.DataFrame,
    token_col: str = "token",
    text_col: str = "sentence",
    metadata_col: str = "category",
    era_col: str | None = "Era",
    era_value: str | None = "先秦",
    min_tokens: int = 1,
) -> pd.DataFrame:
    """DMR 학습·문장 추출에 필요한 최소 컬럼을 정리한다.

    Parameters
    ----------
    token_col:
        토큰 리스트 컬럼. 문장 단위 자료는 보통 ``token``,
        문단 단위 자료는 ``pg_token``을 사용한다.
    text_col:
        대표 문장/문단 컬럼. 문장 단위는 ``sentence``,
        문단 단위는 ``paragraph``를 사용한다.
    metadata_col:
        DMR의 metadata로 넣을 컬럼. 기존 노트북은 ``category``를 사용했다.
    era_col, era_value:
        선진시기만 분석할 경우 ``Era == '先秦'`` 필터를 적용한다.
        전체 자료를 사용할 경우 둘 중 하나를 ``None``으로 둔다.
    """
    out = normalize_token_column(df, token_col=token_col)

    if era_col and era_value and era_col in out.columns:
        out = out[out[era_col] == era_value].copy()

    required = [token_col, text_col, metadata_col]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise KeyError(f"필수 컬럼 누락: {missing}")

    keep_cols = [col for col in REQUIRED_BASE_COLUMNS if col in out.columns]
    keep_cols = list(dict.fromkeys([metadata_col, *keep_cols, text_col, token_col]))

    out = out[keep_cols].dropna(subset=[token_col, text_col, metadata_col]).copy()
    out = out[out[token_col].apply(lambda tokens: len(tokens) >= min_tokens)].copy()
    out = out.reset_index(drop=True)

    # 내부 처리 표준명으로 맞춘다. 원본 컬럼명은 README에 기록한다.
    if token_col != "token":
        out = out.rename(columns={token_col: "token"})
    if text_col != "sentence":
        out = out.rename(columns={text_col: "sentence"})
    if metadata_col != "category":
        out = out.rename(columns={metadata_col: "category"})

    return out


def summarize_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """학파/분류별 문헌 수와 토큰 수를 집계한다."""
    summary = (
        df.assign(token_count=df["token"].apply(len))
        .groupby("category", dropna=False)
        .agg(
            documents=("sentence", "count"),
            total_tokens=("token_count", "sum"),
            mean_tokens=("token_count", "mean"),
        )
        .reset_index()
        .sort_values("documents", ascending=False)
    )
    return summary
