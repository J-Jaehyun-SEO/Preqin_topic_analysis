# Data files

이 저장소에는 원자료와 대용량 전처리 파일을 포함하지 않는다.  
재현 실행을 위해서는 아래 형식의 전처리 완료 DataFrame을 `data/processed/`에 별도로 배치해야 한다.

## 필수 파일

### 1. 문장 단위 분석

권장 파일명:

```text
data/processed/st_df_All_fin_251120_nega.pkl
```

필수 컬럼:

| 컬럼 | 의미 |
|---|---|
| `category` | DMR metadata. 예: 儒家, 道家, 法家 등 |
| `book_title` | 서명 |
| `chapter_title` | 편/장명 |
| `sentence` | 원문 문장 |
| `token` | 전처리 완료 토큰 리스트 |
| `Era` | 시대 구분. 선진시기 필터에는 `先秦` 사용 |

### 2. 문단 단위 분석

권장 파일명:

```text
data/processed/para_only_fin_251119_nega.pkl
```

필수 컬럼:

| 컬럼 | 의미 |
|---|---|
| `category` | DMR metadata |
| `book_title` | 서명 |
| `chapter_title` | 편/장명 |
| `paragraph` | 원문 문단 |
| `pg_token` | 문단 단위 전처리 토큰 리스트 |
| `Era` | 시대 구분 |

## 선택 파일

이미 학습된 모델을 공개하거나 내부 재현에 사용할 경우:

```text
models/250820_hakpa_dmr_model_k100_min30_rm0.bin
```

단, 모델 파일은 크기가 클 수 있으므로 GitHub 일반 커밋보다는 Git LFS, Release, OSF, Zenodo 등을 사용하는 편이 좋다.

## 원자료부터 재현할 경우 추가로 필요한 자료

현재 노트북은 이미 전처리된 `token`/`pg_token` 컬럼을 전제로 한다. 원문에서 토큰을 다시 만들려면 다음 자료도 필요하다.

1. 원문 코퍼스: `category`, `book_title`, `chapter_title`, `sentence` 또는 `paragraph`, `Era` 포함.
2. 정규화/치환 딕셔너리: 이체자·오자·고유명사 결합 규칙.
3. 불용어 목록: 문장부호, 장식 기호, 분석에서 제외한 한자/기호 목록.
4. 전처리 로그: 어떤 판본/출처의 텍스트를 어떤 순서로 병합했는지 설명하는 문서.
