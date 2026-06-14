# Pre-Qin Topic Modeling Pipeline

선진시기 문헌 코퍼스에 대해 `tomotopy`의 DMR(DMRModel)을 적용하고, 학습된 토픽에서 토픽어 및 대표 문장을 추출하기 위한 정리본이다.  

## 1. 저장소 구조

```text
preqin-topic-modeling/
├─ src/preqin_topic_modeling/
│  ├─ data.py                # 데이터 로드·컬럼 정규화·선진시기 필터
│  ├─ train_dmr.py           # DMR 학습 및 토픽어 저장
│  └─ extract_sentences.py   # 토픽어·대표문장 추출
├─ notebooks/
│  └─ 01_topic_modeling_clean.ipynb
├─ data/
│  ├─ README.md
│  └─ sample_schema.csv
├─ docs/
│  └─ CODE_CLEANING_REPORT.md
├─ config.example.yaml
├─ requirements.txt
├─ pyproject.toml
└─ .gitignore
```

## 2. 설치

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## 3. 필요한 데이터

이 저장소는 연구 데이터까지를 완전히 포함하지 않는다. 
그 대신 샘플 경량 파일을 제공한다. 다음 파일 중 하나를 별도로 준비한다.

### 문장 단위

```text
data/processed/st_df_All_fin_251120_nega.pkl
```

필수 컬럼: `category`, `book_title`, `chapter_title`, `sentence`, `token`, `Era`

### 문단 단위

```text
data/processed/para_only_fin_251119_nega.pkl
```

필수 컬럼: `category`, `book_title`, `chapter_title`, `paragraph`, `pg_token`, `Era`

`token` 또는 `pg_token`은 이미 전처리된 토큰 리스트여야 한다.

## 4. DMR 학습 실행

### 문장 단위 예시

```bash
python -m preqin_topic_modeling.train_dmr   --data-path data/processed/st_df_All_fin_251120_nega.pkl   --output-dir outputs   --model-name dmr_k100_min30_rm0.bin   --token-col token   --text-col sentence   --metadata-col category   --era-col Era   --era-value 先秦   --k 100   --min-df 30   --rm-top 0   --seed 2025   --term-weight PMI   --iterations 500   --step 20   --top-n-words 30
```

### 문단 단위 예시

```bash
python -m preqin_topic_modeling.train_dmr   --data-path data/processed/para_only_fin_251119_nega.pkl   --output-dir outputs   --model-name dmr_para_k35_min15_rm0.bin   --token-col pg_token   --text-col paragraph   --metadata-col category   --era-col Era   --era-value 先秦   --k 35   --min-df 15   --rm-top 0   --seed 2025   --term-weight PMI   --iterations 500   --step 20
```

실행 결과는 `outputs/run_YYYYMMDD_HHMMSS/` 아래에 저장된다.

## 5. 대표문장 추출

### 특정 키워드가 상위 토픽어에 포함된 토픽 추출

```bash
python -m preqin_topic_modeling.extract_sentences   --data-path data/processed/st_df_All_fin_251120_nega.pkl   --model-path outputs/run_YYYYMMDD_HHMMSS/models/dmr_k100_min30_rm0.bin   --output-dir outputs   --target-keywords 心,禮,法   --top-n-rank 30   --sentences-per-topic 50   --keyword-method hybrid   --sentence-method entropy
```

### 특정 토픽 ID의 대표문장 추출

```bash
python -m preqin_topic_modeling.extract_sentences   --data-path data/processed/st_df_All_fin_251120_nega.pkl   --model-path outputs/run_YYYYMMDD_HHMMSS/models/dmr_k100_min30_rm0.bin   --output-dir outputs   --topic-ids 0,5,12   --sentences-per-topic 50
```

## 6. GitHub 업로드 권장 방식

```bash
git init
git add README.md requirements.txt pyproject.toml config.example.yaml src notebooks data/README.md data/sample_schema.csv docs .gitignore
git commit -m "Add cleaned DMR topic modeling pipeline"
git branch -M main
git remote add origin https://github.com/<USER>/<REPOSITORY>.git
git push -u origin main
```

대용량 코퍼스와 모델 파일은 기본적으로 `.gitignore`에 의해 제외된다. 공개가 필요한 경우 Git LFS 또는 별도 연구 데이터 저장소를 사용한다
