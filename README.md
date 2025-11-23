# AI+X: Deep Learning 2025-2
AI+X: 딥러닝 2025-2 기말 프로젝트

-----

**정보시스템학과 2024062042 김규민 oplgk0576@hanyang.ac.kr**

**정보시스템학과 2023092606 송정빈 happysongjb@hanyang.ac.kr**

-----

# 한국어 영화 리뷰 데이터를 활용한 감성 분석: 성능 비교 및 개선 연구

**Sentiment Analysis on Korean Movie Reviews: Performance Comparison and Improvement**

본 프로젝트는 **[NSMC (Naver Sentiment Movie Corpus)](https://github.com/e9t/nsmc)** 데이터셋을 활용하여 한국어 텍스트의 긍정/부정 감성을 분류하는 다양한 모델링 방법론을 비교 분석합니다. 고전적인 머신러닝 기법부터 딥러닝 모델까지 단계적으로 적용하며, 전처리 수준과 토큰화 단위가 모델 성능에 미치는 영향을 실증적으로 규명합니다.

본 리포트는 **Google Colab** 환경에서 누구나 실험 결과를 재현할 수 있도록 작성되었습니다.

-----

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [데이터셋 및 환경](#2-데이터셋-및-환경)
3. [방법론 및 실험 설계](#3-방법론-및-실험-설계)
    - [3.1. 통계 기반 머신러닝 (Baseline)](#31-통계-기반-머신러닝-baseline)
        - [3.1.1. TF-IDF 기반 모델링](#311-tf-idf-기반-모델링)
        - [3.1.2. Word2Vec 임베딩 모델링](#312-word2vec-임베딩-모델링)
        - [3.1.3. 결정 트리 (Decision Tree)](#313-결정-트리-decision-tree)
    - [3.2. 딥러닝 모델 (1D-CNN)](#32-딥러닝-모델-1d-cnn)
        - [3.2.1. Morpheme-level 1D-CNN (형태소 단위)](#321-morpheme-level-1d-cnn-형태소-단위)
        - [3.2.2. Syllable-level 1D-CNN (음절 단위)](#322-syllable-level-1d-cnn-음절-단위)
        - [3.2.3. Jamo-level 1D-CNN (자소 단위)](#323-jamo-level-1d-cnn-자소-단위)
    - [3.3. 딥러닝 모델 (LSTM)](#33-딥러닝-모델-lstm)
        - [3.3.1. Morpheme-level LSTM (형태소 단위)](#331-morpheme-level-lstm-형태소-단위)
4. [전처리 파이프라인 상세](#4-전처리-파이프라인-상세-preprocessing-strategy)
5. [최종 성능 평가 및 분석 (Final Evaluation)](#5-최종-성능-평가-및-분석-final-evaluation)
    - [5.1. 예측 결과 저장 (Inference & Logging)](#51-예측-결과-저장-inference--logging)
    - [5.2. 성능 지표 계산 및 시각화 (Metrics & Visualization)](#52-성능-지표-계산-및-시각화-metrics--visualization)


-----


## 1\. 프로젝트 개요

감성 분석(Sentiment Analysis)은 텍스트에 내재된 주관적 의견을 식별하는 과제입니다. 한국어는 교착어 특성상 조사와 어미 처리가 중요하므로, 본 연구에서는 단순 모델링을 넘어 \*\*"어떤 전처리(Preprocessing)와 어떤 토큰화(Tokenization)가 성능에 결정적인가?"\*\*를 핵심 질문으로 설정하고 실험을 진행합니다.

-----

## 2\. 데이터셋 및 환경

* **데이터셋:** [NSMC (Naver Sentiment Movie Corpus)](https://github.com/e9t/nsmc)
- Each file is consisted of three columns: `id`, `document`, `label`
    - `id`: The review id, provieded by Naver
    - `document`: The actual review
    - `label`: The sentiment class of the review. (0: negative, 1: positive)
    - Columns are delimited with tabs (i.e., `.tsv` format; but the file extension is `.txt` for easy access for novices)
- 200K reviews in total
    - `ratings.txt`: All 200K reviews
    - `ratings_test.txt`: 50K reviews held out for testing
    - `ratings_train.txt`: 150K reviews for training   
- All reviews are shorter than 140 characters
- Each sentiment class is sampled equally (i.e., random guess yields 50% accuracy)
    - 100K negative reviews (originally reviews of ratings 1-4)
    - 100K positive reviews (originally reviews of ratings 9-10)
    - Neutral reviews (originally reviews of ratings 5-8) are excluded

* **실험 환경 (Experimental Environment):**
  - **Platform:** Google Colab (GPU T4 Runtime)
  - **Python Version:** Python 3.12.12
  - **Key Libraries:** `scikit-learn`, `konlpy`, `gensim`, `jamo`, `torch`, `pandas`, `numpy`, `matplotlib`, `tqdm`
  - **Library Installation:** 본 프로젝트는 단일 `.ipynb` 환경에서 수행됩니다. 빠른 속도와 높은 정확도를 위해 **Mecab** 형태소 분석기를 사용하며, 필요한 라이브러리 설치 코드는 아래와 같습니다.

  ```python
  # 1. Install Mecab (Korean Morphological Analyzer)
  # C++ based, extremely fast compared to Java-based KoNLPy
  !pip install python-mecab-ko

  # 2. Install Common Libraries
  # jamo: 자소 분리 / gensim: Word2Vec / torch: 딥러닝
  !pip install jamo gensim torch scikit-learn pandas numpy matplotlib tqdm
  ```

-----

## 3\. 방법론 및 실험 설계

본 프로젝트는 전처리 수준과 모델의 복잡도가 성능에 미치는 영향을 다각도로 분석하기 위해 다음과 같이 실험군을 세분화하여 비교합니다.

### 3.1. 통계 기반 머신러닝 (Baseline)

가장 기본적인 모델인 로지스틱 회귀(Logistic Regression)와 결정 트리(Decision Tree)를 사용하여, **텍스트 전처리(Morphological Analysis & Stopwords Removal)의 유무**가 성능에 미치는 영향을 직접적으로 비교합니다.

#### 3.1.1. TF-IDF 기반 모델링
단어의 빈도를 가중치로 사용하는 TF-IDF 벡터화 방식을 사용합니다.
*   **Exp 1-1. TF-IDF (Raw):** 형태소 분석 없이 띄어쓰기(Whitespace) 기준으로 토큰화하며, 불용어 처리를 하지 않은 순수 데이터.
*   **Exp 1-2. TF-IDF (Preprocessed):** **`Mecab`** 분석기를 사용하여 조사/어미 등을 정교하게 분리하고 불용어를 제거한 데이터.
*   **[사용 라이브러리]**
    *   **Preprocessing:** `mecab.MeCab` (Exp 1-2 only)
    *   **Feature Extraction:** `sklearn.feature_extraction.text.TfidfVectorizer`
    *   **Model:** `sklearn.linear_model.LogisticRegression`

#### 3.1.2. Word2Vec 임베딩 모델링
단어의 의미 정보를 반영하는 Word2Vec 임베딩을 적용한 후, 문장 내 단어 벡터들의 평균(Mean Pooling)값을 입력으로 사용합니다.
*   **Exp 2-1. Word2Vec (Raw):** 전처리 없는 띄어쓰기 기준 토큰의 임베딩 학습.
*   **Exp 2-2. Word2Vec (Preprocessed):** 형태소 분절 및 불용어 제거를 수행한 토큰의 임베딩 학습.
*   **[사용 라이브러리]**
    *   **Preprocessing:** `mecab.MeCab` (Exp 2-2 only)
    *   **Embedding:** `gensim.models.Word2Vec`
    *   **Operation:** `numpy` (Mean Pooling)
    *   **Model:** `sklearn.linear_model.LogisticRegression`

#### 3.1.3. 결정 트리 (Decision Tree)
희소(Sparse)한 텍스트 데이터에서 트리 모델이 갖는 한계를 확인하고, 전처리가 트리의 복잡도 완화에 미치는 영향을 확인합니다.
*   **Exp 3-1. Decision Tree (Raw):** 전처리 없는 고차원 데이터 학습.
*   **Exp 3-2. Decision Tree (Preprocessed):** 핵심 형태소만 남긴 데이터 학습.
*   **[사용 라이브러리]**
    *   **Preprocessing:** `mecab.MeCab` (Exp 3-2 only)
    *   **Feature Extraction:** `sklearn.feature_extraction.text.TfidfVectorizer`
    *   **Model:** `sklearn.tree.DecisionTreeClassifier`

### 3.2. 딥러닝 모델 (1D-CNN)

딥러닝 모델에서는 **"입력 토큰의 단위(Granularity)"**를 핵심 변수로 설정하여, 한국어의 언어적 특성에 가장 적합한 표현 방식을 탐구합니다. 모든 CNN 모델은 `PyTorch`를 사용하여 구현하며, 동일한 아키텍처(Kernel sizes, Filters)를 공유합니다.

#### 3.2.1. Morpheme-level 1D-CNN (형태소 단위)
*   **특징:** 의미의 최소 단위인 형태소를 입력으로 사용합니다. 문법적 구조를 가장 잘 반영하는 정석적인 방법입니다.
*   **입력:** `Mecab`을 통해 분절된 형태소 시퀀스.
*   **[사용 라이브러리]**
    *   **Preprocessing:** `mecab.MeCab`
    *   **Model:** `torch.nn` (Embedding, Conv1d, Linear)
    *   **Optim:** `torch.optim`

#### 3.2.2. Syllable-level 1D-CNN (음절 단위)
*   **특징:** '글자(Character)' 단위로 문장을 쪼개어 입력합니다. OOV(Out of Vocabulary) 문제가 거의 발생하지 않으며, 오탈자에 비교적 강건합니다.
*   **입력:** 한국어 음절 시퀀스 (예: "영화" -> "영", "화")
*   **[사용 라이브러리]**
    *   **Preprocessing:** Python Native String Methods (Slicing)
    *   **Model:** `torch.nn` (Embedding, Conv1d, Linear)
    *   **Optim:** `torch.optim`

#### 3.2.3. Jamo-level 1D-CNN (자소 단위)
*   **특징:** 한글의 자모(초성, 중성, 종성)를 분리하여 입력합니다. 음절보다 더 작은 단위의 패턴을 학습하여, 신조어나 비표준어(초성체 등)가 많은 리뷰 데이터에서 잠재력을 가집니다.
*   **입력:** `jamo` 라이브러리의 `j2hcj` 함수를 활용한 자소 시퀀스 (예: "영화" -> "ㅇ", "ㅕ", "ㅇ", "ㅎ", "ㅘ")
*   **[사용 라이브러리]**
    *   **Preprocessing:** `jamo.h2j`, `jamo.j2hcj`
    *   **Model:** `torch.nn` (Embedding, Conv1d, Linear)
    *   **Optim:** `torch.optim`

### 3.3. 딥러닝 모델 (LSTM)

순차 데이터(Sequential Data) 처리에 특화된 RNN 계열의 LSTM을 사용하여, 문맥의 장기 의존성(Long-term Dependency)을 학습합니다. CNN이 지역적 특징(Local Feature) 추출에 강하다면, LSTM은 문장 전체의 흐름을 파악하는 데 강점이 있습니다.

#### 3.3.1. Morpheme-level LSTM (형태소 단위)
*   **특징:** 3.2.1의 CNN 모델과 동일한 '형태소 전처리 데이터'를 사용하여, 아키텍처(CNN vs LSTM)에 따른 성능 차이를 공정하게 비교합니다. 정교한 형태소 분절과 불용어 제거를 통해 핵심 의미 단위의 시퀀스를 학습합니다.
*   **입력:** `Mecab`을 통해 분절된 형태소 시퀀스.
*   **[사용 라이브러리]**
    *   **Preprocessing:** `mecab.MeCab`
    *   **Model:** `torch.nn` (Embedding, LSTM, Linear, Dropout)
    *   **Sequence Handling:** `torch.nn.utils.rnn` (`pad_sequence`, `pack_padded_sequence`, `pad_packed_sequence`)
        *   *Note: 가변 길이 시퀀스의 효율적 학습을 위해 패딩(Padding) 및 패킹(Packing) 기법을 적용합니다.*
    *   **Optim:** `torch.optim`


-----

## 4\. 전처리 파이프라인 상세 (Preprocessing Strategy)

본 연구는 모델과 실험 목적에 따라 총 3가지의 상이한 전처리 파이프라인을 적용합니다.

### Pipeline A: Minimal Preprocessing (Raw)
> **적용 대상:** Exp 1-1, 2-1, 3-1 (Raw 비교군) 및 3.2.2 (Syllable CNN)
*   가장 기초적인 정제만 수행하여 데이터 본연의 노이즈를 보존합니다.
1.  **Data Cleaning:** 결측치(Null) 및 중복 데이터 제거.
2.  **Regex Cleaning:** 한글, 영어, 숫자, 공백을 제외한 특수문자 제거 (`[^ㄱ-ㅎ가-힣0-9a-zA-Z ]`).
3.  **Tokenization:** 공백(띄어쓰기) 기준 분절 (Syllable CNN의 경우 글자 단위 분절).

### Pipeline B: Linguistic Preprocessing (Morphology)
> **적용 대상:** Exp 1-2, 2-2, 3-2 (Preprocessed 비교군) 및 **3.2.1 (Morpheme CNN), 3.3.1 (Morpheme LSTM)**
*   한국어 문법 지식을 활용하여 의미 단위로 데이터를 압축합니다.
1.  **Basic Cleaning:** Pipeline A와 동일 (결측치 및 특수문자 제거).
2.  **Morphological Analysis:** **`Mecab` (python-mecab-ko)** 사용.
    *   **Precise Segmentation:** 문장을 형태소 단위로 정밀하게 분해 (예: '재밌었다' -> '재밌', '었', '다').
    *   *Note: Okt와 달리 인위적인 원형 복원(Stemming)을 수행하지 않고, 형태소 본연의 의미를 보존하여 학습에 활용함.*
3.  **Stopwords Removal:** 조사, 접속사 등 의미 기여도가 낮은 불용어 리스트 정의 및 제거.
4.  **Filtering:** 길이가 1 이하인 토큰 제거.

### Pipeline C: Sub-character Preprocessing (Jamo)
> **적용 대상:** 3.2.3 (Jamo CNN)
*   한글의 구성 원리를 활용하여 자소 단위로 데이터를 분해합니다.
1.  **Basic Cleaning:** Pipeline A와 동일.
2.  **Jamo Decomposition:** `jamo` 패키지 활용.
    *   `h2j()`: 한글 음절을 초/중/종성으로 분리.
    *   `j2hcj()`: 분리된 자모를 호환 자모(Compatibility Jamo) 코드로 변환하여 학습 가능한 시퀀스로 생성.

### Data Efficiency Strategy (Pickle Caching)
Google Colab의 런타임 초기화 문제에 대응하고 실험 효율성을 높이기 위해, 각 파이프라인을 거친 데이터는 **`pickle`** 형식으로 로컬에 캐싱(Caching)합니다.
*   **목적:** 형태소 분석(Pipeline B) 및 자소 분리(Pipeline C) 과정을 매 실험마다 반복하지 않고, 저장된 리스트 객체를 즉시 로드하여 학습 시간을 단축합니다.
*   **파일 포맷:** `train_morphs.pkl`, `train_jamo.pkl` 등.

-----

## 5\. 최종 성능 평가 및 분석 (Final Evaluation)

모든 모델의 학습이 완료된 후, Test Set(50,000개)에 대한 객관적인 성능 비교를 위해 통일된 평가 프로세스를 진행합니다.

### 5.1. 예측 결과 저장 (Inference & Logging)
각 모델(3.1 ~ 3.3)로 테스트 데이터셋에 대한 추론을 수행하고, 예측된 라벨(0 또는 1)을 텍스트 파일로 저장합니다. 이를 통해 실험 코드를 다시 돌리지 않고도 결과를 영구적으로 보존하고 분석할 수 있습니다.
*   **Output Format:** `prediction_[ModelName].txt` (각 행에 0 또는 1 기록)
*   **[사용 라이브러리]**
    *   `torch` (Inference mode: `with torch.no_grad():`)
    *   `sklearn` (Model `predict` method for ML models)
    *   `numpy` (Array handling)

### 5.2. 성능 지표 계산 및 시각화 (Metrics & Visualization)
저장된 예측 파일들과 정답 라벨(`ratings_test.txt`)을 로드하여 최종 성능을 계산합니다. 단순 정확도(Accuracy)뿐만 아니라, 모델이 어떤 클래스를 헷갈려하는지 파악하기 위해 혼동 행렬(Confusion Matrix)을 시각화합니다.
1.  **Accuracy Calculation:** 전체 테스트 데이터 중 올바르게 분류한 비율 계산.
2.  **Confusion Matrix:** True Positive, True Negative, False Positive, False Negative 분포 확인.
*   **[사용 라이브러리]**
    *   **Metrics:** `sklearn.metrics.accuracy_score`, `sklearn.metrics.confusion_matrix`
    *   **Data Handling:** `pandas` (결과 집계), `numpy`
    *   **Visualization:** `matplotlib.pyplot`, `seaborn` (Heatmap 시각화)

-----
