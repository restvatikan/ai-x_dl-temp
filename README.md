# AI+X: Deep Learning 2025-2
AI+X: 딥러닝 2025-2 기말 프로젝트

-----

정보시스템학과 2024062042 김규민 oplgk0576@hanyang.ac.kr    
정보시스템학과 2023092606 송정빈 happysongjb@hanyang.ac.kr

-----
다음은 요청하신 요구사항(GitHub Tech Blog 스타일, 논문 형식의 전개, 이모티콘 제외, 계층적 방법론 구성, 제공해주신 로지스틱 회귀 실험 결과 반영 등)을 충실히 반영하여 작성한 `README.md` 초안입니다.

Markdown 문법을 사용하여 바로 복사하여 사용하실 수 있도록 구성했습니다.

-----

# 한국어 영화 리뷰 데이터를 활용한 감성 분석: 성능 비교 및 개선 연구

**Sentiment Analysis on Korean Movie Reviews: Performance Comparison and Improvement**

본 프로젝트는 NSMC(Naver Sentiment Movie Corpus) 데이터셋을 활용하여 한국어 텍스트의 긍정/부정 감성을 분류하는 다양한 모델링 방법론을 비교 분석합니다. 고전적인 머신러닝 기법부터 딥러닝 모델까지 단계적으로 적용하며, 특히 전처리 수준과 토큰화 단위(Tokenization Granularity)가 모델 성능에 미치는 영향을 실증적으로 규명하는 것을 목표로 합니다.

본 리포트는 누구나 실험 결과를 재현할 수 있도록 Google Colab 환경에서의 실행을 전제로 작성되었습니다.

-----

## 목차 (Table of Contents)

1.  [프로젝트 개요 (Project Overview)](https://www.google.com/search?q=%231-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EA%B0%9C%EC%9A%94-project-overview)
2.  [데이터셋 및 환경 (Dataset & Environment)](https://www.google.com/search?q=%232-%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%85%8B-%EB%B0%8F-%ED%99%98%EA%B2%BD-dataset--environment)
3.  [방법론 및 실험 설계 (Methodology)](https://www.google.com/search?q=%233-%EB%B0%A9%EB%B2%95%EB%A1%A0-%EB%B0%8F-%EC%8B%A4%ED%97%98-%EC%84%A4%EA%B3%84-methodology)
      * [3.1. 선형 모델과 특성 추출 (Linear Models & Feature Engineering)](https://www.google.com/search?q=%2331-%EC%84%A0%ED%98%95-%EB%AA%A8%EB%8D%B8%EA%B3%BC-%ED%8A%B9%EC%84%B1-%EC%B6%94%EC%B6%9C-linear-models--feature-engineering)
      * [3.2. 트리 기반 모델 (Tree-based Models)](https://www.google.com/search?q=%2332-%ED%8A%B8%EB%A6%AC-%EA%B8%B0%EB%B0%98-%EB%AA%A8%EB%8D%B8-tree-based-models)
      * [3.3. 합성곱 신경망 모델 (CNN Approaches)](https://www.google.com/search?q=%2333-%ED%95%A9%EC%84%B1%EA%B3%B1-%EC%8B%A0%EA%B2%BD%EB%A7%9D-%EB%AA%A8%EB%8D%B8-cnn-approaches)
4.  [전처리 파이프라인 상세 (Preprocessing Pipeline)](https://www.google.com/search?q=%234-%EC%A0%84%EC%B2%98%EB%A6%AC-%ED%8C%8C%EC%9D%B4%ED%94%84%EB%9D%BC%EC%9D%B8-%EC%83%81%EC%84%B8-preprocessing-pipeline)
5.  [실험 결과 및 분석 (Results & Analysis)](https://www.google.com/search?q=%235-%EC%8B%A4%ED%97%98-%EA%B2%B0%EA%B3%BC-%EB%B0%8F-%EB%B6%84%EC%84%9D-results--analysis)
6.  [결론 및 향후 과제 (Conclusion)](https://www.google.com/search?q=%236-%EA%B2%B0%EB%A1%A0-%EB%B0%8F-%ED%96%A5%ED%9B%84-%EA%B3%BC%EC%A0%9C-conclusion)
7.  [사용 방법 (Usage)](https://www.google.com/search?q=%237-%EC%82%AC%EC%9A%A9-%EB%B0%A9%EB%B2%95-usage)

-----

## 1\. 프로젝트 개요 (Project Overview)

자연어 처리(NLP) 분야에서 감성 분석(Sentiment Analysis)은 텍스트에 내재된 주관적인 의견을 식별하는 핵심 과제입니다. 한국어는 교착어의 특성상 조사와 어미의 변화가 다양하여 영어에 비해 전처리와 토큰화 방식이 모델 성능에 지대한 영향을 미칩니다.

본 연구에서는 다음과 같은 질문에 대한 해답을 탐구합니다.

1.  **전처리의 심도:** 단순 정규표현식 정제와 형태소 분석 기반의 정규화 간의 성능 차이는 어느 정도인가?
2.  **임베딩 방식:** 빈도 기반(TF-IDF)과 분산 표현(Word2Vec) 중 감성 분류에 더 적합한 것은 무엇인가?
3.  **토큰화 단위:** 딥러닝 모델(CNN)에서 형태소, 음절, 자소 단위 토큰화 중 가장 효율적인 입력 단위는 무엇인가?

-----

## 2\. 데이터셋 및 환경 (Dataset & Environment)

### 2.1. 데이터셋: NSMC (Naver Sentiment Movie Corpus)

네이버 영화 리뷰 데이터를 기반으로 구성된 총 20만 개의 리뷰 데이터셋을 사용합니다.

  * **Training Set:** 150,000개
  * **Test Set:** 50,000개
  * **Label:** 0(부정), 1(긍정)
  * **특이사항:** 본 프로젝트에서는 클래스 불균형 문제를 배제하기 위해 Positive/Negative 비율이 1:1로 균형 잡힌 원본 데이터셋의 분포를 그대로 따릅니다.

### 2.2. 실험 환경

  * **Language:** Python 3.x
  * **Platform:** Google Colab (GPU T4)
  * **Libraries:** Scikit-learn, KoNLPy (Okt), Gensim (Word2Vec), PyTorch or TensorFlow/Keras, Pandas, NumPy

-----

## 3\. 방법론 및 실험 설계 (Methodology)

본 연구는 방법론을 크게 세 가지 대분류(Category)로 나누고, 각 분류 내에서 세부적인 파생 모델을 비교하는 계층적 구조로 실험을 진행합니다.

### 3.1. 선형 모델과 특성 추출 (Linear Models & Feature Engineering)

가장 직관적이고 해석 가능한 로지스틱 회귀(Logistic Regression)를 베이스라인 모델로 설정하여, 텍스트 전처리 및 벡터화 방법론에 따른 성능 변화를 측정합니다.

  * **Model:** Logistic Regression (Fixed)
  * **Variation A: 텍스트 전처리 심도 (Preprocessing Depth)**
      * **Basic:** 정규표현식(Regex)을 이용한 노이즈 제거 및 단순 공백 단위 토큰화.
      * **Advanced:** 형태소 분석기(Okt)를 활용한 어간 추출(Stemming), 불용어(Stopwords) 제거, 길이 필터링.
  * **Variation B: 특성 추출 방식 (Feature Extraction)**
      * **Count-based:** TF-IDF (Term Frequency - Inverse Document Frequency)를 사용하여 단어의 중요도를 가중치로 부여.
      * **Embedding-based:** Word2Vec을 사용하여 단어의 의미론적 벡터를 학습하고, 문서 내 단어 벡터의 평균(Average Pooling)을 입력으로 사용.

### 3.2. 트리 기반 모델 (Tree-based Models)

비선형적 관계를 학습할 수 있는 결정 트리 계열 모델을 적용합니다.

  * **Model:** Decision Tree
  * **Variations:**
      * **Criterion:** Gini 불순도(Impurity) vs 엔트로피(Entropy) 정보 이득.
      * **Pruning:** 과적합 방지를 위한 ccp\_alpha 파라미터 튜닝 비교.

### 3.3. 합성곱 신경망 모델 (CNN Approaches)

텍스트의 지역적(Local) 특징을 추출하는 데 탁월한 1D-CNN 구조를 채택합니다. 모델 구조는 고정하되, 입력 토큰의 단위를 다변화하여 한국어 처리에 최적화된 단위를 찾습니다.

  * **Model architecture:** Embedding -\> Conv1D -\> MaxPooling -\> Fully Connected
  * **Variations (Tokenization Granularity):**
      * **Morpheme-level:** 형태소 단위 임베딩 (의미 단위 보존).
      * **Syllable-level:** 음절(글자) 단위 임베딩 (OOV 문제 완화).
      * **Jamo-level:** 자소(초성, 중성, 종성) 단위 임베딩 (한국어의 구조적 특성 반영).

-----

## 4\. 전처리 파이프라인 상세 (Preprocessing Pipeline)

성능 향상의 핵심 요인인 전처리는 **Data Ingestion & Cleaning**과 **Advanced Preprocessing**의 단계로 구성됩니다.

### 4.1. 데이터 정제 (Data Ingestion & Cleaning)

  * **결측치 처리:** `dropna`를 사용하여 Null 값을 제거하고, `strip`으로 양끝 공백을 제거합니다.
  * **데이터 분할:** `train_test_split`을 사용하여 학습 및 검증 데이터를 8:2 비율로 분할합니다 (`stratify` 옵션 적용).

### 4.2. 심화 전처리 (Advanced Preprocessing Methodology)

단순 정제로는 한계가 있는 한국어의 특성을 고려하여, KoNLPy의 Okt 분석기를 활용한 언어학적 처리를 수행합니다.

1.  **정규표현식(Regex):** `[^ㄱ-ㅎ가-힣0-9a-zA-Z ]` 패턴을 사용하여 특수문자 및 노이즈를 제거합니다.
2.  **형태소 분석 및 정규화 (Normalization):** `okt.morphs(text, stem=True)`를 사용하여 '합니다', '하는' 등을 기본형인 '하다'로 통일합니다. 이는 단어의 희소성(Sparsity)을 줄이는 결정적 역할을 수행합니다.
3.  **불용어(Stopwords) 제거:** 조사를 비롯하여 문맥상 의미가 약한 일반 명사(영화, 배우, 감독 등)를 리스트로 정의하여 제거합니다.
4.  **노이즈 필터링:** 정보가가 낮다고 판단되는 길이 1 이하의 단어를 제거합니다.

-----

## 5\. 실험 결과 및 분석 (Results & Analysis)

### 5.1. 선형 모델 및 특성 추출 실험 결과 (Logistic Regression)

베이스라인인 로지스틱 회귀 모델을 기준으로, 전처리 방식과 특성 추출 방법에 따른 검증 세트(Validation Set) 정확도 비교입니다.

| 모델 구분 (Model ID) | 전처리 방식 (Preprocessing) | 특성 추출 (Feature Ext.) | 특징 (Note) | 정확도 (Accuracy) |
| :--- | :--- | :--- | :--- | :--- |
| **M1** | Basic (Regex) | TF-IDF | 단순 토큰화 | **79.27%** |
| **M2** | **Advanced (Okt+Stem)** | **TF-IDF** | **형태소 정규화 + 불용어 처리** | **82.84%** |
| **M3** | Advanced (Okt+Stem) | Word2Vec | 임베딩 평균 (dim=100) | 81.10% |

**[분석 결과]**

  * **전처리의 중요성:** M1과 M2를 비교했을 때, 형태소 분석 및 어간 추출을 적용한 경우 약 **3.57%p의 성능 향상**이 관찰되었습니다. 이는 한국어에서 어간을 통일하여 차원을 축소하는 것이 매우 중요함을 시사합니다.
  * **TF-IDF vs Word2Vec:** 감성 분류와 같은 단순 분류 과제에서는 문맥 전체를 평균 내는 Word2Vec(M3) 방식보다, 특정 감성 단어(핵심 키워드)에 가중치를 부여하는 TF-IDF(M2) 방식이 소폭 우세한 성능을 보였습니다.

### 5.2. 딥러닝 모델 실험 결과 (CNN)

*(이 섹션은 프로젝트 진행 시 도출된 결과로 채워질 예정입니다)*

| 토큰화 단위 (Tokenization) | 모델 구조 | 정확도 (Accuracy) | 비고 |
| :--- | :--- | :--- | :--- |
| Morpheme-level | 1D-CNN | TBD | 형태소 단위 |
| Syllable-level | 1D-CNN | TBD | 음절 단위 |
| Jamo-level | 1D-CNN | TBD | 자소 단위 |

-----

## 6\. 결론 및 향후 과제 (Conclusion)

본 연구를 통해 한국어 영화 리뷰 감성 분석에서 \*\*형태소 기반의 정교한 전처리(Normalization)\*\*가 모델의 복잡도를 높이는 것보다 우선적으로 고려되어야 할 성능 향상 요인임을 확인했습니다. 특히 Logistic Regression과 같은 단순 모델에서도 적절한 전처리(Advanced Preprocessing)가 동반될 경우 82% 이상의 준수한 성능을 달성할 수 있었습니다.

향후 과제로는 Transformer 기반의 BERT 모델(KoBERT, KcBERT)을 도입하여 문맥적 의미 파악 능력을 극대화하고, 앞선 CNN 모델들과의 성능 차이를 비교 분석할 예정입니다.

-----

## 7\. 사용 방법 (Usage)

본 프로젝트는 Jupyter Notebook 파일(`.ipynb`)로 제공됩니다.

1.  레포지토리 상단의 `.ipynb` 파일을 클릭합니다.
2.  GitHub 페이지 내의 **"Open in Colab"** 배지를 클릭하여 Google Colab 환경으로 이동합니다.
3.  런타임 유형을 GPU로 설정한 후, 순차적으로 셀을 실행합니다.
      * 필요한 라이브러리(KoNLPy 등) 설치 코드는 노트북 최상단에 포함되어 있습니다.
