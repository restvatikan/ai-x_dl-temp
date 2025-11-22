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

1.  [프로젝트 개요](https://www.google.com/search?q=%231-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EA%B0%9C%EC%9A%94-project-overview)
2.  [데이터셋 및 환경](https://www.google.com/search?q=%232-%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%85%8B-%EB%B0%8F-%ED%99%98%EA%B2%BD-dataset--environment)
3.  [방법론 및 실험 설계](https://www.google.com/search?q=%233-%EB%B0%A9%EB%B2%95%EB%A1%A0-%EB%B0%8F-%EC%8B%A4%ED%97%98-%EC%84%A4%EA%B3%84-methodology)
      * [3.1. 선형 모델 및 특성 추출](https://www.google.com/search?q=%2331-%EC%84%A0%ED%98%95-%EB%AA%A8%EB%8D%B8-%EB%B0%8F-%ED%8A%B9%EC%84%B1-%EC%B6%94%EC%B6%9C-linear-models--feature-engineering)
          * [3.1.1. TF-IDF + Logistic Regression](https://www.google.com/search?q=%23311-tf-idf--logistic-regression)
          * [3.1.2. Word2Vec + Logistic Regression](https://www.google.com/search?q=%23312-word2vec--logistic-regression)
      * [3.2. 트리 기반 모델](https://www.google.com/search?q=%2332-%ED%8A%B8%EB%A6%AC-%EA%B8%B0%EB%B0%98-%EB%AA%A8%EB%8D%B8-tree-based-models)
          * [3.2.1. Decision Tree](https://www.google.com/search?q=%23321-decision-tree)
      * [3.3. 딥러닝 모델 (CNN)](https://www.google.com/search?q=%2333-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EB%AA%A8%EB%8D%B8-deep-learning---cnn)
          * [3.3.1. Morpheme-level 1D-CNN](https://www.google.com/search?q=%23331-morpheme-level-1d-cnn)
          * [3.3.2. Syllable/Jamo-level 1D-CNN](https://www.google.com/search?q=%23332-syllablejamo-level-1d-cnn)
4.  [전처리 파이프라인 상세](https://www.google.com/search?q=%234-%EC%A0%84%EC%B2%98%EB%A6%AC-%ED%8C%8C%EC%9D%B4%ED%94%84%EB%9D%BC%EC%9D%B8-%EC%83%81%EC%84%B8-preprocessing-pipeline)
5.  [실험 결과 및 분석](https://www.google.com/search?q=%235-%EC%8B%A4%ED%97%98-%EA%B2%B0%EA%B3%BC-%EB%B0%8F-%EB%B6%84%EC%84%9D-results--analysis)
6.  [결론 및 향후 과제](https://www.google.com/search?q=%236-%EA%B2%B0%EB%A1%A0-%EB%B0%8F-%ED%96%A5%ED%9B%84-%EA%B3%BC%EC%A0%9C-conclusion)
7.  [사용 방법](https://www.google.com/search?q=%237-%EC%82%AC%EC%9A%A9-%EB%B0%A9%EB%B2%95-usage)

-----

## 1\. 프로젝트 개요

감성 분석(Sentiment Analysis)은 텍스트에 내재된 주관적 의견을 식별하는 과제입니다. 한국어는 교착어 특성상 조사와 어미 처리가 중요하므로, 본 연구에서는 단순 모델링을 넘어 \*\*"어떤 전처리(Preprocessing)와 어떤 토큰화(Tokenization)가 성능에 결정적인가?"\*\*를 핵심 질문으로 설정하고 실험을 진행합니다.

-----

## 2\. 데이터셋 및 환경

  * **Dataset:** [NSMC (Naver Sentiment Movie Corpus)](https://github.com/e9t/nsmc)
      * 총 20만 개 (Train 150,000개 / Test 50,000개)
      * Label: 0 (부정), 1 (긍정)
  * **Environment:** Python 3.x, Google Colab (GPU T4)
  * **Key Libraries:** `scikit-learn`, `konlpy`, `gensim`, `jamo`, `torch` (or `tensorflow`), `pandas`, `numpy`

-----

## 3\. 방법론 및 실험 설계

### 3.1. 선형 모델 및 특성 추출

가장 직관적이고 해석 가능한 로지스틱 회귀를 베이스라인으로 고정하고, 텍스트를 벡터화하는 방식에 따른 성능 차이를 비교합니다.

#### 3.1.1. TF-IDF + Logistic Regression

단어의 빈도(TF)와 역문서 빈도(IDF)를 활용하여 문서 내 핵심 단어에 가중치를 부여하는 방식입니다. 문맥보다는 특정 긍/부정 키워드의 존재 여부가 중요한 감성 분석에서 강력한 베이스라인이 됩니다.

  * **[난이도]** 하 (Low)
  * **[소요 시간]** 매우 빠름 (\< 5분)
  * **[예상 성능]** 중상 (Acc: 80% \~ 83%)
  * **[라이브러리]** `scikit-learn`, `konlpy`

#### 3.1.2. Word2Vec + Logistic Regression

단어의 의미를 벡터 공간에 매핑(Embedding)한 뒤, 문장에 포함된 모든 단어 벡터를 평균(Mean Pooling)내어 분류기에 입력합니다. 단어 간의 의미적 유사성을 반영할 수 있다는 장점이 있습니다.

  * **[난이도]** 중 (Medium)
  * **[소요 시간]** 빠름 (\~15분)
  * **[예상 성능]** 중 (Acc: 79% \~ 81%) - 단순 평균 시 키워드 정보 희석 가능성 존재
  * **[라이브러리]** `gensim`, `scikit-learn`, `konlpy`

### 3.2. 트리 기반 모델

#### 3.2.1. 결정 트리 (Decision Tree)

데이터의 특성을 기준으로 규칙(Rule)을 생성하여 분류하는 모델입니다. 직관적인 해석이 가능하나, 희소한(Sparse) 텍스트 데이터 특성상 과적합(Overfitting)이 발생하기 쉽습니다.

  * **[난이도]** 하 (Low)
  * **[소요 시간]** 보통 (\~10분)
  * **[예상 성능]** 하\~중 (Acc: 70% \~ 76%) - 가지치기(Pruning) 튜닝 필수
  * **[라이브러리]** `scikit-learn`

### 3.3. 딥러닝 모델 (CNN)

이미지 처리에 주로 쓰이는 CNN을 텍스트(1D-CNN)에 적용하여 문맥의 지역적 특징(Local feature)을 추출합니다. 입력 토큰의 단위를 다변화하여 비교합니다.

#### 3.3.1. Morpheme-level 1D-CNN

형태소 분석기를 통해 의미 단위로 분절된 토큰을 입력으로 사용합니다. 한국어의 문법적 특성을 가장 잘 반영하는 정석적인 접근법입니다.

  * **[난이도]** 중상 (Medium-High)
  * **[소요 시간]** 보통 (\~30분, GPU 필수)
  * **[예상 성능]** 상 (Acc: 83% \~ 85%)
  * **[라이브러리]** `torch`, `konlpy`

#### 3.3.2. Syllable/Jamo-level 1D-CNN

'음절(글자)' 혹은 '자소(자모)' 단위로 토큰화하여 입력합니다. 특히 자소 분리 실험에서는 **`jamo` 라이브러리의 `j2hcj(h2j())`** 방식을 적용합니다. 이는 텍스트를 초/중/종성으로 분리(`h2j`)한 뒤, 이를 다시 호환 자모(Compatibility Jamo)로 변환(`j2hcj`)하여 학습에 적합한 표준 형태로 가공하는 과정입니다. 형태소 분석의 오분석 가능성을 배제하고, 오탈자나 신조어(OOV)에 강건한 모델을 구축할 수 있습니다.

  *   **[난이도]** 상 (High) - 시퀀스 길이가 길어 모델 구조 설계 주의 필요
  *   **[소요 시간]** 느림 (\~40분 이상, GPU 필수)
  *   **[예상 성능]** 상 (Acc: 82% \~ 85%) - 구어체 데이터에서 효과적
  *   **[라이브러리]** `torch`, `jamo`

-----

## 4\. 전처리 파이프라인 상세

성능 향상의 핵심인 전처리는 **Basic**과 **Advanced** 두 단계로 나누어 진행하며, 그 효과를 정량적으로 비교합니다.

1.  **Data Cleaning:** 결측치(Null) 제거 및 중복 데이터 처리.
2.  **Basic Preprocessing:** 정규표현식(`[^ㄱ-ㅎ가-힣0-9a-zA-Z ]`)을 이용한 특수문자 제거.
3.  **Advanced Preprocessing (Key Step):**
      * **형태소 분석 및 정규화(Normalization):** `Okt.morphs(text, stem=True)`를 사용하여 용언의 활용형(예: 합니다 -\> 하다)을 원형으로 통일.
      * **불용어(Stopwords) 제거:** 조사 및 의미 없는 일반 명사 제거.
      * **필터링:** 길이가 1 이하인 토큰 제거.

-----

## 5\. 실험 결과 및 분석

### 5.1. 선형 모델 성능 비교

Logistic Regression 모델을 고정한 상태에서 전처리 및 특성 추출 방식에 따른 정확도 비교입니다.

| 모델 ID | 방법론 | 전처리 | 정확도 |
| :--- | :--- | :--- | :--- |
| **M1** | **TF-IDF + LR** | Basic (Regex Only) | 79.27% |
| **M2** | **TF-IDF + LR** | **Advanced (Stemming)** | **82.84%** |
| **M3** | **Word2Vec + LR** | Advanced (Stemming) | 81.10% |

**[분석]**

  * **전처리 효과:** 형태소 분석 기반의 정규화(Stemming)를 적용한 M2가 M1 대비 **약 3.57%p** 높은 성능을 기록했습니다. 이는 한국어 NLP에서 형태소 정규화가 필수적임을 시사합니다.
  * **방법론 비교:** 단순 감성 분류 과제에서는 전체 문맥을 평균 내는 Word2Vec(M3)보다, 핵심 감성 어휘에 가중치를 주는 TF-IDF(M2) 방식이 더 효율적이었습니다.

### 5.2. 딥러닝 모델 성능 비교

*(본 섹션은 프로젝트 진행 후 도출된 결과를 기입합니다)*

| 모델 | 토큰화 단위 | 정확도 |
| :--- | :--- | :--- |
| 1D-CNN | Morpheme (형태소) | TBD |
| 1D-CNN | Syllable (음절) | TBD |

-----

## 6\. 결론 및 향후 과제

본 연구를 통해 \*\*"형태소 기반의 정규화(Normalization)"\*\*가 모델의 복잡성을 높이는 것보다 성능 향상에 더 큰 기여를 함을 확인했습니다. 향후 과제로는 Transformer 기반의 BERT 모델(KoBERT)을 도입하여 문맥 파악 능력을 극대화하고 본 연구의 결과와 비교 분석할 예정입니다.

-----

## 7\. 사용 방법

본 프로젝트는 Jupyter Notebook(`.ipynb`) 형태로 제공됩니다.

1.  GitHub 저장소 상단의 `.ipynb` 파일을 클릭합니다.
2.  **"Open in Colab"** 버튼을 클릭하여 Google Colab 환경으로 이동합니다.
3.  런타임 유형을 **GPU**로 설정합니다.
4.  상단 셀부터 순차적으로 실행하면 데이터 다운로드부터 모델 학습까지 자동으로 진행됩니다.
