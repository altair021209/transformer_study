# Transformer Architecture & Training: A Linear Algebra Perspective

## 0. Key Concepts & Definitions (Pre-requisites)

문서를 읽기 전, 질문하신 핵심 변수와 용어에 대한 정의입니다.

### 1. $d_k$ (Dimension of Key)
* **정의:** Multi-Head Attention에서 **각 헤드(Head)가 사용하는 부분 공간(Subspace)의 차원 크기**입니다.
* **계산:** 전체 모델 차원($d_{model}$)을 헤드 개수($h$)로 나눈 값입니다.
    * 논문 기준: $512 / 8 = 64$. 즉, **$d_k = 64$** 입니다.
* **역할:**
    * **Subspace Size:** 512차원의 거대한 정보를 64차원짜리 8개로 쪼개서 병렬 처리할 때, 그 '작은 조각'의 크기입니다.
    * **Scaling Factor:** 내적($Q \cdot K^T$) 값이 차원이 커질수록 기하급수적으로 커지는 것을 막기 위해, 나누는 수($\sqrt{d_k} = 8$)로 사용됩니다.

### 2. 핵심 선형대수 행위 (Action Verbs)
모델이 데이터를 다루는 방식을 3가지로 구분합니다.

* **Mapping (매핑 / 사상)**
    * **의미:** "공간 이동". $A$ 공간에 있는 데이터를 $B$ 공간으로 옮기는 모든 행위.
    * **예시:** `Embedding` (정수 공간 $\to$ 벡터 공간), `FFN Expansion` (512차원 $\to$ 2048차원).
    * **목적:** 데이터를 다루기 쉬운 형태나 더 넓은 공간으로 보내 표현력을 높임.

* **Projection (투영 / 사영)**
    * **의미:** "정보 압축 & 시점 변경". 고차원 데이터를 저차원 부분 공간(Subspace)으로 찌그러뜨려 넣는 행위.
    * **예시:** `Linear Layer` (512 $\to$ 64).
    * **목적:** 불필요한 정보는 버리고(Loss), **핵심 특징(Feature)**만 남기거나, 데이터를 바라보는 **기저(Basis)**를 바꿈.

* **Alignment (정렬 / 맞춤)**
    * **의미:** "방향 일치". 두 벡터 사이의 각도를 좁혀서($\theta \to 0$) 비슷하게 만드는 행위.
    * **예시:** `Dot Product Attention` ($Q \cdot K^T$), `Cross Entropy Loss`.
    * **목적:** Query가 찾고자 하는 정보(Key)를 정확히 가리키게 하거나, 모델의 출력($Z$)이 정답($W$)과 같아지도록 유도함.

---

## 1. Mathematical Formulation Setup

* **$B$:** Batch Size
* **$L$:** Sequence Length ($L_{enc}$ or $L_{dec}$)
* **$d$:** Hidden Dimension ($d_{model} = 512$)
* **$V$:** Vocabulary Size (약 30,000 ~ 50,000)
* **$h$:** Number of Heads (8)
* **$d_k$:** Head Dimension ($64$)

---

## 2. Preprocessing & Embedding

### 2-1. Tokenize
텍스트를 정수 인덱스로 변환합니다. 학습 시 `Input(Source)`와 `Input(Target)` 두 가지 텐서를 준비합니다.
* **형태:** `[Batch, Seq_len]` (Integer Vector)

### 2-2. Embedding & Scaling
$$X_{input} = (X_{token} \times W_{emb}) \cdot \sqrt{d}$$
* **Mapping:** 이산적인 정수 인덱스를 연속적인 고차원 벡터 공간 $\mathbb{R}^{512}$로 매핑합니다.
* **Scaling ($\sqrt{d}$):** 차원이 커질수록 내적 값의 분산이 커지는 것을 막기 위해, 벡터의 Norm(크기)을 키워 **분산의 균형(Variance Balance)**을 맞춥니다.
* **Dims:** `(B, L)` $\rightarrow$ `(B, L, 512)`

### 2-3. Positional Encoding
$$X_{final} = X_{input} \oplus PE$$
* **Mapping:** 위치 정보(Index)를 삼각함수를 이용해 고유한 벡터로 변환합니다.
* **Element-wise Sum:** 위치 정보와 의미 정보를 서로 다른 부분 공간(Subspace)에 중첩(Superposition) 시킵니다.

* ### 0-3. Positional Encoding: Matrix Formulation & Linear Algebra View

위치 정보가 없는 벡터 공간에 순서를 부여하기 위해, **고유한 패턴을 가진 상수 행렬 $P$**를 입력 행렬 $X$에 더합니다.

#### 1. 행렬 연산 정의 (Matrix Operation)
$$X_{final} = X_{embedding} + P$$

* **$X_{embedding}$:** `(L, d)` 크기의 학습 가능한 임베딩 행렬.
* **$P$:** `(L, d)` 크기의 **고정된(Fixed) 위치 인코딩 행렬**.

#### 2. 행렬 $P$의 내부 구조 (Construction)
행렬 $P$의 $pos$번째 행(Row), $i$번째 열(Column)의 원소는 다음과 같이 정의됩니다.

$$
P \in \mathbb{R}^{L \times d}
$$

이 행렬은 **짝수 열(Even columns)**에는 사인(Sin) 파형을, **홀수 열(Odd columns)**에는 코사인(Cos) 파형을 채워 넣습니다.

* **각속도 (Angular Frequency):** $\omega_k = \frac{1}{10000^{2k/d}}$
    * $pos$: 문장 내 위치 인덱스 ($0, 1, \dots, L-1$)
    * $k$: 차원 인덱스 ($0, 1, \dots, d/2-1$)

이를 행렬 형태로 시각화하면 다음과 같습니다.

$$P = 
\begin{bmatrix}
\sin(\omega_0 \cdot 0) & \cos(\omega_0 \cdot 0) & \dots & \sin(\omega_{\frac{d}{2}-1} \cdot 0) & \cos(\omega_{\frac{d}{2}-1} \cdot 0) \\
\sin(\omega_0 \cdot 1) & \cos(\omega_0 \cdot 1) & \dots & \sin(\omega_{\frac{d}{2}-1} \cdot 1) & \cos(\omega_{\frac{d}{2}-1} \cdot 1) \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
\sin(\omega_0 \cdot t) & \cos(\omega_0 \cdot t) & \dots & \sin(\omega_{\frac{d}{2}-1} \cdot t) & \cos(\omega_{\frac{d}{2}-1} \cdot t) \\
\vdots & \vdots & \ddots & \vdots & \vdots 
\end{bmatrix}$$

* **좌측 열 (저주파):** $\omega$가 큼. 파장이 매우 길어서 천천히 변함. (전체 숲을 보는 위치 정보)
* **우측 열 (고주파):** $\omega$가 작음. 파장이 짧아서 빠르게 변함. (나무를 보는 세밀한 위치 정보)

#### 3. 선형대수적 의미: 회전 변환 (Rotation Transformation)

왜 하필 `sin`, `cos`을 섞어서 쓸까요? 이것은 **상대적 위치(Relative Position)**를 **선형 변환(Linear Transformation)**으로 표현하기 위해서입니다.



임의의 차원 쌍 $(2k, 2k+1)$을 묶어서 하나의 **2차원 부분 공간(2D Subspace)**으로 생각해보겠습니다.
위치 $t$에서의 벡터는 $(\sin(\omega_k t), \cos(\omega_k t))$ 입니다.

이때, $t$에서 $\phi$만큼 떨어진 위치 $t+\phi$의 벡터는 다음과 같이 표현됩니다.

$$
\begin{bmatrix}
\sin(\omega_k (t+\phi)) \\
\cos(\omega_k (t+\phi))
\end{bmatrix}
=\begin{bmatrix}
\cos(\omega_k \phi) & \sin(\omega_k \phi) \\
-\sin(\omega_k \phi) & \cos(\omega_k \phi)
\end{bmatrix}
\begin{bmatrix}
\sin(\omega_k t) \\
\cos(\omega_k t)
\end{bmatrix}$$

$$P_{(t+\phi)} = R_{\phi} \cdot P_{(t)}$$

* **핵심 통찰:** 어떤 위치 $t$의 벡터에 **회전 행렬(Rotation Matrix) $R_{\phi}$**를 곱하기만 하면, $\phi$만큼 떨어진 위치의 정보를 **선형적으로** 만들어낼 수 있습니다.
* **결론:** 모델(Attention)이 단순한 행렬 곱셈(Linear Layer)만으로도 **"아, 이 단어는 저 단어보다 3칸 뒤에 있구나"**라는 상대적 거리를 기하학적으로 학습할 수 있게 됩니다.

---

## 3. Encoder Layer

입력: $X \in \mathbb{R}^{B \times L \times 512}$

### 3-1. Multi-head Self Attention
거대한 512차원 벡터를 8개의 64차원 **부분 공간(Subspace)**으로 쪼개서(Isomorphism) 병렬 처리합니다.

#### A. Linear Projection (In Subspaces)
$$Q_i = X W_Q^i, \quad K_i = X W_K^i, \quad V_i = X W_V^i$$
* **Projection:** 512차원 데이터를 64차원($d_k$) 부분 공간으로 투영합니다.
* **Dims:** `(B, L, 512)` $\times$ `(512, 64)` $\rightarrow$ `(B, L, 64)` (per head)
* **선형대수적 관점:** $Q$는 탐색을 위해, $K$는 검색되기 위해 최적화된 기저(Basis)로 회전 변환됩니다.

#### B. Scaled Dot-Product Attention (Alignment)
$$Head_i = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i$$
* **Alignment ($Q \cdot K^T$):** Query 벡터와 가장 방향이 비슷한(유사한) Key 벡터를 찾습니다.
* **Scaling ($1/\sqrt{d_k}$):** 분산을 1로 정규화하여 Gradient Vanishing을 방지합니다.
* **Mixing ($\cdot V$):** 확률값(Attention Weight)을 이용해 $V$ 벡터들을 **선형 결합(Linear Combination)**합니다. 문맥이 반영된 새로운 벡터가 탄생합니다.

#### C. Concat & Linear (Output Projection)
$$MultiHead(X) = \text{Concat}(Head_1, \dots, Head_h) W_O$$
* **Direct Sum (Concat):** 8개의 부분 공간 결과를 물리적으로 합칩니다.
* **Mapping ($W_O$):** 서로 독립적이던 8개의 정보를 섞어서(Mixing) 다시 512차원 공간으로 통합합니다.

### 3-2. Add & Norm
$$X_{attn} = \text{LayerNorm}(X + MultiHead(X))$$
* **Add:** Residual Connection (Gradient Highway).
* **LayerNorm:** 벡터를 구(Hypersphere) 표면 근처로 정규화하여 **Loss Surface를 평탄하게(Smoothing)** 만듭니다.

### 3-3. Position-wise FFN
$$FFN(x) = \text{ReLU}(x W_1 + b_1) W_2 + b_2$$
* **Mapping (Expansion):** 512차원 $\to$ 2048차원.
* **Activation (ReLU):** 비선형성을 부여하여 꼬여있는 데이터의 **매니폴드(Manifold)**를 펴줍니다.
* **Projection (Compression):** 2048차원



## 2. Decoder Layer

**입력 (Inputs):**
* **$X_{dec} \in \mathbb{R}^{B \times L_{dec} \times d}$**: Target 문장의 임베딩 (Shifted Right, 즉 `<sos>`로 시작).
* **$X_{enc} \in \mathbb{R}^{B \times L_{enc} \times d}$**: Encoder가 처리를 마친 Source 문장의 문맥 정보 (Key, Value 제공용).

### 2-1. Masked Self Attention
Encoder의 Self Attention과 메커니즘은 같지만, **인과성(Causality)**을 지키기 위해 미래 정보를 물리적으로 차단합니다.

* **Operation:**
    $$M_{score} = \frac{Q_{dec} K_{dec}^T}{\sqrt{d_k}} + \text{Mask}$$
    * **Mask:** Score Map의 우상단(Upper Triangle)에 $-\infty$를 더해줍니다.
* **Dimensions:**
    * Score Map: `(B, L_dec, L_dec)` (정사각형 행렬)
* **[선형대수적 관점]**
    * **Subspace Restriction (공간 제약):** $t$시점의 Query 벡터가 $t+1$ 이후의 Key 벡터들과 내적했을 때, 그 값을 강제로 음의 무한대로 보냅니다.
    * Softmax를 통과하면 $e^{-\infty} \to 0$이 되므로, 미래 시점의 기저(Basis)와는 **직교(Orthogonal, 관계없음)** 상태가 되도록 강제하여 정보의 흐름을 차단합니다.

### 2-2. Cross Attention (Encoder-Decoder Attention)
서로 다른 두 벡터 공간(Source & Target)을 연결하는 **다리(Bridge)** 역할을 합니다.

* **Linear Projection:**
    * **Query (from Decoder):** $Q = X_{dec} W_Q$ $\rightarrow$ "현재 번역할 위치에서 내가 궁금한 정보는..."
    * **Key (from Encoder):** $K = X_{enc} W_K$ $\rightarrow$ "원본 문장이 가진 정보들의 인덱스(Key)는..."
    * **Value (from Encoder):** $V = X_{enc} W_V$ $\rightarrow$ "원본 문장의 실제 의미(Value)는..."
* **Operation (Alignment & Mixing):**
    $$Context = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
* **Dimensions:**
    * $Q$: `(B, L_dec, d_k)`
    * $K^T$: `(B, d_k, L_enc)`
    * **Score Map ($QK^T$):** **`(B, L_dec, L_enc)`** $\rightarrow$ 디코더의 각 단어(행)가 인코더의 어떤 단어(열)와 연관되는지를 나타내는 직사각형 맵.
* **[선형대수적 관점]**
    * **Basis Mapping:** Target 언어 공간에 있는 벡터($Q$)를 Source 언어 공간($K, V$)에 투영하여, 가장 유사한 방향(**Alignment**)을 찾습니다.
    * **Injection:** 찾아낸 Source의 정보($V$)를 가중합(Weighted Sum)하여 Decoder의 벡터 공간으로 **주입(Injection)**합니다. 즉, 번역에 필요한 힌트를 원문 공간에서 가져와 현재 공간에 섞는 과정입니다.

### 2-3. Position-wise FFN
Cross Attention을 통해 외부 정보를 흡수한 뒤, 이를 현재 문맥에 맞게 정제합니다.

* **Operation:**
    $$FFN(x) = \text{ReLU}(x W_1 + b_1) W_2 + b_2$$
* **[선형대수적 관점]**
    * 외부(Encoder)에서 가져온 정보와 내부(Decoder) 정보가 섞인 벡터를, 고차원으로 보냈다가($W_1$) 다시 압축($W_2$)하면서 **매니폴드(Manifold)를 매끄럽게 정리**하는 단계입니다.

---

## 3. Output Projection & Validation

### 3-1. Final Linear Layer (Projection to Vocab)
모델의 내부 차원($d=512$)을 단어 집합의 크기($V \approx 30,000$)로 확장합니다.

* **Operation:**
    $$Z = X_{final} W_{vocab}^T$$
* **Dimensions:**
    * $X_{final}$: `(B, L_dec, 512)`
    * $W_{vocab}^T$: `(512, V)`
    * **Logits ($Z$):** **`(B, L_dec, V)`**
* **[선형대수적 관점]**
    * **Similarity Check:** 디코더가 만든 최종 벡터($X_{final}$)와 단어장 내의 **모든 단어 임베딩 벡터($W_{vocab}$의 행들)** 간의 **내적(Dot Product)**을 한 번에 수행합니다.
    * 내적 값이 클수록 두 벡터의 방향이 비슷하다는 뜻이며, 이는 곧 해당 단어일 확률이 높다는 것을 의미합니다. (Cosine Similarity와 유사)

### 3-2. Softmax & Label Smoothing
확률 분포를 만들고, 정답지와의 오차를 계산할 준비를 합니다.

* **Label Smoothing (정답지 수정):**
    * Hard Label (One-hot): $y = [0, 1, 0, \dots]$
    * Soft Label: $y_{ls} = [0.01, 0.9, 0.01, \dots]$
    * **이유:** 정답 벡터가 너무 뾰족하면(Impulse), 모델이 무한히 큰 가중치를 가지려 합니다(**Overconfidence**). 이를 방지해 벡터 공간상에서 무리하게 찢어지는 것을 막고 **일반화 성능**을 높입니다.
* **Softmax:**
    $$P(y_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$
    * Logits 벡터를 **확률 심플렉스(Probability Simplex, 합이 1인 공간)**로 투영합니다.

### 3-3. Loss Calculation (Cross Entropy)
예측된 분포 $P$와 실제 분포 $y_{ls}$ 사이의 거리를 측정합니다.

* **Formula:**
    $$Loss = - \sum_{i=1}^{V} y_{ls}^{(i)} \log(P^{(i)})$$
* **[선형대수/통계적 관점]**
    * **KL-Divergence:** 두 확률 분포(모델의 예측 vs 실제 정답)가 정보 이론적으로 얼마나 다른지를 측정하는 **거리 함수(Distance Metric)**입니다.
    * 이 값이 0이 되도록(두 분포가 완벽하게 겹치도록) 만드는 것이 학습의 목표입니다.

### 3-4. Backpropagation & Optimizer
* **Gradient Descent:**
    $$\theta_{new} = \theta_{old} - \eta \cdot \nabla_{\theta} Loss$$
* **[기하학적 관점]**
    * 수억 차원의 파라미터 공간(Hyperspace)에 존재하는 **Loss Surface(오차 지형)**에서, 현재 위치의 기울기(Gradient)를 따라 가장 낮은 계곡(Global Minimum)으로 **하강(Descent)**하는 과정입니다.
    * 이때 **Adam** Optimizer는 관성(Momentum)과 적응형 보폭(Adaptive Rate)을 이용해, 좁은 골짜기나 평평한 구간(Plateau)을 효율적으로 탈출하여 최적해에 도달합니다.
