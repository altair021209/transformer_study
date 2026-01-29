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

$$
P = 
\begin{bmatrix}
\sin(\omega_0 \cdot 0) & \cos(\omega_0 \cdot 0) & \dots & \sin(\omega_{\frac{d}{2}-1} \cdot 0) & \cos(\omega_{\frac{d}{2}-1} \cdot 0) \\
\sin(\omega_0 \cdot 1) & \cos(\omega_0 \cdot 1) & \dots & \sin(\omega_{\frac{d}{2}-1} \cdot 1) & \cos(\omega_{\frac{d}{2}-1} \cdot 1) \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
\sin(\omega_0 \cdot t) & \cos(\omega_0 \cdot t) & \dots & \sin(\omega_{\frac{d}{2}-1} \cdot t) & \cos(\omega_{\frac{d}{2}-1} \cdot t) \\
\vdots & \vdots & \ddots & \vdots & \vdots 
\end{bmatrix}
$$

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
=
\begin{bmatrix}
\cos(\omega_k \phi) & \sin(\omega_k \phi) \\
-\sin(\omega_k \phi) & \cos(\omega_k \phi)
\end{bmatrix}
\begin{bmatrix}
\sin(\omega_k t) \\
\cos(\omega_k t)
\end{bmatrix}
$$

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
