# Transformer Architecture & Training Process (Revised & Enhanced)

## 0. Preprocessing

### 0-1. Tokenize
텍스트를 정수 인덱스로 변환합니다. 일반적으로 공백을 기준으로 하여 단어 하나로 분할합니다.
모델 Training 과정에서는 2개의 input을 준비합니다. 이들의 index는 vocab 내에서 해당 단어가 위치하는 인덱스입니다.

- **input(source)** => `[12, 154, 32, 660... <eos>]` (Padding까지 포함한 길이를 `seq_len`이라고 합니다.)
- **input(target)** => `[<sos>, 1, 55, 66, 234...]` (shifted Right)

현재의 형태는 `[Batch, seq_len]`인 정수 벡터입니다.

### 0-2. Embedding
Embedding matrix의 lookup 결과를 반환합니다. 벡터의 인덱스가 `a`인 경우, Embedding matrix의 `a`번째 행을 가져오는 연산이 됩니다. 이들을 row-wise로 concat 합니다.

- **Embedding matrix:** `(vocab_size, hidden_dim)`의 크기.
    - 여기서 `hidden_dim`은 하나의 단어를 벡터 공간에 나타내는 기준 차원을 의미합니다. (논문에서는 512 사용)
    - **[선형대수적 관점]** 이산적인(Discrete) 정수 인덱스를 연속적인(Continuous) 고차원 벡터 공간 $\mathbb{R}^{512}$으로 매핑(Mapping)하는 과정입니다.

- **Scaling:** `* (512)^(1/2)`
    - 초기 Embedding matrix는 1/512 정도로 매우 작게 설정되어, dim이 클수록 각 원소의 값이 0에 가깝게 매우 작아집니다. 스케일링 없이 더하면 상대적으로 값이 큰 위치 정보(Positional Encoding)가 단어의 의미 정보를 덮어 버릴 수 있습니다.
    - **[추가 설명]** 차원이 커질수록 내적 값의 분산이 커지는 것을 막아주기 위해, 벡터의 Norm(크기)을 키워서 **분산의 균형(Variance Balance)**을 맞춰주는 역할도 합니다.

=> `(seq_len, hidden_dim)`의 크기가 됩니다.

### 0-3. Positional Encoding
Embedded matrix의 문장 내에서의 절대적인 순서를 정해줍니다.
이때 상대적인 위치를 볼 수 있는 이유는, 특정 거리 `K`만큼 떨어진 위치의 벡터를 현재 위치 벡터의 **선형 변환(Linear Transformation, 회전 변환)**만으로 표현할 수 있기 때문입니다. (논문에서는 sin/cos 연산 이용)

### 0-4. Element-wise sum
Embedded matrix, Positional Encoding을 element-wise sum 해줍니다.

- 충분히 공간이 큰 고차원에서는 위치 정보와 단어의 의미 정보가 서로 다른 **부분 공간(Subspace)**에 분포하도록 학습할 수 있습니다.
- 이는 ResNet의 철학과도 유사한데, 정보를 덮어쓰는 게 아니라 얹어주는 것(Superposition)입니다.
- Concat을 하게 되면 봐야 하는 param이 너무 늘어나서 연산 속도가 느려지므로 Sum을 사용합니다.

=> `(seq_len, hidden_dim=512)`의 크기가 됩니다.

---

## 1. Encoder
입력 행렬 `(seq_len, hidden_dim)`이 들어갑니다.

### 1-1. Multi-head Self Attention
여러 개의 헤드에 대해 병렬 연산이 이뤄집니다. 각 헤드는 특화된 부분이 존재합니다.
원본 입력 데이터에 가중치 행렬을 곱해서 **Projection(투영)**을 해주는 연산인데, 이때 $W_q, W_k, W_v$의 세 개의 행렬이 필요합니다.

#### 1-1-1. Linear Projection
Input에 $W_q, W_k, W_v$를 곱해 `(seq_len, attention_dim=64)`인 **Subspace(부분 공간)**로 Projection 하는 결과입니다.

- 이때 $Q, K, V$의 용도는 각각 다릅니다.
    - **V:** 각 벡터의 의미(Context)를 잘 보존하도록 행렬을 구성.
    - **Q, K:** 서로 내적을 최대화하는 방향으로 행렬이 구성됨.
- **[선형대수적 관점]** 64차원은 Query에 걸맞은 핵심 특징을 정의한 기준축(Basis)입니다. $Q$는 벡터를 '탐색'하기 좋게 회전시키고, $K$는 '발견'되기 좋게 회전시킵니다. 고차원에서 벡터들이 너무 흩어져 만나지 못하는 것을 방지하고, 내적 값을 최대화하는 방향으로 정렬(Alignment)합니다.

#### 1-1-2. Scaled Dot Product
Attention Score를 계산합니다. $Q \times K^T$를 구합니다.
문장 내에서 이뤄지는 이 Attention은 하나의 vector에 대해 다른 vector와의 관련성을 구합니다. Dot Product의 특성상, **방향(Cosine Similarity)**이 유사할수록 값이 커집니다.

=> `(seq_len, 64) * (64, seq_len) = (seq_len, seq_len)` Score map

- **Scaling:** `1 / sqrt(dim_k)` (표준편차로 나눔)
    - 내적 값의 분산은 차원 수만큼 커져 있습니다. 분산이 크면 Softmax를 지날 때 기울기(Gradient)가 0이 되어 학습이 안 되는 문제가 발생합니다.
    - 따라서 값의 차이를 유지하면서도 분포를 **정규화(Normalization)**하여 학습을 안정적으로 만들기 위해 표준편차로 나누어줍니다.

#### 1-1-3. Softmax, Mixing
`softmax(score) * V`의 연산을 합니다.
Softmax를 거치면서 확률값이 되었는데, 여기서 $V$와의 행렬 연산을 통해 각 단어들과의 연관성을 모두 이을 수 있습니다.

- **[결과]** 문장 내 다른 모든 단어의 정보를 연관성 비중만큼 가져와 **가중합(Weighted Sum)**, 즉 **선형 결합(Linear Combination)**을 해서 독립적이던 단어들에 문맥을 더한 상태로 재정의한 벡터입니다.

#### 1-1-4. Concat & Linear
여러 개의 연산 결과를 Concat 합니다. => `(seq_len, hidden_dim)`
$W_o(512, 512)$와의 행렬 곱을 통해, 서로 다른 Subspace에서 가져온 정보들을 하나로 **혼합(Mixing)**해 줍니다.

### 1-2. Add & Norm
- **Add:** Residual adding은 Vanishing Gradient를 방지하기 위해 초기의 값을 더해주는 연산입니다. (Gradient Highway)
- **Layer Normalization:** **[수정됨]** 한 벡터 내부에서 정규화를 수행해 줍니다. 이는 Gradient Explode 방지뿐만 아니라, **Loss Surface(손실 지형)를 평탄하게(Smoothing)** 만들어 Optimizer가 최적해를 더 빠르고 안정적으로 찾게 해줍니다.

### 1-3. Position-wise FFN
**[수정 및 추가됨]** 초기에 Expansion `(seq_len, 512) * (512, 2048)`을 하고, **반드시 비선형 활성화 함수(ReLU)를 거친 뒤**, 다시 Projection `(seq_len, 2048) * (2048, 512)`을 수행합니다.

$$FFN(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

- **[선형대수적 관점]** 단순히 차원을 늘렸다 줄이는 것이 아니라, **ReLU**를 통해 비선형성을 부여함으로써 꼬여있는 데이터의 **매니폴드(Manifold)**를 펴주는 역할을 합니다.
- 의미적으로는 Kernel size가 1인 Convolution과 같으며, 각 위치별 정보를 개별적으로 가공하여 특징을 극대화합니다.

---

## 2. Decoder
Input (target) `(seq_len_dec, hidden_dim)` 행렬이 들어옵니다. (첫 행은 `<sos>`)

### 2-1. Masked Self Attention
인코더와 동일하지만, `(L, L)` Score map에 우상단(미래 정보)에 **-infinite ($-\infty$)** Masking을 하고 Softmax를 취합니다.

- **[수정됨]** 0으로 하지 않는 이유: Softmax 함수는 $e^x$를 사용합니다. $e^0 = 1$이 되어 확률이 남지만, $e^{-\infty} = 0$이 되어야 **확률을 완벽하게 0**으로 만들어 미래 정보를 차단할 수 있기 때문입니다.

### 2-2. Cross Attention
- **Q:** 디코더의 현재 상태 행렬 `(seq_len_dec, 512)`
- **K, V:** 인코더의 최종 출력 행렬 `(seq_len_enc, 512)`

#### 2-2-1. Dot product (Alignment)
디코더의 Q와 인코더의 K가 각각 얼만큼 연관성을 지니는지 계산합니다. (Source와 Target 공간의 매핑)
=> `(seq_len_dec, seq_len_enc)` Attention score

#### 2-2-2. Scaling
필수. 안 하면 내적 값이 폭주해서 특정 단어 하나만 쳐다보고 나머지는 무시하게 됨(Gradient Vanishing).

#### 2-2-3. Softmax & Mixing
Attention score를 확률값으로 변형하고 $V$와 Mixing 합니다.
이렇게 나온 결과값 벡터는 **Source 언어의 정보를 흡수한 512차원 좌표값**이 됩니다.

### **[추가] 2-2-4. Position-wise FFN**
Decoder에도 Cross Attention 이후에 Encoder와 똑같은 FFN(Expansion -> ReLU -> Projection)이 존재합니다. 외부에서 가져온 정보를 디코더 문맥에 맞게 다시 한번 정제하는 과정입니다.

### 2-3. Linear
모든 레이어를 거치고 난 뒤 나온 결과 벡터 $Z$.
**[수정됨]** 훈련(Training) 시에는 `(Batch, seq_len, 512)` 전체 덩어리가 들어갑니다.

- 이를 Vocab 가중치 행렬 `(512, vocab_len)`과 내적을 구합니다.
- **[선형대수적 관점]** $W^T$의 한 열은 하나의 단어에 대한 임베딩 벡터입니다. 내적을 한다는 것은, 현재 디코더가 만든 벡터($Z$)와 단어장 내의 모든 단어 벡터 간의 **유사도(각도)**를 측정하는 것입니다.
- 결과는 `(Batch, seq_len, vocab_len)`이 되며, 시퀀스의 **모든 위치에 대해 다음에 올 단어의 점수(Logits)**를 한 번에 생성합니다.

### 2-4. Softmax
결과값(Logits)을 Softmax를 이용해 확률값(Probability)으로 변환합니다.

---

## 3. Validation

### **[추가] 3-0. Label Smoothing**
Loss를 계산하기 전에, 정답지(One-hot Encoding)를 조금 뭉툭하게 만듭니다.
- `[0, 1, 0]` 대신 `[0.1, 0.8, 0.1]` 처럼 정답 확률을 조금 깎아서 오답 클래스에 나눠줍니다.
- **효과:** 모델이 정답을 너무 과신(Overconfidence)하여 벡터 공간에서 너무 극단적인 위치로 가는 것을 막아주어, **일반화(Generalization)** 성능을 높입니다.

### 3-1. Loss Calculation
예측값(Probability)과 실제값(Smoothed Label)에 대한 **Cross Entropy Loss**를 확인합니다.

### 3-2. Backpropagation
Loss를 줄이는 방향을 찾기 위해, 거꾸로(Output -> Input) 돌아가면서 각 Layer에 있는 가중치들의 오차 기여도(Gradient)를 측정합니다. (Chain Rule 이용)

### 3-3. Optimizer (Adam)
분석(Backward)이 끝나면 가중치를 실제로 업데이트합니다.
- SGD는 단순히 기울기 방향으로 내려가지만, **Adam**은 관성(Momentum)과 보폭(Adaptive Rate)을 고려하여 고차원 공간에서도 최적해를 효율적으로 찾아갑니다.