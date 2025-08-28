# 🎯 Unbiased Teacher for Image Classification

본 프로젝트는 Object Detection을 위해 제안된 Unbiased Teacher 방법론을 이미지 분류(Image Classification) 문제에 적용하여 그 성능을 검증한다.

이를 위해 CIFAR-10 데이터셋에 대한 학습 및 평가 코드를 구현하며, 대표적인 준지도학습 방법론인 FixMatch와의 정확도 비교 실험을 통해 Unbiased Teacher의 효과를 분석한다.


---

## 💡 모델 동작 방식 상세 설명

이 모델은 Student와 Teacher의  두 개의 네트워크를 통해 준지도학습을 수행한다. 두 네트워크는 동일한 아키텍처(WideResNet-28-2)를 공유하지만, 가중치를 업데이트하는 방식에서 결정적인 차이를 가진다. 하나의 학습 배치는 64개의 라벨 데이터와 448개의 언라벨 데이터(=64×7)로 구성된다. 전체 학습 과정은 아래의 4단계로 반복 진행한다.

### **1️⃣ Teacher 모델을 이용한 의사 레이블 생성**

Teacher 모델은 레이블이 없는 데이터에 대해 신뢰할 수 있는 **의사 라벨(Pseudo Label)** 을 생성한다. 

1. **입력**: 레이블이 없는 이미지에 **약한 증강(Weak Augmentation)** 을 적용한다. Weak Augmentation은 Cifar10 이미지에 무작위 수평 뒤집기와 4픽셀 반사 패딩 후 32x32 무작위 Crop을 적용한다. 원본 형태를 거의 유지하는 수준의 변환만을 포함한다.
2. **예측**: Teacher 모델은 Weak Augmentation이 적용된 이미지를 입력받아 각 클래스에 대한 확률을 예측한다. 
3. **필터링**: 예측된 확률 값 중 가장 높은 값이 미리 설정된 **임계값(threshold, 기본값 0.95)** 을 넘는 경우에만 해당 예측을 Student 모델에서 활용한다. 이 과정을 통해 부정확한 레이블을 배제하고, 모델이 확신하는 데이터만 학습에 사용할 수 있도록 한다. 

### **2️⃣ Student 모델 학습**

Student 모델은 Teacher가 만들어준 Pseudo Laebl을 활용하여 학습을 진행한다. 

1.  **입력**: Teacher가 사용했던 동일한 원본 이미지에 **강한 증강(Strong Augmentation)** 을 적용한다. Strong Augmentation은 회전, 기울이기, 색상 왜곡 등의 14가지 증강 중 2가지가 선택되어 적용되고, 추가적으로 Cutout이 적용된다.
2.  **Student 모델 학습**: 레이블이 있는 64개의 데이터에 대해서는 일반적인 지도학습 방식과 동일하게, 모델의 예측과 실제 정답 간의 손실(Cross-Entropy Loss)을 계산한다. 강하게 증강된 레이블 없는 이미지에 대해서는 모델의 예측 결과와 **1단계에서 Teacher가 생성한 의사 레이블** 간의 손실을 계산한다. 이는 Student가 이미지의 형태가 심하게 변형되더라도, 원본이 가진 핵심적인 특징을 파악하여 동일한 정답을 맞히도록 훈련하는 과정이다.

### **3️⃣ 손실 결합 및 Student 업데이트**

Student 모델은 두 종류의 손실을 모두 반영하여 가중치를 업데이트한다.

-   **최종 손실**: `최종 손실 = 지도 학습 손실 + λ * 비지도 학습 손실`
-   **업데이트**: 계산된 최종 손실을 기반으로 **역전파(Backpropagation)를 수행하여 Student 모델의 가중치만 업데이트**한다. `λ`는 비지도 학습 손실의 중요도를 조절하는 가중치 파라미터이다. Teacher 모델은 역전파를 통해 학습되지 않으며, 대신 4단계에서 설명할 EMA를 통해 업데이트 된다. 

### **4️⃣ EMA를 통한 Teacher 업데이트**

Teacher 모델의 가중치를 업데이트하는 방식으로, Student 모델의 가중치를 조금씩 Teacher에 적용한다.

-   **EMA (Exponential Moving Average)**: Teacher 모델은 다음과 같은 EMA 공식을 통해 자신의 가중치를 업데이트한다.
    `Teacher 가중치 = α * (이전 Teacher 가중치) + (1-α) * (현재 Student 가중치)`
-   `α` (ema_decay, 기본값 0.999)는 매우 높은 값으로 설정되어, Student의 가중치가 아주 조금씩만 Teacher에게 반영되도록 하였다. 이는 Teacher 모델이 Student의 순간적인 학습 변동에 흔들리지 않게 하기 위함이다. 즉, 천천히 안정적으로 일반화 지식을 가질 수 있도록 하였다.

위 실험에서는 총 100 Epoch에 대해 다음 4단계를 반복하였으며, Top-1 Accuracy, Top-5 Accuracy, Best Top-1 Accuracy, Mean Top-1 Accuracy의 평가지표를 사용하였다. 또한, Fixmatch와의 성능 비교를 통해 Unbiased Teacher의 효과를 확인하였다. 

---

## 📦 실험 환경 및 방법

### 실험 환경
- Google Colab Pro+
- GPU: NVIDIA A100 (40GB)

### 학습 (Training)

아래 스크립트를 실행하여 모델을 학습시켰다. 주요 하이퍼파라미터를 인자로 전달하여 실험 조건을 변경할 수 있다.

-   **CIFAR-10, 레이블 4000개, WideResNet, Unbiased_Teacher 사용 예시**

    ```bash
    !python train_UT.py \
        --dataset cifar10 \
        --num-labeled 4000 \
        --arch wideresnet \
        --batch-size 64 \
        --lr 0.03 \
        --expand-labels \
        --seed 5 \
        --out results/cifar10@4000.5
    ```

-   **CIFAR-10, 레이블 4000개, WideResNet, Fixmatch 사용 예시**
    ```bash
    !python train.py \
        --dataset cifar10 \
        --num-labeled 4000 \
        --arch wideresnet \
        --batch-size 64 \
        --lr 0.03 \
        --expand-labels \
        --seed 5 \
        --out results/cifar10@4000.5
    ```
    
---

## 📊 학습 결과

Dataset: CIFAR-10
Architecture: WideResNet-28-2
Epoch: 100 

| Model | Labeled Data | Top-1 Accuracy (%) | Top-5 Accuracy (%) | Best Top-1 Accuracy (%) | Mean Top-1 Accuracy (%) |
| :-----: | :----------: | :----------: | :----------------: | :----------------: | :----------------: |
| Fixmatch | 250 | 86.30 | 99.10 | 86.65 | 86.10 |
| Unbiased Teacher | 250 | **88.07** | **99.49** | **88.07** | **87.72** |
| Fixmatch | 4000 | 93.83 | 99.82 | 94.00 | 93.76 |
| Unbiased Teacher | 4000 | **94.37** | **99.86** | **94.37** | **93.91** |

---

##  📝 결론

### Labeled Data의 양에 따른 정확도 비교
  - Labeled Data가 250개, 4000개인 두 경우 모두, Unbiased Teacher이 Fixmatch보다 Top-1 Accuracy, Top-5 Accuracy, Best Top-1 Accuracy, Mean Top-1 Accuracy 네 가지의 성능 지표에 대해 우위를 보였다. 기존 논문의 경우에는 300 Epoch을 기준으로 비교를 했기 때문에 차이가 있을 수 있지만 Accuracy 그래프를 보았을 때 에폭이 더 진행되더라도 Unbiased Teacher의 성능이 더 높을 것으로 예상된다.

### Top-1 Accuracy (%) vs. Epoch
<img width="1339" height="412" alt="image" src="https://github.com/user-attachments/assets/f3e34782-d64b-417c-8bdd-c7d91ed9eeed" />


  - **생각해볼 부분** : Labeled Data가 40개인 경우: 60에폭 정도에서 Pseudo Label로 사용하고 있는 데이터의 비율이 80%가 넘어갔음에도 불구하고, Top-1 Accuracy가 약 40%대에 머물며 효과적인 학습이 진행되지 못했다.(Top-5 Accuracy는 약 88% 정도이었다.) 기존 Fixmatch 논문에서 총 300 에폭의 결과를 비교했기에 300 에폭의 학습이 진행되면 유의미한 결과가 나올 수도 있다고 생각이 들지만, GPU 성능 및 시간의 이유로 인해 실험을 진행하지 못하였다.
    다만, 라벨 수가 적은 경우, Teacher 모델이 잘못된 Pseudo Label을 만들 가능성이 보다 커지게 된다. 그러면 Student 모델이 잘못 태깅된 라벨로 학습을 하게 되므로, Teacher 모델에게 오염된 파라미터를 제공하게 된다. 이로 인해 Teacher 모델이 점점 더 오염되면서 정확도가 크게 오르지 못하는 현상으로 이어질 수 있다. 추후 더 큰 에폭으로 학습을 진행해서 정확도의 경향을 파악해보는 것이 중요하다고 생각한다.  
