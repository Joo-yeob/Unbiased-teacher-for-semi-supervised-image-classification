# 🎯 Unbiased Teacher for Image Classification

본 프로젝트는 Object Detection을 위해 제안된 Unbiased Teacher 방법론을 이미지 분류(Image Classification) 문제에 적용하여 그 성능을 검증한다.

이를 위해 CIFAR-10 데이터셋에 대한 학습 및 평가 코드를 구현하며, 대표적인 준지도학습 방법론인 FixMatch와의 정확도 비교 실험을 통해 Unbiased Teacher의 효과를 분석한다.


---

## 💡 모델 동작 방식 상세 설명

이 모델은 Student와 Teacher의  두 개의 네트워크를 통해 준지도학습을 수행한다. 두 네트워크는 동일한 아키텍처(WideResNet-28-2)를 공유하지만, 가중치를 업데이트하는 방식에서 결정적인 차이를 가진다. 전체 학습 과정은 아래의 4단계로 반복 진행한다.

### **1️⃣ Teacher 모델을 이용한 의사 레이블 생성**

Teacher 모델은 레이블이 없는 데이터에 대해 신뢰할 수 있는 **의사 라벨(Pseudo Label)** 을 생성한다. 

1. **입력**: 레이블이 없는 이미지에 **약한 증강(Weak Augmentation)** 을 적용한다. Weak Augmentation은 Cifar10 이미지에 무작위 수평 뒤집기와 4픽셀 반사 패딩 후 32x32 무작위 Crop을 적용한다. 원본 형태를 거의 유지하는 수준의 변환만을 포함한다.
2. **예측**: Teacher 모델은 Weak Augmentation이 적용된 이미지를 입력받아 각 클래스에 대한 확률을 예측한다. 
3. **필터링**: 예측된 확률 값 중 가장 높은 값이 미리 설정된 **임계값(threshold, 기본값 0.95)** 을 넘는 경우에만 해당 예측을 Student 모델에서 활용한다. 이 과정을 통해 부정확한 레이블을 배제하고, 모델이 확신하는 데이터만 학습에 사용할 수 있도록 한다. 

### **2️⃣ 강한 증강 및 Student 학습**

Student 모델은 Teacher가 만들어준 Pseudo Laebl을 활용하여 강인한(robust) 표현을 학습합니다.

1.  **입력**: Teacher가 사용했던 **동일한 원본 이미지**에 이번에는 **강한 증강(Strong Augmentation)**을 적용합니다. 강한 증강은 `RandAugment` 기법을 통해 회전, 기울이기, 색상 왜곡 등 매우 심한 변형을 가하고, 이미지 일부를 가리는 `Cutout`을 적용하여 거의 새로운 이미지처럼 만듭니다.
2.  **지도 학습 (Supervised Loss)**: 레이블이 있는 소량의 데이터에 대해서는 일반적인 지도학습 방식과 동일하게, 모델의 예측과 실제 정답 간의 손실(Cross-Entropy Loss)을 계산합니다.
3.  **비지도 학습 (Unsupervised Loss)**: 강하게 증강된 레이블 없는 이미지에 대해 Student 모델이 예측을 수행합니다. 이 예측 결과와 **1단계에서 Teacher가 생성한 의사 레이블** 간의 손실을 계산합니다. 이는 Student가 이미지의 형태가 심하게 변형되더라도, 원본이 가진 핵심적인 특징을 파악하여 동일한 정답을 맞히도록 훈련하는 과정입니다.

### **3️⃣ 손실 결합 및 Student 업데이트**

Student 모델은 두 종류의 손실을 모두 반영하여 가중치를 업데이트합니다.

-   **최종 손실**: `최종 손실 = 지도 학습 손실 + λ * 비지도 학습 손실`
-   **업데이트**: 계산된 최종 손실을 기반으로 **역전파(Backpropagation)를 수행하여 Student 모델의 가중치만 업데이트**합니다. `λ`는 비지도 학습 손실의 중요도를 조절하는 가중치 파라미터입니다.

### **4️⃣ EMA를 통한 Teacher 업데이트**

이 프로젝트의 핵심적인 부분으로, **Teacher 모델은 역전파를 통해 학습되지 않습니다.** 대신, 더 똑똑해진 Student 모델의 지식을 부드럽게 이어받습니다.

-   **EMA (Exponential Moving Average)**: Teacher 모델은 다음과 같은 EMA 공식을 통해 자신의 가중치를 업데이트합니다.
    `Teacher 가중치 = α * (이전 Teacher 가중치) + (1-α) * (현재 Student 가중치)`
-   `α` (ema_decay, 기본값 0.999)는 매우 높은 값으로 설정되어, Student의 가중치가 아주 조금씩만 Teacher에게 반영됩니다. 이 덕분에 Teacher는 Student의 순간적인 학습 변동에 크게 흔들리지 않고, 시간의 흐름에 따라 축적된 안정적이고 일반화된 지식 체계(앙상블 모델과 유사한 효과)를 갖추게 됩니다.

이러한 4단계의 순환 과정을 통해, Student는 점차 어려운 문제를 푸는 능력을 기르고 Teacher는 더 안정적인 지식을 제공하는 선순환 구조가 완성됩니다.

---

## 📦 실험 환경 및 방법

### 실험 환경
- Google Colab Pro+
- GPU: NVIDIA A100 (40GB)

### 학습 (Training)

아래 스크립트를 실행하여 모델을 학습시켰다. 주요 하이퍼파라미터를 인자로 전달하여 실험 조건을 변경할 수 있다.

-   **CIFAR-10, 레이블 4000개, WideResNet 사용 예시**

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
    
---

## 📊 학습 결과

Dataset: CIFAR-10
Architecture: WideResNet-28-2
Epoch: 100 

| Model | Labeled Data | Top-1 Accuracy (%) | Top-5 Accuracy (%) | Best Top-1 Accuracy (%) | Mean Top-1 Accuracy (%) |
| :-----: | :----------: | :----------: | :----------------: | :----------------: | :----------------: |
| Fixmatch(X) | 250 | **94.37** | **99.86** | **94.37** | **93.91** |
| Unbiased Teacher(X) | 250 | **94.37** | **99.86** | **94.37** | **93.91** |
| Fixmatch | 4000 | 93.83 | 99.82 | 94.00 | 93.76 |
| Unbiased Teacher | 4000 | **94.37** | **99.86** | **94.37** | **93.91** |

---

##  📝 결론

1. Labeled Data의 양에 따른 정확도 비교
   Labeled Data가 40개인 경우에는 60 에폭에서 이미 Pseudo Label 비율이 80%까지 상승했음에도 불구하고 Top-1 Accuracy가 약 40% 대에 머물며 효과적인 학습이 진행되지 못했다. Fixmatch 기법과 다르게 Unbiased Teacher은 
