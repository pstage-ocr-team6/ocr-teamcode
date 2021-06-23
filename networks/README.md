<div align="center">
    <h1>Supported Models</h1>
</div>

# Supported Models

## SATRN
<div align="center">
    <img src="https://github.com/pstage-ocr-team6/ocr-teamcode/blob/main/assets/SATRN.png" width="660" height="771">
</div>

[Transformer](https://arxiv.org/abs/1706.03762)의 encoder-decoder 구조를 STR 테스크에 적합하게 변경한 모델입니다. [On Recognizing Texts of Arbitrary Shapes with 2D Self-Attention](https://arxiv.org/abs/1910.04396)에서 제안되었으며, 주요 특징은 다음과 같습니다.

### Shallow CNN
2D feature map을 생성하는 CNN의 깊이를 크게 줄임으로써, self-attention encoder block이 spatial dependency를 더 잘 포착하도록 할 수 있습니다. `ShallowConvLayer`를 두 겹 쌓는 형태로 구현되었으며, config file에서 `SATRN.encoder.shallow_cnn` 을 `True`로 하여 설정(`False`일 때는 `DeepCNN300`)할 수 있습니다.

### Adaptive 2D positional encoding
Transformer의 positional encoding을 2D로 확장하기 위한 방안입니다. 두 방향의 sinusoidal positional encoding을 weighted sum하는 형태로 구현하였으며, 이는 위 논문에서 제시한 바와 같습니다. 이 때, weighted sum의 'weight'는 이미지에 따라 두 방향 정보의 중요도를 조절하는 adaptive gate로서, `Linear-ReLU-Linear-sigmoid`의 구조를 가집니다. 코드에서 adaptive gate는 `AdaptiveGate`으로, positional encoding은 `AdaptivePositionalEncoding2D`으로 구현되어 있습니다. Config file에서 `SATRN.encoder.adaptive_gate`을 `True`로 하여 설정(`False`일 때는 `PositionalEncoding2D` (non-adaptive concat))할 수 있습니다.

### Locality-aware feedforward layer
Transformer의 point-wise feedforward layer를 3x3 convolutional layer를 활용한 구조로 변경하여, short-term dependency를 더욱 효과적으로 포착할 수 있도록 했습니다. Config file에서 `SATRN.encoder.conv_ff`를 `True`로 하면 self-attention block의 feedforward layer를 `LocalityAwareFeedforward`로 설정(`False`일 때 `Feedforward`)할 수 있습니다. `SATRN.encoder.seprable_ff`가 `True`일 때 separable, `False`일 때 convolution으로 설정할 수 있습니다(아래 그림 참고).
<div align="center">
    <img src="https://github.com/pstage-ocr-team6/ocr-teamcode/blob/main/assets/SATRN_feedforward.png" width="246" height="148">
</div>

## Attention
<div align="center">
    <img src="https://github.com/pstage-ocr-team6/ocr-teamcode/blob/main/assets/Attention.png" width="275" height="329">
</div>

[ASTER: An Attentional Scene Text Recognizer with Flexible Rectification](https://ieeexplore.ieee.org/document/8395027)에서 제안된 ASTER에서 Bi-LSTM을 제거한 구조입니다. CNN encoder와 RNN+attention decoder로 이루어져 있으며, decoder의 RNN은 LSTM 혹은 GRU로 설정할 수 있습니다(config file의  `Attention.encoder.cell_type`).