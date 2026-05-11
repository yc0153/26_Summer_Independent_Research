# 050 Morphology 강의 교안

Morphology는 "모폴로지"라고 읽습니다.
binary image에서 흰색 영역의 모양을 깎거나 키우거나 정리하는 연산입니다.

실행 파일:

```text
010_erosion.py
011_erosion_trackbar.py
020_dilation.py
021_dilation_trackbar.py
030_opening.py
031_opening_trackbar.py
040_closing.py
041_closing_trackbar.py
050_morph_gradient.py
051_morph_gradient_trackbar.py
```

`*_trackbar.py` 파일은 `iterations` 값을 움직이면서 morphology 연산을 몇 번 반복할 때 결과가 어떻게 변하는지 봅니다.
창에서 `q`를 누르면 종료하고, `s`를 누르면 현재 `input | result` 비교 화면을 저장합니다.

트랙바 예제는 노이즈가 잘 보이도록 원본 크기로 표시합니다.
결과가 너무 네모나게 깨져 보이지 않도록 `3x3` 타원 kernel을 씁니다.
네모 kernel을 여러 번 반복하면 원이나 곡선도 네모 방향으로 깎이고 커집니다.

## 1. 왜 edge와 분리했나?

Edge detection은 grayscale 이미지에서 밝기 변화로 경계를 찾는 작업입니다.
Morphology는 이미 만들어진 binary mask의 모양을 정리하는 작업입니다.

둘은 함께 자주 쓰이지만, 목적이 다릅니다.

```text
edge: 경계 찾기
morphology: mask 모양 정리하기
```

그래서 수업에서는 독립된 파트로 나누는 것이 더 이해하기 쉽습니다.

## 2. 기본 관점

Morphology에서 중요한 약속은 다음입니다.

```text
흰색 = 전경
검은색 = 배경
```

그리고 작은 kernel을 이미지 위에서 움직이며 주변 흰색 영역의 모양을 봅니다.
kernel 모양이 결과 모양에 직접 영향을 줍니다.
네모 kernel은 각진 결과를 만들기 쉽고, 타원 kernel은 곡선 형태를 조금 더 자연스럽게 유지합니다.

## 3. Erosion과 Dilation

Erosion은 "이로전"이라고 읽고, 흰색 영역을 깎습니다.

수식:

$$
A \ominus B
$$

텍스트로 쓰면:

```text
kernel 안의 모든 픽셀이 흰색일 때만 중심을 흰색으로 남긴다.
```

정성적 의미:

- 흰색 물체가 작아집니다.
- 작은 흰색 노이즈가 사라집니다.
- 얇게 이어진 부분이 끊어질 수 있습니다.

Dilation은 "다일레이션"이라고 읽고, 흰색 영역을 키웁니다.

수식:

$$
A \oplus B
$$

텍스트로 쓰면:

```text
kernel 안에 흰색 픽셀이 하나라도 있으면 중심을 흰색으로 만든다.
```

정성적 의미:

- 흰색 물체가 커집니다.
- 작은 검은 구멍이 메워집니다.
- 가까운 물체가 붙을 수 있습니다.

## 4. Opening

Opening은 "오프닝"이라고 읽습니다.
예제에서는 두 흰 물체가 얇은 다리로 붙어 있는 `mask_opening.png`를 사용합니다.

$$
A \circ B = (A \ominus B) \oplus B
$$

텍스트로 쓰면:

```text
opening = erosion -> dilation
```

정성적 의미:

먼저 깎으면서 얇은 흰색 연결부를 끊고, 다시 키워서 큰 물체의 크기를 어느 정도 되돌립니다.
그래서 두 물체 사이의 검은 틈이 점점 열리는지 보면 됩니다.
작은 흰 점 노이즈도 같이 사라집니다.

## 5. Closing

Closing은 "클로징"이라고 읽습니다.
예제에서는 작은 검은 구멍과 얇은 검은 틈이 있는 `mask_closing.png`를 사용합니다.

$$
A \bullet B = (A \oplus B) \ominus B
$$

텍스트로 쓰면:

```text
closing = dilation -> erosion
```

정성적 의미:

먼저 키워서 작은 검은 구멍을 메우고, 다시 깎아서 원래 크기에 가깝게 돌립니다.
그래서 흰 물체 내부의 검은 구멍이나 얇은 끊김이 메워지는지 보면 됩니다.

## 6. Morphological Gradient

Morphological gradient는 "모폴로지컬 그래디언트"라고 읽습니다.

$$
\operatorname{gradient}
= (A \oplus B) - (A \ominus B)
$$

텍스트로 쓰면:

```text
gradient = dilation - erosion
```

정성적 의미:

키운 물체와 깎은 물체의 차이를 보면, 물체의 바깥쪽 경계만 남습니다.

## 수업 포인트

Morphology는 thresholding 뒤에 자연스럽게 이어지는 파트입니다.
가장 기본이 되는 연산은 `erosion`과 `dilation`이고, `opening`과 `closing`은 이 둘을 순서대로 붙인 조합입니다.

```text
thresholding으로 mask 만들기
-> morphology로 mask 정리하기
-> contour로 물체 분석하기
```
