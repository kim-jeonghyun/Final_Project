# Final_Project IBOBA : AI 패션 코디네이트 플랫폼


본 서비스는 [KDT 프로그래머스 인공지능 데브코스 2기](https://programmers.co.kr/learn/courses/11612) 과정의 일부로 진행된 팀 프로젝트의 결과물입니다.
웹 서비스는 여기서 확인할 수 있습니다. (향후 업데이트 예정)

또한 본 서비스는 Parser-Free Virtual Try-on via Distilling Appearance Flows [논문](https://paperswithcode.com/paper/parser-free-virtual-try-on-via-distilling)과 [github](https://github.com/geyuying/PF-AFN)을 바탕으로 하여 이를 발전시키고, 웹서비스로 개발한 결과물임을 밝힙니다.

본 서비스에 사용한 DL모델의 학습 데이터는 AIHub의 [패션 상품 및 착용 이미지](https://aihub.or.kr/aidata/30755) 데이터를 재가공한 것입니다.

AIhub 데이터의 재가공과 관련한 보다 자세한 설명 및 코드는 [전처리 레포지토리](https://github.com/Programmers-B-2-Team/preprocess_functions)를 참고해 주세요.

프론트엔드에 활용한 bootstrap theme은 https://technext.github.io/majestic-2/v1.0.1/# 이며, 이를 디자인에 맞게 변형하여 사용하였습니다.

## 주요 기능

본 서비스에서 제공하는 핵심 기능은 인물의 전신 사진과 의류 사진을 input으로 넣으면, 의류를 착용한 인물 사진을 output으로 생성하는 것입니다.

의류 사진은 웹사이트에서 기본으로 제공합니다. 이는 일반적인 쇼핑몰에서 의류를 판매하면서, 이와 같은 가상 착용 기능을 부가적으로 서비스하는 것과 비슷한 환경입니다. 
인물의 전신 사진은 1. 웹사이트에서 기본으로 제공하는 모델의 사진을 활용하거나, 2. 직접 사진을 업로드하여 생성 모델의 input으로 활용할 수 있습니다.

## 딥러닝 모델

본 서비스에서 활용하는 딥러닝 모델은 크게 2가지 입니다.

1. 사용자가 직접 업로드한 사진을 이미지 생성 모델의 input으로 활용하기 위해 전처리 과정을 거칩니다. 이 때 u-net 기반의 모델을 거칩니다. 이 과정은 [전처리 레포지토리](https://github.com/Programmers-B-2-Team/preprocess_functions)에 상세히 설명되어 있으므로 참고해 주세요.
2. 사람 이미지와 의류 이미지를 input으로 하여 그 의류를 착용한 인물 이미지를 새롭게 생성하는 생성 모델이 있습니다. 이는 [PF-AFN 깃허브](https://github.com/geyuying/PF-AFN) 의 PF-AFN_test 폴더를 기반으로 하여 본 서비스에 맞게 수정하였으며, inference folder에 관련 코드를 정리하였습니다.


## 필요 패키지 설치

```
pip install -r requirements.txt
```
