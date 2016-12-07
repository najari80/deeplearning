---
layout: page
mathjax: true
permalink: assignments2016/assignment2/
---

이번 숙제에서 여러분은 backpropagation 코드를 작성하는 법을 연습하고, 기본 형태의 뉴럴 네트워크(신경망)와 컨볼루션 신경망을 학습해볼 것입니다. 이번 숙제의 목표는 다음과 같습니다.

- **뉴럴 네트워크(신경망)** 에 대해 이해하고 레이어가 있는 구조가 어떻게 배치되어 있는지 이해하기
- **backpropagation** 에 대해 이해하고 (벡터화된) 코드로 구현하기
- 뉴럴 네트워크를 학습시키는데 필요한 여러 가지 **업데이트 규칙** 구현하기
- 딥 뉴럴 네트워크를 학습하는데 필요한 **batch normalization** 구현하기
- 네트워크를 regularization 할 때 필요한 **dropout** 구현하기
- 효과적인 **교차 검증(cross validation)** 을 통해 뉴럴 네트워크 구조에서 사용되는 여러 가지 hyperparameter 들의 최적값 찾기
- **컨볼루션 신경망** 구조에 대해 이해하고 이 모델들을 실제 데이터에 학습해보는 것을 경험하기

## 설치
여러분은 다음 두가지 방법으로 숙제를 시작할 수 있습니다: Terminal.com을 이용한 가상 환경 또는 로컬 환경.

### Terminal에서의 가상 환경.
Terminal에는 우리의 수업을 위한 서브도메인이 만들어져 있습니다. [www.stanfordterminalcloud.com](https://www.stanfordterminalcloud.com) 계정을 등록하세요. 이번 숙제에 대한 스냅샷은 [여기](https://www.stanfordterminalcloud.com/snapshot/6c95ca2c9866a962964ede3ea5813d4c2410ba48d92cf8d11a93fbb13e08b76a)에서 찾아볼 수 있습니다. 만약 수업에 등록되었다면, TA(see Piazza for more information)에게 이 수업을 위한 Terminal 예산을 요구할 수 있습니다. 처음 스냅샷을 실행시키면, 수업을 위한 모든 것이 설치되어 있어서 바로 숙제를 시작할 수 있습니다. [여기](/terminal-tutorial)에 Terminal을 위한 간단한 튜토리얼을 작성해 뒀습니다.

### 로컬 환경
[여기](http://vision.stanford.edu/teaching/cs231n/winter1516_assignment2.zip)에서 압축파일을 다운받고 다음을 따르세요.

**[선택 1] Use Anaconda:**
과학, 수학, 공학, 데이터 분석을 위한 대부분의 주요 패키지들을 담고있는 [Anaconda](https://www.continuum.io/downloads)를 사용하여 설치하는 것이 흔히 사용하는 방법입니다. 설치가 다 되면 모든 요구사항(dependency)을 넘기고 바로 숙제를 시작해도 좋습니다.

**[선택 2] 수동 설치, virtual environment:**
만약 Anaconda 대신 좀 더 일반적이면서 까다로운 방법을 택하고 싶다면 이번 과제를 위한 [virtual environment](http://docs.python-guide.org/en/latest/dev/virtualenvs/)를 만들 수 있습니다. 만약 virtual environment를 사용하지 않는다면 모든 코드가 컴퓨터에 전역적으로 종속되게 설치됩니다. Virtual environment의 설정은 아래를 참조하세요.

~~~bash
cd assignment1
sudo pip install virtualenv      # 아마 먼저 설치되어 있을 겁니다.
virtualenv .env                  # virtual environment를 만듭니다.
source .env/bin/activate         # virtual environment를 활성화 합니다.
pip install -r requirements.txt  # dependencies 설치합니다.
# Work on the assignment for a while ...
deactivate                       # virtual environment를 종료합니다.
~~~

**데이터셋 다운로드:**
먼저 숙제를 시작하기전에 CIFAR-10 dataset를 다운로드해야 합니다. 아래 코드를 `assignment2` 폴더에서 실행하세요:

~~~bash
cd cs231n/datasets
./get_datasets.sh
~~~

**Cython extension 컴파일하기:** 컨볼루션 신경망은 매우 효율적인 구현을 필요로 합니다. 이 숙제를 위해서 [Cython](http://cython.org/)을 활용하여 여러 기능들을 구현해 놓았는데, 이를 위해 코드를 돌리기 전에 Cython extension을 컴파일 해야 합니다. `cs231n` 디렉토리에서 아래 명령어를 실행하세요:

~~~bash
python setup.py build_ext --inplace
~~~

**IPython 시작:**
CIFAR-10 data를 받았다면, `assignment1` 폴더의 IPython notebook server를 시작할 수 있습니다. IPython에 친숙하지 않다면 작성해둔 [IPython tutorial](/ipython-tutorial)를 읽어보는 것을 권장합니다.

**NOTE:** OSX에서 virtual environment를 실행하면, matplotlib 에러가 날 수 있습니다([이 문제에 관한 이슈](http://matplotlib.org/faq/virtualenv_faq.html)).  IPython 서버를 `assignment2`폴더의 `start_ipython_osx.sh`로 실행하면 이 문제를 피해갈 수 있습니다; 이 스크립트는 virtual environment가 `.env`라고 되어있다고 가정하고 작성되었습니다.

### 과제 제출:
로컬 환경이나 Terminal에 상관없이, 이번 숙제를 마쳤다면 `collectSubmission.sh`스크립트를 실행하세요. 이 스크립트는 `assignment2.zip`파일을 만듭니다. 이 파일을 [the coursework](https://coursework.stanford.edu/portal/site/W16-CS-231N-01/)에 업로드하세요.

### Q1: Fully-connected 뉴럴 네트워크 (30 points)
`FullyConnectedNets.ipynb` IPython notebook 파일에서 모듈화된 레이어 디자인을 소개하고, 이 레이어들을 이용해서 임의의 깊이를 갖는 fully-connected 네트워크를 구현할 것입니다. 이 모델들을 최적화하기 위해서 자주 사용되는 여러 가지 업데이트 규칙들을 구현해야 할 것입니다.

### Q2: Batch Normalization (30 points)
`BatchNormalization.ipynb` IPython notebook 파일에서는 batch normalization 을 구현하고, 이를 사용하여 깊은(deep) fully-connected 네트워크를 학습할 것입니다.

### Q3: Dropout (10 points)
`Dropout.ipynb` IPython notebook 파일에서는 Dropout을 구현하고, 이것이 모델의 일반화 성능에 어떤 영향을 미치는지 살펴볼 것입니다.

### Q4: CIFAR-10 에서의 컨볼루션 신경망 (30 points)
`ConvolutionalNetworks.ipynb` IPython notebook 파일에서는 컨볼루션 신경망에서 흔히 사용되는 여러 새로운 레이어들을 구현할 것입니다. 먼저 CIFAR-10 데이터셋에 대해 (얕은, 깊지않은, 작은 규모의) 컨볼루션 신경망을 학습하고, 이후에는 가능한 한 최선의 노력을 다해서 최고의 성능을 뽑아내보길 바랍니다.

### Q5: 추가 과제: 뭔가 더 해보세요! (up to +10 points)
네트워크를 학습하는 과정 속에서, 더 좋은 성능을 위해 필요한 것이 있다면 얼마든지 추가적으로 구현하기 바랍니다. 최적화 기법(solver)을 바꿔도 좋고, 추가적인 레이어를 구현하거나, 다른 종류의 regularization 을 사용하고나, 모델 ensemble 등 생각나는 모든 것을 시도해 보세요. 이번 숙제에서 다루지 않은 새로운 아이디어를 구현한다면 추가 점수를 받을 수 있을 것입니다.
