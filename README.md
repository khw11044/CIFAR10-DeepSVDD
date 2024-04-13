
## CIFA10으로 Anomaly Detection - DeepSVDD 기반 


DeepSVDD가 2019년에 나온 논문으로 가장 대표적인 Anomaly detection 기법인데 

MNIST에 대해서는 뭐 98% 나오지만 CIFAR10에 대해서는 58~60% 정확도만 나온다. 

아래 paperwithcode를 참고하면 32위로 65.7%라고 한다. 

더 좋은 방법을 찾아야 한다. 

[paperwithcode: DeepSVDD-CIFAR10](https://paperswithcode.com/sota/anomaly-detection-on-one-class-cifar-10)


```python

AUC
airplane : 67.72%
automobile : 51.84%
bird : 63.19% 
cat : 52.94%
deer : 73.46%
dog : 46.85%
frog : 76.55%
horse : 54.11%
ship : 46.80%
truck : 49.64%

```