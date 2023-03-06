## 세종시 뭐하니 

### 서비스 배경
세종시의 상가공실률은 전국 최상위 수준이며 이를 해소하기 위해 정부는 상가 허용 업종 규제 완화를 추진하는 등 적극적으로 힘쓰고 있다. <br>
이에 따라 창업자나 소상공인의 관심이 증가할 것이라 예상된다.  <br>
따라서 예비 창업자 및 소상공인을 위한 **업종별 상권분석 및 매출 예측 서비스**를 제공한다면 맞춤 업종 및 위치 선정과 안정적인 가게 운영에 도움이 될 것이다. 



### Tech stack & Frameworks
- 시각화 : matplotlib, seaborn
- 데이터 전처리 : numpy, pandas
- 모델링 : sckit-learn, huber regressor
- 웹 구현 : html, css, js, Flask, Mysql, QGIS

### model
huber regressor

- MAE: 9597225.586886
- RMSE: 3578714022611885.500000
- R2: 0.984369

### web
<img src='https://user-images.githubusercontent.com/49296139/223034382-e6258d62-1f27-49f3-8bee-eb5b3731cf03.png' width="400" height="300">
<img src='https://user-images.githubusercontent.com/49296139/223035405-a22e7a22-b500-401d-a92d-af6c77321332.png' width="400" height="300">
<img src='https://user-images.githubusercontent.com/49296139/223035550-1e0b8e9b-6ed2-4c1c-8546-8d7503cda9b2.png' width="400" height="300">





### 개선점

- 구체적인 카드 매출 발생 가맹점별 위치 데이터를 확보한다면 상권기반예측으로 다양한 피처를 추가할 수 있다. (ex. 상권 종류, 업력, 지하철/버스 정류장 500m 이내 유무 등등)
- 장기간의 데이터를 확보한다면 예측할 수 있는 기간이 길어지고 시계열 모델 고도화를 할 수 있을 것이다. 
- 업종별로 모델링을 한다면 정확도가 상승할 것이라 예상한다. 
