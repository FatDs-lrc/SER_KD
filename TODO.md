## 2021.02.19

1. 尝试使用1dcnn    [DONE]
+ 结果:

| acc    | uar    | f1     |
|--------|--------|--------|
| 0.4562 | 0.4766 | 0.4389 |


## 2021.02.20

2. 更换成双向LSTM

3. 在CNN出来的结果后面+DNN

4. 参照SincNet的DNN设计, 在DNN里增加BN和LN

5. SincNet存在爆显存的问题:
+ 如果输入平均是4s, 则输入Tensor大小为[bs, 400, 3200], 这时bs设置为32的时候都会爆显存
+ 调研E2E-SincNet(ASR) https://github.com/TParcollet/E2E-SincNet
+ 调研 "EEG Emotion Classiﬁcation Using an Improved SincNet-Based Deep Learning Model"论文中的方案(略读)
+ [PENDING]