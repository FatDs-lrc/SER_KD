## 2021.02.20

1. 尝试使用1dcnn ------- [DONE]
    + 结果:
        32channel + 单向LSTM
        | acc    | uar    | f1     |
        |--------|--------|--------|
        | 0.4562 | 0.4766 | 0.4389 |


## 2021.02.21

2. 更换成双向LSTM, 增大1dcnn的参数量, 使用mean 
    + biLSTM - 32channel norm-None ------- [DONE] 

        | run | acc    | uar    | f1     |
        |-----|--------|--------|--------|
        | 1   | 0.4817 | 0.4943 | 0.4603 |
        | 2   | 0.4656 | 0.4945 | 0.4530 |
        -- **提升2个点** --


    + mean on utt ------- [ABORT]

        原因: 发现输入数据绝对值很小, 基本都在1e-3这个量级, 尝试归一化到[-1, 1]
        python
        ```
        mean = signal.mean()
        std = signal.std()
        signal = (signal - mean) / std
        ```
        -- **不太行, 一直卡在25%不动** --

    + mean on trn 
        python
        ```
        mean = -7.395432666646425e-06
        std = 0.058110240439255334
        signal = (signal - mean) / std
        ```

        -- **trn提升速度稍微加快, val最大值有所提升, tst上提升幅度不大** --

+ 大参数1dcnn 64channel && 128channel


3. 在CNN出来的结果后面+DNN
    + 128,128,128 norm-None -- 泛化性能提高了很多,真·玄学, 但是变得难学了很多, trn上50epoch下去只能到68%左右

## 2021.02.22
4. 参照SincNet的DNN设计, 在DNN里增加BN和LN

5. 将第一层cnn替换成SincNet 

6. SincNet存在爆显存的问题:
    + 如果输入平均是4s, 则输入Tensor大小为[bs, 400, 3200], 这时bs设置为32的时候都会爆显存
    + 调研E2E-SincNet(ASR) https://github.com/TParcollet/E2E-SincNet
    + 调研 "EEG Emotion Classiﬁcation Using an Improved SincNet-Based Deep Learning Model"论文中的方案(略读)
    + [PENDING]

7. 替换LSTM为transformer