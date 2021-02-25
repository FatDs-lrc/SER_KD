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
        | run | acc    | uar    | f1     |
        |-----|--------|--------|--------|
        | 1   | 0.4764 | 0.4997 | 0.4602 |
        | 2   | 0.4588 | 0.4735 | 0.4457 |

        -- **trn提升速度稍微加快, val最大值有所提升, tst上提升幅度不大** --

    

3. 在CNN出来的结果后面+DNN
    + 128,128,128 norm-None 
    -- 泛化性能提高了很多,真·玄学, 但是变得难学了很多, trn上50epoch下去只能到68%左右 
    | run | acc    | uar    | f1     |
    |-----|--------|--------|--------|
    | 1   | 0.4643 | 0.4821 | 0.4350 |
    -- **效果不稳定**, 有一组实验出现了多个cv卡在0.25的情况 --

    + 128,128,128 norm-on-trn
    | run | acc    | uar    | f1     |
    |-----|--------|--------|--------|
    | 1   | 0.3934 | 0.4307 | 0.3556 |
    | 2   | 0.4643 | 0.4821 | 0.4350 |
    
    + 大参数1dcnn 64channel && 128channel
    64 channel
    | run | acc    | uar    | f1     |
    |-----|--------|--------|--------|
    | 1   | 0.5068 | 0.4973 | 0.4666 |
    | 2   | 0.5305 | 0.5000	| 0.4799 |

## 2021.02.22
4. 修改SincNet的shift为100ms，先跑一版出来看看结果
    + bs 32, shift 0.1s, 卡在0.25不动
    + **跑的特别慢** GPU占用基本是满的，性能提升空间不大，可以尝试加半精度
    + 学习率改成 1e-4 能动了
    + 4.1 使用amp混合精度加速

5. 参照SincNet的DNN设计, 在DNN里增加BN和LN [PENDING]

6. 将1dcnn中第一层cnn替换成SincNet 

7. SincNet存在爆显存的问题:                [PENDING]
    + 如果输入平均是4s, 则输入Tensor大小为[bs, 400, 3200], 这时bs设置为32的时候都会爆显存
    + 调研E2E-SincNet(ASR) https://github.com/TParcollet/E2E-SincNet
    + 调研 "EEG Emotion Classiﬁcation Using an Improved SincNet-Based Deep Learning Model"论文中的方案(略读)
    + [PENDING]

8. 替换LSTM为transformer

9. 回头来看看vggish + LSTM
    + 重新抽一下特征
    
