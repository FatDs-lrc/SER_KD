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
    


## 
Resnet 改成2dresnet试试
1. conv1d + 2d Resnet 直接做分类
2. 2dResnet 改前两层的kernel接LSTM


## Transformer 实验记录
init method改为normal有效，xavier和kaiming都难以收敛
transformer速度很快，比LSTM快很多

+ 直接用resample_poly降采样的comparE
    + maxpool
    - 没有用warmup，直接训练
        - 1层
            acc     uar     f1
            0.5397	0.5641	0.5325
            0.5323	0.5548	0.5302
            0.5387	0.5584	0.5263
        
        - 2层
            - 学习率调整为5e-4, 学习率为1e-4时效果不好
            两层效果差一些，不太稳定
            acc     uar     f1
            0.5165	0.5338	0.5075
            0.5264	0.5366	0.5002

    - 添加warmup解决初始化的问题, 学习率统一调整为5e-4, 重新实验
        - 1层
            acc     uar     f1
            0.5308	0.5482	0.5260
            0.5343	0.5542	0.5271
        - 2层
            acc     uar     f1


        - 3层
            acc     uar     f1
            0.5233	0.5429	0.5183
            0.5348	0.5556	0.5303

        - 4层
            acc     uar     f1
            0.5423	0.5604	0.5405
            0.5225	0.5453	0.5163

        - 6层
            acc     uar     f1
            0.5303	0.5423	0.5249
            0.5371	0.5585	0.5326
        
        效果基本差不多，和LSTM相比有一点差距
    
    + mean pool
        - 1层
            acc     uar     f1
            0.5423	0.5529	0.5364
            0.5334	0.5412	0.5243

        - 2层
            acc     uar     f1
            0.5198	0.5340	0.5178
            0.5358	0.5452	0.5329

        - 3层
            acc     uar     f1
            0.5248	0.5416	0.5146
            0.5172	0.5317	0.5135

        - 4层

        - 6层
            acc     uar     f1
            0.5218	0.5376	0.5077
            0.5178	0.5416	0.5139

    + 用最后一个时刻
        - 1层
            acc     uar     f1
            0.5032	0.5268	0.4995
            0.5116	0.5301	0.5054

        - 2层
            acc     uar     f1
            0.5218	0.5376	0.5077
            0.5210	0.5296	0.5185

        - 3层
            acc     uar     f1
            0.5319	0.5403	0.5227
            0.5368	0.5472	0.5332

        - 4层
            acc     uar     f1
            

        - 6层
            acc     uar     f1
            0.5137	0.5352	0.5102
            0.5178	0.5416	0.5139

    + 结论：transformer层数对实验结果影响不大，transformer实验不是很稳定

+ cnn1d + Transformer
    + 根据上面的实验发现，transformer 层数影响不大，使用maxpool效果较好
    - 1 层
    - 2 层
    - 4 层
    

## Resnet 实验记录
- 完全仿照IJICAI的实现 + bi-LSTM
    acc     uar     f1
    0.5311	0.5569	0.5280
    0.5087	0.5227	0.5014
    0.5251	0.5575	0.5200
    0.5297	0.5475	0.5206

- 在第一层conv中改成bias=False, 并添加bn+relu
    acc     uar     f1
    0.5238	0.5383	0.5189
    0.5063	0.5378	0.5001
    效果很不稳定

- 修改将采样为 1*1 卷积, 并添加bn，relu
    效果不太行




1. 处理MELD, Val + tst 进入 V4
2. 重新训练6个实验: KD, 1mse, 2mse, 3mse, 4mse, nokd 4mse
3. finetune多个epoch找最好结果, 每个2次
4. meld 单独训练的实验

ablation:
1. 消融实验
2. teacher student match的程度 打印一个confusion matrix
3. MELD中fear和disguest讨论, re-balance, re-weight, 能否加强


