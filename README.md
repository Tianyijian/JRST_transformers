# 2019 CCF BDCI 金融信息负面及主体判定

## 赛题介绍

给定文本以及实体列表，判断实体的正负性，是ABSA（Aspect-Based Sentiment Analysis）任务。

赛题链接：[https://www.datafountain.cn/competitions/353](https://www.datafountain.cn/competitions/353)

## 方法简介

- 将文本与实体一一对应，作为句子对使用BERT二分类
- 将实体在文本中的位置标记出来，使用BERT做二分类

## 代码模板

使用PyTorch版的BERT，即Huggingface的[transformers](https://github.com/huggingface/transformers)。

在代码中增加了对比赛数据的对接，五折交叉验证，输出日志的优化，tensorboard的记录优化。

## 技巧结果

- 训练集与最终的结果文件中存在很多的实体包含关系，需要做实体清理。
- 最终的结果为0.9479，复赛排名30名左右，引以为戒。