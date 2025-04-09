from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 示例文本数据
sentences = [
    ['I', 'love', 'natural', 'language', 'processing'],
    ['Python', 'is', 'a', 'powerful', 'programming', 'language'],
    ['Tsinghua', 'is', 'a', 'name', 'of', 'university'],
    ['Tsinghua University', 'is', 'a', 'name', 'of', 'university'],
    ['清华', 'is', 'a', 'name', 'of', 'university'],
    ['清华大学', 'is', 'a', 'name', 'of', 'university'],
    ['北大', 'is', 'a', 'name', 'of', 'university'],
    ['东北大学', 'is', 'a', 'name', 'of', 'university'],
    ['北京大学', 'is', 'a', 'name', 'of', 'university'],
    ['Peking University', 'is', 'a', 'name', 'of', 'university'],
    ['Machine', 'learning', 'is', 'very', 'interesting']
]

# 训练 Word2Vec 模型
model = Word2Vec(sentences, min_count=1)

# 定义两个要比较的词
# word1 = '清华'
# word2 = 'Tsinghua'
word1 = 'Peking University'
word2 = '北大'

# 检查词是否在词汇表中
if word1 in model.wv and word2 in model.wv:
    # 获取词向量
    vector1 = model.wv[word1]
    vector2 = model.wv[word2]

    # 调整向量形状以适应 cosine_similarity 函数
    vector1 = vector1.reshape(1, -1)
    vector2 = vector2.reshape(1, -1)

    # 计算余弦相似度
    similarity = cosine_similarity(vector1, vector2)[0][0]
    print(f"'{word1}' 和 '{word2}' 的相似度是: {similarity}")
else:
    print("至少有一个词不在词汇表中。")
    