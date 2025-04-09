from gensim.models import Word2Vec
import nltk
from nltk.corpus import brown

# 下载 brown 语料库
nltk.download('brown')

# 获取 brown 语料库中的句子
sentences = brown.sents()

# 训练 Word2Vec 模型
model = Word2Vec(sentences, min_count=1)

# 计算 "king" 和 "queen" 的相似度
similarity = model.wv.similarity('king', 'queen')
print(f"“king” 和 “queen” 的相似度为: {similarity}")
    