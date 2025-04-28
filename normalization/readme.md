# 测试学校归一化

## university_normalization.py
这个是通过openai生成embedding的方式，需要设置OPENAI_API_KEY.

内含简单的测试，运行方式：
1. `export OPENAI_API_KEY="sk-proj-xxxxxxxxx"`
1. 开启VPN
2. `python normalization/university_normalization.py`

## cross_lingual_retrieval_bert.py
这个是通过 mBERT (`bert-base-multilingual-cased`) 生成 embedding 的方式, 在内存进行检索。主要是想测试跨语言学校名称匹配的效果。效果非常差, 北京大学都匹配错误。

内含简单的测试，运行方式：
1. 确保已安装依赖: `pip install torch transformers faiss-cpu numpy`
2. `python normalization/cross_lingual_retrieval_bert.py`

## crosslingual-match.py

这个是测试BGE跨语言匹配效果的。 效果比mBERT好, 但是比openai的效果差很多。没有做rerank。
内含简单的测试，运行方式：
1. 确保已安装依赖: `pip install torch transformers faiss-cpu numpy`
2. `python normalization/crosslingual-match.py`

# 总结
还是openai的效果好。上面的测试都是没有加rerank，加了rerank的效果更好。

openai之后是bge的embedding, 或者通义千问的embedding.

mbert不行.

如果没有rerank模型，也可以用LLM来做rerank，这个效果比rerank模型效果更好。