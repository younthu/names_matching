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

这个是测试BGE跨语言匹配效果的。 效果比mBERT好, 但是比openai的效果差很多。没有做rerank。 rerank很简单，用一个模型，做语义相似度比较就可以了。
内含简单的测试，运行方式：
1. 确保已安装依赖: `pip install torch transformers faiss-cpu numpy`
2. `python normalization/crosslingual-match.py`

# 总结
还是openai的效果好。上面的测试都是没有加rerank，加了rerank的效果更好。

openai之后是bge的embedding, 或者通义千问的embedding.

mbert不行.

如果没有rerank模型，也可以用LLM来做rerank，这个效果比rerank模型效果更好。


# 测试文件描述
1. cross-lingual-rag.txt, 大学名称列表，直接生成embedding，query的时候也不做任何修饰。
    1. 测试跨语言embedding. 可以输入剑桥大学，输出combridge university 或者输入huazhong university of science and technology, 输出华中科技大学。 或者输入PKU，输出北京大学。
2. cross-lingual-rag-templated.txt, 大学名称列表，做了模板处理: University named: {university}。没啥效果提升。
3. cross-lingual-rag-rerank-templated2.txt, 大学名称列表， 做了模板处理， 加入了大学的特征信息：University named: Peking University, located in Beijing, China 。
    1. 效果提升了很多。
    1. embedding的时候需要用llm填充特征信息。填充该学校所在的国家和城市。
    1. query之前要用llm做同样的信息填充。填充该学校所在的国家和城市。
    1. 只做向量检索，不做rerank，挺准确的。开了rerank之后，效果变差了, 不准了。
    1. 用dify测试的。
4. cross-lingual-rag-represent.txt, 大学名称列表，做了模板处理， 加入了大学的特征信息：The represent of university: Peking University。
    1. 效果没有提升。