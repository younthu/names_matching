

from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_names(file_path):
    names = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
        for line in lines:
            if not line:
                continue
            names.append(line)
    return names

def get_model(model_name: str, file_path: str):
    model = SentenceTransformer(model_name)
    names = get_names(file_path)
    model.encode(names, normalize_embeddings=True)
    return model

def find_matches(query: str, model: SentenceTransformer, embeddings, names: list[str], threshold: float = 0.6, top_k: int = 5) -> List[Tuple[str, float]]:
     # 计算查询名称的向量表示
    query_embedding = model.encode([query], normalize_embeddings=True)
    
    # 计算余弦相似度
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # 获取相似度最高的结果
    matches = []
    seen_primaries = set()
    
    # 按相似度排序获取索引
    sorted_indices = np.argsort(similarities)[::-1]
    return [(names[i], float(similarities[i])) for i in sorted_indices if similarities[i] >= threshold][:top_k]

for m in ["BAAI/bge-m3", "BAAI/bge-large-zh-V1.5"]:
    model = SentenceTransformer(m)
    names = get_names('./data/cross-lingual-rag.txt')
    embeddings = model.encode(names, normalize_embeddings=True)


    for name in [
        "剑桥大学",
        "北京大学",
        "清华大学",
        "Huazhong University of Science and Technology",
        "Zhejiang University",
        "武汉纺织大学",
        "东北大学",
        "中国东北大学"
        ]:
        matches = find_matches(name, model, embeddings, names)
        print(f'{m=},{name=}\t=>{matches}')
