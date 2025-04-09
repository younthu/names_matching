from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

class UniversityMatcher:
    def __init__(self, primary_keys_file: str, model_name: str = 'all-mpnet-base-v2'):
        # 加载预训练模型
        self.model = SentenceTransformer(model_name)
        
        # 读取标准大学名称列表
        with open(primary_keys_file, 'r', encoding='utf-8') as f:
            self.standard_names = [line.strip() for line in f if line.strip()]
        
        # 预处理标准名称
        self.processed_names = [self._preprocess_name(name) for name in self.standard_names]
        
        # 计算标准名称的向量表示
        self.name_embeddings = self.model.encode(self.processed_names)
    
    def _preprocess_name(self, name: str) -> str:
        """预处理大学名称"""
        # 转换为小写
        name = name.lower()
        # 移除特殊字符和多余空格
        name = re.sub(r'[^\w\s]', ' ', name)
        # 替换多个空格为单个空格
        name = re.sub(r'\s+', ' ', name)
        # 移除前后空格
        return name.strip()
    
    def find_matches(self, query: str, threshold: float = 0.6, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        查找最匹配的大学名称
        
        Args:
            query: 要查询的大学名称
            threshold: 相似度阈值，低于此值的匹配将被过滤
            top_k: 返回的最大匹配数量
            
        Returns:
            列表，包含(匹配的标准名称, 相似度分数)的元组，按相似度降序排序
        """
        # 预处理查询名称
        processed_query = self._preprocess_name(query)
        
        # 计算查询名称的向量表示
        query_embedding = self.model.encode([processed_query])
        
        # 计算余弦相似度
        similarities = cosine_similarity(query_embedding, self.name_embeddings)[0]
        
        # 获取相似度最高的结果
        matches = []
        for idx in np.argsort(similarities)[::-1]:
            if similarities[idx] >= threshold:
                matches.append((self.standard_names[idx], float(similarities[idx])))
            if len(matches) >= top_k:
                break
        
        return matches

def main():
    # 初始化匹配器
    matcher = UniversityMatcher('/Users/zhiyongyang/sourcecode/names_matching/data/university_primary_keys.txt')
    
    while True:
        # 获取用户输入
        query = input("\n请输入大学名称（输入 'q' 退出）: ")
        if query.lower() == 'q':
            break
            
        # 查找匹配
        matches = matcher.find_matches(query)
        
        if matches:
            print("\n找到以下可能的匹配：")
            for name, score in matches:
                print(f"- {name} (相似度: {score:.2f})")
        else:
            print("\n未找到匹配的大学名称。")

if __name__ == "__main__":
    main()