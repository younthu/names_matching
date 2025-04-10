from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

class UniversityMatcher:
    # BAAI/bge-large-zh-V1.5, 
    # BAAI/bge-m3
    # https://huggingface.co/BAAI/bge-large-zh-v1.5
    def __init__(self, primary_keys_file: str, model_name: str = 'BAAI/bge-large-zh-V1.5', parent_separator: str = '\n\n', child_separator: str = '\n'):
        # 加载预训练模型
        self.model = SentenceTransformer(model_name)
        self.parent_separator = parent_separator
        self.child_separator = child_separator
        
        # 读取并解析大学名称文件
        self.university_data = self._parse_university_file(primary_keys_file)
        
        # 计算所有名称的向量表示
        all_names = []
        for names in self.university_data.values():
            all_names.extend(names)
        self.all_names = all_names
        self.name_embeddings = self.model.encode(all_names, normalize_embeddings=True)
        
        # 创建名称到标准名称的映射
        self.name_to_primary = {name: primary for primary, names in self.university_data.items() 
                              for name in names}
    
    def _parse_university_file(self, file_path: str) -> Dict[str, List[str]]:
        """解析大学名称文件，返回标准名称到所有别名的映射"""
        university_data = {}
        current_names = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            
        for line in lines:
            if not current_names:
                current_names.append(line)
            elif line == "":
                if current_names:
                    university_data[current_names[0]] = current_names
                current_names = []
            else:
                current_names.append(line)
                
        if current_names:
            university_data[current_names[0]] = current_names
            
        return university_data
    
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
            threshold: 相似度阈值
            top_k: 返回的最大匹配数量
            
        Returns:
            列表，包含(标准名称, 相似度分数)的元组，按相似度降序排序
        """
        # 计算查询名称的向量表示
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # 计算余弦相似度
        similarities = cosine_similarity(query_embedding, self.name_embeddings)[0]
        
        # 获取相似度最高的结果
        matches = []
        seen_primaries = set()
        
        # 按相似度排序获取索引
        sorted_indices = np.argsort(similarities)[::-1]
        
        for idx in sorted_indices:
            if similarities[idx] >= threshold:
                matched_name = self.all_names[idx]
                primary_name = self.name_to_primary[matched_name]
                
                # 避免重复的标准名称
                if primary_name not in seen_primaries:
                    matches.append((primary_name, float(similarities[idx])))
                    seen_primaries.add(primary_name)
                    
                    # 同时返回中文名（如果存在）
                    chinese_name = next((name for name in self.university_data[primary_name] 
                                      if any('\u4e00' <= c <= '\u9fff' for c in name)), None)
                    if chinese_name:
                        matches[-1] = (f"{primary_name} ({chinese_name})", float(similarities[idx]))
                        
            if len(matches) >= top_k:
                break
                
        return matches

def main():
    # 初始化匹配器
    matcher = UniversityMatcher('/Users/zhiyongyang/sourcecode/names_matching/data/university_primary_keys.primary_key.cn_name.all_names.all_chars.newline_separator.txt')
    
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