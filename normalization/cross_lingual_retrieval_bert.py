import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
import os # 导入 os 模块

def load_university_names(filepath):
    """从文件加载大学名称列表"""
    # Indent the following block
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            names = [line.strip() for line in f if line.strip()]
        print(f"成功加载 {len(names)} 个大学名称从 {filepath}")
        return names
    except FileNotFoundError:
        print(f"错误：文件未找到 {filepath}")
        return []
    except Exception as e:
        print(f"加载文件时出错 {filepath}: {e}")
        return []

def generate_embeddings(model, texts, batch_size=32):
    """使用模型为文本列表生成嵌入"""
    # Indent the following block
    print(f"开始为 {len(texts)} 个文本生成嵌入...")
    start_time = time.time()
    # 确保模型在合适的设备上 (GPU if available, else CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    model.to(device)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, device=device)
    end_time = time.time()
    print(f"生成嵌入完成，耗时: {end_time - start_time:.2f} 秒")
    return embeddings

def find_top_matches(query_embedding, database_embeddings, database_names, top_n=5):
    """查找与查询嵌入最相似的前 N 个匹配项"""
    # Indent the following block
    # 计算余弦相似度
    similarities = cosine_similarity(query_embedding.reshape(1, -1), database_embeddings)[0]
    # 获取排序后的索引
    sorted_indices = np.argsort(similarities)[::-1]
    # 获取前 N 个匹配项及其分数
    top_matches = [(database_names[i], similarities[i]) for i in sorted_indices[:top_n]]
    return top_matches

# --- 主程序 ---
if __name__ == "__main__":
    # 获取当前脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建相对于脚本位置的数据文件路径
    # Correctly identify the project root (one level up from script_dir)
    project_root = os.path.dirname(script_dir) # Corrected this line
    university_list_path = os.path.join(project_root, 'data', 'university_primary_keys.txt')
    # 或者，如果坚持使用绝对路径:
    # university_list_path = '/Users/zhiyongyang/sourcecode/names_matching/data/university_primary_keys.txt'

    print(f"尝试从以下路径加载大学列表: {university_list_path}")

    # 1. 加载大学名称
    university_names = load_university_names(university_list_path)

    if not university_names:
        print("未能加载大学名称列表，程序退出。")
        exit()

    # 2. 加载 mBERT 模型
    print("正在加载 mBERT 模型 (bert-base-multilingual-cased)...")
    start_time = time.time()
    # 你可以选择其他多语言模型，例如 'paraphrase-multilingual-mpnet-base-v2' 可能效果更好
    model = SentenceTransformer('bert-base-multilingual-cased')
    # model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    end_time = time.time()
    print(f"模型加载完成，耗时: {end_time - start_time:.2f} 秒")


    # 3. 生成数据库嵌入 (如果嵌入已存在，可以考虑加载它们以节省时间)
    # 注意：为大量名称生成嵌入可能需要一些时间
    database_embeddings = generate_embeddings(model, university_names)

    # 4. 定义测试查询 (键: 查询名称, 值: 预期英文名称)
    test_queries_with_expected = {
        "Peking University": "Peking University",              # 英语
        "北京大学": "Peking University",                       # 中文
        "北大": "Peking University",                       # 中文
        "清华大学": "Tsinghua University",                       # 英文
        "清华": "Tsinghua University",                       # 中文
        "Université de Paris": "University of Paris",            # 法语 (注意: 实际列表可能没有完全对应的 'University of Paris', 取决于 university_primary_keys.txt 内容)
        "Universidad de Buenos Aires": "University of Buenos Aires",    # 西班牙语
        "Universität Heidelberg": "Heidelberg University",         # 德语
        "東京大学": "The University of Tokyo",                       # 日语
        "서울대학교": "Seoul National University",                     # 韩语
        "Московский государственный университет": "Lomonosov Moscow State University", # 俄语
        "جامعة القاهرة": "Cairo University",                  # 阿拉伯语
        "Universidade de São Paulo": "University of São Paulo",      # 葡萄牙语
        "Università di Bologna": "University of Bologna",          # 意大利语
        "Universiteit van Amsterdam": "University of Amsterdam",     # 荷兰语
        "Uniwersytet Warszawski": "University of Warsaw",         # 波兰语
        "Đại học Quốc gia Hà Nội": "Vietnam National University, Hanoi",        # 越南语
        "มหาวิทยาลัยจุฬาลงกรณ์": "Chulalongkorn University",          # 泰语
        "İstanbul Üniversitesi": "Istanbul University",          # 土耳其语
        "Πανεπιστήμιο Αθηνών": "National and Kapodistrian University of Athens",            # 希腊语
        "Tel Aviv University": "Tel Aviv University",            # 英语/希伯来语
        "香港大学": "The University of Hong Kong",                       # 中文
        "ETH Zurich": "ETH Zurich - Swiss Federal Institute of Technology" # 英语/德语缩写 (注意: 预期名称可能需要根据实际列表调整)
    }

    print("\n--- 开始跨语言检索测试 ---")

    # 5. & 6. & 7. 为每个查询生成嵌入并查找匹配项
    for query, expected_name in test_queries_with_expected.items():
        print(f"\n查询: \"{query}\" (预期英文名: \"{expected_name}\")")
        # 注意：每次都为单个查询生成嵌入效率较低，但在测试场景下可以接受
        # 如果有大量查询，最好一次性生成所有查询的嵌入
        query_embedding = generate_embeddings(model, [query])
        top_matches = find_top_matches(query_embedding, database_embeddings, university_names, top_n=5)

        print("Top 5 匹配结果:")
        for name, score in top_matches:
            print(f"  - {name} (相似度: {score:.4f})")

    print("\n--- 检索测试完成 ---")