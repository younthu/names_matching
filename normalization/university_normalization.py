import atexit # atexit 可能不再需要，但保留以防万一
import yaml
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
from openai import OpenAI
# 仅导入 MilvusClient 和可能的辅助类型
from pymilvus import MilvusClient, DataType # DataType 可能需要用于 schema

# --- 配置常量 ---
# Milvus Lite 数据库文件名 (MilvusClient 直接使用)
MILVUS_DB_FILE = "university_embeddings_client.db" # 使用新文件名避免与旧方式冲突
# Milvus 集合名称
COLLECTION_NAME = "universities_client" # 使用新集合名
# OpenAI 嵌入模型
OPENAI_EMBED_MODEL = "text-embedding-3-small"
# OpenAI 嵌入维度
EMBEDDING_DIM = 1536
# Milvus 字段名 (保持不变)
FIELD_ID = "id" # MilvusClient 默认需要一个主键，通常是 int64
FIELD_NAME = "name"
FIELD_EMBEDDING = "embedding"
# Milvus 索引参数 (格式可能略有不同，或通过 MilvusClient 方法设置)
INDEX_PARAMS = {
    "metric_type": "IP", # Inner Product for cosine similarity on normalized vectors
    "index_type": "AUTOINDEX", # MilvusClient 推荐使用 AUTOINDEX，它会自动选择合适的索引类型
    "params": {} # AUTOINDEX 通常不需要额外参数
}
# 搜索参数 (格式可能略有不同)
SEARCH_PARAMS = {
    "metric_type": "IP",
    "params": {} # 搜索参数通常在 search 调用时指定，如 consistency_level
}
# 大学列表文件名
UNIVERSITY_LIST_FILE = "university_primary_keys.txt"
# 配置文件名
CONFIG_FILE = "config.yaml"

class UniversityNormalization:
    """
    使用 OpenAI 嵌入和 MilvusClient (Milvus Lite) 对大学名称进行规范化。
    """
    def __init__(self):
        self.persist_dir = Path(__file__).parent
        self.db_path = self.persist_dir / MILVUS_DB_FILE
        self.collection_name = COLLECTION_NAME # 存储集合名称
        self.universities: List[str] = []
        self.client_openai = self._init_openai_client() # 重命名避免冲突

        # 初始化 MilvusClient
        print(f"初始化 MilvusClient，使用数据库文件: {self.db_path}")
        # MilvusClient 直接接收数据库文件名
        self.client_milvus = MilvusClient(uri=str(self.db_path))
        print("MilvusClient 初始化完成。")

        self._load_universities()
        self._init_milvus_collection()
        # atexit 不再需要手动管理 Milvus Lite 服务器的启停

    # --- 移除 _start_milvus_lite, _stop_milvus_lite, _connect_milvus ---

    def _init_openai_client(self) -> OpenAI:
        """从配置文件初始化 OpenAI 客户端，使用 openai-4o 的密钥。"""
        # 配置文件路径相对于当前脚本文件
        config_path = Path(__file__).parent.parent / CONFIG_FILE
        try:
            # 从环境变量获取 OpenAI API 密钥
            openai_api_key = os.getenv("OPENAI_API_KEY")

            print("OpenAI 客户端初始化成功 (使用 openai-4o 密钥)。")
            return OpenAI(api_key=openai_api_key)
        except Exception as e:
            print(f"初始化 OpenAI 客户端时发生意外错误: {e}")
            raise ValueError("无法初始化 OpenAI 客户端。请检查 config.yaml 文件和配置结构。") from e

    def _load_universities(self):
        """从文件加载大学列表。"""
        university_file = self.persist_dir / "../data" / UNIVERSITY_LIST_FILE
        if not university_file.exists():
            raise FileNotFoundError(f"大学列表文件未找到: {university_file}")
        try:
            with open(university_file, 'r', encoding='utf-8') as f:
                self.universities = [line.strip() for line in f if line.strip()]
            if not self.universities:
                raise ValueError("大学列表文件为空或无法读取有效名称。")
            print(f"成功加载 {len(self.universities)} 个大学名称。")
        except Exception as e:
            print(f"加载大学列表时出错: {e}")
            self.universities = []

    def _get_embedding(self, text: str) -> List[float]:
        """使用 OpenAI API 获取单个文本的嵌入。"""
        try:
            response = self.client_openai.embeddings.create(
                model=OPENAI_EMBED_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"获取 '{text}' 的嵌入时出错: {e}")
            raise RuntimeError(f"无法为 '{text}' 生成嵌入。") from e

    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """使用 OpenAI API 获取一批文本的嵌入。"""
        if not texts:
            return []
        try:
            response = self.client_openai.embeddings.create(
                model=OPENAI_EMBED_MODEL,
                input=texts
            )
            if len(response.data) != len(texts):
                 raise ValueError(f"嵌入请求返回了 {len(response.data)} 个结果，预期为 {len(texts)} 个。")
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"批量获取嵌入时出错: {e}")
            raise RuntimeError("无法生成批量嵌入。") from e

    def _init_milvus_collection(self):
        """使用 MilvusClient 初始化 Milvus 集合。"""
        try:
            has_collection = self.client_milvus.has_collection(self.collection_name)
            if has_collection:
                print(f"找到现有集合 '{self.collection_name}'。")
                # MilvusClient 通常不需要显式加载，可以直接操作
                stats = self.client_milvus.get_collection_stats(self.collection_name)
                print(f"集合统计信息: {stats}")
                # 可选：检查实体数
                if stats['row_count'] != len(self.universities):
                     print(f"警告：集合中的实体数 ({stats['row_count']}) 与文件中的大学数 ({len(self.universities)}) 不匹配。可能需要重建集合。")
                     # 如果需要重建，可以先删除旧集合
                     # print("正在删除旧集合以重建...")
                     # self.client_milvus.drop_collection(self.collection_name)
                     # has_collection = False # 强制进入创建逻辑
            else:
                 print(f"集合 '{self.collection_name}' 不存在。正在创建和填充...")

            if not has_collection: # 如果集合不存在 (或被删除后需要重建)
                # 1. 创建集合 (MilvusClient 可以自动创建 schema 或显式定义)
                # 显式定义更安全
                print(f"正在创建集合 '{self.collection_name}'...") # 添加日志
                self.client_milvus.create_collection(
                    collection_name=self.collection_name,
                    dimension=EMBEDDING_DIM, # 主要向量字段的维度
                    primary_field_name=FIELD_ID, # 指定主键字段名 ("id")
                    vector_field_name=FIELD_EMBEDDING, # 指定向量字段名 ("embedding")
                    # id_type="int64", # 移除此行，让 auto_id 处理类型
                    metric_type="IP", # 索引的度量类型
                    auto_id=True, # 让 Milvus 自动生成 int64 主键 ID
                    # 可以添加其他字段定义
                    schema_extra=[
                        {"name": FIELD_NAME, "type": DataType.VARCHAR, "params": {"max_length": 1024}}
                    ],
                    consistency_level="Strong" # 明确指定一致性级别，有时有助于避免问题
                )
                print(f"集合 '{self.collection_name}' 创建成功。")

                # 2. 生成嵌入
                print("正在为大学名称生成嵌入...")
                batch_size = 100
                all_embeddings = []
                for i in range(0, len(self.universities), batch_size):
                    batch_texts = self.universities[i:i + batch_size]
                    batch_embeddings = self._get_embeddings_batch(batch_texts)
                    all_embeddings.extend(batch_embeddings)
                    print(f"已处理 {min(i + batch_size, len(self.universities))}/{len(self.universities)} 个名称...")

                if len(all_embeddings) != len(self.universities):
                     raise RuntimeError("生成的嵌入数量与大学名称数量不匹配。")

                # 3. 准备插入的数据 (list of dictionaries)
                data_to_insert: List[Dict[str, Any]] = []
                for name, embedding in zip(self.universities, all_embeddings):
                    data_to_insert.append({
                        FIELD_NAME: name,
                        FIELD_EMBEDDING: embedding
                        # FIELD_ID 会被 auto_id 自动填充
                    })

                # 4. 插入数据
                print(f"正在将 {len(data_to_insert)} 条数据插入 Milvus...")
                # MilvusClient 的 insert 返回插入结果信息
                insert_result = self.client_milvus.insert(
                    collection_name=self.collection_name,
                    data=data_to_insert
                )
                print(f"数据插入完成。插入的主键数量: {insert_result['insert_count']}, 前几个主键: {insert_result['ids'][:10]}...")
                # MilvusClient 通常会自动 flush，无需手动调用

                # 5. 创建索引 (对于 MilvusClient，通常在创建集合时或之后创建)
                # 如果创建集合时未指定索引类型，可以在此创建
                # 但我们已在 create_collection 中指定了 metric_type，AUTOINDEX 会自动处理
                # 如果需要特定索引，可以取消注释并调整：
                # print("正在为嵌入字段创建索引...")
                # index_params = self.client_milvus.prepare_index_params()
                # index_params.add_index(
                #     field_name=FIELD_EMBEDDING,
                #     index_type="IVF_FLAT", # 或者其他类型
                #     metric_type="IP",
                #     params={"nlist": 128}
                # )
                # self.client_milvus.create_index(self.collection_name, index_params)
                print(f"索引将由 AUTOINDEX 自动创建或已在集合创建时定义。")

                # 6. MilvusClient 不需要显式加载集合

        except Exception as e:
            print(f"初始化 Milvus 集合时出错: {e}")
            # MilvusClient 对象仍然存在，但集合可能未就绪
            raise RuntimeError("无法初始化 Milvus 集合。") from e

    def find_closest_university(self, query: str, top_k: int = 1) -> List[str]:
        """
        使用 MilvusClient 查找与查询最相似的大学名称。
        """
        if not self.client_milvus: # 检查 client 是否初始化
            print("错误：MilvusClient 未初始化。")
            return []
        if not query:
            print("错误：查询字符串不能为空。")
            return []

        try:
            # 1. 获取查询的嵌入
            query_embedding = self._get_embedding(query)

            # 2. 在 Milvus 中搜索
            # MilvusClient 的 search 方法参数略有不同
            search_results = self.client_milvus.search(
                collection_name=self.collection_name,
                data=[query_embedding], # 查询向量列表
                anns_field=FIELD_EMBEDDING, # 要搜索的向量字段
                limit=top_k, # 返回结果数量
                output_fields=[FIELD_NAME], # 需要返回的字段列表
                # search_params 在这里指定，例如：
                # params={"nprobe": 10} # 如果使用 IVF_FLAT 等需要 nprobe 的索引
                # 对于 AUTOINDEX 或 FLAT，通常不需要额外搜索参数
            )

            # 3. 处理结果
            # search_results 是一个列表，每个元素对应一个查询向量的结果
            # 我们只有一个查询向量，所以取 search_results[0]
            hits = search_results[0]
            # 每个 hit 是一个字典，类似 {'id': ..., 'distance': ..., 'entity': {'name': ...}}
            closest_universities = [hit['entity'][FIELD_NAME] for hit in hits]

            return closest_universities

        except Exception as e:
            print(f"使用 MilvusClient 搜索大学 '{query}' 时出错: {e}")
            return []

# --- 示例用法 ---
if __name__ == "__main__":
    normalizer = None
    try:
        print("正在初始化 UniversityNormalization (使用 MilvusClient)...")
        normalizer = UniversityNormalization()
        print("初始化完成。")

        # 测试查询
        queries = [
            "MIT",
            "Stanford University",
            "University of California, Berkeley",
            "清华大学",
            "北京大学",
            "Harvard University",
            "University of Oxford",
            "University of Cambridge",
            "University of Edinburgh",
            "University of Warwick",
            "University of London",
            "University of Manchester",
            "University of Bristol",
            "University of Sussex",
            "University of Birmingham",
            "University of Manchester",
            "University of Sheffield",
            "University of York",
            "PKU",
            "Cambridge"
        ]

        for q in queries:
            print(f"\n查询: '{q}'")
            results = normalizer.find_closest_university(q, top_k=9)
            if results:
                print("找到的最相似大学:")
                for i, uni in enumerate(results):
                    print(f"  {i+1}. {uni}")
            else:
                print("未能找到匹配的大学或发生错误。")

    except (FileNotFoundError, ConnectionError, ValueError, RuntimeError) as e:
         print(f"\n发生严重错误，程序终止: {e}")
    except Exception as e:
         print(f"\n发生未知错误: {e}")