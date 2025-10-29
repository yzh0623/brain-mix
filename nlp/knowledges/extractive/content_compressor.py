"""
Copyright (c) 2025 by Zhenhui Yuan. All right reserved.
FilePath: /brain-mix/nlp/knowledges/extractive/content_compressor.py
Author: yuanzhenhui
Date: 2025-10-27 00:07:32
LastEditTime: 2025-10-29 21:15:46
"""

import os
import re
import math
import jieba
import jieba.posseg as pseg
import numpy as np
import torch.multiprocessing as mp
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from concurrent.futures import ThreadPoolExecutor, as_completed

import warnings
warnings.filterwarnings("ignore")

import sys
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(project_dir, 'utils'))

from yaml_util import YamlUtil
import const_util as CU
from logging_util import LoggingUtil
logger = LoggingUtil(os.path.basename(__file__).replace(".py", ""))

# 句子分割正则（中文）
SENTENCE_SPLIT_RE = re.compile(r'([。！？；;!?]|\n+)')
# 数字日期正则
NUMBER_RE = re.compile(r'\d+')
DATE_RE = re.compile(r'(\d{4}[-年/.]\d{1,2}[-月/.]\d{1,2}|\d{1,2}月\d{1,2}日)')
# 核心数
CORE_COUNT = os.cpu_count()

class ContentCompressor:
    def __init__(self,
                 sentence_length_limit: int = 400,
                 prefilter_ratio: float = 0.7,
                 prefilter_k: Optional[int] = None,
                 min_sentences: int = 1,
                 max_sentences: Optional[int] = 15,
                 batch_size: int = 32):
        """
            初始化内容压缩器
            :param sentence_length_limit: 句子长度限制
            :param prefilter_ratio: 预过滤比例
            :param prefilter_k: 预过滤数量
            :param min_sentences: 最小句子数量
            :param max_sentences: 最大句子数量
            :param batch_size: 批量处理大小
        """

        mp.set_start_method('spawn', force=True)

        self.sentence_length_limit = sentence_length_limit
        self.prefilter_ratio = prefilter_ratio
        self.prefilter_k = prefilter_k
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
        self.batch_size = batch_size
        
        self.model_cnf = os.path.join(project_dir, 'resources', 'config', CU.ACTIVATE, 'nlp_cnf.yml')
        cnf = YamlUtil(self.model_cnf)
        self._internal_model = SentenceTransformer(
            model_name_or_path=cnf.get_value('models.compress.path'),
            device=cnf.get_value('models.compress.device')
        )

        # 初始化 jieba（只在主进程初始化一次）
        if not hasattr(jieba, '_initialized'):
            jieba.lcut("初始化jieba")
            jieba._initialized = True

    def split_sentences(self, text: str) -> List[str]:
        """
            将文本分割成句子

            使用 SENTENCE_SPLIT_RE 正则表达式将文本分割成句子

            :param text: 需要分割的文本
            :return sentences: 分割后的句子列表
        """
        if not text:
            return []

        # 使用 SENTENCE_SPLIT_RE 正则表达式将文本分割成句子
        parts = SENTENCE_SPLIT_RE.split(text)
        sentences = []
        for i in range(0, len(parts), 2):
            piece = parts[i].strip()
            punct = parts[i + 1] if i + 1 < len(parts) else ''
            if piece:
                sentences.append(piece + (punct if punct else ''))

        # 过除空白符的句子
        return [s.strip() for s in sentences if len(s.strip()) > 0]

    def tokenize_for_tfidf(self, text: str) -> List[str]:
        """使用 jieba 分词器对文本进行分词"""
        return jieba.lcut(text)

    def has_proper_noun(self, sentence: str) -> bool:
        """检查句子中是否包含专有名词"""
        try:
            for _, flag in pseg.cut(sentence):
                if flag and flag.startswith(('nr', 'ns', 'nz')):
                    return True
        except Exception:
            return False
        return False

    def contains_number_or_date(self, sentence: str) -> bool:
        """检查句子中是否包含数字或日期"""
        return bool(NUMBER_RE.search(sentence) or DATE_RE.search(sentence))

    def _get_prefilter_k(self, n: int) -> int:
        """
        计算预过滤的句子数量

        如果 self.prefilter_k 不为空，则直接返回该值
        否则，计算预过滤的句子数量，取自 min_sentences 和 n * self.prefilter_ratio 中的最大值

        Args:
            n (int): 句子的数量

        Returns:
            int: 预过滤的句子数量
        """
        if self.prefilter_k:
            return int(self.prefilter_k)
        return max(self.min_sentences, int(math.ceil(n * self.prefilter_ratio)))

    def tfidf_prefilter(self, sentences: List[str], full_text: str) -> List[int]:
        """
        使用 TF-IDF 算法对句子进行预过滤

        Args:
            sentences (List[str]): 句子的列表
            full_text (str): 原始文本

        Returns:
            List[int]: 排序后的句子索引列表
        """
        n = len(sentences)
        if n <= 0:
            return []

        vectorizer = TfidfVectorizer(
            # 使用 jieba 分词器对文本进行分词
            tokenizer=self.tokenize_for_tfidf,
            # 不对分词进行lowercase处理
            token_pattern=None,
            lowercase=False
        )
        try:
            # 对句子进行TF-IDF变换
            tfidf_matrix = vectorizer.fit_transform(sentences)
            # 对原始文本进行TF-IDF变换
            doc_vec = vectorizer.transform([full_text])
            # 计算每个句子与原始文本的cosine相似度
            sim = cosine_similarity(tfidf_matrix, doc_vec).reshape(-1)
        except Exception as e:
            # 如果TF-IDF失败，使用句子长度进行排序
            logger.warning(f"TF-IDF failed: {e}, fallback to length-based ranking.")
            sim = np.array([len(s) for s in sentences], dtype=float)

        k = min(self._get_prefilter_k(n), n)
        # 选择前k个句子
        topk_idx = np.argsort(-sim)[:k]
        return sorted(topk_idx.tolist())

    def mmr_select(self,
                   embedding_matrix: np.ndarray,
                   relevance_scores: np.ndarray,
                   k: int,
                   lambda_param: float = 0.6) -> List[int]:
        """
        使用 MMR 算法选择句子

        Args:
            embedding_matrix (np.ndarray): 句子的嵌入向量矩阵
            relevance_scores (np.ndarray): 句子的相关性分数组
            k (int): 选择的句子数量
            lambda_param (float, optional): MMR 算法的参数，default to 0.6

        Returns:
            List[int]: 选择的句子索引列表
        """
        n = embedding_matrix.shape[0]
        if n == 0 or k <= 0:
            return []
        if k >= n:
            return list(range(n))

        sim = cosine_similarity(embedding_matrix)
        # 将对角元素设置为 0，以避免选择同一个句子
        np.fill_diagonal(sim, 0.0)
        selected = [int(np.argmax(relevance_scores))]
        candidates = set(range(n)) - set(selected)

        while len(selected) < k and candidates:
            cand_list = list(candidates)
            # 计算每个候选句子到已选择句子集合的最大相似度
            max_sim_to_sel = np.array([sim[c, selected].max() for c in cand_list])
            # 计算每个候选句子的 MMR 评分
            mmr_scores = lambda_param * relevance_scores[cand_list] - (1 - lambda_param) * max_sim_to_sel
            chosen = cand_list[int(np.argmax(mmr_scores))]
            selected.append(chosen)
            candidates.remove(chosen)
        return sorted(selected)

    def compress_text(self,
                      text: str,
                      compression_ratio: float = 0.4638,
                      lambda_mmr: float = 0.7374,
                      position_weight: float = 0.2642,
                      named_entity_weight: float = 0.2048,
                      number_weight: float = 0.351,
                      length_weight: float = 0.1192) -> str:
        """
        对文本进行摘要

        Args:
            text (str): 输入文本
            compression_ratio (float, optional): 摘要保留的句子数量占总句子数量的比例，default to 0.4638
            lambda_mmr (float, optional): MMR 算法的参数，default to 0.7374
            position_weight (float, optional): 位置分数的权重，default to 0.2642
            named_entity_weight (float, optional): 命名实体分数的权重，default to 0.2048
            number_weight (float, optional): 数字或日期分数的权重，default to 0.351
            length_weight (float, optional): 句子长度分数的权重，default to 0.1192

        Returns:
            str: 摘要生成的摘要
        """
        # 如果输入文本为空，直接返回空字符串
        if not text:
            return ""

        # 将文本分割成句子
        sentences = self.split_sentences(text)
        n = len(sentences)
        # 如果句子数量为0或文本长度小于限制，直接返回原文本
        if n == 0 or len(text) <= self.sentence_length_limit:
            return text

        # 计算需要保留的句子数量
        compression_ratio = max(0.0, min(1.0, compression_ratio))
        est_k = max(self.min_sentences, int(math.ceil(n * compression_ratio)))
        # 如果设置了最大句子数量，确保不超过最大值
        if self.max_sentences:
            est_k = min(est_k, self.max_sentences)

        # 使用TF-IDF预过滤候选句子
        candidate_idxs = self.tfidf_prefilter(sentences, text)
        # 如果候选句子数量不足，补充最长句子
        if len(candidate_idxs) < est_k:
            remain = [i for i in range(n) if i not in candidate_idxs]
            extra = sorted(remain, key=lambda i: len(sentences[i]), reverse=True)[:est_k - len(candidate_idxs)]
            candidate_idxs = sorted(candidate_idxs + extra)

        # 获取候选句子
        candidate_sentences = [sentences[i] for i in candidate_idxs]
        # 构建句子嵌入
        embeddings = self.build_embeddings(candidate_sentences)
        # 如果嵌入为空，返回前est_k个句子
        if embeddings.size == 0:
            return ' '.join(sentences[:est_k])

        # 计算文档向量
        doc_vec = np.mean(embeddings, axis=0)
        # 计算语义相似度分数
        sem_scores = cosine_similarity(embeddings, [doc_vec]).reshape(-1)
        sem_scores = (sem_scores - sem_scores.min()) / (sem_scores.max() - sem_scores.min() + 1e-12)

        # 计算位置分数
        pos_scores = np.linspace(1, 0, len(candidate_idxs))
        # 计算命名实体分数
        named_scores = np.array([1.0 if self.has_proper_noun(s) else 0.0 for s in candidate_sentences])
        # 计算数字或日期分数
        num_scores = np.array([1.0 if self.contains_number_or_date(s) else 0.0 for s in candidate_sentences])
        # 计算句子长度分数
        len_scores = np.array([len(jieba.lcut(s)) for s in candidate_sentences], dtype=float)
        len_scores = (len_scores - len_scores.min()) / (len_scores.max() - len_scores.min() + 1e-12)

        # 计算剩余权重
        remain_w = max(0.0, 1.0 - (position_weight + named_entity_weight + number_weight + length_weight))
        # 计算总分数
        total_score = (
            sem_scores * remain_w +
            pos_scores * position_weight +
            named_scores * named_entity_weight +
            num_scores * number_weight +
            len_scores * length_weight
        )
        total_score = np.nan_to_num(total_score, nan=0.0)

        # 使用MMR算法选择句子
        selected_local = self.mmr_select(embeddings, total_score, k=est_k, lambda_param=lambda_mmr)
        # 获取全局选择的句子索引
        selected_global = sorted([candidate_idxs[i] for i in selected_local])
        # 生成摘要
        summary = ' '.join(sentences[i].strip() for i in selected_global)

        # 如果摘要长度超过限制，截断
        if len(summary) > self.sentence_length_limit:
            summary = summary[:self.sentence_length_limit]
        return summary

    def _split_and_compress(self, split_keyword, docs: List[str], results: List[str], **compress_kwargs):
        """
        将文档列表压缩成摘要列表
        :param split_keyword: 分割关键字，用于分割文档
        :param docs: 需要压缩的文档列表
        :param results: 压缩后的摘要列表
        :param compress_kwargs: 压缩参数
        :return: None
        """
        for doc in docs:
            if split_keyword:
                # 查找 split_keyword 在字符串中的位置
                idx = doc.find(split_keyword)
                if idx != -1:
                    # 获取关键字后内容
                    before = doc[:idx + len(split_keyword)]
                    after = doc[idx + len(split_keyword):]
                    compressed = self.compress_text(after, **compress_kwargs)
                    results.append(before + compressed)
                else:
                    # 未找到关键字，直接压缩全文
                    results.append(self.compress_text(doc, **compress_kwargs))
            else:
                # 未指定关键字，直接压缩全文
                results.append(self.compress_text(doc, **compress_kwargs))

    def build_embeddings(self, sentences: List[str]) -> np.ndarray:
        
        """
        将句子列表编码成嵌入向量矩阵

        使用 SentenceTransformer 对句子列表进行编码

        :param sentences: 需要编码的句子列表
        :return embeddings: 编码后的嵌入向量矩阵
        """
        if not sentences:
            return np.empty((0, 0))

        # 使用 SentenceTransformer 内部多线程
        # batch_size 可以根据需要调整，device 可以设置为 cpu
        emb = self._internal_model.encode(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=False
        )
        return np.asarray(emb)
    
    def compress_documents(self, docs: List[str], split_keyword: Optional[str] = None, **compress_kwargs) -> List[str]:
        """
        将文档列表压缩成摘要列表

        Args:
            docs (List[str]): 需要压缩的文档列表
            split_keyword (Optional[str]): 分割关键字，用于分割文档
            compress_kwargs: 压缩参数

        Returns:
            List[str]: 压缩后的摘要列表
        """
        logger.info(f"Compressing {len(docs)} docs in multi-thread mode...")
        results = []
        with ThreadPoolExecutor(max_workers=CORE_COUNT) as executor:
            futures = [
                # 将文档列表split into多个子列表，然后对每个子列表进行压缩
                executor.submit(self._split_and_compress, split_keyword, [doc], results, **compress_kwargs)
                for doc in docs
            ]
            # 等待所有线程完成
            for _ in as_completed(futures):
                pass
        return results
    
if __name__ == "__main__":
    docs_list = []
    compressor = ContentCompressor()
    summary = compressor.compress_documents(
        docs=docs_list,
        split_keyword="【正文】"
    )
    logger.info(f"---- Summary ----{len(summary)}\n")