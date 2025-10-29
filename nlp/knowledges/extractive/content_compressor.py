"""
Copyright (c) 2025 by Zhenhui Yuan. All right reserved.
FilePath: /brain-mix/nlp/knowledges/extractive/content_compressor.py
Author: yuanzhenhui
Date: 2025-10-27 00:07:32
LastEditTime: 2025-10-29 00:05:48
"""

import os
import re
import math
import jieba
import jieba.posseg as pseg
import numpy as np
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

import sys
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(project_dir, 'utils'))

import const_util as CU
from yaml_util import YamlUtil
from logging_util import LoggingUtil
logger = LoggingUtil(os.path.basename(__file__).replace(".py", ""))

# 句子分割正则（中文）
SENTENCE_SPLIT_RE = re.compile(r'([。！？；;!?]|\n+)')
# 数字日期正则
NUMBER_RE = re.compile(r'\d+')
DATE_RE = re.compile(r'(\d{4}[-年/.]\d{1,2}[-月/.]\d{1,2}|\d{1,2}月\d{1,2}日)')


class ContentCompressor:
    def __init__(self,
                 sentence_length_limit: int = 800,
                 prefilter_ratio: float = 0.6,
                 prefilter_k: Optional[int] = None,
                 min_sentences: int = 1,
                 max_sentences: Optional[int] = 20,
                 batch_size: int = 32,
                 model: Optional[SentenceTransformer] = None,
                 model_name: Optional[str] = None,
                 device: Optional[str] = None):
        """
        初始化内容压缩器
        :param sentence_length_limit: 句子长度限制，默认为 800
        :param prefilter_ratio: 预过滤比例，默认为 0.6
        :param prefilter_k: 预过滤数量，可选
        :param min_sentences: 最小句子数量，默认为 1
        :param max_sentences: 最大句子数量，可选
        :param batch_size: 批量处理大小，默认为 32
        :param model: 外部传入的已加载模型（优先使用）
        :param model_name: 模型名称（当 model 为 None 时使用）
        :param device: 设备（当 model 为 None 时使用）
        """
        self.sentence_length_limit = sentence_length_limit
        self.prefilter_ratio = prefilter_ratio
        self.prefilter_k = prefilter_k
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
        self.batch_size = batch_size
        
        # 优先使用外部传入的模型
        self._external_model = model
        
        # 如果没有外部模型，保存模型配置以便延迟加载
        if model is None:
            if model_name and device:
                self.model_name = model_name
                self.device = device
            else:
                self.model_cnf = os.path.join(project_dir, 'resources', 'config', CU.ACTIVATE, 'nlp_cnf.yml')
                cnf = YamlUtil(self.model_cnf)
                self.model_name = cnf.get_value('models.compress.path')
                self.device = cnf.get_value('models.compress.device')
        
        # 内部模型实例（延迟加载）
        self._internal_model = None

        # 初始化 jieba（只在主进程初始化一次）
        if not hasattr(jieba, '_initialized'):
            jieba.lcut("初始化jieba")
            jieba._initialized = True

    def _get_model(self) -> SentenceTransformer:
        """
        获取模型实例（懒加载 + 优先使用外部模型）
        """
        # 1. 优先使用外部传入的模型
        if self._external_model is not None:
            return self._external_model
        
        # 2. 如果没有内部模型，则创建
        if self._internal_model is None:
            self._internal_model = SentenceTransformer(
                model_name_or_path=self.model_name, 
                device=self.device
            )
        
        return self._internal_model

    def split_sentences(self, text: str) -> List[str]:
        """将文本分割成句子"""
        if not text:
            return []
        parts = SENTENCE_SPLIT_RE.split(text)
        sentences = []
        for i in range(0, len(parts), 2):
            piece = parts[i].strip()
            punct = parts[i + 1] if i + 1 < len(parts) else ''
            if piece:
                sentences.append(piece + (punct if punct else ''))
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
        """计算预过滤的句子数量"""
        if self.prefilter_k:
            return int(self.prefilter_k)
        return max(self.min_sentences, int(math.ceil(n * self.prefilter_ratio)))

    def tfidf_prefilter(self, sentences: List[str], full_text: str) -> List[int]:
        """使用 TF-IDF 算法对句子进行预过滤"""
        n = len(sentences)
        if n <= 0:
            return []
        
        vectorizer = TfidfVectorizer(
            tokenizer=self.tokenize_for_tfidf,
            token_pattern=None,
            lowercase=False
        )
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            doc_vec = vectorizer.transform([full_text])
            sim = cosine_similarity(tfidf_matrix, doc_vec).reshape(-1)
        except Exception as e:
            logger.warning(f"TF-IDF failed: {e}, fallback to length-based ranking.")
            sim = np.array([len(s) for s in sentences], dtype=float)
        
        k = min(self._get_prefilter_k(n), n)
        topk_idx = np.argsort(-sim)[:k]
        return sorted(topk_idx.tolist())

    def build_embeddings(self, sentences: List[str]) -> np.ndarray:
        """使用 SentenceTransformer 模型生成句子的嵌入向量"""
        if not sentences:
            return np.empty((0, 0))
        
        model = self._get_model()
        emb = model.encode(sentences, batch_size=self.batch_size, show_progress_bar=False)
        return np.asarray(emb)

    def mmr_select(self,
                   embedding_matrix: np.ndarray,
                   relevance_scores: np.ndarray,
                   k: int,
                   lambda_param: float = 0.6) -> List[int]:
        """使用 MMR 算法选择句子"""
        n = embedding_matrix.shape[0]
        if n == 0 or k <= 0:
            return []
        if k >= n:
            return list(range(n))

        sim = cosine_similarity(embedding_matrix)
        np.fill_diagonal(sim, 0.0)
        selected = [int(np.argmax(relevance_scores))]
        candidates = set(range(n)) - set(selected)

        while len(selected) < k and candidates:
            cand_list = list(candidates)
            max_sim_to_sel = np.array([sim[c, selected].max() for c in cand_list])
            mmr_scores = lambda_param * relevance_scores[cand_list] - (1 - lambda_param) * max_sim_to_sel
            chosen = cand_list[int(np.argmax(mmr_scores))]
            selected.append(chosen)
            candidates.remove(chosen)
        return sorted(selected)

    def compress_text(self,
                      text: str,
                      compression_ratio: float = 0.3,
                      lambda_mmr: float = 0.6,
                      position_weight: float = 0.15,
                      named_entity_weight: float = 0.25,
                      number_weight: float = 0.20,
                      length_weight: float = 0.05) -> str:
        """压缩文本，返回摘要"""
        
        if not text:
            return ""
        
        sentences = self.split_sentences(text)
        n = len(sentences)
        if n == 0 or len(text) <= self.sentence_length_limit:
            return text

        est_k = max(self.min_sentences, int(math.ceil(n * compression_ratio)))
        if self.max_sentences:
            est_k = min(est_k, self.max_sentences)
            
        candidate_idxs = self.tfidf_prefilter(sentences, text)
        if len(candidate_idxs) < est_k:
            remain = [i for i in range(n) if i not in candidate_idxs]
            extra = sorted(remain, key=lambda i: len(sentences[i]), reverse=True)[:est_k - len(candidate_idxs)]
            candidate_idxs = sorted(candidate_idxs + extra)

        candidate_sentences = [sentences[i] for i in candidate_idxs]
        embeddings = self.build_embeddings(candidate_sentences)
        if embeddings.size == 0:
            return ' '.join(sentences[:est_k])

        doc_vec = np.mean(embeddings, axis=0)
        sem_scores = cosine_similarity(embeddings, [doc_vec]).reshape(-1)
        sem_scores = (sem_scores - sem_scores.min()) / (sem_scores.max() - sem_scores.min() + 1e-12)

        pos_scores = np.linspace(1, 0, len(candidate_idxs))
        named_scores = np.array([1.0 if self.has_proper_noun(s) else 0.0 for s in candidate_sentences])
        num_scores = np.array([1.0 if self.contains_number_or_date(s) else 0.0 for s in candidate_sentences])
        len_scores = np.array([len(jieba.lcut(s)) for s in candidate_sentences], dtype=float)
        len_scores = (len_scores - len_scores.min()) / (len_scores.max() - len_scores.min() + 1e-12)

        remain_w = max(0.0, 1.0 - (position_weight + named_entity_weight + number_weight + length_weight))
        total_score = (
            sem_scores * remain_w +
            pos_scores * position_weight +
            named_scores * named_entity_weight +
            num_scores * number_weight +
            len_scores * length_weight
        )
        total_score = np.nan_to_num(total_score, nan=0.0)

        selected_local = self.mmr_select(embeddings, total_score, k=est_k, lambda_param=lambda_mmr)
        selected_global = sorted([candidate_idxs[i] for i in selected_local])
        summary = ' '.join(sentences[i].strip() for i in selected_global)
        
        if len(summary) > self.sentence_length_limit:
            summary = summary[:self.sentence_length_limit]
        return summary

    def compress_documents(self, docs: List[str], **compress_kwargs) -> List[str]:
        """批量压缩文档（单进程模式）"""
        logger.info(f"Compressing {len(docs)} docs in single-process mode...")
        results = []
        
        for doc in docs:
            try:
                results.append(self.compress_text(doc, **compress_kwargs))
            except Exception as e:
                logger.error(f"Compression failed: {e}")
                results.append("")
        
        return results


# ---------------- 示例运行 ----------------
if __name__ == "__main__":
    sample = """
    鸡血藤 行情疲软鸡血藤近期行情疲软，商家积极销售货源，市场少商关注，货源走销不快，进口货多要价在9-10元，国产货6元左右。 金银花 购销一般 市场金银花购销一般，货源以实际需求购销为主，行情暂稳，现市场河南统货要价120-125元/千克，山东统货要价110元左右，河北青花货要价130元左右。
    """

    compressor = ContentCompressor(
        sentence_length_limit=400,
        prefilter_ratio=0.6,
        max_sentences=4
    )

    summary = compressor.compress_text(
        sample,
        compression_ratio=0.4,
        lambda_mmr=0.6,
        position_weight=0.15,
        named_entity_weight=0.3,
        number_weight=0.25,
        length_weight=0.05
    )

    logger.info("---- Summary ----")
    logger.info(summary)