"""
Copyright (c) 2025 by Zhenhui Yuan. All right reserved.
FilePath: /brain-mix/nlp/knowledges/extractive/content_compressor.py
Author: yuanzhenhui
Date: 2025-10-27 00:07:32
LastEditTime: 2025-11-25 09:37:11
"""

import sys
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

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(project_dir, 'utils'))

import const_util as CU
from yaml_util import YamlUtil
from logging_util import LoggingUtil
logger = LoggingUtil(os.path.basename(__file__).replace(".py", ""))

SENTENCE_SPLIT_RE = re.compile(r'([。！？；;!?]|\n+)')
NUMBER_RE = re.compile(r'\d+')
DATE_RE = re.compile(r'(\d{4}[-年/.]\d{1,2}[-月/.]\d{1,2}|\d{1,2}月\d{1,2}日)')
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
        Initialize the ContentCompressor object.

        Parameters:
            sentence_length_limit (int): The maximum length of a sentence in characters.
            prefilter_ratio (float): The ratio of the length of the summary to the length of the original text.
            prefilter_k (Optional[int]): The number of sentences to keep in the summary.
            min_sentences (int): The minimum number of sentences to keep in the summary.
            max_sentences (Optional[int]): The maximum number of sentences to keep in the summary.
            batch_size (int): The number of texts to process in parallel.
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
        """
        Load the model using the config file.
        """
        self._internal_model = SentenceTransformer(
            model_name_or_path=cnf.get_value('models.compress.path'),
            device=cnf.get_value('models.compress.device')
        )

        if not hasattr(jieba, '_initialized'):
            """
            Initialize Jieba if it has not been initialized.
            """
            jieba.lcut("初始化jieba")
            jieba._initialized = True

    def split_sentences(self, text: str) -> List[str]:
        """
        Split the text into sentences.

        The text is split into parts based on the sentence split regex.
        Then, each pair of parts is concatenated into a sentence, with
        punctuation attached to the end of the sentence.

        Parameters:
            text (str): The text to be split into sentences.

        Returns:
            List[str]: A list of sentences.

        """
        if not text:
            return []

        parts = SENTENCE_SPLIT_RE.split(text)
        sentences = []
        for i in range(0, len(parts), 2):
            piece = parts[i].strip()
            punct = parts[i + 1] if i + 1 < len(parts) else ''
            if piece:
                sentences.append(piece + (punct if punct else ''))
        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
        return sentences

    def tokenize_for_tfidf(self, text: str) -> List[str]:
        """
        Tokenize the text into words using Jieba.

        Parameters:
            text (str): The text to be tokenized.

        Returns:
            List[str]: A list of words.

        Note:
            Jieba is a Japanese text analysis library.
            It is used here to tokenize the text into words.
        """
        return jieba.lcut(text)

    def has_proper_noun(self, sentence: str) -> bool:
        """
        Check if a sentence contains a proper noun.

        A proper noun is a word that refers to a specific, unique entity.
        Examples of proper nouns include names of people, places, and organizations.

        Parameters:
            sentence (str): The sentence to be checked.

        Returns:
            bool: True if the sentence contains a proper noun, False otherwise.

        Notes:
            This function uses the Jieba library to tokenize the sentence and extract parts of speech.
            It checks if any of the parts of speech are tagged as proper nouns.
        """
        try:
            for _, flag in pseg.cut(sentence):
                if flag and flag.startswith(('nr', 'ns', 'nz')):
                    return True
        except Exception:
            return False
        return False

    def contains_number_or_date(self, sentence: str) -> bool:
        """
        Check if a sentence contains a number or a date.

        Parameters:
            sentence (str): The sentence to be checked.

        Returns:
            bool: True if the sentence contains a number or a date, False otherwise.

        Notes:
            This function uses regular expressions to check for numbers and dates.
            The regular expression for numbers matches any sequence of digits.
            The regular expression for dates matches dates in the format YYYY-MM-DD or MM-DD-DD.
        """
        return bool(NUMBER_RE.search(sentence) or DATE_RE.search(sentence))

    def _get_prefilter_k(self, n: int) -> int:
        """
        Get the prefilter k value based on the number of sentences.

        If prefilter_k is set, return it.
        Otherwise, return the maximum of min_sentences and the number of sentences multiplied by prefilter_ratio.

        Parameters:
            n (int): The number of sentences.

        Returns:
            int: The prefilter k value.
        """
        if self.prefilter_k:
            # If prefilter_k is set, return it
            return int(self.prefilter_k)
        # Otherwise, return the maximum of min_sentences and the number of sentences multiplied by prefilter_ratio
        return max(self.min_sentences, int(math.ceil(n * self.prefilter_ratio)))

    def tfidf_prefilter(self, sentences: List[str], full_text: str) -> List[int]:
        """
        Preprocess sentences using TF-IDF and cosine similarity.

        The TF-IDF prefiltering step is designed to filter out sentences
        that are not relevant to the full text. The relevance of a sentence
        is determined by its cosine similarity to the full text, measured using
        the TF-IDF vector representation of the sentence and the full text.

        Parameters:
            sentences (List[str]): The list of sentences to be filtered.
            full_text (str): The full text to which the sentences are compared.

        Returns:
            List[int]: The indices of the sentences that are relevant to the full text.

        Notes:
            The TF-IDF prefilter uses the cosine similarity metric to measure
            the relevance of a sentence to the full text. The cosine similarity
            is calculated as the dot product of the TF-IDF vectors of the sentence
            and the full text, normalized by the magnitudes of the vectors.

            If the TF-IDF prefiltering step fails, the function returns the indices
            of the sentences in the order of their lengths.
        """
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
            logger.warning(
                f"TF-IDF failed: {e}, fallback to length-based ranking.")
            sim = np.array([len(s) for s in sentences], dtype=float)

        k = min(self._get_prefilter_k(n), n)
        topk_idx = np.argsort(-sim)[:k]
        return sorted(topk_idx.tolist())

    def mmr_select(self,
                   embedding_matrix: np.ndarray,
                   relevance_scores: np.ndarray,
                   k: int,
                   lambda_param: float = 0.6) -> List[int]:
        """
        Select the top-k sentences from the embedding matrix and relevance scores
        using the Maximal Margin Relevance (MMR) algorithm.

        The MMR algorithm selects the top-k sentences that maximize the relevance
        and minimize the redundancy. The relevance is measured by the relevance scores,
        and the redundancy is measured by the cosine similarity between the selected sentences.

        Parameters:
            embedding_matrix (np.ndarray): The embedding matrix of the sentences.
            relevance_scores (np.ndarray): The relevance scores of the sentences.
            k (int): The number of sentences to select.
            lambda_param (float, optional): The parameter that controls the trade-off between relevance and redundancy. Defaults to 0.6.

        Returns:
            List[int]: The indices of the selected sentences.
        """
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
            max_sim_to_sel = np.array(
                [sim[c, selected].max() for c in cand_list])
            mmr_scores = lambda_param * \
                relevance_scores[cand_list] - \
                (1 - lambda_param) * max_sim_to_sel
            chosen = cand_list[int(np.argmax(mmr_scores))]
            selected.append(chosen)
            candidates.remove(chosen)
        return sorted(selected)

    def compress_text(self,
                      text: str,
                      compression_ratio: float = 0.25,
                      lambda_mmr: float = 0.6,
                      position_weight: float = 0.10,
                      named_entity_weight: float = 0.20,
                      number_weight: float = 0.35,
                      length_weight: float = 0.10) -> str:
        """
        Compress the text by selecting the top-k sentences that maximize the relevance and minimize the redundancy.

        Parameters:
            text (str): The text to be compressed.
            compression_ratio (float, optional): The compression ratio of the text. Defaults to 0.25.
            lambda_mmr (float, optional): The parameter that controls the trade-off between relevance and redundancy in the MMR algorithm. Defaults to 0.6.
            position_weight (float, optional): The weight of the position feature in the total score. Defaults to 0.10.
            named_entity_weight (float, optional): The weight of the named entity feature in the total score. Defaults to 0.20.
            number_weight (float, optional): The weight of the number feature in the total score. Defaults to 0.35.
            length_weight (float, optional): The weight of the length feature in the total score. Defaults to 0.10.

        Returns:
            str: The compressed text.
        """
        # Special case: if the text is empty, return an empty string
        if not text:
            return ""

        # Split the text into sentences
        sentences = self.split_sentences(text)

        # Calculate the number of sentences to keep based on the compression ratio
        n = len(sentences)
        if n == 0 or len(text) <= self.sentence_length_limit:
            return text

        compression_ratio = max(0.0, min(1.0, compression_ratio))
        est_k = max(self.min_sentences, int(math.ceil(n * compression_ratio)))
        if self.max_sentences:
            est_k = min(est_k, self.max_sentences)

        # Pre-filter the sentences using TF-IDF
        candidate_idxs = self.tfidf_prefilter(sentences, text)

        # If the number of sentences after pre-filtering is less than est_k, add some extra sentences
        if len(candidate_idxs) < est_k:
            remain = [i for i in range(n) if i not in candidate_idxs]
            extra = sorted(remain, key=lambda i: len(sentences[i]), reverse=True)[:est_k - len(candidate_idxs)]
            candidate_idxs = sorted(candidate_idxs + extra)

        # Get the candidate sentences
        candidate_sentences = [sentences[i] for i in candidate_idxs]

        # Build the embeddings of the candidate sentences
        embeddings = self.build_embeddings(candidate_sentences)

        # If the embeddings are empty, return the original text
        if embeddings.size == 0:
            return ' '.join(sentences[:est_k])

        # Calculate the document vector
        doc_vec = np.mean(embeddings, axis=0)

        # Calculate the semantic similarity scores
        sem_scores = cosine_similarity(embeddings, [doc_vec]).reshape(-1)
        sem_scores = (sem_scores - sem_scores.min()) / (sem_scores.max() - sem_scores.min() + 1e-12)

        # Calculate the position scores
        pos_scores = np.linspace(1, 0, len(candidate_idxs))

        # Calculate the named entity scores
        named_scores = np.array([1.0 if self.has_proper_noun(s) else 0.0 for s in candidate_sentences])

        # Calculate the number scores
        num_scores = np.array([1.0 if self.contains_number_or_date(s) else 0.0 for s in candidate_sentences])

        # Calculate the length scores
        len_scores = np.array([len(jieba.lcut(s)) for s in candidate_sentences], dtype=float)
        len_scores = (len_scores - len_scores.min()) / (len_scores.max() - len_scores.min() + 1e-12)

        # Calculate the total scores
        remain_w = max(0.0, 1.0 - (position_weight + named_entity_weight + number_weight + length_weight))
        total_score = (
            sem_scores * remain_w +
            pos_scores * position_weight +
            named_scores * named_entity_weight +
            num_scores * number_weight +
            len_scores * length_weight
        )
        total_score = np.nan_to_num(total_score, nan=0.0)

        # Select the top-k sentences using the MMR algorithm
        selected_local = self.mmr_select(embeddings, total_score, k=est_k, lambda_param=lambda_mmr)

        # Select the top-k sentences based on the total scores
        selected_global = sorted([candidate_idxs[i] for i in selected_local])

        # Get the summary
        summary = ' '.join(sentences[i].strip() for i in selected_global)

        # If the summary is too long, truncate it
        if len(summary) > self.sentence_length_limit:
            summary = summary[:self.sentence_length_limit]

        # Clean and format the summary
        return self.clean_and_format(summary)

    def clean_and_format(self, compress_text: str) -> str:
        """
        Clean and format the compressed text.

        Parameters:
            compress_text (str): The compressed text to be cleaned and formatted.

        Returns:
            str: The cleaned and formatted text.
        """
        # Replace multiple "，" with a single "，"
        s = re.sub('，，+', '，', compress_text)
        # Replace multiple whitespace characters with a single space
        s = re.sub(r'\s+', ' ', s).strip()
        # Replace "。 " with "。\n"
        s = s.replace('。 ', '。\n')
        # Return the cleaned and formatted text
        return s

    def _split_and_compress(self, split_keyword: Optional[str], docs: List[str], results: List[str], **compress_kwargs) -> None:
        """
        Split the documents based on the split keyword and compress them.

        Parameters:
            split_keyword (Optional[str]): The split keyword to split the documents.
            docs (List[str]): The list of documents to be compressed.
            results (List[str]): The list to store the compressed documents.
            **compress_kwargs: The keyword arguments to be passed to the compress_text method.
        """
        for doc in docs:
            if split_keyword:
                # Find the index of the split keyword
                idx = doc.find(split_keyword)
                if idx != -1:
                    # Split the document into two parts
                    before = doc[:idx + len(split_keyword)]
                    after = doc[idx + len(split_keyword):]
                    # Compress the second part
                    compressed = self.compress_text(after, **compress_kwargs)
                    # Append the compressed document to the results
                    results.append(before + compressed)
                else:
                    # Compress the entire document
                    results.append(self.compress_text(doc, **compress_kwargs))
            else:
                # Compress the entire document
                results.append(self.compress_text(doc, **compress_kwargs))

    def build_embeddings(self, sentences: List[str]) -> np.ndarray:
        """
        Build embeddings for a list of sentences.

        Parameters:
            sentences (List[str]): The list of sentences to build embeddings for.

        Returns:
            np.ndarray: The embeddings of the sentences.
        """
        if not sentences:
            # If the list of sentences is empty, return an empty numpy array
            return np.empty((0, 0))

        # Use the internal model to encode the sentences
        emb = self._internal_model.encode(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=False
        )
        # Return the embeddings as a numpy array
        return np.asarray(emb)

    def compress_documents(self, docs: List[str], split_keyword: Optional[str] = None, **compress_kwargs) -> List[str]:
        """
        Compress a list of documents in multi-thread mode.

        Parameters:
            docs (List[str]): The list of documents to be compressed.
            split_keyword (Optional[str]): The split keyword to split the documents.
            **compress_kwargs: The keyword arguments to be passed to the compress_text method.

        Returns:
            List[str]: The list of compressed documents.
        """
        logger.info(f"Compressing {len(docs)} docs in multi-thread mode...")
        results = []
        with ThreadPoolExecutor(max_workers=CORE_COUNT) as executor:
            futures = [
                executor.submit(self._split_and_compress, split_keyword, [doc], results, **compress_kwargs)
                for doc in docs
            ]
            for _ in as_completed(futures):
                pass
        return results


if __name__ == "__main__":
    content_str = "鸡血藤 行情疲软鸡血藤近期行情疲软，商家积极销售货源，市场少商关注，货源走销不快，进口货多要价在9-10元，国产货6元左右。山茱萸 行情疲软山茱萸市场行情疲软，现药厂货售价53-54元，饮片货63-64元，筒子皮70元，市场需求一般。苦杏仁 走销稍快苦杏仁市场货源走销稍快，正值销售旺季，价格平稳，内蒙货25元左右，山西货23-24元。金银花 购销一般 市场金银花购销一般，货源以实际需求购销为主，行情暂稳，现市场河南统货要价120-125元/千克，山东统货要价110元左右，河北青花货要价130元左右。山楂 走动尚可市场山楂新货上市购销有商购货走销，行情小幅调整，目前市场山楂价格机器统片8-9元左右，手工统片12-13元左右，山楂无籽货20-21元左右，中心选货30-35元左右。黑枸杞 购销一般市场黑枸杞新货来货增多，货源流通需求暂时一般，行情暂稳，现市场黑枸杞小米统货售价20-25元左右，中等货售价40-50元左右，选货售价60-90元不等/千克。蛇床子 行情暂稳市场蛇床子购销尚可，市场货源批量购销，行情小幅波动，目前市场蛇床子统货售价26-27元/千克，选货售价31-32元左右。甘草 货源充足 市场甘草货源增多，需求购销走动一般，，行情小幅震荡调整，现在市场新疆甘草统片20-25元左右/千克，甘肃统片30-35元上下，药厂货售价15-25元不等。白花蛇舌草 走动一般 市场白花蛇舌草来货增多货源充足，行情调整，目前市场白花蛇舌草家种统货价8元左右/千克，统片售价11-12元上下。公丁香 行情平稳市场公丁香购销不快，货源暂时充足需求一般,货源小批量购销为主，行情暂稳，目前市场公丁香大丁货价格在58-60元左右。黄连 行情震荡市场黄连购销一般，近期随着新货上市购销，市面货源走动一般，来货量增多，货源需求暂时不佳小批量走动，行情震荡，目前市场鸡爪连售价330-350元/千克，单枝售价350-360元左右。康美中药城供求信息平台药材买卖，就来康美中药城Mini Program"
    compressor = ContentCompressor()
    summary = compressor.compress_text(text=content_str)
    logger.info("---- Summary ----\n")
    logger.info(summary)

    # docs_list = []
    # summary = compressor.compress_documents(
    #     docs=docs_list,
    #     split_keyword="【正文】"
    # )
    # logger.info(f"---- Summary ----{len(summary)}\n")
