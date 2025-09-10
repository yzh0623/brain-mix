"""
Copyright (c) 2025 by Zhenhui Yuan All right reserved.
FilePath: /brain-mix/utils/persistence/clean_util.py
Author: Zhenhui Yuan
Date: 2025-09-05 09:56:19
LastEditTime: 2025-09-10 15:48:06
"""

from sklearn.cluster import DBSCAN
import numpy as np
import elasticsearch
from tqdm import tqdm
from collections import defaultdict
import os
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logging_util import LoggingUtil
logger = LoggingUtil(os.path.basename(__file__).replace(".py", ""))

class CleanUtil:

    def find_and_remove_duplicate_vectors(self,
                                          es,
                                          index_name,
                                          vector_field,
                                          date_field=None,
                                          text_field=None,
                                          similarity_threshold=0.9999,
                                          exclude_words=[],
                                          batch_size=1000,
                                          min_samples=1,
                                          callback_function=None):
        """
        Find and remove duplicate vectors from an Elasticsearch index using Scroll API.

        :param es: Elasticsearch client
        :param index_name: Elasticsearch index name
        :param vector_field: name of the vector field in the index
        :param date_field: name of the date field in the index
        :param text_field: name of the text field in the index
        :param similarity_threshold: minimum cosine similarity to consider two vectors as duplicates
        :param exclude_words: list of words to exclude from the search
        :param batch_size: number of documents to retrieve in a single scroll request
        :param min_samples: minimum number of samples required to form a dense region for DBSCAN clustering
        :param callback_function: a callable that takes a list of documents to delete as an argument
        """
        logger.info("Starting duplicate vector removal process...")

        total_vectors = 0
        scroll_timeout = "5m"
        # Build the query condition
        scroll_query = self._search_with_nopageing_custom(
            vector_field,
            date_field,
            text_field,
            batch_size,
            exclude_words
        )

        # Use scroll API to retrieve all documents in batches
        resp = es.search(
            index=index_name,
            body=scroll_query,
            scroll=scroll_timeout
        )
        scroll_id = resp.get('_scroll_id')
        hits = resp['hits']['hits'] if 'hits' in resp and 'hits' in resp['hits'] else []

        try:
            while hits:
                vectors, doc_ids, metadata, to_delete = [], [], [], []

                for hit in hits:
                    # Extract the vector and metadata from the hit
                    if vector_field in hit['_source']:
                        vectors.append(hit['_source'][vector_field])
                        doc_ids.append(hit['_id'])
                        metadata.append({
                            "date": hit['_source'].get(date_field) if date_field else None,
                            "text_length": len(hit['_source'].get(text_field, "")) if text_field else 0
                        })

                total_vectors += len(hits)
                logger.info(f"Processed {total_vectors} vectors so far...")

                if vectors:
                    # Convert the vectors to numpy array
                    vectors_np = np.array(vectors)
                    logger.info(f"Clustering {len(vectors_np)} vectors in the current batch...")
                    eps = 1 - similarity_threshold

                    # Perform DBSCAN clustering
                    clustering = DBSCAN(
                        eps=eps,
                        min_samples=min_samples,
                        metric='cosine'
                    ).fit(vectors_np)

                    labels = clustering.labels_
                    clusters = defaultdict(list)
                    for idx, label in enumerate(labels):
                        if label != -1:
                            clusters[label].append(
                                (doc_ids[idx], metadata[idx]))
                    # Find and collect all documents that need to be deleted
                    batch_to_delete = self._find_and_collect_all_need_delete(clusters, date_field)
                    to_delete.extend(batch_to_delete)
                    if to_delete:
                        # Delete the documents in batches
                        self._delete_with_batch(es, to_delete, batch_size, index_name)
                        logger.info("Deletion completed")
                        # Call the callback function if provided
                        if callback_function:
                            try:
                                callback_function(to_delete)
                            except Exception as e:
                                logger.error(f"Error executing callback function: {e}")
                    else:
                        logger.info("No duplicate documents to delete")

                # Get the next batch of documents
                try:
                    resp = es.scroll(scroll_id=scroll_id,scroll=scroll_timeout)
                    scroll_id = resp.get('_scroll_id', None)
                    hits = resp['hits']['hits'] if resp and 'hits' in resp and 'hits' in resp['hits'] else [
                    ]
                except elasticsearch.NotFoundError as e:
                    logger.error(f"Scroll context expired: {e}. Reinitializing scroll...")
                    resp = es.search(index=index_name,body=scroll_query, scroll=scroll_timeout)
                    scroll_id = resp.get('_scroll_id')
                    hits = resp['hits']['hits'] if 'hits' in resp and 'hits' in resp['hits'] else [
                    ]
                except Exception as e:
                    logger.error(f"Scroll failed: {e}")
                    break

        finally:
            if scroll_id:
                try:
                    # Clear the scroll context
                    es.clear_scroll(scroll_id=scroll_id)
                    logger.info("Scroll context cleared.")
                except Exception as e:
                    logger.warning(f"Failed to clear scroll context: {e}")

    def _search_with_nopageing_custom(self, vector_field, date_field, text_field, batch_size, exclude_words):
        """
        Search for all documents in the index with the given vector field, date field, and text field,
        excluding documents with the given words, and ensure the vector field exists.

        Use the scroll API to retrieve all documents in batches.
        """
        # Use scroll API to retrieve all documents in batches
        scroll_query = {
            "_source": [vector_field],
            "size": batch_size
        }

        if date_field:
            scroll_query["_source"].append(date_field)
        if text_field:
            scroll_query["_source"].append(text_field)

        # Build the query condition
        query_conditions = {
            "exists": {
                "field": vector_field
            }
        }

        if exclude_words:
            # Add exclude conditions for specific words
            must_not_conditions = []
            for word in exclude_words:
                must_not_conditions.append({"match": {text_field: word}})

            query_conditions = {
                "bool": {
                    "must": [query_conditions],
                    "must_not": must_not_conditions
                }
            }

        scroll_query["query"] = query_conditions
        return scroll_query

    def _find_and_collect_all_need_delete(self, clusters, date_field):
        """
        Find all documents in the given clusters that need to be deleted.

        Each cluster is a list of documents that are considered duplicates.
        The best document in each cluster is determined by the following criteria:
        1. If date_field is provided, the document with the latest date.
        2. If there are multiple documents with the same latest date, the one with the longest text.

        All other documents in the cluster are marked for deletion.

        Parameters:
            clusters (dict): A dictionary where each key is a cluster ID and each value is a list of documents in the cluster.
            date_field (str): The field name for the date, if provided.

        Returns:
            list: A list of document IDs that need to be deleted.
        """
        to_delete = []
        for _, docs in tqdm(clusters.items()):
            if len(docs) <= 1:
                continue

            if date_field:
                best_doc = max(docs, key=lambda x: (
                    x[1]["date"] if x[1]["date"] is not None else "",
                    x[1]["text_length"]
                ))
            else:
                best_doc = max(docs, key=lambda x: x[1]["text_length"])

            for doc_id, _ in docs:
                if doc_id != best_doc[0]:
                    to_delete.append(doc_id)

        return to_delete

    def _delete_with_batch(self, es, to_delete, batch_size, index_name):
        """
        Delete the given document IDs in batches.

        Parameters:
            es (Elasticsearch): The Elasticsearch client.
            to_delete (list): A list of document IDs to delete.
            batch_size (int): The number of documents to delete in each batch.
            index_name (str): The name of the Elasticsearch index.
        """
        logger.info(f"Preparing to delete {len(to_delete)} duplicate documents...")

        # Delete the documents in batches
        for i in tqdm(range(0, len(to_delete), batch_size)):
            batch = to_delete[i:i + batch_size]

            # Construct the bulk delete request body
            body = [
                {"delete": {"_index": index_name, "_id": doc_id}}
                for doc_id in batch
            ]

            # Execute the bulk delete request
            es.bulk(body=body)