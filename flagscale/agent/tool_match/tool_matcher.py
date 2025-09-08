"""Intelligent tool matcher using semantic embeddings"""

import logging

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ToolMatcher:
    """Semantic tool matcher with caching and optimization."""

    def __init__(self, max_tools: int = 3, min_similarity: float = 0.1):
        self.max_tools = max_tools
        self.min_similarity = min_similarity
        self.tools = []
        self.tool_embeddings = []
        self.model = None
        self._query_cache = OrderedDict()  # LRU Cache for query embeddings
        self._cache_max_size = 100  # Maximum cache size
        self.logger = logging.getLogger(__name__)
        self._init_model()

    def _init_model(self):
        """Initialize sentence transformer model with improved error handling"""
        try:
            # Check if sentence-transformers is available first
            from sentence_transformers import SentenceTransformer

            # Check network connectivity with multiple fallbacks
            if not self._check_network_connectivity():
                self.logger.warning(
                    "Network unavailable for model download. Will return all tools on match."
                )
                self.model = None
                return

            # Try to load model with local cache first
            try:
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                self.logger.info("Semantic model initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load model (may need download): {e}")
                self.logger.warning("Will return all tools on match.")
                self.model = None

        except ImportError:
            self.logger.warning(
                "sentence-transformers not available. Install with: pip install sentence-transformers"
            )
            self.logger.warning("Will return all tools on match.")
            self.model = None
        except Exception as e:
            self.logger.warning(
                f"Failed to initialize semantic model: {e}. Will return all tools on match."
            )
            self.model = None

    def _check_network_connectivity(self) -> bool:
        """Check network connectivity with  endpoint"""
        import socket
        import urllib.request

        endpoints = ['https://huggingface.co']

        for endpoint in endpoints:
            try:
                urllib.request.urlopen(endpoint, timeout=3)
                return True
            except (urllib.error.URLError, socket.timeout):
                continue

        return False

    def fit(self, tools: List[Dict[str, Any]]):
        """Train matcher with tools"""
        self.tools = tools
        if self.model:
            self._fit_embeddings()

    def _fit_embeddings(self):
        """Generate embeddings for tools"""
        try:
            tool_texts = []
            for tool in self.tools:
                func = tool.get("function", {})
                name = func.get("name", "")
                desc = func.get("description", "")
                tool_texts.append(f"{name} {desc}".strip())

            if tool_texts:
                self.tool_embeddings = self.model.encode(tool_texts, convert_to_tensor=True)
                self.logger.info(f"Generated embeddings for {len(self.tools)} tools")
        except Exception as e:
            self.logger.warning(f"Failed to generate embeddings: {e}")
            self.tool_embeddings = []

    def match_tools(self, task: str) -> List[Tuple[str, float]]:
        """Match task with relevant tools.
        If semantic model is unavailable, return all tools (no truncation).
        """
        if not self.tools:
            return []

        if self.model and len(self.tool_embeddings) > 0:
            return self._semantic_match(task)
        else:
            return self._get_all_tools_fallback()

    def _get_all_tools_fallback(self) -> List[Tuple[str, float]]:
        """Fallback method to return all tools with score 1.0"""
        results: List[Tuple[str, float]] = []
        for i, tool in enumerate(self.tools):
            name = tool.get("function", {}).get("name", f"tool_{i}")
            results.append((name, 1.0))
        return results

    def _semantic_match(self, task: str) -> List[Tuple[str, float]]:
        """Semantic matching using embeddings with caching"""
        try:
            # Check cache first
            task_embedding = self._get_cached_embedding(task)
            similarities = self._cosine_similarity(task_embedding, self.tool_embeddings)

            tool_scores = []
            for i, (tool, similarity) in enumerate(zip(self.tools, similarities)):
                if similarity >= self.min_similarity:
                    name = tool.get("function", {}).get("name", f"tool_{i}")
                    tool_scores.append((name, float(similarity)))

            tool_scores.sort(key=lambda x: x[1], reverse=True)
            return tool_scores[: self.max_tools]
        except Exception as e:
            self.logger.warning(f"Semantic matching failed: {e}")
            # If semantic path fails, return all tools
            return self._get_all_tools_fallback()

    def _get_cached_embedding(self, task: str):
        """Get embedding from cache or compute and cache it - LRU cache"""
        if task in self._query_cache:
            # Move to end (most recently used)
            self._query_cache.move_to_end(task)
            return self._query_cache[task]

        # Compute new embedding
        embedding = self.model.encode([task], convert_to_tensor=True)

        # Manage cache size with LRU eviction
        if len(self._query_cache) >= self._cache_max_size:
            # Remove least recently used entry (first item)
            self._query_cache.popitem(last=False)

        self._query_cache[task] = embedding
        return embedding

    def _cosine_similarity(self, a, b):
        """Calculate cosine similarity"""
        try:
            import torch

            if torch.is_tensor(a) and torch.is_tensor(b):
                a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
                b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
                similarity = torch.mm(a_norm, b_norm.t())
                return similarity.squeeze().cpu().numpy()
        except ImportError:
            pass

        # Fallback to numpy
        try:
            a_np = a.cpu().numpy() if hasattr(a, 'cpu') else np.array(a)
            b_np = b.cpu().numpy() if hasattr(b, 'cpu') else np.array(b)

            a_norm = a_np / np.linalg.norm(a_np, axis=1, keepdims=True)
            b_norm = b_np / np.linalg.norm(b_np, axis=1, keepdims=True)

            similarity = np.dot(a_norm, b_norm.T)
            return similarity.squeeze()
        except Exception as e:
            self.logger.warning(f"Cosine similarity calculation failed: {e}")
            return np.zeros(len(self.tools))
