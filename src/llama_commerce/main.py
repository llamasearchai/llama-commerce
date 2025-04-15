# Directory Structure
"""
llama_commerce/
├── __init__.py
├── commerce_manager.py
├── product_search/
│   ├── __init__.py
│   ├── product_vector_search.py
│   ├── visual_similarity_search.py
├── recommendation/
│   ├── __init__.py
│   ├── recommendation_engine.py
├── pricing/
│   ├── __init__.py
│   ├── price_predictor.py
├── ranking/
│   ├── __init__.py
│   ├── inventory_aware_ranker.py
├── conversational/
│   ├── __init__.py
│   ├── conversational_commerce_agent.py
├── security/
│   ├── __init__.py
│   ├── fraud_detector.py
├── export/
│   ├── __init__.py
│   ├── core_ml_exporter.py
├── ar/
│   ├── __init__.py
│   ├── ar_preview_generator.py
├── utils/
│   ├── __init__.py
│   ├── config.py
│   ├── logging_utils.py
│   ├── vector_utils.py
└── tests/
    ├── __init__.py
    ├── test_commerce_manager.py
    ├── test_product_vector_search.py
    ├── test_visual_similarity_search.py
    ├── test_recommendation_engine.py
    ├── test_price_predictor.py
    ├── test_inventory_aware_ranker.py
    ├── test_conversational_commerce_agent.py
    ├── test_fraud_detector.py
    ├── test_core_ml_exporter.py
    ├── test_ar_preview_generator.py
    └── conftest.py
"""

# __init__.py
"""
"""
from .commerce_manager import CommerceManager

__version__ = "0.1.0"
__all__ = ["CommerceManager"]


# commerce_manager.py
"""
Main orchestrator for the llama_commerce package.
"""
import logging
from typing import Any, Dict, List, Optional, Union

from .ar.ar_preview_generator import ARPreviewGenerator
from .conversational.conversational_commerce_agent import ConversationalCommerceAgent
from .export.core_ml_exporter import CoreMLExporter
from .pricing.price_predictor import PricePredictor
from .product_search.product_vector_search import ProductVectorSearch
from .product_search.visual_similarity_search import VisualSimilaritySearch
from .ranking.inventory_aware_ranker import InventoryAwareRanker
from .recommendation.recommendation_engine import RecommendationEngine
from .security.fraud_detector import FraudDetector
from .utils.config import Config
from .utils.logging_utils import setup_logging


class CommerceManager:
    """
    Orchestrates the e-commerce components of the llama_commerce package.

    This class serves as the main entry point for the package, providing a unified
    interface to access all the e-commerce functionalities like product search,
    recommendations, pricing, ranking, conversational commerce, fraud detection,
    ML model export, and AR preview generation.

    Attributes:
        config (Config): Configuration object containing settings for all components.
        product_vector_search (ProductVectorSearch): Component for vector-based product search.
        visual_similarity_search (VisualSimilaritySearch): Component for visual similarity search.
        recommendation_engine (RecommendationEngine): Component for product recommendations.
        price_predictor (PricePredictor): Component for price prediction.
        inventory_aware_ranker (InventoryAwareRanker): Component for inventory-aware ranking.
        conversational_commerce_agent (ConversationalCommerceAgent): Component for conversational commerce.
        fraud_detector (FraudDetector): Component for fraud detection.
        core_ml_exporter (CoreMLExporter): Component for exporting models to Core ML format.
        ar_preview_generator (ARPreviewGenerator): Component for generating AR previews.
        logger (logging.Logger): Logger for this class.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initializes the CommerceManager with all its components.

        Args:
            config_path (Optional[str]): Path to the configuration file. If None,
                the default configuration will be used.
        """
        # Set up logging
        setup_logging()
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing CommerceManager")

        # Load configuration
        self.config = Config(config_path)

        # Initialize components
        self.product_vector_search = ProductVectorSearch(self.config)
        self.visual_similarity_search = VisualSimilaritySearch(self.config)
        self.recommendation_engine = RecommendationEngine(self.config)
        self.price_predictor = PricePredictor(self.config)
        self.inventory_aware_ranker = InventoryAwareRanker(self.config)
        self.conversational_commerce_agent = ConversationalCommerceAgent(self.config)
        self.fraud_detector = FraudDetector(self.config)
        self.core_ml_exporter = CoreMLExporter(self.config)
        self.ar_preview_generator = ARPreviewGenerator(self.config)

        self.logger.info("CommerceManager initialized successfully")

    def search_products(
        self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Searches for products based on a text query.

        Args:
            query (str): The search query.
            top_k (int): Number of top results to return. Defaults to 10.
            filters (Optional[Dict[str, Any]]): Filters to apply to the search results.
                Defaults to None.

        Returns:
            List[Dict[str, Any]]: List of product dictionaries matching the query.
        """
        self.logger.info(f"Searching products with query: {query}")
        results = self.product_vector_search.search(query, top_k, filters)
        return results

    def search_similar_products_by_image(
        self, image_path: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Searches for products visually similar to the provided image.

        Args:
            image_path (str): Path to the image file.
            top_k (int): Number of top results to return. Defaults to 10.
            filters (Optional[Dict[str, Any]]): Filters to apply to the search results.
                Defaults to None.

        Returns:
            List[Dict[str, Any]]: List of product dictionaries visually similar to the image.
        """
        self.logger.info(f"Searching visually similar products for image: {image_path}")
        results = self.visual_similarity_search.search(image_path, top_k, filters)
        return results

    def get_product_recommendations(
        self, user_id: str, product_id: Optional[str] = None, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Gets product recommendations for a user, optionally based on a specific product.

        Args:
            user_id (str): ID of the user.
            product_id (Optional[str]): ID of the product to base recommendations on.
                If None, recommendations will be based on user's history. Defaults to None.
            top_k (int): Number of recommendations to return. Defaults to 5.

        Returns:
            List[Dict[str, Any]]: List of recommended product dictionaries.
        """
        self.logger.info(f"Getting product recommendations for user: {user_id}")
        recommendations = self.recommendation_engine.get_recommendations(
            user_id, product_id, top_k
        )
        return recommendations

    def predict_price(self, product_data: Dict[str, Any]) -> float:
        """
        Predicts the optimal price for a product.

        Args:
            product_data (Dict[str, Any]): Product data including features needed for prediction.

        Returns:
            float: Predicted optimal price.
        """
        self.logger.info(
            f"Predicting price for product: {product_data.get('id', 'unknown')}"
        )
        predicted_price = self.price_predictor.predict(product_data)
        return predicted_price

    def rank_products(
        self, products: List[Dict[str, Any]], user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Ranks products considering inventory levels and user preferences.

        Args:
            products (List[Dict[str, Any]]): List of product dictionaries to rank.
            user_id (Optional[str]): ID of the user. If provided, user preferences will be
                considered in ranking. Defaults to None.

        Returns:
            List[Dict[str, Any]]: Ranked list of product dictionaries.
        """
        self.logger.info(
            f"Ranking {len(products)} products for user: {user_id or 'anonymous'}"
        )
        ranked_products = self.inventory_aware_ranker.rank(products, user_id)
        return ranked_products

    def process_conversation(self, user_id: str, message: str) -> Dict[str, Any]:
        """
        Processes a user message in a conversational commerce context.

        Args:
            user_id (str): ID of the user.
            message (str): User message.

        Returns:
            Dict[str, Any]: Response containing message, recommended products, and actions.
        """
        self.logger.info(f"Processing conversation for user {user_id}: {message}")
        response = self.conversational_commerce_agent.process_message(user_id, message)
        return response

    def detect_fraud(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detects potential fraud in a transaction.

        Args:
            transaction_data (Dict[str, Any]): Transaction data to analyze.

        Returns:
            Dict[str, Any]: Fraud analysis results including fraud score and risk factors.
        """
        self.logger.info(
            f"Detecting fraud for transaction: {transaction_data.get('id', 'unknown')}"
        )
        fraud_analysis = self.fraud_detector.detect(transaction_data)
        return fraud_analysis

    def export_model_to_coreml(
        self, model_name: str, model_data: Any, target_path: str
    ) -> bool:
        """
        Exports a machine learning model to Core ML format.

        Args:
            model_name (str): Name of the model.
            model_data (Any): Model data to export.
            target_path (str): Path where the Core ML model will be saved.

        Returns:
            bool: True if export was successful, False otherwise.
        """
        self.logger.info(f"Exporting model {model_name} to Core ML format")
        success = self.core_ml_exporter.export(model_name, model_data, target_path)
        return success

    def generate_ar_preview(
        self, product_id: str, environment: str = "living_room"
    ) -> str:
        """
        Generates an AR preview for a product.

        Args:
            product_id (str): ID of the product.
            environment (str): Environment in which to place the product.
                Defaults to "living_room".

        Returns:
            str: Path to the generated AR preview image.
        """
        self.logger.info(
            f"Generating AR preview for product {product_id} in {environment}"
        )
        preview_path = self.ar_preview_generator.generate_preview(
            product_id, environment
        )
        return preview_path


# product_search/product_vector_search.py
"""
Vector-based product search implementation.
"""
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import llama_vector
except ImportError:
    llama_vector = None

from ..utils.config import Config
from ..utils.vector_utils import normalize_vector


class ProductVectorSearch:
    """
    Provides vector-based search for products using llama_vector integration.

    This class enables semantic search capabilities for product catalogs by
    converting text queries into vector embeddings and finding products with
    similar embeddings.

    Attributes:
        config (Config): Configuration object.
        index (Optional[Any]): Vector index for product search.
        embedding_model (Optional[Any]): Model for generating embeddings.
        logger (logging.Logger): Logger for this class.
    """

    def __init__(self, config: Config):
        """
        Initializes the ProductVectorSearch component.

        Args:
            config (Config): Configuration object containing settings.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing ProductVectorSearch")

        self.config = config
        self._initialize_vector_search()

    def _initialize_vector_search(self) -> None:
        """
        Initializes the vector search components.

        This method sets up the embedding model and vector index required for
        vector-based product search.

        Raises:
            ImportError: If llama_vector is not installed.
        """
        if llama_vector is None:
            self.logger.warning(
                "llama_vector not installed. Using fallback search method."
            )
            self.index = None
            self.embedding_model = None
            return

        try:
            # Initialize embedding model
            model_name = self.config.get(
                "product_search.embedding_model", "default_embedding_model"
            )
            self.embedding_model = llama_vector.EmbeddingModel(model_name)

            # Initialize vector index
            index_path = self.config.get("product_search.index_path")
            if index_path and llama_vector.Index.exists(index_path):
                self.index = llama_vector.Index.load(index_path)
                self.logger.info(f"Loaded existing vector index from {index_path}")
            else:
                self.logger.info("Creating new vector index")
                self.index = llama_vector.Index(
                    dimension=self.embedding_model.dimension
                )
                # Save index if path is provided
                if index_path:
                    self.index.save(index_path)
                    self.logger.info(f"Saved new vector index to {index_path}")
        except Exception as e:
            self.logger.error(f"Failed to initialize vector search: {str(e)}")
            self.index = None
            self.embedding_model = None

    def build_index(self, products: List[Dict[str, Any]]) -> bool:
        """
        Builds the vector index from a list of products.

        Args:
            products (List[Dict[str, Any]]): List of product dictionaries to index.

        Returns:
            bool: True if index was built successfully, False otherwise.
        """
        if llama_vector is None or self.embedding_model is None:
            self.logger.warning("Cannot build index: llama_vector not available")
            return False

        try:
            # Create a new index
            self.index = llama_vector.Index(dimension=self.embedding_model.dimension)

            # Process and add each product
            for product in products:
                # Generate embedding from product description or concatenated fields
                text = self._get_product_text(product)
                embedding = self.embedding_model.embed(text)

                # Add to index
                self.index.add(
                    id=str(product["id"]), vector=embedding, metadata=product
                )

            # Save the index if path is configured
            index_path = self.config.get("product_search.index_path")
            if index_path:
                self.index.save(index_path)
                self.logger.info(f"Vector index built and saved to {index_path}")

            return True
        except Exception as e:
            self.logger.error(f"Failed to build index: {str(e)}")
            return False

    def _get_product_text(self, product: Dict[str, Any]) -> str:
        """
        Extracts text from a product for embedding generation.

        Args:
            product (Dict[str, Any]): Product dictionary.

        Returns:
            str: Text representation of the product.
        """
        # Concatenate relevant fields with weights
        text_parts = []

        if "name" in product:
            text_parts.append(product["name"] * 3)  # Weight name higher

        if "description" in product:
            text_parts.append(product["description"])

        if "categories" in product and isinstance(product["categories"], list):
            text_parts.append(" ".join(product["categories"]))

        if "attributes" in product and isinstance(product["attributes"], dict):
            for key, value in product["attributes"].items():
                if isinstance(value, (str, int, float)):
                    text_parts.append(f"{key}: {value}")

        # Fallback to product ID if no text is available
        if not text_parts and "id" in product:
            return f"Product {product['id']}"

        return " ".join(text_parts)

    def search(
        self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Searches for products similar to the query text.

        Args:
            query (str): The search query.
            top_k (int): Number of top results to return. Defaults to 10.
            filters (Optional[Dict[str, Any]]): Filters to apply to the search results.
                Defaults to None.

        Returns:
            List[Dict[str, Any]]: List of product dictionaries matching the query.
        """
        if llama_vector is None or self.embedding_model is None or self.index is None:
            return self._fallback_search(query, top_k, filters)

        try:
            # Generate embedding for the query
            query_embedding = self.embedding_model.embed(query)

            # Prepare filter function if filters are provided
            filter_func = None
            if filters:
                filter_func = lambda metadata: all(
                    self._check_filter(metadata, key, value)
                    for key, value in filters.items()
                )

            # Perform the search
            results = self.index.search(
                vector=query_embedding, top_k=top_k, filter_func=filter_func
            )

            # Format results
            formatted_results = []
            for result in results:
                item = result.metadata.copy()
                item["score"] = float(result.score)
                formatted_results.append(item)

            return formatted_results
        except Exception as e:
            self.logger.error(f"Vector search failed: {str(e)}")
            return self._fallback_search(query, top_k, filters)

    def _check_filter(self, metadata: Dict[str, Any], key: str, value: Any) -> bool:
        """
        Checks if metadata matches a filter condition.

        Args:
            metadata (Dict[str, Any]): Product metadata.
            key (str): Filter key, can use dot notation for nested fields.
            value (Any): Expected value or callable predicate.

        Returns:
            bool: True if filter condition is satisfied, False otherwise.
        """
        # Support for nested fields with dot notation
        if "." in key:
            parts = key.split(".")
            current = metadata
            for part in parts[:-1]:
                if part not in current:
                    return False
                current = current[part]
            key = parts[-1]
            if key not in current:
                return False
            actual_value = current[key]
        else:
            if key not in metadata:
                return False
            actual_value = metadata[key]

        # Handle different filter types
        if callable(value):
            # If value is a callable, use it as a predicate
            return value(actual_value)
        elif isinstance(value, list):
            # If value is a list, check if actual_value is in the list
            return actual_value in value
        elif isinstance(value, dict) and actual_value is not None:
            # Handle range queries
            if "gt" in value and actual_value <= value["gt"]:
                return False
            if "gte" in value and actual_value < value["gte"]:
                return False
            if "lt" in value and actual_value >= value["lt"]:
                return False
            if "lte" in value and actual_value > value["lte"]:
                return False
            return True
        else:
            # Direct comparison
            return actual_value == value

    def _fallback_search(
        self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Fallback search method when vector search is not available.

        This method uses basic keyword matching as a fallback when vector search
        cannot be performed.

        Args:
            query (str): The search query.
            top_k (int): Number of top results to return. Defaults to 10.
            filters (Optional[Dict[str, Any]]): Filters to apply to the search results.
                Defaults to None.

        Returns:
            List[Dict[str, Any]]: List of product dictionaries matching the query.
        """
        self.logger.info("Using fallback keyword search")

        # In a real implementation, this would access a database of products
        # For simulation, we'll return dummy results
        results = []
        for i in range(min(top_k, 5)):  # Limit to 5 results in fallback mode
            results.append(
                {
                    "id": f"product_{i+1}",
                    "name": f"Fallback Product {i+1} for '{query}'",
                    "description": "This is a fallback result because vector search is not available.",
                    "price": 99.99,
                    "score": 0.5 - (i * 0.1),  # Decreasing score
                }
            )

        return results


# product_search/visual_similarity_search.py
"""
Visual similarity search implementation.
"""
import io
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import llama_vector
    import PIL
    from PIL import Image
except ImportError:
    llama_vector = None
    PIL = None

from ..utils.config import Config


class VisualSimilaritySearch:
    """
    Provides visual similarity search for products using image embeddings.

    This class enables searching for products that are visually similar to a provided
    image using computer vision techniques and vector similarity.

    Attributes:
        config (Config): Configuration object.
        index (Optional[Any]): Vector index for visual similarity search.
        vision_model (Optional[Any]): Model for generating image embeddings.
        logger (logging.Logger): Logger for this class.
    """

    def __init__(self, config: Config):
        """
        Initializes the VisualSimilaritySearch component.

        Args:
            config (Config): Configuration object containing settings.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing VisualSimilaritySearch")

        self.config = config
        self._initialize_visual_search()

    def _initialize_visual_search(self) -> None:
        """
        Initializes the visual search components.

        This method sets up the vision model and vector index required for
        visual similarity search.

        Raises:
            ImportError: If required dependencies are not installed.
        """
        if llama_vector is None or PIL is None:
            self.logger.warning(
                "Required dependencies not installed. Using fallback search method."
            )
            self.index = None
            self.vision_model = None
            return

        try:
            # Initialize vision model
            model_name = self.config.get(
                "visual_search.vision_model", "default_vision_model"
            )
            self.vision_model = llama_vector.VisionModel(model_name)

            # Initialize vector index
            index_path = self.config.get("visual_search.index_path")
            if index_path and llama_vector.Index.exists(index_path):
                self.index = llama_vector.Index.load(index_path)
                self.logger.info(
                    f"Loaded existing visual search index from {index_path}"
                )
            else:
                self.logger.info("Creating new visual search index")
                self.index = llama_vector.Index(dimension=self.vision_model.dimension)
                # Save index if path is provided
                if index_path:
                    self.index.save(index_path)
                    self.logger.info(f"Saved new visual search index to {index_path}")
        except Exception as e:
            self.logger.error(f"Failed to initialize visual search: {str(e)}")
            self.index = None
            self.vision_model = None

    def build_index(self, products: List[Dict[str, Any]]) -> bool:
        """
        Builds the visual similarity index from a list of products with image paths.

        Args:
            products (List[Dict[str, Any]]): List of product dictionaries with image paths.

        Returns:
            bool: True if index was built successfully, False otherwise.
        """
        if llama_vector is None or PIL is None or self.vision_model is None:
            self.logger.warning(
                "Cannot build index: required dependencies not available"
            )
            return False

        try:
            # Create a new index
            self.index = llama_vector.Index(dimension=self.vision_model.dimension)

            # Process and add each product
            for product in products:
                # Skip products without images
                if "image_path" not in product:
                    self.logger.warning(
                        f"Skipping product {product.get('id', 'unknown')} without image_path"
                    )
                    continue

                # Load and process image
                try:
                    image_path = product["image_path"]
                    image = Image.open(image_path)
                    embedding = self.vision_model.embed(image)

                    # Add to index
                    self.index.add(
                        id=str(product["id"]), vector=embedding, metadata=product
                    )
                except Exception as img_error:
                    self.logger.warning(
                        f"Failed to process image for product {product.get('id', 'unknown')}: {str(img_error)}"
                    )

            # Save the index if path is configured
            index_path = self.config.get("visual_search.index_path")
            if index_path:
                self.index.save(index_path)
                self.logger.info(
                    f"Visual similarity index built and saved to {index_path}"
                )

            return True
        except Exception as e:
            self.logger.error(f"Failed to build visual similarity index: {str(e)}")
            return False

    def search(
        self, image_path: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Searches for products visually similar to the provided image.

        Args:
            image_path (str): Path to the image file.
            top_k (int): Number of top results to return. Defaults to 10.
            filters (Optional[Dict[str, Any]]): Filters to apply to the search results.
                Defaults to None.

        Returns:
            List[Dict[str, Any]]: List of product dictionaries visually similar to the image.
        """
        if (
            llama_vector is None
            or PIL is None
            or self.vision_model is None
            or self.index is None
        ):
            return self._fallback_search(image_path, top_k, filters)

        try:
            # Load and process the query image
            image = Image.open(image_path)
            image_embedding = self.vision_model.embed(image)

            # Prepare filter function if filters are provided
            filter_func = None
            if filters:
                filter_func = lambda metadata: all(
                    self._check_filter(metadata, key, value)
                    for key, value in filters.items()
                )

            # Perform the search
            results = self.index.search(
                vector=image_embedding, top_k=top_k, filter_func=filter_func
            )

            # Format results
            formatted_results = []
            for result in results:
                item = result.metadata.copy()
                item["visual_similarity_score"] = float(result.score)
                formatted_results.append(item)

            return formatted_results
        except Exception as e:
            self.logger.error(f"Visual similarity search failed: {str(e)}")
            return self._fallback_search(image_path, top_k, filters)

    def _check_filter(self, metadata: Dict[str, Any], key: str, value: Any) -> bool:
        """
        Checks if metadata matches a filter condition.

        Args:
            metadata (Dict[str, Any]): Product metadata.
            key (str): Filter key, can use dot notation for nested fields.
            value (Any): Expected value or callable predicate.

        Returns:
            bool: True if filter condition is satisfied, False otherwise.
        """
        # Support for nested fields with dot notation
        if "." in key:
            parts = key.split(".")
            current = metadata
            for part in parts[:-1]:
                if part not in current:
                    return False
                current = current[part]
            key = parts[-1]
            if key not in current:
                return False
            actual_value = current[key]
        else:
            if key not in metadata:
                return False
            actual_value = metadata[key]

        # Handle different filter types
        if callable(value):
            # If value is a callable, use it as a predicate
            return value(actual_value)
        elif isinstance(value, list):
            # If value is a list, check if actual_value is in the list
            return actual_value in value
        elif isinstance(value, dict) and actual_value is not None:
            # Handle range queries
            if "gt" in value and actual_value <= value["gt"]:
                return False
            if "gte" in value and actual_value < value["gte"]:
                return False
            if "lt" in value and actual_value >= value["lt"]:
                return False
            if "lte" in value and actual_value > value["lte"]:
                return False
            return True
        else:
            # Direct comparison
            return actual_value == value

    def _fallback_search(
        self, image_path: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Fallback search method when visual search is not available.

        This method returns dummy results when visual search cannot be performed.

        Args:
            image_path (str): Path to the image file.
            top_k (int): Number of top results to return. Defaults to 10.
            filters (Optional[Dict[str, Any]]): Filters to apply to the search results.
                Defaults to None.
        """
        self.logger.warning("Visual search unavailable, using fallback dummy search")

        # Create dummy results
        dummy_results = []
        for i in range(top_k):
            product_id = f"dummy_{i+1}"
            metadata = {
                "product_id": product_id,
                "name": f"Dummy Product {i+1}",
                "category": "Dummy",
            }

            # Check filters if provided
            if filters:
                match = True
                for key, value in filters.items():
                    if not self._check_filter(metadata, key, value):
                        match = False
                        break
                if not match:
                    continue

            dummy_results.append(
                {"id": product_id, "distance": float(i), "metadata": metadata}
            )

        return dummy_results
