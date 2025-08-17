"""
Vector Store for fast face similarity search using ChromaDB
"""
import chromadb
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
import json
from datetime import datetime

from config import VECTOR_DB_PATH, COLLECTION_NAME, SIMILARITY_THRESHOLD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceVectorStore:
    """
    Vector database for storing and searching face embeddings
    """
    
    def __init__(self, db_path: str = VECTOR_DB_PATH, collection_name: str = COLLECTION_NAME):
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Face embeddings for similarity search"}
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def add_face_embedding(
        self, 
        face_id: str, 
        embedding: np.ndarray, 
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Add a face embedding to the vector store
        
        Args:
            face_id: Unique identifier for the face
            embedding: Face embedding vector
            metadata: Additional metadata (image_path, location, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert embedding to list for ChromaDB
            embedding_list = embedding.tolist()
            
            # Add metadata
            full_metadata = {
                **metadata,
                "added_at": datetime.now().isoformat(),
                "embedding_dimension": len(embedding)
            }
            
            # Add to collection
            self.collection.add(
                embeddings=[embedding_list],
                metadatas=[full_metadata],
                ids=[face_id]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding face embedding {face_id}: {e}")
            return False
    
    def add_face_embeddings_batch(
        self, 
        face_ids: List[str], 
        embeddings: List[np.ndarray], 
        metadatas: List[Dict[str, Any]]
    ) -> int:
        """
        Add multiple face embeddings in batch
        
        Args:
            face_ids: List of face identifiers
            embeddings: List of face embeddings
            metadatas: List of metadata dictionaries
            
        Returns:
            Number of successfully added embeddings
        """
        try:
            if len(face_ids) != len(embeddings) or len(face_ids) != len(metadatas):
                raise ValueError("Lists must have the same length")
            
            # Convert embeddings to lists
            embedding_lists = [emb.tolist() for emb in embeddings]
            
            # Add metadata
            full_metadatas = []
            for metadata in metadatas:
                full_metadata = {
                    **metadata,
                    "added_at": datetime.now().isoformat(),
                    "embedding_dimension": len(embeddings[0])
                }
                full_metadatas.append(full_metadata)
            
            # Add to collection
            self.collection.add(
                embeddings=embedding_lists,
                metadatas=full_metadatas,
                ids=face_ids
            )
            
            logger.info(f"Added {len(face_ids)} face embeddings to vector store")
            return len(face_ids)
            
        except Exception as e:
            logger.error(f"Error adding batch embeddings: {e}")
            return 0
    
    def search_similar_faces(
        self, 
        query_embedding: np.ndarray, 
        n_results: int = 10,
        min_similarity: float = SIMILARITY_THRESHOLD
    ) -> List[Dict[str, Any]]:
        """
        Search for similar faces with enhanced similarity calculation and better thresholding
        
        Args:
            query_embedding: Query face embedding (should be normalized)
            n_results: Maximum number of results to return
            min_similarity: Minimum similarity threshold (0-1 scale)
            
        Returns:
            List of similar faces with metadata and enhanced similarity scores
        """
        try:
            # Ensure query embedding is properly normalized
            query_norm = np.linalg.norm(query_embedding)
            if query_norm > 0:
                query_embedding_normalized = query_embedding / query_norm
            else:
                query_embedding_normalized = query_embedding
                logger.warning("Query embedding has zero norm")
                
            # Convert embedding to list for ChromaDB
            query_list = query_embedding_normalized.tolist()
            
            # Search with expanded results to allow better filtering
            search_n = min(n_results * 5, 500)  # Get more candidates for better filtering
            
            results = self.collection.query(
                query_embeddings=[query_list],
                n_results=search_n,
                include=['embeddings', 'metadatas', 'distances']
            )
            
            # Process results with enhanced similarity calculation
            similar_faces = []
            
            if results['ids'] and len(results['ids'][0]) > 0:
                for i, face_id in enumerate(results['ids'][0]):
                    try:
                        distance = results['distances'][0][i]
                        stored_embedding = np.array(results['embeddings'][0][i], dtype=np.float32)
                        
                        # Calculate comprehensive similarity metrics
                        similarity_metrics = self._calculate_enhanced_similarity(
                            query_embedding_normalized, 
                            stored_embedding
                        )
                        
                        # Use the primary similarity score for filtering and ranking
                        primary_similarity = similarity_metrics['primary_similarity']
                        
                        # Apply similarity threshold filter
                        if primary_similarity >= min_similarity:
                            face_data = {
                                'face_id': face_id,
                                'similarity': primary_similarity,
                                'similarity_percentage': primary_similarity * 100,
                                'cosine_similarity': similarity_metrics['cosine_similarity'],
                                'euclidean_distance': distance,
                                'euclidean_similarity': similarity_metrics.get('euclidean_similarity', 0.0),
                                'correlation_similarity': similarity_metrics.get('correlation_similarity', 0.0),
                                'confidence_score': self._calculate_confidence_score(similarity_metrics),
                                'metadata': results['metadatas'][0][i],
                                'similarity_metrics': similarity_metrics
                            }
                            similar_faces.append(face_data)
                            
                    except Exception as e:
                        logger.warning(f"Error processing result {i}: {e}")
                        continue
            
            # Sort by confidence score (combination of similarity and reliability)
            similar_faces.sort(key=lambda x: x['confidence_score'], reverse=True)
            
            # Limit to requested number of results
            similar_faces = similar_faces[:n_results]
            
            # Log search results for debugging
            if similar_faces:
                top_result = similar_faces[0]
                logger.info(f"Found {len(similar_faces)} similar faces - Top match: {top_result['similarity_percentage']:.1f}% confidence")
                logger.debug(f"Top result metrics: cosine={top_result['cosine_similarity']:.4f}, euclidean_sim={top_result['euclidean_similarity']:.4f}")
            else:
                logger.info(f"No similar faces found above threshold {min_similarity:.3f}")
            
            return similar_faces
            
        except Exception as e:
            logger.error(f"Error searching similar faces: {e}")
            return []
    
    def _calculate_enhanced_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive similarity metrics between two embeddings
        
        Args:
            embedding1: First embedding (query, should be normalized)
            embedding2: Second embedding (from database)
            
        Returns:
            Dictionary containing various similarity metrics
        """
        try:
            # Normalize both embeddings to ensure consistent comparison
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                logger.warning("One of the embeddings has zero norm")
                return self._get_zero_similarity_metrics()
                
            embedding1_normalized = embedding1 / norm1
            embedding2_normalized = embedding2 / norm2
            
            metrics = {}
            
            # 1. Cosine similarity (most reliable for normalized embeddings)
            cosine_similarity = np.dot(embedding1_normalized, embedding2_normalized)
            cosine_similarity = float(np.clip(cosine_similarity, -1.0, 1.0))
            metrics['cosine_similarity'] = cosine_similarity
            
            # 2. Euclidean distance and derived similarity
            euclidean_distance = float(np.linalg.norm(embedding1_normalized - embedding2_normalized))
            # For unit vectors, euclidean distance ranges from 0 (identical) to 2 (opposite)
            euclidean_similarity = max(0.0, 1.0 - (euclidean_distance / 2.0))
            metrics['euclidean_distance'] = euclidean_distance
            metrics['euclidean_similarity'] = euclidean_similarity
            
            # 3. Dot product similarity (similar to cosine for normalized vectors)
            dot_product = float(np.dot(embedding1_normalized, embedding2_normalized))
            metrics['dot_product'] = dot_product
            
            # 4. Calculate primary similarity using proven face recognition approach
            # For face embeddings, cosine similarity above 0.4 typically indicates same person
            # Convert to 0-1 scale where higher values indicate higher similarity
            if cosine_similarity >= 0.4:
                # High confidence range - scale from [0.4, 1.0] to [0.6, 1.0]
                primary_similarity = 0.6 + (cosine_similarity - 0.4) * 0.67  # 0.4 gap scaled to 0.67
            elif cosine_similarity >= 0.0:
                # Medium confidence range - scale from [0.0, 0.4] to [0.2, 0.6]
                primary_similarity = 0.2 + cosine_similarity * 1.0
            else:
                # Low confidence range - scale from [-1.0, 0.0] to [0.0, 0.2]
                primary_similarity = max(0.0, 0.2 + cosine_similarity * 0.2)
            
            metrics['primary_similarity'] = float(primary_similarity)
            
            # 5. Angular distance (converted from cosine similarity)
            try:
                angular_distance = np.arccos(np.clip(cosine_similarity, -1.0, 1.0)) / np.pi
                angular_similarity = 1.0 - angular_distance
                metrics['angular_similarity'] = float(angular_similarity)
            except:
                metrics['angular_similarity'] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating enhanced similarity: {e}")
            return self._get_zero_similarity_metrics()
    
    def _get_zero_similarity_metrics(self) -> Dict[str, float]:
        """Return zero similarity metrics for error cases"""
        return {
            'cosine_similarity': 0.0,
            'euclidean_distance': 2.0,
            'euclidean_similarity': 0.0,
            'dot_product': 0.0,
            'primary_similarity': 0.0,
            'angular_similarity': 0.0
        }
    
    def _calculate_confidence_score(self, similarity_metrics: Dict[str, float]) -> float:
        """
        Calculate confidence score based on multiple similarity metrics
        
        Args:
            similarity_metrics: Dictionary of similarity metrics
            
        Returns:
            Confidence score (0-1)
        """
        try:
            primary_sim = similarity_metrics.get('primary_similarity', 0.0)
            cosine_sim = similarity_metrics.get('cosine_similarity', 0.0)
            euclidean_sim = similarity_metrics.get('euclidean_similarity', 0.0)
            
            # Weight the metrics based on their reliability for face recognition
            confidence = (
                primary_sim * 0.5 +      # Primary similarity (50%)
                cosine_sim * 0.3 +       # Cosine similarity (30%)
                euclidean_sim * 0.2      # Euclidean similarity (20%)
            )
            
            # Apply confidence boost for very high cosine similarity
            if cosine_sim > 0.7:
                confidence = min(1.0, confidence * 1.1)
            
            return float(max(0.0, min(1.0, confidence)))
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 0.0
    
    def _calculate_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding (should be normalized)
            embedding2: Second embedding
            
        Returns:
            Cosine similarity (-1 to 1, where 1 is most similar)
        """
        try:
            # Normalize both embeddings to unit length
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            embedding1_normalized = embedding1 / norm1
            embedding2_normalized = embedding2 / norm2
            
            # Calculate cosine similarity (dot product of normalized vectors)
            similarity = np.dot(embedding1_normalized, embedding2_normalized)
            
            # Ensure result is in [-1, 1] range
            similarity = np.clip(similarity, -1.0, 1.0)
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def get_face_by_id(self, face_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific face by ID
        
        Args:
            face_id: Face identifier
            
        Returns:
            Face data or None if not found
        """
        try:
            results = self.collection.get(
                ids=[face_id],
                include=['embeddings', 'metadatas']
            )
            
            if results['ids'] and len(results['ids']) > 0:
                return {
                    'face_id': results['ids'][0],
                    'embedding': np.array(results['embeddings'][0]),
                    'metadata': results['metadatas'][0]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting face {face_id}: {e}")
            return None
    
    def update_face_metadata(self, face_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a face
        
        Args:
            face_id: Face identifier
            metadata: New metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get existing face
            existing_face = self.get_face_by_id(face_id)
            if not existing_face:
                return False
            
            # Update metadata
            updated_metadata = {
                **existing_face['metadata'],
                **metadata,
                "updated_at": datetime.now().isoformat()
            }
            
            # Update in collection
            self.collection.update(
                ids=[face_id],
                metadatas=[updated_metadata]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating face metadata {face_id}: {e}")
            return False
    
    def delete_face(self, face_id: str) -> bool:
        """
        Delete a face from the vector store
        
        Args:
            face_id: Face identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection.delete(ids=[face_id])
            logger.info(f"Deleted face {face_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting face {face_id}: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection
        
        Returns:
            Collection statistics
        """
        try:
            # Get collection info
            collection_count = self.collection.count()
            
            # Get sample of metadata to analyze
            sample_results = self.collection.get(
                limit=min(100, collection_count),
                include=['metadatas']
            )
            
            stats = {
                "total_faces": collection_count,
                "collection_name": self.collection_name,
                "db_path": self.db_path
            }
            
            # Analyze metadata
            if sample_results['metadatas']:
                image_paths = set()
                embedding_dimensions = []
                
                for metadata in sample_results['metadatas']:
                    if 'image_path' in metadata:
                        image_paths.add(metadata['image_path'])
                    if 'embedding_dimension' in metadata:
                        embedding_dimensions.append(metadata['embedding_dimension'])
                
                stats.update({
                    "unique_images": len(image_paths),
                    "avg_embedding_dimension": np.mean(embedding_dimensions) if embedding_dimensions else 0
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    def clear_collection(self) -> bool:
        """
        Clear all data from the collection
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete the collection
            self.client.delete_collection(self.collection_name)
            
            # Recreate the collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Face embeddings for similarity search"}
            )
            
            logger.info(f"Cleared collection {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False