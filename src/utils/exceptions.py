"""Custom exceptions for the RAG system."""


class RAGException(Exception):
    """Base exception for all RAG system errors."""
    pass


class CorpusError(RAGException):
    """Raised when corpus loading or processing fails."""
    pass


class RetrievalError(RAGException):
    """Raised when document retrieval fails."""
    pass


class GenerationError(RAGException):
    """Raised when LLM generation fails."""
    pass


class EvaluationError(RAGException):
    """Raised when evaluation process fails."""
    pass


class ConfigurationError(RAGException):
    """Raised when configuration is invalid."""
    pass
