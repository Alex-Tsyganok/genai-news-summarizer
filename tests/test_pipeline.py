"""
Test script for the AI News Summarizer components.
"""
import os
import sys
import unittest
from unittest.mock import Mock, patch
from dotenv import load_dotenv

# Add the parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import Article
from src.extractors import NewsExtractor
from src.summarizers import AIsummarizer
from src.storage import VectorStorage
from src.search import SemanticSearcher
from config import settings

class TestNewsExtractor(unittest.TestCase):
    """Test cases for NewsExtractor."""
    
    def setUp(self):
        self.extractor = NewsExtractor()
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        dirty_text = "This  is   a    test!!! @#$%^&*()  with   extra    spaces."
        clean_text = self.extractor._clean_text(dirty_text)
        
        self.assertIsInstance(clean_text, str)
        self.assertNotIn("  ", clean_text)  # No double spaces
        self.assertTrue(len(clean_text) > 0)
    
    def test_extract_article_invalid_url(self):
        """Test extraction with invalid URL."""
        result = self.extractor.extract_article("invalid-url")
        
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        self.assertIsNone(result.article)

class TestAIsummarizer(unittest.TestCase):
    """Test cases for AIsummarizer."""
    
    def setUp(self):
        # Only run if API key is available
        if not settings.OPENAI_API_KEY:
            self.skipTest("OpenAI API key not available")
        
        self.summarizer = AIsummarizer()
        self.sample_article = Article(
            title="Sample News Article",
            body="This is a sample news article about technology. " * 20,
            source_url="https://example.com/article"
        )
    
    def test_prepare_content(self):
        """Test content preparation."""
        content = self.summarizer._prepare_content(self.sample_article)
        
        self.assertIsInstance(content, str)
        self.assertIn("Title:", content)
        self.assertIn("Content:", content)
        self.assertIn(self.sample_article.title, content)
    
    def test_validate_summary(self):
        """Test summary validation."""
        long_summary = "This is a very long summary. " * 50
        validated = self.summarizer._validate_summary(long_summary)
        
        self.assertIsInstance(validated, str)
        self.assertTrue(len(validated) <= settings.MAX_SUMMARY_LENGTH * 2)
    
    def test_validate_topics(self):
        """Test topics validation."""
        topics = ["technology", "ai", "machine learning", "news", "innovation", "extra"]
        validated = self.summarizer._validate_topics(topics)
        
        self.assertIsInstance(validated, list)
        self.assertTrue(len(validated) <= settings.MAX_TOPICS)
        self.assertTrue(all(isinstance(topic, str) for topic in validated))

class TestVectorStorage(unittest.TestCase):
    """Test cases for VectorStorage."""
    
    def setUp(self):
        # Use a test-specific storage directory
        test_dir = "./test_data/chromadb"
        os.makedirs(test_dir, exist_ok=True)
        
        # Temporarily override settings for testing
        original_dir = settings.CHROMADB_PERSIST_DIRECTORY
        settings.CHROMADB_PERSIST_DIRECTORY = test_dir
        
        self.storage = VectorStorage()
        self.sample_article = Article(
            title="Test Article",
            body="This is a test article for unit testing.",
            source_url="https://test.com/article",
            summary="Test summary",
            topics=["test", "article", "unit testing"]
        )
        
        # Restore original setting
        settings.CHROMADB_PERSIST_DIRECTORY = original_dir
    
    def test_generate_article_id(self):
        """Test article ID generation."""
        article_id = self.storage._generate_article_id(self.sample_article)
        
        self.assertIsInstance(article_id, str)
        self.assertTrue(article_id.startswith("article_"))
    
    def test_prepare_document_text(self):
        """Test document text preparation."""
        doc_text = self.storage._prepare_document_text(self.sample_article)
        
        self.assertIsInstance(doc_text, str)
        self.assertIn(self.sample_article.title, doc_text)
        self.assertIn(self.sample_article.summary, doc_text)
    
    def test_prepare_metadata(self):
        """Test metadata preparation."""
        metadata = self.storage._prepare_metadata(self.sample_article)
        
        self.assertIsInstance(metadata, dict)
        self.assertIn('title', metadata)
        self.assertIn('source_url', metadata)
        self.assertIn('topics', metadata)

class TestModels(unittest.TestCase):
    """Test cases for data models."""
    
    def test_article_creation(self):
        """Test Article model creation."""
        article = Article(
            title="Test Title",
            body="Test body content",
            source_url="https://test.com"
        )
        
        self.assertEqual(article.title, "Test Title")
        self.assertEqual(article.body, "Test body content")
        self.assertEqual(article.source_url, "https://test.com")
        self.assertIsNotNone(article.extracted_at)
        self.assertIsInstance(article.metadata, dict)
        self.assertIsInstance(article.topics, list)
    
    def test_article_to_dict(self):
        """Test Article to dictionary conversion."""
        article = Article(
            title="Test Title",
            body="Test body",
            source_url="https://test.com",
            summary="Test summary",
            topics=["test", "article"]
        )
        
        article_dict = article.to_dict()
        
        self.assertIsInstance(article_dict, dict)
        self.assertEqual(article_dict['title'], "Test Title")
        self.assertEqual(article_dict['topics'], ["test", "article"])
    
    def test_article_from_dict(self):
        """Test Article from dictionary creation."""
        article_data = {
            'title': "Test Title",
            'body': "Test body",
            'source_url': "https://test.com",
            'summary': "Test summary",
            'topics': ["test"]
        }
        
        article = Article.from_dict(article_data)
        
        self.assertEqual(article.title, "Test Title")
        self.assertEqual(article.topics, ["test"])

def run_tests():
    """Run all tests."""
    print("🧪 Running AI News Summarizer Tests")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Create test suite
    test_classes = [
        TestNewsExtractor,
        TestAIsummarizer,
        TestVectorStorage,
        TestModels
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, failure in result.failures:
            print(f"  - {test}: {failure}")
    
    if result.errors:
        print("\n🔥 Errors:")
        for test, error in result.errors:
            print(f"  - {test}: {error}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
