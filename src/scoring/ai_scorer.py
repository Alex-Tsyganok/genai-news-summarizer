"""
AI-based confidence scoring for news articles.
"""
import json
from typing import Dict, Any, Tuple

import openai
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from ..models import Article, ProcessingResult
from config import settings, logger

class AIConfidenceScorer:
    """
    Uses AI to evaluate if content is a legitimate news article.
    """
    
    def __init__(self, openai_config=None, min_confidence_score=None):
        """
        Initialize the AI confidence scorer.
        
        Args:
            openai_config: Optional custom OpenAI configuration
            min_confidence_score: Optional minimum confidence score threshold
        """
        self.openai_config = openai_config or settings.get_openai_config()
        self.min_confidence_score = min_confidence_score or settings.MIN_CONFIDENCE_SCORE
        openai.api_key = self.openai_config['api_key']
        
    def score_article(self, article: Article) -> ProcessingResult:
        """
        Calculate AI-based confidence score for an article.
        
        Args:
            article: Article to evaluate
            
        Returns:
            ProcessingResult containing confidence score and analysis
        """
        try:
            # Prepare article content for analysis
            content_for_analysis = self._prepare_content(article)
            
            # Get AI analysis
            score, analysis = self._analyze_with_ai(content_for_analysis)
            
            # Add results to article metadata
            article.metadata['ai_confidence_score'] = score
            article.metadata['ai_analysis'] = analysis
            
            return ProcessingResult(
                success=True,
                article=article,
                processing_time=0
            )
            
        except Exception as e:
            error_msg = f"Failed to calculate AI confidence score: {str(e)}"
            logger.error(error_msg)
            return ProcessingResult(
                success=False,
                error=error_msg,
                processing_time=0
            )
    
    def _prepare_content(self, article: Article) -> str:
        """Prepare article content for AI analysis."""
        # Combine relevant article parts for analysis
        content = {
            "title": article.title,
            "content_excerpt": article.body[:1000],  # First 1000 chars
            "metadata": {
                k: v for k, v in (article.metadata or {}).items()
                if k in ['authors', 'publish_date', 'extraction_method', 'source_url']
            }
        }
        return json.dumps(content)
    
    def _analyze_with_ai(self, content: str) -> Tuple[float, Dict[str, Any]]:
        """
        Use OpenAI with LangChain to analyze article authenticity.
        
        Returns:
            Tuple of (confidence_score, detailed_analysis)
        """
        # Create LangChain prompt template
        system_prompt = """You are an expert news article validator with STRICT CRITERIA.
            Analyze the provided content and determine if it's a legitimate news article.

            SCORING CRITERIA:
            1. Content Type Determination:
               DOCUMENTATION (-0.8):
               - Technical documentation, tutorials
               - API documentation, README files
               - Installation guides, how-to content
               - Must be labeled as "documentation"
               
               FORUM/Q&A (-0.7):
               - Stack Overflow style posts
               - Forum discussions
               - Technical Q&A content
               - Must be labeled as "forum_post"
               
               PRODUCT/MARKETING (-0.6):
               - Product listings
               - Marketing materials
               - Promotional content
               - Must be labeled as "product_page"
               
               BLOG (-0.3 to 0):
               - Technical blogs: -0.3
               - Educational blogs: -0.2
               - News-focused blogs: 0
               - Must be labeled appropriately
            
            2. Professional News Elements:
               Required for News Classification:
               - Clear headline/title (+0.2)
               - Author attribution (+0.1)
               - Publication date (+0.1)
               - News-worthy event/topic (+0.3)
               - Objective reporting style (+0.2)
               - Multiple sources/quotes (+0.1)
            
            3. Automatic Rejection Triggers:
               - Code snippets/commands = documentation
               - Q&A format = forum_post
               - Product specs = product_page
               - Installation steps = documentation
               
            Return a JSON response with:
            {{
                "confidence_score": float (0-1),
                "is_news_article": boolean,
                "content_type": "news_article" | "blog_post" | "documentation" | "forum_post" | "product_page" | "other",
                "analysis": {{
                    "style_score": float (0-1),
                    "content_quality_score": float (0-1),
                    "structure_score": float (0-1),
                    "news_relevance": float (0-1),
                    "reasons": [str],  # Why it was accepted/rejected
                    "flags": [str]     # Content type warnings
                }}
            }}
            
            STRICT SCORING RULES:
            - Documentation MUST score <= 0.4
            - Technical blogs without news focus MUST score <= 0.5
            - Forum/Q&A content MUST score <= 0.3
            - Product/marketing pages MUST score <= 0.4
            - Pure tutorials MUST score <= 0.3
            - Only news articles and news-focused blogs can score > 0.6
            
            On content_type:
            - Be VERY aggressive in detecting documentation
            - Any how-to content = documentation
            - Any technical guides = documentation
            - Any API/SDK docs = documentation"""
            
        human_prompt = "Analyze this content: {input_content}"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])

        # Create chain with LangChain
        model = ChatOpenAI(
            model=self.openai_config['model'],
            temperature=0.1,  # Low temperature for consistent analysis
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        chain = prompt | model | StrOutputParser()
        
        try:
            # Execute chain with tracing
            response = chain.invoke({"input_content": content})
            
            # Parse response
            result = json.loads(response)
            confidence_score = result['confidence_score']
            
            return confidence_score, result['analysis']
            
        except Exception as e:
            logger.error(f"LangChain/OpenAI API error: {str(e)}")
            # Return conservative default scores
            return 0.0, {
                "error": str(e),
                "style_score": 0.0,
                "content_quality_score": 0.0,
                "structure_score": 0.0,
                "reasons": ["AI analysis failed"],
                "flags": ["analysis_error"]
            }
