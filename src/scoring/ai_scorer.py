"""
AI-based confidence scoring for news articles.
"""
import json
from typing import Dict, Any, Tuple

import openai
from ..models import Article, ProcessingResult
from config import settings, logger

class AIConfidenceScorer:
    """
    Uses AI to evaluate if content is a legitimate news article.
    """
    
    def __init__(self):
        """Initialize the AI confidence scorer."""
        self.openai_config = settings.get_openai_config()
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
                if k in ['authors', 'publish_date', 'extraction_method']
            }
        }
        return json.dumps(content)
    
    def _analyze_with_ai(self, content: str) -> Tuple[float, Dict[str, Any]]:
        """
        Use OpenAI to analyze article authenticity.
        
        Returns:
            Tuple of (confidence_score, detailed_analysis)
        """
        system_prompt = """You are an expert news article validator. 
        Analyze the provided content and determine if it's a legitimate news article.
        Consider factors like:
        - Professional writing style and structure
        - News-worthy content
        - Objectivity and lack of promotional content
        - Presence of typical news article elements
        
        Return a JSON response with:
        {
            "confidence_score": float (0-1),
            "is_news_article": boolean,
            "analysis": {
                "style_score": float (0-1),
                "content_quality_score": float (0-1),
                "structure_score": float (0-1),
                "reasons": [str],
                "flags": [str]
            }
        }"""
        
        try:
            response = openai.chat.completions.create(
                model=self.openai_config['model'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this content: {content}"}
                ],
                temperature=0.1,  # Low temperature for consistent analysis
                response_format={"type": "json_object"}
            )
            
            # Parse response
            result = json.loads(response.choices[0].message.content)
            confidence_score = result['confidence_score']
            
            return confidence_score, result['analysis']
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            # Return conservative default scores
            return 0.0, {
                "error": str(e),
                "style_score": 0.0,
                "content_quality_score": 0.0,
                "structure_score": 0.0,
                "reasons": ["AI analysis failed"],
                "flags": ["analysis_error"]
            }
