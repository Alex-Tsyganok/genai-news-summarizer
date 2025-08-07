"""
AI-powered summarization and topic detection using OpenAI GPT.
"""
import openai
from typing import List, Tuple, Optional
import json
import re
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from ..models import Article
from config import settings, logger

class AIsummarizer:
    """
    AI-powered article summarizer using OpenAI GPT models.
    
    Generates concise summaries and identifies key topics from article content.
    """
    
    def __init__(self):
        """Initialize the AI summarizer."""
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        self.chat_model = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
            temperature=0.3
        )
    
    def summarize_article(self, article: Article) -> Tuple[str, List[str]]:
        """
        Generate summary and topics for an article.
        
        Args:
            article: Article object to summarize
            
        Returns:
            Tuple of (summary, topics_list)
        """
        try:
            logger.info(f"Summarizing article: {article.title[:50]}...")
            
            # Prepare the content for summarization
            content = self._prepare_content(article)
            
            # Generate summary and topics in one call for efficiency
            result = self._generate_summary_and_topics(content)
            
            summary = result.get('summary', '').strip()
            topics = result.get('topics', [])
            
            # Validate and clean results
            summary = self._validate_summary(summary)
            topics = self._validate_topics(topics)
            
            logger.info(f"Successfully summarized article with {len(topics)} topics")
            return summary, topics
            
        except Exception as e:
            logger.error(f"Failed to summarize article: {str(e)}")
            # Return fallback summary
            return self._create_fallback_summary(article), []
    
    def _prepare_content(self, article: Article) -> str:
        """
        Prepare article content for summarization.
        
        Args:
            article: Article to prepare
            
        Returns:
            Cleaned and truncated content
        """
        # Combine title and body
        content = f"Title: {article.title}\n\nContent: {article.body}"
        
        # Truncate if too long (GPT token limits)
        max_chars = 8000  # Conservative estimate for token limits
        if len(content) > max_chars:
            content = content[:max_chars] + "..."
        
        return content
    
    def _generate_summary_and_topics(self, content: str) -> dict:
        """
        Generate summary and topics using OpenAI API.
        
        Args:
            content: Article content to process
            
        Returns:
            Dictionary with summary and topics
        """
        system_prompt = """You are an expert news analyst. Analyze the provided news article and extract:

1. A concise summary (2-3 sentences, max 200 words) that captures the key points
2. A list of 2-5 main topics/keywords that represent the article's themes

Return your response as a JSON object with this exact format:
{
  "summary": "Your summary here...",
  "topics": ["topic1", "topic2", "topic3"]
}

Focus on:
- Key facts and developments
- Main people, organizations, or entities involved
- Important implications or outcomes
- Clear, actionable topics that would help in semantic search"""

        user_prompt = f"Please analyze this news article:\n\n{content}"
        
        try:
            response = self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            content_text = response.choices[0].message.content
            
            # Parse JSON response
            result = json.loads(content_text)
            return result
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            # Try to extract summary and topics from malformed response
            return self._parse_malformed_response(content_text)
        
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise
    
    def _parse_malformed_response(self, response_text: str) -> dict:
        """
        Parse malformed response that might not be valid JSON.
        
        Args:
            response_text: Raw response text from API
            
        Returns:
            Dictionary with extracted summary and topics
        """
        result = {"summary": "", "topics": []}
        
        # Try to extract summary
        summary_match = re.search(r'"summary"\s*:\s*"([^"]*)"', response_text, re.DOTALL)
        if summary_match:
            result["summary"] = summary_match.group(1).strip()
        
        # Try to extract topics
        topics_match = re.search(r'"topics"\s*:\s*\[(.*?)\]', response_text, re.DOTALL)
        if topics_match:
            topics_text = topics_match.group(1)
            topics = re.findall(r'"([^"]*)"', topics_text)
            result["topics"] = [topic.strip() for topic in topics if topic.strip()]
        
        return result
    
    def _validate_summary(self, summary: str) -> str:
        """
        Validate and clean the generated summary.
        
        Args:
            summary: Generated summary text
            
        Returns:
            Cleaned and validated summary
        """
        if not summary:
            return "Summary not available."
        
        # Ensure reasonable length
        if len(summary) > settings.MAX_SUMMARY_LENGTH * 2:
            # Truncate at sentence boundary
            sentences = summary.split('. ')
            truncated = []
            total_length = 0
            
            for sentence in sentences:
                if total_length + len(sentence) <= settings.MAX_SUMMARY_LENGTH:
                    truncated.append(sentence)
                    total_length += len(sentence)
                else:
                    break
            
            summary = '. '.join(truncated)
            if not summary.endswith('.'):
                summary += '.'
        
        return summary.strip()
    
    def _validate_topics(self, topics: List[str]) -> List[str]:
        """
        Validate and clean the generated topics.
        
        Args:
            topics: List of generated topics
            
        Returns:
            Cleaned and validated topics list
        """
        if not topics:
            return []
        
        # Clean and filter topics
        cleaned_topics = []
        for topic in topics:
            if isinstance(topic, str):
                cleaned_topic = topic.strip().lower()
                if cleaned_topic and len(cleaned_topic) > 2:
                    cleaned_topics.append(cleaned_topic)
        
        # Remove duplicates while preserving order
        unique_topics = []
        seen = set()
        for topic in cleaned_topics:
            if topic not in seen:
                unique_topics.append(topic)
                seen.add(topic)
        
        # Limit number of topics
        return unique_topics[:settings.MAX_TOPICS]
    
    def _create_fallback_summary(self, article: Article) -> str:
        """
        Create a fallback summary when AI summarization fails.
        
        Args:
            article: Article to create fallback for
            
        Returns:
            Fallback summary text
        """
        # Use first few sentences of the article
        sentences = article.body.split('. ')[:3]
        summary = '. '.join(sentences)
        
        if len(summary) > settings.MAX_SUMMARY_LENGTH:
            summary = summary[:settings.MAX_SUMMARY_LENGTH] + "..."
        
        return summary
    
    def batch_summarize(self, articles: List[Article]) -> List[Tuple[str, List[str]]]:
        """
        Summarize multiple articles in batch.
        
        Args:
            articles: List of articles to summarize
            
        Returns:
            List of (summary, topics) tuples
        """
        results = []
        
        for article in articles:
            try:
                summary, topics = self.summarize_article(article)
                results.append((summary, topics))
            except Exception as e:
                logger.error(f"Failed to summarize article {article.title}: {e}")
                fallback_summary = self._create_fallback_summary(article)
                results.append((fallback_summary, []))
        
        return results
