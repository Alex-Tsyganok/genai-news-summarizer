"""
AI-powered summarization and topic detection using OpenAI GPT.
"""
import openai
from typing import List, Tuple, Optional
import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ..models import Article
from config import settings, logger

class AIsummarizer:
    """
    AI-powered article summarizer using OpenAI GPT models.
    
    Generates concise summaries and identifies key topics from article content.
    """
    
    def __init__(self, openai_config=None):
        """
        Initialize the AI summarizer.
        
        Args:
            openai_config: Optional custom OpenAI configuration
        """
        self.openai_config = openai_config or settings.get_openai_config()
        self.client = openai.OpenAI(api_key=self.openai_config['api_key'])
        self.chat_model = ChatOpenAI(
            api_key=self.openai_config['api_key'],
            model=self.openai_config['model'],
            temperature=0.3
        )
    
    def summarize_article(self, article: Article) -> Tuple[str, List[str]]:
        """
        Generate summary and topics for an article.
        
        Uses OpenAI GPT for high-quality AI summarization by default. When USE_FALLBACK_ONLY
        environment variable is enabled, bypasses AI calls entirely for development/testing
        purposes and uses basic text processing instead.
        
        Args:
            article: Article object to summarize
            
        Returns:
            Tuple of (summary, topics_list)
            
        Note:
            Fallback-only mode is intended for:
            - Development and testing without API costs
            - CI/CD pipelines that don't require API keys
            - Offline environments or demo scenarios
            - Cost control during development
        """
        try:
            logger.info(f"Summarizing article: {article.title[:50]}...")
            
            # Check if fallback-only mode is enabled (for development/testing purposes)
            # This bypasses AI calls entirely to avoid API costs during development
            if settings.USE_FALLBACK_ONLY:
                logger.info("Using fallback-only mode (AI summarization disabled)")
                fallback_summary = self._create_fallback_summary(article)
                fallback_topics = self._extract_fallback_topics(article)
                return fallback_summary, fallback_topics
            
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
            Cleaned content (no aggressive truncation)
        """
        # Combine title and body (do not aggressively truncate here)
        content = f"Title: {article.title}\n\nContent: {article.body}"
        return content
    
    def _generate_summary_and_topics(self, content: str) -> dict:
        """
        Generate summary and topics using OpenAI API.
        
        Args:
            content: Article content to process
            
        Returns:
            Dictionary with summary and topics
        """
        # If content fits within one chunk, summarize directly; otherwise, do iterative reduction
        chunk_size = getattr(settings, "SUMMARY_CHUNK_SIZE", 8000)
        if len(content) <= chunk_size:
            system_prompt = f"""You are a news analyst. Extract:

1. Summary ({settings.SUMMARY_SENTENCES}, max {settings.SUMMARY_WORD_LIMIT} words)
2. Topics (2-4 keywords)

Return strict JSON only:
{{
  "summary": "Brief summary...",
  "topics": ["topic1", "topic2"]
}}"""

            user_prompt = f"Analyze the following article and return JSON only.\n\n{content[:chunk_size]}"

            try:
                response = self.client.chat.completions.create(
                    model=self.openai_config['model'],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=350  # Increased from 200 to accommodate 5-10 sentence summaries
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

        # Iterative summarization path for long content
        try:
            chunks = self._chunk_text(content, chunk_size=chunk_size, overlap=200)
            logger.debug(f"Summarizing long article in {len(chunks)} chunks (size ~{chunk_size})")
            partial_summaries: List[str] = []
            for idx, chunk in enumerate(chunks, start=1):
                try:
                    s = self._summarize_chunk(chunk, idx, len(chunks))
                    partial_summaries.append(s)
                except Exception as ce:
                    logger.warning(f"Chunk {idx}/{len(chunks)} summarization failed: {ce}")

            if not partial_summaries:
                raise RuntimeError("All chunk summarizations failed")

            # Combine partial summaries into final summary and topics
            combined = self._finalize_summary_and_topics(partial_summaries)
            return combined
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response during combine: {e}")
            # Try to recover from malformed final response
            return self._parse_malformed_response(str(e))
        except Exception as e:
            logger.error(f"Iterative summarization failed: {e}")
            raise

    def _chunk_text(self, text: str, chunk_size: int, overlap: int = 200) -> List[str]:
        """Split text into overlapping character chunks, trying to break at paragraph boundaries.

        Args:
            text: Full article text (including title/body)
            chunk_size: Desired max characters per chunk
            overlap: Characters of overlap between consecutive chunks

        Returns:
            List of chunk strings
        """
        if chunk_size <= 0:
            return [text]

        # Prefer splitting by paragraph markers to reduce mid-sentence cuts
        paragraphs = re.split(r"\n\s*\n", text)
        chunks: List[str] = []
        current = ""
        for p in paragraphs:
            candidate = (current + "\n\n" + p).strip() if current else p
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # If a single paragraph is too large, hard-split it
                while len(p) > chunk_size:
                    chunks.append(p[:chunk_size])
                    p = p[chunk_size - overlap :]
                current = p
        if current:
            chunks.append(current)

        # Add overlap between chunks to preserve context
        if overlap > 0 and len(chunks) > 1:
            with_overlap: List[str] = []
            for i, ch in enumerate(chunks):
                prefix = chunks[i - 1][-overlap:] + "\n" if i > 0 else ""
                with_overlap.append(prefix + ch)
            chunks = with_overlap

        return chunks

    def _summarize_chunk(self, chunk: str, idx: int, total: int) -> str:
        """Summarize a single chunk using configurable summary settings."""
        system_prompt = (
            f"You are a helpful news analyst. Summarize the given chunk into {settings.SUMMARY_SENTENCES} "
            f"(max {settings.SUMMARY_WORD_LIMIT} words) capturing only the essential facts. Return plain text."
        )
        user_prompt = f"Chunk {idx}/{total}:\n\n{chunk}"
        response = self.client.chat.completions.create(
            model=self.openai_config['model'],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=300,  # Increased from 180 to accommodate longer summaries
        )
        text = response.choices[0].message.content or ""
        return text.strip()

    def _finalize_summary_and_topics(self, partial_summaries: List[str]) -> dict:
        """Reduce partial chunk summaries into final summary and topics (JSON)."""
        system_prompt = f"""You are a news analyst. Given multiple partial summaries of a single article,
produce a final result with:
1) Summary: {settings.SUMMARY_SENTENCES}, max {settings.SUMMARY_WORD_LIMIT} words
2) Topics: 2-4 keywords

Return strict JSON only:
{{
  "summary": "...",
  "topics": ["..."]
}}
"""
        joined = "\n- " + "\n- ".join(s for s in partial_summaries if s)
        user_prompt = f"Combine the following partial summaries into final JSON.\n{joined}"
        response = self.client.chat.completions.create(
            model=self.openai_config['model'],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=400,  # Increased from 220 to accommodate longer final summaries
        )
        content_text = response.choices[0].message.content or "{}"
        return json.loads(content_text)
    
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
    
    def _extract_fallback_topics(self, article: Article) -> List[str]:
        """
        Extract basic topics from article when AI is disabled.
        
        This method provides simple keyword extraction for development and testing
        purposes when USE_FALLBACK_ONLY mode is enabled. It uses basic text processing
        techniques instead of AI-powered topic identification.
        
        Args:
            article: Article to extract topics from
            
        Returns:
            List of basic topics extracted from title and content
            
        Note:
            This is a simplified approach for testing/development. Production usage
            should rely on AI-powered topic extraction for better quality results.
        """
        import re
        from collections import Counter
        
        # Combine title and first paragraph for topic extraction
        text = f"{article.title} {article.body[:500]}".lower()
        
        # Remove common stop words and extract meaningful words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'our', 'their',
            'said', 'says', 'can', 'one', 'two', 'new', 'also', 'more', 'than', 'only',
            'after', 'before', 'when', 'where', 'why', 'how', 'what', 'who', 'which'
        }
        
        # Extract words (3+ characters, alphabetic)
        words = re.findall(r'\b[a-z]{3,}\b', text)
        meaningful_words = [word for word in words if word not in stop_words]
        
        # Get most common words as topics
        word_counts = Counter(meaningful_words)
        topics = [word for word, count in word_counts.most_common(5) if count > 1]
        
        # If no good topics found, try to extract from title
        if not topics:
            title_words = re.findall(r'\b[a-z]{3,}\b', article.title.lower())
            topics = [word for word in title_words if word not in stop_words][:3]
        
        return topics[:settings.MAX_TOPICS]
    
    def batch_summarize(self, articles: List[Article]) -> List[Tuple[str, List[str]]]:
        """
        Summarize multiple articles in batch.
        
        Processes articles using AI summarization by default, or fallback mode when
        USE_FALLBACK_ONLY is enabled for development/testing purposes.
        
        Args:
            articles: List of articles to summarize
            
        Returns:
            List of (summary, topics) tuples
            
        Note:
            In fallback-only mode, all articles are processed using basic text extraction
            without any OpenAI API calls, making it suitable for testing and development.
        """
        results = []
        
        # Check if fallback-only mode is enabled (development/testing mode)
        if settings.USE_FALLBACK_ONLY:
            logger.info(f"Batch processing {len(articles)} articles in fallback-only mode")
            for article in articles:
                fallback_summary = self._create_fallback_summary(article)
                fallback_topics = self._extract_fallback_topics(article)
                results.append((fallback_summary, fallback_topics))
            return results
        
        for article in articles:
            try:
                summary, topics = self.summarize_article(article)
                results.append((summary, topics))
            except Exception as e:
                logger.error(f"Failed to summarize article {article.title}: {e}")
                fallback_summary = self._create_fallback_summary(article)
                results.append((fallback_summary, []))
        
        return results
