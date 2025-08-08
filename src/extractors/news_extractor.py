"""
News article extraction and parsing functionality.
"""
import requests
from newspaper import Article as NewspaperArticle
from bs4 import BeautifulSoup
from typing import Optional, Dict, Any
import time
from urllib.parse import urljoin, urlparse
import re

from ..models import Article, ProcessingResult
from config import settings, logger

class NewsExtractor:
    """
    News article extractor that handles multiple extraction methods.
    
    Uses newspaper3k as primary method with BeautifulSoup as fallback.
    """
    
    def __init__(self):
        """Initialize the news extractor."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def extract_article(self, url: str) -> ProcessingResult:
        """
        Extract article content from a URL.
        
        Args:
            url: The URL to extract content from
            
        Returns:
            ProcessingResult containing the extracted article or error
        """
        start_time = time.time()
        
        try:
            logger.info(f"Extracting article from: {url}")
            
            # First try with newspaper3k
            article = self._extract_with_newspaper(url)
            
            if not article or not article.body.strip():
                logger.warning(f"Newspaper3k failed for {url}, trying BeautifulSoup")
                article = self._extract_with_beautifulsoup(url)
            
            if not article or not article.body.strip():
                raise ValueError("Failed to extract meaningful content")
            
            processing_time = time.time() - start_time
            logger.info(f"Successfully extracted article from {url} in {processing_time:.2f}s")
            
            return ProcessingResult(
                success=True,
                article=article,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Failed to extract article from {url}: {str(e)}"
            logger.error(error_msg)
            
            return ProcessingResult(
                success=False,
                error=error_msg,
                processing_time=processing_time
            )
    
    def _extract_with_newspaper(self, url: str) -> Optional[Article]:
        """
        Extract article using newspaper3k library.
        
        Args:
            url: The URL to extract from
            
        Returns:
            Article object or None if extraction failed
        """
        try:
            article = NewspaperArticle(url)
            article.download()
            article.parse()
            
            # Basic validation
            if not article.title or not article.text:
                return None
            
            # Extract metadata
            metadata = {
                'authors': article.authors,
                'publish_date': article.publish_date.isoformat() if article.publish_date else None,
                'top_image': article.top_image,
                'extraction_method': 'newspaper3k'
            }
            
            return Article(
                title=self._clean_text(article.title),
                body=self._clean_text(article.text),
                source_url=url,
                metadata=metadata
            )
            
        except Exception as e:
            logger.debug(f"Newspaper3k extraction failed: {str(e)}")
            return None
    
    def _extract_with_beautifulsoup(self, url: str) -> Optional[Article]:
        """
        Extract article using BeautifulSoup as fallback.
        
        Args:
            url: The URL to extract from
            
        Returns:
            Article object or None if extraction failed
        """
        try:
            response = self.session.get(url, timeout=settings.REQUEST_TIMEOUT)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = self._extract_title(soup)
            if not title:
                return None
            
            # Extract body text
            body = self._extract_body(soup)
            if not body:
                return None
            
            metadata = {
                'extraction_method': 'beautifulsoup',
                'content_type': response.headers.get('content-type', ''),
                'status_code': response.status_code
            }
            
            return Article(
                title=self._clean_text(title),
                body=self._clean_text(body),
                source_url=url,
                metadata=metadata
            )
            
        except Exception as e:
            logger.debug(f"BeautifulSoup extraction failed: {str(e)}")
            return None
    
    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract title from HTML soup."""
        # Try various title selectors
        selectors = [
            'h1',
            'title',
            '[class*="title"]',
            '[class*="headline"]',
            'h2'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text().strip()
                if text and len(text) > 10:  # Reasonable title length
                    snippet = text[:100].replace("\n", " ")
                    logger.debug(f"Title extracted via selector '{selector}'; length={len(text)}; snippet={snippet!r}")
                    return text
        
        logger.debug("Title extraction failed with default selectors")
        return None
    
    def _extract_body(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract body text from HTML soup.

        Strategy:
        1) Remove non-content elements (scripts, navs, etc.).
        2) Find candidate containers via common selectors and pick the densest by
           paragraph count and text length.
        3) If none found, scan all div/section/article for densest content.
        4) Aggregate text from paragraphs and relevant headings in DOM order.
        5) Fallback to concatenating all <p> tags when needed.
        """

        # 1) Remove clearly unwanted elements early
        for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside', 'form', 'button', 'noscript']):
            element.decompose()

        def score_element(el) -> tuple[int, int]:
            ps = el.find_all('p')
            text = ' '.join(p.get_text(separator=' ', strip=True) for p in ps)
            return (len(ps), len(text))

        def pick_best(candidates) -> Optional[BeautifulSoup]:
            best = None
            best_score = (0, 0)
            for el in candidates:
                pcount, tlen = score_element(el)
                # Require some minimum content to avoid navs
                if pcount >= 3 or tlen > 800:
                    if (pcount, tlen) > best_score:
                        best = el
                        best_score = (pcount, tlen)
            return best

        # 2) Gather candidates using common selectors
        selector_list = [
            'article',
            'main',
            '[role="main"]',
            '[class*="content"]',
            '[class*="article"]',
            '[class*="post"]',
            '[class*="story"]',
            '.entry-content',
            '#content',
            '.article-body',
            '.article__content',
            '.content__article-body',
        ]
        candidates = []
        seen = set()
        for sel in selector_list:
            for el in soup.select(sel):
                # Deduplicate by object id
                key = id(el)
                if key not in seen:
                    candidates.append(el)
                    seen.add(key)

        best = pick_best(candidates)
        if best is not None:
            pcount, tlen = score_element(best)
            logger.debug(
                f"Body container selected from selectors: <{best.name} class={best.get('class')} id={best.get('id')}>; "
                f"paragraphs={pcount}; text_len={tlen}"
            )

        # 3) If nothing from selectors, scan structural blocks for densest content
        if best is None:
            block_candidates = soup.find_all(['article', 'section', 'div'])
            best = pick_best(block_candidates)
            if best is not None:
                pcount, tlen = score_element(best)
                logger.debug(
                    f"Body container selected from structural scan: <{best.name} class={best.get('class')} id={best.get('id')}>; "
                    f"paragraphs={pcount}; text_len={tlen}"
                )

        # 4) Aggregate text in order from the chosen container
        def aggregate(el) -> str:
            lines = []
            for node in el.descendants:
                if getattr(node, 'name', None) in ('p', 'li'):
                    txt = node.get_text(separator=' ', strip=True)
                    if txt:
                        lines.append(txt)
                elif getattr(node, 'name', None) in ('h1', 'h2', 'h3'):
                    # Keep important headings to preserve structure
                    txt = node.get_text(separator=' ', strip=True)
                    if txt:
                        lines.append(txt)
            text = '\n'.join(lines)
            # Normalize whitespace
            text = re.sub(r'\s+\n', '\n', text)
            text = re.sub(r'\n{3,}', '\n\n', text)
            return text.strip()

        if best is not None:
            text = aggregate(best)
            if text and len(text) > 200:
                logger.debug(
                    f"Aggregated body from selected container; length={len(text)}; approx_lines={text.count('\n') + 1}"
                )
                return text

        # 5) Fallback: concatenate all paragraph text in document order
        paragraphs = soup.find_all('p')
        text = '\n'.join([p.get_text(separator=' ', strip=True) for p in paragraphs])
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
        logger.debug(
            f"Fallback paragraph concatenation used; paragraphs={len(paragraphs)}; length={len(text)}"
        )
        return text if len(text) > 200 else None
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)\"\']+', ' ', text)
        
        # Remove multiple spaces again
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def batch_extract(self, urls: list) -> Dict[str, ProcessingResult]:
        """
        Extract articles from multiple URLs.
        
        Args:
            urls: List of URLs to extract from
            
        Returns:
            Dictionary mapping URLs to ProcessingResults
        """
        results = {}
        
        for url in urls:
            results[url] = self.extract_article(url)
            time.sleep(1)  # Be respectful to servers
        
        return results
