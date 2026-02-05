"""
PubMed Evidence Retrieval
Searches scientific literature to support medical answers
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
import time
import requests

logger = logging.getLogger(__name__)

# Try to import Bio.Entrez
try:
    from Bio import Entrez
    BIO_AVAILABLE = True
except ImportError:
    logger.warning("Biopython not available, using REST API fallback")
    BIO_AVAILABLE = False


@dataclass
class PubMedArticle:
    """PubMed article metadata"""
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    year: int
    relevance_score: float
    url: str


class PubMedRetriever:
    """
    PubMed evidence retrieval using Entrez E-utilities API
    
    Usage:
        retriever = PubMedRetriever(email="your@email.com")
        articles = retriever.search("diabetes neuropathy treatment", max_results=10)
    """
    
    def __init__(
        self,
        email: str = "medgemma@example.com",
        api_key: Optional[str] = None,
        max_retries: int = 3
    ):
        self.email = email
        self.api_key = api_key
        self.max_retries = max_retries
        
        if BIO_AVAILABLE:
            Entrez.email = email
            if api_key:
                Entrez.api_key = api_key
        
        # Base URL for REST API fallback
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    def search(
        self,
        query: str,
        max_results: int = 10,
        min_year: Optional[int] = 2015
    ) -> List[PubMedArticle]:
        """
        Search PubMed for articles
        
        Args:
            query: Search query
            max_results: Maximum number of results
            min_year: Minimum publication year
        
        Returns:
            List of PubMedArticle objects
        """
        
        try:
            # Build search query with filters
            full_query = query
            if min_year:
                full_query += f" AND {min_year}:3000[pdat]"
            
            logger.info(f"Searching PubMed: {full_query}")
            
            if BIO_AVAILABLE:
                return self._search_entrez(full_query, max_results)
            else:
                return self._search_rest(full_query, max_results)
        
        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            return []
    
    def _search_entrez(self, query: str, max_results: int) -> List[PubMedArticle]:
        """Search using Bio.Entrez"""
        
        # Search for PMIDs
        handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_results,
            sort="relevance"
        )
        search_results = Entrez.read(handle)
        handle.close()
        
        pmids = search_results["IdList"]
        
        if not pmids:
            return []
        
        # Fetch article details
        handle = Entrez.efetch(
            db="pubmed",
            id=pmids,
            rettype="abstract",
            retmode="xml"
        )
        articles_xml = Entrez.read(handle)
        handle.close()
        
        # Parse articles
        articles = []
        for i, article_data in enumerate(articles_xml["PubmedArticle"]):
            article = self._parse_article_entrez(article_data, i, len(pmids))
            if article:
                articles.append(article)
        
        return articles
    
    def _parse_article_entrez(self, article_data, rank: int, total: int) -> Optional[PubMedArticle]:
        """Parse article from Entrez XML"""
        
        try:
            article = article_data["MedlineCitation"]["Article"]
            
            # Extract PMID
            pmid = str(article_data["MedlineCitation"]["PMID"])
            
            # Extract title
            title = article.get("ArticleTitle", "")
            
            # Extract abstract
            abstract_parts = article.get("Abstract", {}).get("AbstractText", [])
            if isinstance(abstract_parts, list):
                abstract = " ".join([str(part) for part in abstract_parts])
            else:
                abstract = str(abstract_parts)
            
            # Extract authors
            authors = []
            author_list = article.get("AuthorList", [])
            for author in author_list[:3]:  # First 3 authors
                if "LastName" in author and "Initials" in author:
                    authors.append(f"{author['LastName']} {author['Initials']}")
            
            # Extract journal
            journal = article.get("Journal", {}).get("Title", "")
            
            # Extract year
            pub_date = article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
            year = int(pub_date.get("Year", 0))
            
            # Calculate relevance score (based on rank)
            relevance_score = 1.0 - (rank / total)
            
            return PubMedArticle(
                pmid=pmid,
                title=title,
                abstract=abstract,
                authors=authors,
                journal=journal,
                year=year,
                relevance_score=relevance_score,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            )
        
        except Exception as e:
            logger.error(f"Failed to parse article: {e}")
            return None
    
    def _search_rest(self, query: str, max_results: int) -> List[PubMedArticle]:
        """Search using REST API fallback"""
        
        try:
            # Search for PMIDs
            search_url = f"{self.base_url}/esearch.fcgi"
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json"
            }
            
            response = requests.get(search_url, params=search_params, timeout=10)
            response.raise_for_status()
            search_data = response.json()
            
            pmids = search_data.get("esearchresult", {}).get("idlist", [])
            
            if not pmids:
                return []
            
            # Fetch article summaries
            summary_url = f"{self.base_url}/esummary.fcgi"
            summary_params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "json"
            }
            
            response = requests.get(summary_url, params=summary_params, timeout=10)
            response.raise_for_status()
            summary_data = response.json()
            
            # Parse articles
            articles = []
            for i, pmid in enumerate(pmids):
                article_data = summary_data.get("result", {}).get(pmid, {})
                article = self._parse_article_rest(pmid, article_data, i, len(pmids))
                if article:
                    articles.append(article)
            
            return articles
        
        except Exception as e:
            logger.error(f"REST API search failed: {e}")
            return []
    
    def _parse_article_rest(self, pmid: str, article_data: Dict, rank: int, total: int) -> Optional[PubMedArticle]:
        """Parse article from REST API response"""
        
        try:
            title = article_data.get("title", "")
            
            # Abstract not available in summary - would need separate fetch
            abstract = "(Abstract not available via REST API - use Bio.Entrez for full abstracts)"
            
            # Authors
            authors = []
            author_list = article_data.get("authors", [])
            for author in author_list[:3]:
                authors.append(author.get("name", ""))
            
            # Journal
            journal = article_data.get("source", "")
            
            # Year
            pub_date = article_data.get("pubdate", "")
            year = 0
            if pub_date:
                try:
                    year = int(pub_date.split()[0])
                except:
                    pass
            
            # Relevance score
            relevance_score = 1.0 - (rank / total)
            
            return PubMedArticle(
                pmid=pmid,
                title=title,
                abstract=abstract,
                authors=authors,
                journal=journal,
                year=year,
                relevance_score=relevance_score,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            )
        
        except Exception as e:
            logger.error(f"Failed to parse REST article: {e}")
            return None
    
    def search_for_relationship(
        self,
        entity1: str,
        entity2: str,
        relationship_type: str,
        max_results: int = 5
    ) -> List[PubMedArticle]:
        """
        Search for evidence supporting a relationship between entities
        
        Args:
            entity1: First entity (e.g., "diabetes")
            entity2: Second entity (e.g., "neuropathy")
            relationship_type: Type of relationship (e.g., "causes")
        
        Returns:
            List of supporting articles
        """
        
        # Build query based on relationship type
        if relationship_type in ["causes", "leads_to"]:
            query = f"{entity1} AND {entity2} AND (etiology OR pathogenesis OR cause)"
        elif relationship_type in ["treats", "therapy"]:
            query = f"{entity1} AND {entity2} AND (treatment OR therapy OR management)"
        elif relationship_type == "diagnoses":
            query = f"{entity1} AND {entity2} AND (diagnosis OR diagnostic)"
        else:
            query = f"{entity1} AND {entity2}"
        
        return self.search(query, max_results=max_results)


class EvidenceRetriever:
    """
    High-level evidence retrieval with caching and ranking
    """
    
    def __init__(
        self,
        pubmed_email: str = "medgemma@example.com",
        pubmed_api_key: Optional[str] = None,
        cache_size: int = 1000
    ):
        self.pubmed = PubMedRetriever(email=pubmed_email, api_key=pubmed_api_key)
        
        # Simple cache
        self.cache: Dict[str, List[PubMedArticle]] = {}
        self.cache_size = cache_size
    
    def retrieve(
        self,
        query: str,
        max_results: int = 10,
        use_cache: bool = True
    ) -> List[PubMedArticle]:
        """Retrieve evidence with caching"""
        
        cache_key = f"{query}:{max_results}"
        
        if use_cache and cache_key in self.cache:
            logger.info(f"Cache hit for: {query}")
            return self.cache[cache_key]
        
        # Search PubMed
        articles = self.pubmed.search(query, max_results=max_results)
        
        # Update cache
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            self.cache.pop(next(iter(self.cache)))
        
        self.cache[cache_key] = articles
        
        return articles
    
    def retrieve_for_entities(
        self,
        entities: List[str],
        max_results_per_entity: int = 5
    ) -> Dict[str, List[PubMedArticle]]:
        """Retrieve evidence for multiple entities"""
        
        results = {}
        
        for entity in entities:
            articles = self.retrieve(entity, max_results=max_results_per_entity)
            results[entity] = articles
        
        return results
    
    def format_evidence(self, articles: List[PubMedArticle], max_articles: int = 3) -> str:
        """Format articles as evidence context"""
        
        if not articles:
            return "No supporting evidence found."
        
        lines = ["Supporting Evidence:"]
        
        for i, article in enumerate(articles[:max_articles]):
            lines.append(f"\n{i+1}. {article.title}")
            lines.append(f"   Authors: {', '.join(article.authors)}")
            lines.append(f"   Journal: {article.journal} ({article.year})")
            lines.append(f"   PMID: {article.pmid}")
            
            if article.abstract and len(article.abstract) > 50:
                # Truncate abstract
                abstract_preview = article.abstract[:200] + "..."
                lines.append(f"   Abstract: {abstract_preview}")
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Demo
    print("PubMed Evidence Retrieval Demo")
    print("="*60)
    
    # Initialize retriever
    retriever = EvidenceRetriever()
    
    # Test queries
    test_queries = [
        "diabetes neuropathy pathogenesis",
        "metformin mechanism diabetes treatment",
        "diabetic retinopathy risk factors"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-"*60)
        
        articles = retriever.retrieve(query, max_results=5)
        
        print(f"Found {len(articles)} articles:")
        for article in articles[:3]:
            print(f"\n  Title: {article.title}")
            print(f"  Authors: {', '.join(article.authors)}")
            print(f"  Journal: {article.journal} ({article.year})")
            print(f"  PMID: {article.pmid}")
            print(f"  Relevance: {article.relevance_score:.2f}")
    
    print("\n" + "="*60)
    print("Evidence Retrieval Demo Complete")
