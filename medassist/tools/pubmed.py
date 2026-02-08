"""
PubMed API integration for medical literature retrieval
Based on AMG-RAG implementation
"""

import requests
import time
from typing import List, Optional
from xml.etree import ElementTree as ET


class PubMedSearcher:
    """
    PubMed API wrapper for medical literature search
    
    Usage:
        pubmed = PubMedSearcher(api_key="your-key")  # api_key optional
        results = pubmed.search("diabetes treatment", max_results=5)
    """
    
    def __init__(self, api_key: Optional[str] = None, email: Optional[str] = None):
        """
        Initialize PubMed searcher
        
        Args:
            api_key: NCBI API key (optional, increases rate limits)
            email: Email for NCBI tracking (optional but recommended)
        """
        self.api_key = api_key
        self.email = email
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        
    def search(self, query: str, max_results: int = 5) -> List[str]:
        """
        Search PubMed and return article abstracts
        
        Args:
            query: Medical search query
            max_results: Maximum number of articles to retrieve
            
        Returns:
            List of article abstracts
        """
        try:
            # Step 1: Search for PMIDs
            pmids = self._search_pubmed(query, max_results)
            
            if not pmids:
                return []
            
            # Step 2: Fetch article details
            abstracts = self._fetch_articles(pmids)
            
            return abstracts
            
        except Exception as e:
            print(f"PubMed search error: {e}")
            return []
    
    def _search_pubmed(self, query: str, max_results: int) -> List[str]:
        """Search PubMed for article IDs"""
        
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json"
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email
        
        try:
            response = requests.get(
                f"{self.base_url}/esearch.fcgi",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            pmids = data.get("esearchresult", {}).get("idlist", [])
            
            return pmids
            
        except Exception as e:
            print(f"PubMed search API error: {e}")
            return []
    
    def _fetch_articles(self, pmids: List[str]) -> List[str]:
        """Fetch article abstracts by PMID"""
        
        if not pmids:
            return []
        
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml"
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email
        
        try:
            response = requests.get(
                f"{self.base_url}/efetch.fcgi",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            abstracts = []
            
            for article in root.findall(".//PubmedArticle"):
                # Extract abstract text
                abstract_texts = article.findall(".//AbstractText")
                if abstract_texts:
                    abstract = " ".join([
                        text.text for text in abstract_texts 
                        if text.text
                    ])
                    abstracts.append(abstract)
            
            # Rate limiting
            time.sleep(0.34)  # NCBI requires max 3 requests/second without API key
            
            return abstracts
            
        except Exception as e:
            print(f"PubMed fetch error: {e}")
            return []


# Example usage
if __name__ == "__main__":
    # Test PubMed search
    pubmed = PubMedSearcher()
    
    query = "type 2 diabetes treatment metformin"
    print(f"Searching PubMed for: {query}\n")
    
    results = pubmed.search(query, max_results=3)
    
    print(f"Found {len(results)} articles:\n")
    for i, abstract in enumerate(results, 1):
        print(f"Article {i}:")
        print(abstract[:200] + "...")
        print()
