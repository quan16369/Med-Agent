"""
Knowledge Base Synchronization System
Automatically updates medical knowledge when internet is available
Scalable architecture with local SQLite + optional cloud backend
"""

from typing import Dict, List, Optional, Any
import sqlite3
import json
import hashlib
import requests
from datetime import datetime, timedelta
from pathlib import Path
import threading
import time
import logging

logger = logging.getLogger(__name__)


class KnowledgeDatabase:
    """
    Scalable local knowledge database with cloud sync capability
    Uses SQLite for local storage, syncs with remote when available
    """
    
    def __init__(
        self,
        db_path: str = "./data/medical_kb/knowledge.db",
        remote_url: Optional[str] = None,
        auto_sync: bool = True
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.remote_url = remote_url or "https://api.medgemma-kb.example.com"
        self.auto_sync = auto_sync
        
        # Initialize database
        self.conn = None
        self._init_database()
        
        # Sync status
        self.last_sync = self._get_last_sync_time()
        self.sync_interval = timedelta(hours=24)  # Sync every 24 hours
        
        # Background sync thread
        if auto_sync:
            self._start_background_sync()
    
    def _init_database(self):
        """Initialize SQLite database with scalable schema"""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        cursor = self.conn.cursor()
        
        # Guidelines table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS guidelines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                guideline_id TEXT UNIQUE NOT NULL,
                source TEXT NOT NULL,
                category TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                version TEXT NOT NULL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                checksum TEXT NOT NULL,
                metadata TEXT
            )
        """)
        
        # Drugs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drugs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                drug_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                generic_name TEXT,
                category TEXT NOT NULL,
                indication TEXT,
                dosing TEXT,
                contraindications TEXT,
                interactions TEXT,
                pregnancy_category TEXT,
                cost_category TEXT,
                version TEXT NOT NULL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                checksum TEXT NOT NULL,
                metadata TEXT
            )
        """)
        
        # Protocols table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS protocols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                protocol_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                content TEXT NOT NULL,
                version TEXT NOT NULL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                checksum TEXT NOT NULL,
                metadata TEXT
            )
        """)
        
        # Sync metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sync_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sync_type TEXT NOT NULL,
                last_sync TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT NOT NULL,
                records_updated INTEGER DEFAULT 0,
                error_message TEXT
            )
        """)
        
        # Full-text search indexes
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS guidelines_fts USING fts5(
                guideline_id,
                title,
                content,
                content='guidelines',
                content_rowid='id'
            )
        """)
        
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS drugs_fts USING fts5(
                drug_id,
                name,
                indication,
                content='drugs',
                content_rowid='id'
            )
        """)
        
        # Indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_guidelines_category ON guidelines(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_drugs_category ON drugs(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_protocols_category ON protocols(category)")
        
        self.conn.commit()
        logger.info(f"Knowledge database initialized at {self.db_path}")
    
    def _get_last_sync_time(self) -> Optional[datetime]:
        """Get timestamp of last successful sync"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT last_sync FROM sync_metadata 
            WHERE status = 'success' 
            ORDER BY last_sync DESC LIMIT 1
        """)
        result = cursor.fetchone()
        
        if result:
            return datetime.fromisoformat(result[0])
        return None
    
    def needs_sync(self) -> bool:
        """Check if database needs synchronization"""
        if not self.last_sync:
            return True
        
        time_since_sync = datetime.now() - self.last_sync
        return time_since_sync > self.sync_interval
    
    def check_internet_connectivity(self) -> bool:
        """Check if internet connection is available"""
        try:
            response = requests.get(
                "https://www.google.com",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
    
    def sync_from_remote(self, force: bool = False) -> Dict[str, Any]:
        """
        Sync knowledge base from remote server
        
        Args:
            force: Force sync even if not needed
        
        Returns:
            Sync result with statistics
        """
        if not force and not self.needs_sync():
            return {
                "status": "skipped",
                "message": "Sync not needed yet",
                "last_sync": self.last_sync
            }
        
        if not self.check_internet_connectivity():
            return {
                "status": "failed",
                "message": "No internet connectivity",
                "last_sync": self.last_sync
            }
        
        logger.info("Starting knowledge base sync from remote...")
        
        try:
            # Get current version info
            local_version = self._get_local_version()
            
            # Check remote version
            remote_version = self._get_remote_version()
            
            if local_version == remote_version and not force:
                logger.info("Knowledge base is up to date")
                return {
                    "status": "success",
                    "message": "Already up to date",
                    "version": local_version
                }
            
            # Fetch updates
            stats = {
                "guidelines_updated": 0,
                "drugs_updated": 0,
                "protocols_updated": 0
            }
            
            # Sync guidelines
            guidelines_updates = self._fetch_remote_data("guidelines", local_version)
            stats["guidelines_updated"] = self._apply_updates("guidelines", guidelines_updates)
            
            # Sync drugs
            drugs_updates = self._fetch_remote_data("drugs", local_version)
            stats["drugs_updated"] = self._apply_updates("drugs", drugs_updates)
            
            # Sync protocols
            protocols_updates = self._fetch_remote_data("protocols", local_version)
            stats["protocols_updated"] = self._apply_updates("protocols", protocols_updates)
            
            # Update sync metadata
            self._record_sync("success", sum(stats.values()))
            self.last_sync = datetime.now()
            
            logger.info(f"Sync completed: {stats}")
            
            return {
                "status": "success",
                "message": "Sync completed successfully",
                "stats": stats,
                "new_version": remote_version,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Sync failed: {e}")
            self._record_sync("failed", 0, str(e))
            
            return {
                "status": "failed",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_local_version(self) -> str:
        """Get current local database version"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT MAX(version) as version FROM (
                SELECT MAX(version) as version FROM guidelines
                UNION ALL
                SELECT MAX(version) as version FROM drugs
                UNION ALL
                SELECT MAX(version) as version FROM protocols
            )
        """)
        result = cursor.fetchone()
        return result[0] if result and result[0] else "0.0.0"
    
    def _get_remote_version(self) -> str:
        """Get latest version from remote server"""
        try:
            response = requests.get(
                f"{self.remote_url}/api/v1/version",
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            return data.get("version", "0.0.0")
        except Exception as e:
            logger.error(f"Failed to get remote version: {e}")
            return "0.0.0"
    
    def _fetch_remote_data(
        self,
        data_type: str,
        since_version: str
    ) -> List[Dict]:
        """Fetch updates for specific data type from remote"""
        try:
            response = requests.get(
                f"{self.remote_url}/api/v1/{data_type}/updates",
                params={"since_version": since_version},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data.get("items", [])
        except Exception as e:
            logger.error(f"Failed to fetch {data_type} updates: {e}")
            return []
    
    def _apply_updates(self, table: str, updates: List[Dict]) -> int:
        """Apply updates to database table"""
        if not updates:
            return 0
        
        cursor = self.conn.cursor()
        count = 0
        
        for item in updates:
            try:
                # Calculate checksum
                content_str = json.dumps(item, sort_keys=True)
                checksum = hashlib.sha256(content_str.encode()).hexdigest()
                
                # Check if update needed
                if self._is_update_needed(table, item.get("id"), checksum):
                    if table == "guidelines":
                        self._upsert_guideline(cursor, item, checksum)
                    elif table == "drugs":
                        self._upsert_drug(cursor, item, checksum)
                    elif table == "protocols":
                        self._upsert_protocol(cursor, item, checksum)
                    
                    count += 1
                    
            except Exception as e:
                logger.error(f"Failed to apply update for {table}: {e}")
        
        self.conn.commit()
        return count
    
    def _is_update_needed(self, table: str, item_id: str, new_checksum: str) -> bool:
        """Check if item needs to be updated"""
        cursor = self.conn.cursor()
        
        id_field = f"{table[:-1]}_id" if table.endswith('s') else f"{table}_id"
        
        cursor.execute(
            f"SELECT checksum FROM {table} WHERE {id_field} = ?",
            (item_id,)
        )
        result = cursor.fetchone()
        
        if not result:
            return True  # New item
        
        return result[0] != new_checksum  # Different checksum
    
    def _upsert_guideline(self, cursor, item: Dict, checksum: str):
        """Insert or update guideline"""
        cursor.execute("""
            INSERT OR REPLACE INTO guidelines (
                guideline_id, source, category, title, content,
                version, checksum, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            item["id"],
            item["source"],
            item["category"],
            item["title"],
            json.dumps(item["content"]),
            item["version"],
            checksum,
            json.dumps(item.get("metadata", {}))
        ))
    
    def _upsert_drug(self, cursor, item: Dict, checksum: str):
        """Insert or update drug"""
        cursor.execute("""
            INSERT OR REPLACE INTO drugs (
                drug_id, name, generic_name, category, indication,
                dosing, contraindications, interactions, pregnancy_category,
                cost_category, version, checksum, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            item["id"],
            item["name"],
            item.get("generic_name"),
            item["category"],
            item.get("indication"),
            item.get("dosing"),
            json.dumps(item.get("contraindications", [])),
            json.dumps(item.get("interactions", [])),
            item.get("pregnancy_category"),
            item.get("cost_category"),
            item["version"],
            checksum,
            json.dumps(item.get("metadata", {}))
        ))
    
    def _upsert_protocol(self, cursor, item: Dict, checksum: str):
        """Insert or update protocol"""
        cursor.execute("""
            INSERT OR REPLACE INTO protocols (
                protocol_id, name, category, content,
                version, checksum, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            item["id"],
            item["name"],
            item["category"],
            json.dumps(item["content"]),
            item["version"],
            checksum,
            json.dumps(item.get("metadata", {}))
        ))
    
    def _record_sync(self, status: str, records_updated: int, error: Optional[str] = None):
        """Record sync operation in metadata"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO sync_metadata (sync_type, status, records_updated, error_message)
            VALUES (?, ?, ?, ?)
        """, ("full", status, records_updated, error))
        self.conn.commit()
    
    def _start_background_sync(self):
        """Start background thread for automatic syncing"""
        def sync_worker():
            while True:
                try:
                    if self.needs_sync():
                        logger.info("Background sync triggered")
                        self.sync_from_remote()
                except Exception as e:
                    logger.error(f"Background sync error: {e}")
                
                # Sleep for 1 hour before checking again
                time.sleep(3600)
        
        thread = threading.Thread(target=sync_worker, daemon=True)
        thread.start()
        logger.info("Background sync thread started")
    
    def search(
        self,
        query: str,
        tables: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Full-text search across knowledge base
        
        Args:
            query: Search query
            tables: Tables to search (guidelines, drugs, protocols)
            limit: Maximum results
        
        Returns:
            List of matching documents
        """
        tables = tables or ["guidelines", "drugs", "protocols"]
        results = []
        
        cursor = self.conn.cursor()
        
        if "guidelines" in tables:
            cursor.execute("""
                SELECT g.* FROM guidelines g
                JOIN guidelines_fts fts ON g.id = fts.rowid
                WHERE guidelines_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, limit))
            
            results.extend([
                {"type": "guideline", **dict(row)}
                for row in cursor.fetchall()
            ])
        
        if "drugs" in tables:
            cursor.execute("""
                SELECT d.* FROM drugs d
                JOIN drugs_fts fts ON d.id = fts.rowid
                WHERE drugs_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, limit))
            
            results.extend([
                {"type": "drug", **dict(row)}
                for row in cursor.fetchall()
            ])
        
        if "protocols" in tables:
            cursor.execute("""
                SELECT * FROM protocols
                WHERE name LIKE ? OR content LIKE ?
                LIMIT ?
            """, (f"%{query}%", f"%{query}%", limit))
            
            results.extend([
                {"type": "protocol", **dict(row)}
                for row in cursor.fetchall()
            ])
        
        return results[:limit]
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM guidelines")
        guidelines_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM drugs")
        drugs_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM protocols")
        protocols_count = cursor.fetchone()[0]
        
        return {
            "guidelines": guidelines_count,
            "drugs": drugs_count,
            "protocols": protocols_count,
            "total": guidelines_count + drugs_count + protocols_count,
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "local_version": self._get_local_version(),
            "database_size_mb": self.db_path.stat().st_size / (1024 * 1024)
        }
    
    def export_for_offline(self, output_path: str):
        """Export entire database for offline distribution"""
        import shutil
        
        # Close connection
        self.conn.close()
        
        # Copy database file
        shutil.copy2(self.db_path, output_path)
        
        # Reopen connection
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        logger.info(f"Database exported to {output_path}")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


class CloudSyncService:
    """
    Optional cloud sync service for multi-clinic deployments
    Allows clinics to share anonymized case data and receive updates
    """
    
    def __init__(
        self,
        api_url: str = "https://api.medgemma-kb.example.com",
        api_key: Optional[str] = None
    ):
        self.api_url = api_url
        self.api_key = api_key
    
    def upload_anonymized_cases(self, cases: List[Dict]) -> bool:
        """Upload anonymized cases for knowledge improvement"""
        try:
            response = requests.post(
                f"{self.api_url}/api/v1/cases/upload",
                json={"cases": cases},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to upload cases: {e}")
            return False
    
    def get_regional_insights(self, region: str) -> Dict:
        """Get disease prevalence insights for region"""
        try:
            response = requests.get(
                f"{self.api_url}/api/v1/insights/region/{region}",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get regional insights: {e}")
            return {}


if __name__ == "__main__":
    # Demo
    print("="*60)
    print("Knowledge Database Sync System Demo")
    print("="*60)
    
    # Initialize database
    kb = KnowledgeDatabase(
        db_path="./data/medical_kb/knowledge.db",
        auto_sync=False  # Manual for demo
    )
    
    # Check status
    stats = kb.get_stats()
    print(f"\nDatabase Statistics:")
    print(f"  Guidelines: {stats['guidelines']}")
    print(f"  Drugs: {stats['drugs']}")
    print(f"  Protocols: {stats['protocols']}")
    print(f"  Total: {stats['total']}")
    print(f"  Size: {stats['database_size_mb']:.2f} MB")
    print(f"  Last sync: {stats['last_sync']}")
    
    # Check if sync needed
    print(f"\nNeeds sync: {kb.needs_sync()}")
    print(f"Internet available: {kb.check_internet_connectivity()}")
    
    # Test search
    results = kb.search("malaria fever", limit=3)
    print(f"\nSearch results for 'malaria fever': {len(results)} found")
    
    kb.close()
