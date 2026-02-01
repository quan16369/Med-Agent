"""
Database optimization and data management for large-scale deployments
Handles data pruning, compression, and tiered storage
"""

import sqlite3
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import gzip
import hashlib

logger = logging.getLogger(__name__)


class DatabaseOptimizer:
    """
    Optimize SQLite database for large-scale data
    
    Strategies:
    1. Automatic pruning of old/unused data
    2. Data compression for archived content
    3. Index optimization
    4. Vacuum and analyze
    5. Tiered storage (hot/warm/cold)
    """
    
    def __init__(self, db_path: str, archive_dir: str = "./data/archive"):
        self.db_path = db_path
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Thresholds
        self.max_guidelines = 10000  # Keep most recent 10k guidelines
        self.max_drugs = 5000        # Keep 5k most common drugs
        self.max_protocols = 1000    # Keep 1k protocols
        self.hot_data_days = 30      # Data accessed in last 30 days = hot
        self.warm_data_days = 90     # 30-90 days = warm
        # > 90 days = cold (archive)
        
    def analyze_database_size(self) -> Dict[str, Any]:
        """Analyze current database size and composition"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Total size
            db_size_bytes = Path(self.db_path).stat().st_size
            
            # Count records per table
            stats = {
                "db_size_mb": db_size_bytes / (1024 * 1024),
                "db_path": self.db_path
            }
            
            tables = ["guidelines", "drugs", "protocols"]
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    stats[f"{table}_count"] = count
                    
                    # Check for access_count column
                    cursor.execute(f"PRAGMA table_info({table})")
                    columns = [col[1] for col in cursor.fetchall()]
                    
                    if "access_count" in columns:
                        cursor.execute(f"""
                            SELECT 
                                AVG(access_count) as avg_access,
                                MAX(access_count) as max_access,
                                COUNT(CASE WHEN access_count = 0 THEN 1 END) as unused_count
                            FROM {table}
                        """)
                        row = cursor.fetchone()
                        stats[f"{table}_avg_access"] = row[0] or 0
                        stats[f"{table}_max_access"] = row[1] or 0
                        stats[f"{table}_unused_count"] = row[2] or 0
                        
                except sqlite3.OperationalError:
                    stats[f"{table}_count"] = 0
            
            return stats
            
        finally:
            conn.close()
    
    def add_tracking_columns(self):
        """Add columns for tracking data usage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            tables = ["guidelines", "drugs", "protocols"]
            
            for table in tables:
                # Check if table exists
                cursor.execute(f"""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='{table}'
                """)
                if not cursor.fetchone():
                    continue
                
                # Add access tracking columns if not exist
                columns_to_add = [
                    ("access_count", "INTEGER DEFAULT 0"),
                    ("last_accessed", "TEXT"),
                    ("created_at", "TEXT"),
                    ("data_tier", "TEXT DEFAULT 'hot'")  # hot/warm/cold
                ]
                
                for col_name, col_type in columns_to_add:
                    try:
                        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")
                        logger.info(f"Added column {col_name} to {table}")
                    except sqlite3.OperationalError:
                        # Column already exists
                        pass
            
            conn.commit()
            logger.info("Database tracking columns ready")
            
        finally:
            conn.close()
    
    def update_access_stats(self, table: str, record_id: int):
        """Update access statistics for a record"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(f"""
                UPDATE {table}
                SET 
                    access_count = access_count + 1,
                    last_accessed = ?,
                    data_tier = 'hot'
                WHERE id = ?
            """, (datetime.utcnow().isoformat(), record_id))
            
            conn.commit()
            
        finally:
            conn.close()
    
    def prune_unused_data(self, dry_run: bool = True) -> Dict[str, int]:
        """
        Remove or archive unused data
        
        Strategy:
        - Remove guidelines with 0 accesses and > 90 days old
        - Keep top N most accessed items
        - Archive cold data to compressed files
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        pruned_counts = {}
        
        try:
            # Guidelines: Keep top N by access count
            cursor.execute("""
                SELECT COUNT(*) FROM guidelines 
                WHERE access_count = 0 OR access_count IS NULL
            """)
            unused_guidelines = cursor.fetchone()[0]
            
            if not dry_run and unused_guidelines > self.max_guidelines:
                # Archive cold data first
                cursor.execute("""
                    SELECT id, title, content, category
                    FROM guidelines
                    WHERE (access_count = 0 OR access_count IS NULL)
                    AND created_at < ?
                """, ((datetime.utcnow() - timedelta(days=90)).isoformat(),))
                
                cold_data = cursor.fetchall()
                if cold_data:
                    self._archive_data("guidelines", cold_data)
                
                # Delete archived data
                cursor.execute("""
                    DELETE FROM guidelines
                    WHERE (access_count = 0 OR access_count IS NULL)
                    AND created_at < ?
                """, ((datetime.utcnow() - timedelta(days=90)).isoformat(),))
                
                pruned_counts["guidelines"] = cursor.rowcount
            else:
                pruned_counts["guidelines"] = 0
            
            # Drugs: Similar approach
            cursor.execute("SELECT COUNT(*) FROM drugs")
            total_drugs = cursor.fetchone()[0]
            
            if not dry_run and total_drugs > self.max_drugs:
                # Keep top N by access
                cursor.execute("""
                    DELETE FROM drugs
                    WHERE id NOT IN (
                        SELECT id FROM drugs
                        ORDER BY access_count DESC
                        LIMIT ?
                    )
                """, (self.max_drugs,))
                
                pruned_counts["drugs"] = cursor.rowcount
            else:
                pruned_counts["drugs"] = 0
            
            if not dry_run:
                conn.commit()
                logger.info(f"Pruned data: {pruned_counts}")
            else:
                logger.info(f"Dry run - would prune: {pruned_counts}")
            
            return pruned_counts
            
        finally:
            conn.close()
    
    def _archive_data(self, table: str, data: List[tuple]):
        """Archive data to compressed JSON files"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        archive_file = self.archive_dir / f"{table}_{timestamp}.json.gz"
        
        # Convert to JSON-serializable format
        records = []
        for row in data:
            records.append({
                "id": row[0],
                "title": row[1] if len(row) > 1 else None,
                "content": row[2] if len(row) > 2 else None,
                "category": row[3] if len(row) > 3 else None
            })
        
        # Compress and save
        with gzip.open(archive_file, 'wt', encoding='utf-8') as f:
            json.dump(records, f)
        
        logger.info(f"Archived {len(records)} records to {archive_file}")
    
    def optimize_indexes(self):
        """Optimize database indexes for common queries"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Drop old indexes
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
            existing_indexes = [row[0] for row in cursor.fetchall()]
            
            # Create optimized indexes
            indexes = [
                # Guidelines
                ("idx_guidelines_category", "guidelines", "category"),
                ("idx_guidelines_access", "guidelines", "access_count DESC"),
                ("idx_guidelines_tier", "guidelines", "data_tier"),
                
                # Drugs
                ("idx_drugs_name", "drugs", "name"),
                ("idx_drugs_access", "drugs", "access_count DESC"),
                
                # Protocols
                ("idx_protocols_category", "protocols", "category"),
                ("idx_protocols_access", "protocols", "access_count DESC"),
                
                # FTS5 optimization (if using)
                ("idx_guidelines_fts", "guidelines", "title, content")
            ]
            
            for index_name, table, columns in indexes:
                try:
                    cursor.execute(f"""
                        CREATE INDEX IF NOT EXISTS {index_name} 
                        ON {table}({columns})
                    """)
                    logger.info(f"Created/verified index: {index_name}")
                except sqlite3.OperationalError as e:
                    logger.debug(f"Index creation skipped: {e}")
            
            conn.commit()
            logger.info("Index optimization complete")
            
        finally:
            conn.close()
    
    def vacuum_database(self):
        """Vacuum database to reclaim space and optimize"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            size_before = Path(self.db_path).stat().st_size
            
            # Vacuum
            cursor.execute("VACUUM")
            
            # Analyze for query optimizer
            cursor.execute("ANALYZE")
            
            size_after = Path(self.db_path).stat().st_size
            reclaimed_mb = (size_before - size_after) / (1024 * 1024)
            
            logger.info(f"Vacuum complete. Reclaimed {reclaimed_mb:.2f}MB")
            
            return {
                "size_before_mb": size_before / (1024 * 1024),
                "size_after_mb": size_after / (1024 * 1024),
                "reclaimed_mb": reclaimed_mb
            }
            
        finally:
            conn.close()
    
    def tier_data(self):
        """
        Classify data into tiers based on access patterns
        
        Hot: Accessed in last 30 days
        Warm: Accessed 30-90 days ago
        Cold: Not accessed in 90+ days
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            now = datetime.utcnow()
            hot_cutoff = (now - timedelta(days=self.hot_data_days)).isoformat()
            warm_cutoff = (now - timedelta(days=self.warm_data_days)).isoformat()
            
            tables = ["guidelines", "drugs", "protocols"]
            
            for table in tables:
                # Hot data
                cursor.execute(f"""
                    UPDATE {table}
                    SET data_tier = 'hot'
                    WHERE last_accessed >= ?
                """, (hot_cutoff,))
                hot_count = cursor.rowcount
                
                # Warm data
                cursor.execute(f"""
                    UPDATE {table}
                    SET data_tier = 'warm'
                    WHERE last_accessed < ? AND last_accessed >= ?
                """, (hot_cutoff, warm_cutoff))
                warm_count = cursor.rowcount
                
                # Cold data
                cursor.execute(f"""
                    UPDATE {table}
                    SET data_tier = 'cold'
                    WHERE last_accessed < ? OR last_accessed IS NULL
                """, (warm_cutoff,))
                cold_count = cursor.rowcount
                
                logger.info(f"{table} tiers - Hot: {hot_count}, Warm: {warm_count}, Cold: {cold_count}")
            
            conn.commit()
            
        finally:
            conn.close()
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get recommendations for database optimization"""
        stats = self.analyze_database_size()
        recommendations = []
        
        # Size check
        if stats["db_size_mb"] > 500:
            recommendations.append(
                f"Database is large ({stats['db_size_mb']:.1f}MB). "
                "Consider running pruning and vacuum."
            )
        
        # Unused data check
        for table in ["guidelines", "drugs", "protocols"]:
            unused_key = f"{table}_unused_count"
            total_key = f"{table}_count"
            
            if unused_key in stats and total_key in stats:
                unused = stats[unused_key]
                total = stats[total_key]
                
                if total > 0 and unused / total > 0.3:
                    recommendations.append(
                        f"{table}: {unused}/{total} ({unused/total*100:.1f}%) records unused. "
                        "Consider pruning."
                    )
        
        return recommendations
    
    def full_optimization(self, prune: bool = False):
        """
        Run full optimization pipeline
        
        Args:
            prune: Actually delete data (False = dry run)
        """
        logger.info("Starting full database optimization")
        
        # 1. Analyze current state
        stats_before = self.analyze_database_size()
        logger.info(f"Database size: {stats_before['db_size_mb']:.2f}MB")
        
        # 2. Add tracking columns if needed
        self.add_tracking_columns()
        
        # 3. Classify data tiers
        self.tier_data()
        
        # 4. Prune unused data
        if prune:
            pruned = self.prune_unused_data(dry_run=False)
            logger.info(f"Pruned records: {pruned}")
        else:
            logger.info("Skipping pruning (dry run mode)")
        
        # 5. Optimize indexes
        self.optimize_indexes()
        
        # 6. Vacuum
        vacuum_stats = self.vacuum_database()
        
        # 7. Final analysis
        stats_after = self.analyze_database_size()
        
        return {
            "before": stats_before,
            "after": stats_after,
            "vacuum": vacuum_stats,
            "recommendations": self.get_optimization_recommendations()
        }


class DataSyncManager:
    """
    Manage data synchronization and updates
    
    Strategies:
    1. Incremental updates (only new/changed data)
    2. Priority-based sync (essential data first)
    3. Bandwidth-aware sync
    4. Background sync
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.last_sync = None
    
    def sync_essential_data_only(self):
        """
        Sync only essential data for offline operation
        
        Essential:
        - Emergency protocols (100%)
        - Common drugs (top 500)
        - Frequently accessed guidelines (top 1000)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Mark essential data
            cursor.execute("""
                UPDATE guidelines
                SET data_tier = 'essential'
                WHERE category IN ('emergency', 'critical', 'triage')
                OR access_count > 100
            """)
            
            cursor.execute("""
                UPDATE drugs
                SET data_tier = 'essential'
                WHERE access_count > 50
                LIMIT 500
            """)
            
            conn.commit()
            logger.info("Marked essential data for priority sync")
            
        finally:
            conn.close()
    
    def get_sync_priority_list(self) -> List[Dict[str, Any]]:
        """Get prioritized list of data to sync"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        priority_list = []
        
        try:
            # Priority 1: Emergency/essential
            cursor.execute("""
                SELECT 'guidelines' as type, id, title, 1 as priority
                FROM guidelines
                WHERE data_tier = 'essential'
            """)
            priority_list.extend([dict(zip(["type", "id", "title", "priority"], row)) 
                                 for row in cursor.fetchall()])
            
            # Priority 2: Frequently accessed
            cursor.execute("""
                SELECT 'guidelines' as type, id, title, 2 as priority
                FROM guidelines
                WHERE access_count > 10 AND data_tier != 'essential'
                ORDER BY access_count DESC
                LIMIT 1000
            """)
            priority_list.extend([dict(zip(["type", "id", "title", "priority"], row)) 
                                 for row in cursor.fetchall()])
            
            return priority_list
            
        finally:
            conn.close()


if __name__ == "__main__":
    # Demo
    print("Database Optimization Demo")
    print("="*60)
    
    # Initialize optimizer
    optimizer = DatabaseOptimizer("./data/medical_kb/knowledge.db")
    
    # Analyze current state
    print("\nAnalyzing database...")
    stats = optimizer.analyze_database_size()
    print(f"Database size: {stats['db_size_mb']:.2f}MB")
    
    for key, value in stats.items():
        if key != "db_path":
            print(f"  {key}: {value}")
    
    # Get recommendations
    print("\nOptimization recommendations:")
    recommendations = optimizer.get_optimization_recommendations()
    for rec in recommendations:
        print(f"  - {rec}")
    
    # Run optimization (dry run)
    print("\nRunning optimization (dry run)...")
    result = optimizer.full_optimization(prune=False)
    
    print(f"\nOptimization complete!")
    print(f"  Size before: {result['before']['db_size_mb']:.2f}MB")
    print(f"  Size after: {result['after']['db_size_mb']:.2f}MB")
    if result.get('vacuum'):
        print(f"  Reclaimed: {result['vacuum']['reclaimed_mb']:.2f}MB")
