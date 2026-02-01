# Database Scaling Strategy

## Problem: Local SQLite grows too large

When deploying offline RAG to thousands of clinics, the SQLite database can grow to GB size, causing:
- Slow queries
- Storage constraints (especially on low-end devices)
- Expensive sync operations
- Memory pressure

## Solution: Multi-Tiered Data Management

### 1. **Tiered Storage** (Hot/Warm/Cold)

```
┌─────────────────────────────────────┐
│  HOT DATA (Local SQLite)            │
│  - Last 30 days accessed            │
│  - Emergency protocols (always)     │
│  - Top 1000 guidelines              │
│  - Top 500 drugs                    │
│  Size: ~50-100MB                    │
└─────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  WARM DATA (Compressed Local)       │
│  - 30-90 days accessed              │
│  - Specialty guidelines             │
│  - Uncommon drugs                   │
│  Size: ~200-500MB (compressed)      │
└─────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  COLD DATA (Archived/Cloud)         │
│  - 90+ days not accessed            │
│  - Historical protocols             │
│  - Deprecated drugs                 │
│  - Retrieve on demand when online   │
└─────────────────────────────────────┘
```

### 2. **Automatic Data Management**

#### Tracking
```python
# Every query automatically tracks usage
db.search_guidelines("malaria treatment")
# → Updates: access_count++, last_accessed=now, tier="hot"
```

#### Pruning
```python
# Weekly maintenance (scheduled)
optimizer = DatabaseOptimizer(db_path)
optimizer.full_optimization(prune=True)

# Result:
# - Keep top 10,000 guidelines by access
# - Archive cold data to compressed JSON
# - Remove completely unused data > 90 days
# - Reclaim 50-80% disk space
```

#### Compression
```python
# Cold data archived as gzip JSON
archive/guidelines_20260201_120000.json.gz  # 10MB (was 80MB)
archive/drugs_20260201_120000.json.gz       # 2MB (was 15MB)

# Can retrieve if needed:
optimizer.restore_from_archive("guidelines_20260201_120000.json.gz")
```

### 3. **Sync Optimization**

#### Problem: Syncing 500MB DB over slow network

#### Solution: Differential Sync
```python
# Only sync changes since last sync
sync_manager = DataSyncManager(db_path)

# Priority 1: Essential data (emergency protocols)
sync_manager.sync_essential_data_only()  # 10MB

# Priority 2: Frequently accessed
sync_manager.sync_hot_data()  # 50MB

# Priority 3: Everything else (when bandwidth available)
sync_manager.sync_all()  # 500MB
```

### 4. **Regional Knowledge Hubs**

For massive scale (100,000+ clinics):

```
┌──────────────────────────────────┐
│    Global Cloud (Master)         │
│    - All medical knowledge       │
│    - Updates from WHO, research  │
│    Size: 10GB+                   │
└──────────────────────────────────┘
            │
            ├──────────────┬──────────────┐
            ▼              ▼              ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ Regional Hub 1   │ │ Regional Hub 2   │ │ Regional Hub 3   │
│ (Southeast Asia) │ │ (East Africa)    │ │ (South Asia)     │
│ Size: 2GB        │ │ Size: 2GB        │ │ Size: 2GB        │
└──────────────────┘ └──────────────────┘ └──────────────────┘
      │                     │                     │
      ├────┬────┐          ├────┬────┐          ├────┬────┐
      ▼    ▼    ▼          ▼    ▼    ▼          ▼    ▼    ▼
    Clinics (1000s)      Clinics (1000s)      Clinics (1000s)
    50-100MB each        50-100MB each        50-100MB each
```

**Benefits:**
- Each clinic only syncs with regional hub (lower latency)
- Regional hub filters data by relevance (e.g., tropical diseases for Asia)
- Reduces global bandwidth by 10-100x

### 5. **Implementation Examples**

#### Example 1: Rural Clinic (2GB Storage)
```python
# Minimal configuration
kb = MedicalKnowledgeBase(
    kb_path="./data/medical_kb",
    use_database=True,
    enable_optimization=True
)

# Automatically maintains small DB
# - Keeps only hot data (50MB)
# - Archives warm data locally (200MB compressed)
# - Drops cold data (retrieve from hub when needed)
# Total: 250MB
```

#### Example 2: District Hospital (32GB Storage)
```python
# Keep more data locally
optimizer = DatabaseOptimizer(db_path)
optimizer.max_guidelines = 50000  # More capacity
optimizer.max_drugs = 20000
optimizer.hot_data_days = 90  # Keep 90 days hot

# Total: 500MB-1GB (comprehensive local data)
```

#### Example 3: Regional Hub (500GB Storage)
```python
# No pruning, keep everything
kb = MedicalKnowledgeBase(
    kb_path="./data/medical_kb",
    use_database=True,
    enable_optimization=False  # Disable pruning
)

# Serves 1000s of clinics
# Full knowledge base: 2-10GB
```

### 6. **Performance Targets**

| Deployment | DB Size | Query Time | Sync Time |
|------------|---------|------------|-----------|
| **Rural Clinic** | 50MB | <100ms | 5min (2G) |
| **District Hospital** | 500MB | <200ms | 30min (WiFi) |
| **Regional Hub** | 5GB | <500ms | 2hr (fiber) |

### 7. **Maintenance Schedule**

```python
# Automated maintenance (cron job)
import schedule

# Daily: Tier classification
schedule.every().day.at("02:00").do(optimizer.tier_data)

# Weekly: Prune unused data
schedule.every().sunday.at("03:00").do(
    lambda: optimizer.prune_unused_data(dry_run=False)
)

# Monthly: Full vacuum
schedule.every(30).days.do(optimizer.vacuum_database)
```

### 8. **Monitoring**

```python
# Health check includes DB stats
from medassist.health_checks import HealthChecker

checker = HealthChecker()

# Register DB size check
def check_db_size():
    stats = optimizer.analyze_database_size()
    if stats['db_size_mb'] > 200:  # Threshold for rural clinic
        return ComponentHealth(
            name="database_size",
            status="warn",
            message=f"Database large: {stats['db_size_mb']:.1f}MB"
        )
    return ComponentHealth(
        name="database_size",
        status="pass",
        message=f"Database size OK: {stats['db_size_mb']:.1f}MB"
    )

checker.register_component_checker("database_size", check_db_size)
```

## Best Practices

### ✅ DO:
1. **Enable automatic optimization** in production
2. **Monitor DB size** and access patterns
3. **Archive old data** before deleting
4. **Sync essential data first** (emergency protocols)
5. **Use regional hubs** for massive scale
6. **Run vacuum monthly** to reclaim space
7. **Test restore** from archives periodically

### ❌ DON'T:
1. Keep all data forever (will grow to GB)
2. Sync entire DB over slow networks
3. Delete data without archiving
4. Ignore optimization warnings
5. Deploy without size limits
6. Use single global hub for 100k clinics

## Conclusion

With tiered storage + intelligent pruning + regional hubs:
- **Rural clinic**: 50MB (works on 2GB device)
- **Scales to**: 100,000+ clinics
- **Sync time**: 5min vs 2hr (100x faster)
- **Query speed**: <100ms (same performance at any scale)
- **Data safety**: Everything archived before deletion

System automatically adapts to available storage while maintaining full functionality.
