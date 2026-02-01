"""
Advanced Performance Monitoring and Analytics
Tracks system performance, agent behavior, and quality metrics
"""

import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from datetime import datetime
import statistics


@dataclass
class AgentMetrics:
    """Metrics for individual agent performance"""
    agent_name: str
    invocations: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    success_rate: float = 0.0
    avg_confidence: float = 0.0
    reasoning_steps_avg: float = 0.0
    errors: int = 0


@dataclass
class WorkflowMetrics:
    """Metrics for workflow execution"""
    workflow_id: str
    workflow_type: str
    start_time: str
    end_time: str
    duration: float
    complexity_score: float
    num_agents: int
    num_consultations: int
    success: bool
    final_confidence: float
    reasoning_strategies_used: Dict[str, int]


class PerformanceMonitor:
    """
    Advanced monitoring system for MedAssist
    Tracks performance, identifies bottlenecks, generates insights
    """
    
    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: Number of recent cases to track for rolling metrics
        """
        self.window_size = window_size
        
        # Metrics storage
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        self.workflow_history: deque = deque(maxlen=window_size)
        self.error_log: List[Dict] = []
        
        # Real-time tracking
        self.current_workflow_start: Optional[float] = None
        self.current_workflow_agents: List[str] = []
        
        # Performance stats
        self.total_cases = 0
        self.successful_cases = 0
        self.failed_cases = 0
        
        # Timing breakdown
        self.timing_breakdown: Dict[str, List[float]] = defaultdict(list)
        
        # Quality metrics
        self.confidence_distribution: List[float] = []
        self.complexity_distribution: List[float] = []
    
    def start_workflow(self, workflow_id: str, workflow_type: str):
        """Start tracking a new workflow"""
        self.current_workflow_start = time.time()
        self.current_workflow_agents = []
        
        return {
            'workflow_id': workflow_id,
            'workflow_type': workflow_type,
            'start_time': datetime.now().isoformat()
        }
    
    def track_agent_execution(self, agent_name: str, duration: float, 
                             success: bool, confidence: float, 
                             reasoning_steps: int):
        """Track individual agent execution"""
        # Initialize if first time
        if agent_name not in self.agent_metrics:
            self.agent_metrics[agent_name] = AgentMetrics(agent_name=agent_name)
        
        metrics = self.agent_metrics[agent_name]
        
        # Update counters
        metrics.invocations += 1
        metrics.total_time += duration
        metrics.avg_time = metrics.total_time / metrics.invocations
        
        if not success:
            metrics.errors += 1
        
        # Update success rate
        metrics.success_rate = (metrics.invocations - metrics.errors) / metrics.invocations
        
        # Update confidence (rolling average)
        if metrics.avg_confidence == 0.0:
            metrics.avg_confidence = confidence
        else:
            metrics.avg_confidence = (metrics.avg_confidence * 0.9) + (confidence * 0.1)
        
        # Update reasoning steps (rolling average)
        if metrics.reasoning_steps_avg == 0.0:
            metrics.reasoning_steps_avg = reasoning_steps
        else:
            metrics.reasoning_steps_avg = (metrics.reasoning_steps_avg * 0.9) + (reasoning_steps * 0.1)
        
        # Track for current workflow
        self.current_workflow_agents.append(agent_name)
        
        # Store timing breakdown
        self.timing_breakdown[agent_name].append(duration)
    
    def end_workflow(self, workflow_id: str, workflow_type: str,
                    complexity_score: float, num_consultations: int,
                    success: bool, final_confidence: float,
                    reasoning_strategies: Dict[str, int]):
        """Complete workflow tracking and compute metrics"""
        if self.current_workflow_start is None:
            return None
        
        duration = time.time() - self.current_workflow_start
        
        # Create workflow metrics
        metrics = WorkflowMetrics(
            workflow_id=workflow_id,
            workflow_type=workflow_type,
            start_time=datetime.fromtimestamp(self.current_workflow_start).isoformat(),
            end_time=datetime.now().isoformat(),
            duration=duration,
            complexity_score=complexity_score,
            num_agents=len(set(self.current_workflow_agents)),
            num_consultations=num_consultations,
            success=success,
            final_confidence=final_confidence,
            reasoning_strategies_used=reasoning_strategies
        )
        
        # Store in history
        self.workflow_history.append(metrics)
        
        # Update counters
        self.total_cases += 1
        if success:
            self.successful_cases += 1
        else:
            self.failed_cases += 1
        
        # Update distributions
        self.confidence_distribution.append(final_confidence)
        self.complexity_distribution.append(complexity_score)
        
        # Reset current workflow tracking
        self.current_workflow_start = None
        self.current_workflow_agents = []
        
        return metrics
    
    def log_error(self, agent_name: str, error_type: str, 
                 error_message: str, context: Dict):
        """Log an error with context"""
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'agent': agent_name,
            'error_type': error_type,
            'message': error_message,
            'context': context
        }
        self.error_log.append(error_entry)
    
    def get_agent_performance(self, agent_name: Optional[str] = None) -> Dict:
        """Get performance metrics for agent(s)"""
        if agent_name:
            if agent_name in self.agent_metrics:
                return asdict(self.agent_metrics[agent_name])
            return {}
        
        # Return all agents
        return {
            name: asdict(metrics) 
            for name, metrics in self.agent_metrics.items()
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics"""
        if not self.workflow_history:
            return {'status': 'No data'}
        
        recent_workflows = list(self.workflow_history)[-20:]  # Last 20
        
        # Calculate health metrics
        avg_duration = statistics.mean([w.duration for w in recent_workflows])
        success_rate = sum(1 for w in recent_workflows if w.success) / len(recent_workflows)
        avg_confidence = statistics.mean([w.final_confidence for w in recent_workflows])
        
        # Timing percentiles
        durations = [w.duration for w in recent_workflows]
        durations.sort()
        p50 = durations[len(durations) // 2]
        p95 = durations[int(len(durations) * 0.95)]
        p99 = durations[int(len(durations) * 0.99)]
        
        # Determine health status
        if success_rate > 0.95 and avg_duration < 15:
            status = "Excellent"
        elif success_rate > 0.90 and avg_duration < 20:
            status = "Good"
        elif success_rate > 0.80 and avg_duration < 30:
            status = "Fair"
        else:
            status = "Needs Attention"
        
        return {
            'status': status,
            'total_cases': self.total_cases,
            'success_rate': f"{success_rate:.1%}",
            'avg_duration': f"{avg_duration:.2f}s",
            'avg_confidence': f"{avg_confidence:.2%}",
            'timing_percentiles': {
                'p50': f"{p50:.2f}s",
                'p95': f"{p95:.2f}s",
                'p99': f"{p99:.2f}s"
            },
            'error_count': len(self.error_log),
            'recent_errors': self.error_log[-5:] if self.error_log else []
        }
    
    def get_workflow_insights(self) -> Dict[str, Any]:
        """Generate insights from workflow history"""
        if not self.workflow_history:
            return {}
        
        workflows = list(self.workflow_history)
        
        # Workflow type distribution
        workflow_types = defaultdict(int)
        for w in workflows:
            workflow_types[w.workflow_type] += 1
        
        # Complexity analysis
        if self.complexity_distribution:
            complexity_stats = {
                'mean': statistics.mean(self.complexity_distribution),
                'median': statistics.median(self.complexity_distribution),
                'stdev': statistics.stdev(self.complexity_distribution) if len(self.complexity_distribution) > 1 else 0
            }
        else:
            complexity_stats = {}
        
        # Consultation analysis
        consultations = [w.num_consultations for w in workflows]
        consultation_rate = sum(1 for c in consultations if c > 0) / len(consultations)
        
        # Reasoning strategy popularity
        all_strategies = defaultdict(int)
        for w in workflows:
            for strategy, count in w.reasoning_strategies_used.items():
                all_strategies[strategy] += count
        
        # Performance by complexity
        simple_cases = [w for w in workflows if w.complexity_score < 0.3]
        complex_cases = [w for w in workflows if w.complexity_score > 0.7]
        
        performance_by_complexity = {
            'simple': {
                'avg_duration': statistics.mean([w.duration for w in simple_cases]) if simple_cases else 0,
                'avg_confidence': statistics.mean([w.final_confidence for w in simple_cases]) if simple_cases else 0,
                'count': len(simple_cases)
            },
            'complex': {
                'avg_duration': statistics.mean([w.duration for w in complex_cases]) if complex_cases else 0,
                'avg_confidence': statistics.mean([w.final_confidence for w in complex_cases]) if complex_cases else 0,
                'count': len(complex_cases)
            }
        }
        
        return {
            'workflow_types': dict(workflow_types),
            'complexity_stats': complexity_stats,
            'consultation_rate': f"{consultation_rate:.1%}",
            'reasoning_strategies': dict(all_strategies),
            'performance_by_complexity': performance_by_complexity,
            'total_workflows': len(workflows)
        }
    
    def get_bottleneck_analysis(self) -> Dict[str, Any]:
        """Identify performance bottlenecks"""
        if not self.timing_breakdown:
            return {}
        
        bottlenecks = []
        
        for agent_name, timings in self.timing_breakdown.items():
            if not timings:
                continue
            
            avg_time = statistics.mean(timings)
            max_time = max(timings)
            p95_time = sorted(timings)[int(len(timings) * 0.95)]
            
            # Flag if agent is slow
            if avg_time > 5.0:
                bottlenecks.append({
                    'agent': agent_name,
                    'issue': 'High average latency',
                    'avg_time': f"{avg_time:.2f}s",
                    'recommendation': 'Consider optimization or caching'
                })
            
            # Flag if agent has high variance
            if len(timings) > 1:
                variance = statistics.stdev(timings)
                if variance > avg_time * 0.5:  # High variance
                    bottlenecks.append({
                        'agent': agent_name,
                        'issue': 'Inconsistent performance',
                        'variance': f"{variance:.2f}s",
                        'recommendation': 'Investigate input-dependent slowdowns'
                    })
        
        return {
            'bottlenecks': bottlenecks,
            'timing_summary': {
                agent: {
                    'avg': f"{statistics.mean(timings):.2f}s",
                    'max': f"{max(timings):.2f}s",
                    'count': len(timings)
                }
                for agent, timings in self.timing_breakdown.items()
            }
        }
    
    def generate_report(self) -> str:
        """Generate comprehensive performance report"""
        health = self.get_system_health()
        insights = self.get_workflow_insights()
        bottlenecks = self.get_bottleneck_analysis()
        agent_perf = self.get_agent_performance()
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║           MedAssist Performance Monitoring Report            ║
╚══════════════════════════════════════════════════════════════╝

📊 SYSTEM HEALTH
Status: {health.get('status', 'Unknown')}
Total Cases: {health.get('total_cases', 0)}
Success Rate: {health.get('success_rate', 'N/A')}
Avg Duration: {health.get('avg_duration', 'N/A')}
Avg Confidence: {health.get('avg_confidence', 'N/A')}

Timing Percentiles:
  - P50: {health.get('timing_percentiles', {}).get('p50', 'N/A')}
  - P95: {health.get('timing_percentiles', {}).get('p95', 'N/A')}
  - P99: {health.get('timing_percentiles', {}).get('p99', 'N/A')}

🔍 WORKFLOW INSIGHTS
Total Workflows: {insights.get('total_workflows', 0)}
Consultation Rate: {insights.get('consultation_rate', 'N/A')}

Workflow Types:
"""
        
        for wf_type, count in insights.get('workflow_types', {}).items():
            report += f"  - {wf_type}: {count}\n"
        
        report += "\nReasoning Strategies Used:\n"
        for strategy, count in insights.get('reasoning_strategies', {}).items():
            report += f"  - {strategy}: {count}\n"
        
        report += "\nPerformance by Complexity:\n"
        perf = insights.get('performance_by_complexity', {})
        if 'simple' in perf:
            report += f"  Simple Cases ({perf['simple']['count']}):\n"
            report += f"    Avg Duration: {perf['simple']['avg_duration']:.2f}s\n"
            report += f"    Avg Confidence: {perf['simple']['avg_confidence']:.2%}\n"
        if 'complex' in perf:
            report += f"  Complex Cases ({perf['complex']['count']}):\n"
            report += f"    Avg Duration: {perf['complex']['avg_duration']:.2f}s\n"
            report += f"    Avg Confidence: {perf['complex']['avg_confidence']:.2%}\n"
        
        report += "\n⚡ AGENT PERFORMANCE\n"
        for agent_name, metrics in agent_perf.items():
            report += f"\n{agent_name}:\n"
            report += f"  Invocations: {metrics['invocations']}\n"
            report += f"  Avg Time: {metrics['avg_time']:.2f}s\n"
            report += f"  Success Rate: {metrics['success_rate']:.1%}\n"
            report += f"  Avg Confidence: {metrics['avg_confidence']:.2%}\n"
            report += f"  Avg Reasoning Steps: {metrics['reasoning_steps_avg']:.1f}\n"
        
        report += "\n⚠️  BOTTLENECK ANALYSIS\n"
        if bottlenecks.get('bottlenecks'):
            for bottleneck in bottlenecks['bottlenecks']:
                report += f"\n{bottleneck['agent']}: {bottleneck['issue']}\n"
                report += f"  Recommendation: {bottleneck['recommendation']}\n"
        else:
            report += "No significant bottlenecks detected.\n"
        
        report += "\n" + "="*64 + "\n"
        
        return report
    
    def export_metrics(self, filepath: str):
        """Export all metrics to JSON file"""
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'system_health': self.get_system_health(),
            'workflow_insights': self.get_workflow_insights(),
            'agent_performance': self.get_agent_performance(),
            'bottleneck_analysis': self.get_bottleneck_analysis(),
            'workflow_history': [asdict(w) for w in self.workflow_history],
            'error_log': self.error_log
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)


# Example usage
if __name__ == "__main__":
    monitor = PerformanceMonitor()
    
    # Simulate workflow
    monitor.start_workflow("wf_001", "standard_diagnosis")
    monitor.track_agent_execution("history", 2.3, True, 0.92, 8)
    monitor.track_agent_execution("diagnostic", 3.1, True, 0.88, 12)
    monitor.track_agent_execution("treatment", 2.7, True, 0.90, 10)
    monitor.end_workflow(
        "wf_001", "standard_diagnosis",
        complexity_score=0.45,
        num_consultations=0,
        success=True,
        final_confidence=0.89,
        reasoning_strategies={'ReAct': 2, 'ChainOfThought': 1}
    )
    
    # Generate report
    print(monitor.generate_report())
