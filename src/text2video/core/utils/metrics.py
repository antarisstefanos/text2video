from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import statistics
from collections import defaultdict

def calculate_success_rate(results: List[Dict[str, Any]]) -> float:
    """Calculate success rate from a list of results."""
    if not results:
        return 0.0
    
    success_count = sum(1 for r in results if r.get('success', False))
    return success_count / len(results)

def calculate_average_execution_time(results: List[Dict[str, Any]]) -> float:
    """Calculate average execution time from a list of results."""
    if not results:
        return 0.0
    
    execution_times = [
        r.get('execution_time', 0.0)
        for r in results
        if isinstance(r.get('execution_time'), (int, float))
    ]
    
    return statistics.mean(execution_times) if execution_times else 0.0

def calculate_quality_score(results: List[Dict[str, Any]]) -> float:
    """Calculate average quality score from a list of results."""
    if not results:
        return 0.0
    
    quality_scores = [
        r.get('quality_score', 0.0)
        for r in results
        if isinstance(r.get('quality_score'), (int, float))
    ]
    
    return statistics.mean(quality_scores) if quality_scores else 0.0

def aggregate_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate multiple metrics dictionaries."""
    if not metrics_list:
        return {}
    
    aggregated = defaultdict(list)
    
    # Collect all values for each metric
    for metrics in metrics_list:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                aggregated[key].append(value)
    
    # Calculate statistics for each metric
    result = {}
    for key, values in aggregated.items():
        if values:
            result[f"{key}_mean"] = statistics.mean(values)
            result[f"{key}_median"] = statistics.median(values)
            result[f"{key}_min"] = min(values)
            result[f"{key}_max"] = max(values)
    
    return result

def calculate_performance_trends(results: List[Dict[str, Any]], 
                               time_window: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
    """Calculate performance trends over time."""
    if not results:
        return {}
    
    # Sort results by timestamp
    sorted_results = sorted(
        results,
        key=lambda x: datetime.fromisoformat(x.get('timestamp', '2000-01-01T00:00:00'))
    )
    
    # Group results by time window
    windowed_results = defaultdict(list)
    for result in sorted_results:
        timestamp = datetime.fromisoformat(result.get('timestamp', '2000-01-01T00:00:00'))
        window_start = timestamp - (timestamp % time_window)
        windowed_results[window_start].append(result)
    
    # Calculate metrics for each window
    trends = {}
    for window_start, window_results in windowed_results.items():
        trends[window_start.isoformat()] = {
            'success_rate': calculate_success_rate(window_results),
            'avg_execution_time': calculate_average_execution_time(window_results),
            'avg_quality_score': calculate_quality_score(window_results),
            'total_tasks': len(window_results)
        }
    
    return trends

def calculate_resource_utilization(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Calculate resource utilization from metrics."""
    if not metrics:
        return {}
    
    utilization = {}
    
    # Calculate CPU utilization
    if 'cpu_usage' in metrics:
        utilization['cpu'] = metrics['cpu_usage'] / 100.0
    
    # Calculate memory utilization
    if 'memory_usage' in metrics and 'memory_total' in metrics:
        utilization['memory'] = metrics['memory_usage'] / metrics['memory_total']
    
    # Calculate disk utilization
    if 'disk_usage' in metrics and 'disk_total' in metrics:
        utilization['disk'] = metrics['disk_usage'] / metrics['disk_total']
    
    return utilization

def calculate_error_rates(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate error rates from a list of results."""
    if not results:
        return {}
    
    error_counts = defaultdict(int)
    total_errors = 0
    
    for result in results:
        if not result.get('success', False):
            error_type = result.get('error_type', 'unknown')
            error_counts[error_type] += 1
            total_errors += 1
    
    # Calculate error rates
    error_rates = {}
    for error_type, count in error_counts.items():
        error_rates[error_type] = count / total_errors if total_errors > 0 else 0.0
    
    return error_rates 