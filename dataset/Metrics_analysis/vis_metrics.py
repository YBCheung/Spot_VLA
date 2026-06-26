from turtle import home
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless server

import tensorflow as tf
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from collections import defaultdict
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import sys
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import ot  # Python Optimal Transport library
from sklearn.metrics.pairwise import cosine_similarity

def compute_optimal_transport_distance(seq1, seq2):
    """
    Compute Optimal Transport (Wasserstein) distance between two action sequences
    """
    # Create cost matrix based on Euclidean distances
    cost_matrix = cdist(seq1, seq2, metric='euclidean')
    
    # Uniform distributions for both sequences
    a = np.ones(len(seq1)) / len(seq1)
    b = np.ones(len(seq2)) / len(seq2)
    
    # Compute OT distance using entropic regularization for efficiency
    ot_distance = ot.sinkhorn2(a, b, cost_matrix, reg=0.1)
    
    return float(ot_distance)

def compute_stepwise_cosine_similarity(seq1, seq2):
    """
    Compute step-wise cosine similarity between two action sequences
    Returns mean cosine similarity across aligned steps
    """
    min_len = min(len(seq1), len(seq2))
    
    # Truncate to same length for step-wise comparison
    seq1_trunc = seq1[:min_len]
    seq2_trunc = seq2[:min_len]
    
    # Compute cosine similarity for each step
    step_similarities = []
    for i in range(min_len):
        # Reshape to 2D for sklearn cosine_similarity
        cos_sim = cosine_similarity(seq1_trunc[i:i+1], seq2_trunc[i:i+1])[0, 0]
        step_similarities.append(cos_sim)
    
    return np.array(step_similarities)

def enhanced_dtw_analysis(tfrecord_path, action_dim=7, save_plots=True):
    """
    Comprehensive DTW, OT, and Cosine Similarity analysis for trajectory classification and success evaluation
    """
    # Group actions by instruction
    instruction_to_actions = defaultdict(list)
    instruction_to_episode_ids = defaultdict(list)

    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    episode_id = 0
    
    for raw_record in raw_dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        instruction = None
        actions = None
        
        for key, feature in example.features.feature.items():
            if feature.bytes_list.value and key == 'steps/language_instruction':
                instruction = feature.bytes_list.value[0].decode('utf-8')
            if feature.float_list.value and key == 'steps/action':
                actions = list(feature.float_list.value)
                
        if instruction is not None and actions is not None:
            actions = np.array(actions).reshape(-1, action_dim)
            instruction_to_actions[instruction].append(actions)
            instruction_to_episode_ids[instruction].append(episode_id)
            episode_id += 1

    return analyze_multi_metric_patterns(instruction_to_actions, instruction_to_episode_ids, save_plots)

def analyze_multi_metric_patterns(instruction_to_actions, instruction_to_episode_ids, save_plots=True):
    """
    Analyze DTW, OT, and Cosine Similarity patterns for task distinction and success evaluation
    """
    results = {}
    
    # 1. Intra-task analysis (same instruction)
    print("=== INTRA-TASK MULTI-METRIC ANALYSIS ===")
    intra_task_metrics = {}
    
    for instruction, actions_list in instruction_to_actions.items():
        if len(actions_list) < 2:
            continue
            
        dtw_distances = []
        ot_distances = []
        cos_similarities = []
        
        for (i, a1), (j, a2) in combinations(enumerate(actions_list), 2):
            # DTW distance
            dtw_distance, _ = fastdtw(a1, a2, dist=euclidean)
            dtw_distances.append(dtw_distance)
            
            # OT distance
            ot_distance = compute_optimal_transport_distance(a1, a2)
            ot_distances.append(ot_distance)
            
            # Step-wise cosine similarity (mean)
            step_cos_sim = compute_stepwise_cosine_similarity(a1, a2)
            mean_cos_sim = np.mean(step_cos_sim)
            cos_similarities.append(mean_cos_sim)
        
        intra_task_metrics[instruction] = {
            'dtw': dtw_distances,
            'ot': ot_distances,
            'cosine': cos_similarities
        }
        
        print(f"\n{instruction}:")
        print(f"  Trajectories: {len(actions_list)}")
        print(f"  DTW - Mean: {np.mean(dtw_distances):.4f} ± {np.std(dtw_distances):.4f}")
        print(f"  OT  - Mean: {np.mean(ot_distances):.4f} ± {np.std(ot_distances):.4f}")
        print(f"  COS - Mean: {np.mean(cos_similarities):.4f} ± {np.std(cos_similarities):.4f}")
    
    # 2. Inter-task analysis (different instructions)
    print("\n=== INTER-TASK MULTI-METRIC ANALYSIS ===")
    inter_task_metrics = {}
    instructions = list(instruction_to_actions.keys())
    
    for i, instr1 in enumerate(instructions):
        for j, instr2 in enumerate(instructions[i+1:], i+1):
            dtw_distances = []
            ot_distances = []
            cos_similarities = []
            
            # Sample pairs to avoid too many comparisons
            max_pairs = 10  # Limit pairs for efficiency
            actions1 = instruction_to_actions[instr1][:max_pairs]
            actions2 = instruction_to_actions[instr2][:max_pairs]
            
            for a1 in actions1:
                for a2 in actions2:
                    # DTW distance
                    dtw_distance, _ = fastdtw(a1, a2, dist=euclidean)
                    dtw_distances.append(dtw_distance)
                    
                    # OT distance
                    ot_distance = compute_optimal_transport_distance(a1, a2)
                    ot_distances.append(ot_distance)
                    
                    # Step-wise cosine similarity (mean)
                    step_cos_sim = compute_stepwise_cosine_similarity(a1, a2)
                    mean_cos_sim = np.mean(step_cos_sim)
                    cos_similarities.append(mean_cos_sim)
            
            task_pair = f"{instr1} vs {instr2}"
            inter_task_metrics[task_pair] = {
                'dtw': dtw_distances,
                'ot': ot_distances,
                'cosine': cos_similarities
            }
            
            print(f"\n{task_pair}:")
            print(f"  DTW - Mean: {np.mean(dtw_distances):.4f} ± {np.std(dtw_distances):.4f}")
            print(f"  OT  - Mean: {np.mean(ot_distances):.4f} ± {np.std(ot_distances):.4f}")
            print(f"  COS - Mean: {np.mean(cos_similarities):.4f} ± {np.std(cos_similarities):.4f}")
    
    # 3. Create visualizations
    if save_plots:
        create_multi_metric_visualizations(intra_task_metrics, inter_task_metrics, instruction_to_actions)
    
    # 4. Statistical analysis for task distinguishability
    task_separability = analyze_multi_metric_separability(intra_task_metrics, inter_task_metrics)
    
    # 5. Success evaluation metrics
    success_metrics = evaluate_multi_metric_success(intra_task_metrics, instruction_to_actions)
    
    results = {
        'intra_task_metrics': intra_task_metrics,
        'inter_task_metrics': inter_task_metrics,
        'task_separability': task_separability,
        'success_metrics': success_metrics
    }
    
    return results

def create_multi_metric_visualizations(intra_task_metrics, inter_task_metrics, instruction_to_actions):
    """
    Create comprehensive visualizations for DTW, OT, and Cosine Similarity analysis
    """
    metrics = ['dtw', 'ot', 'cosine']
    metric_names = ['DTW Distance', 'OT Distance', 'Cosine Similarity']
    
    # Create multiple figure sets
    
    # Figure 1: Metric comparison overview
    plt.figure(figsize=(20, 12))
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        # Subplot: Intra-task vs Inter-task for each metric
        plt.subplot(3, 4, idx*4 + 1)
        
        intra_data = []
        inter_data = []
        
        for task, task_metrics in intra_task_metrics.items():
            intra_data.extend(task_metrics[metric])
        
        for task_pair, task_metrics in inter_task_metrics.items():
            inter_data.extend(task_metrics[metric])
        
        all_data = intra_data + inter_data
        all_labels = ['Intra-task'] * len(intra_data) + ['Inter-task'] * len(inter_data)
        
        df = pd.DataFrame({metric_name: all_data, 'Type': all_labels})
        sns.boxplot(data=df, x='Type', y=metric_name)
        plt.title(f'{metric_name}: Intra vs Inter-task')
        if metric in ['dtw', 'ot']:
            plt.yscale('log')
        
        # Subplot: Heatmap for each metric
        plt.subplot(3, 4, idx*4 + 2)
        tasks = list(intra_task_metrics.keys())
        n_tasks = len(tasks)
        metric_matrix = np.zeros((n_tasks, n_tasks))
        
        # Fill diagonal with intra-task means
        for i, task in enumerate(tasks):
            if task in intra_task_metrics:
                metric_matrix[i, i] = np.mean(intra_task_metrics[task][metric])
        
        # Fill off-diagonal with inter-task means
        for task_pair, task_metrics in inter_task_metrics.items():
            tasks_in_pair = task_pair.split(' vs ')
            if len(tasks_in_pair) == 2:
                try:
                    i = tasks.index(tasks_in_pair[0])
                    j = tasks.index(tasks_in_pair[1])
                    mean_metric = np.mean(task_metrics[metric])
                    metric_matrix[i, j] = mean_metric
                    metric_matrix[j, i] = mean_metric
                except ValueError:
                    continue
        
        sns.heatmap(metric_matrix, xticklabels=[t[:10] for t in tasks], 
                    yticklabels=[t[:10] for t in tasks], annot=True, fmt='.3f', 
                    cmap='viridis' if metric != 'cosine' else 'RdYlBu')
        plt.title(f'{metric_name} Matrix')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        
        # Subplot: Distribution per task
        plt.subplot(3, 4, idx*4 + 3)
        task_data = []
        task_labels = []
        
        for task, task_metrics in intra_task_metrics.items():
            task_data.extend(task_metrics[metric])
            task_labels.extend([task[:15]] * len(task_metrics[metric]))
        
        df_task = pd.DataFrame({metric_name: task_data, 'Task': task_labels})
        sns.boxplot(data=df_task, y='Task', x=metric_name)
        plt.title(f'{metric_name} by Task')
        
        # Subplot: Separability scores
        plt.subplot(3, 4, idx*4 + 4)
        separability_scores = []
        task_names = []
        
        for task in intra_task_metrics.keys():
            intra_mean = np.mean(intra_task_metrics[task][metric])
            
            # Find inter-task metrics involving this task
            inter_values = []
            for task_pair, task_metrics in inter_task_metrics.items():
                if task in task_pair:
                    inter_values.extend(task_metrics[metric])
            
            if inter_values:
                inter_mean = np.mean(inter_values)
                # For cosine similarity, we want lower intra and higher inter for good separation
                if metric == 'cosine':
                    separability = (1 - intra_mean) / (1 - inter_mean) if inter_mean != 1 else 0
                else:
                    separability = inter_mean / intra_mean if intra_mean > 0 else 0
                separability_scores.append(separability)
                task_names.append(task[:15])
        
        plt.barh(range(len(task_names)), separability_scores)
        plt.yticks(range(len(task_names)), task_names)
        plt.xlabel(f'Separability Score ({metric_name})')
        plt.title(f'{metric_name} Separability')
    
    plt.tight_layout()
    plt.savefig('multi_metric_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Correlation analysis between metrics
    plt.figure(figsize=(15, 10))
    
    # Collect all metric pairs for correlation analysis
    all_dtw_intra = []
    all_ot_intra = []
    all_cos_intra = []
    
    for task, task_metrics in intra_task_metrics.items():
        all_dtw_intra.extend(task_metrics['dtw'])
        all_ot_intra.extend(task_metrics['ot'])
        all_cos_intra.extend(task_metrics['cosine'])
    
    all_dtw_inter = []
    all_ot_inter = []
    all_cos_inter = []
    
    for task_pair, task_metrics in inter_task_metrics.items():
        all_dtw_inter.extend(task_metrics['dtw'])
        all_ot_inter.extend(task_metrics['ot'])
        all_cos_inter.extend(task_metrics['cosine'])
    
    # Combine intra and inter for correlation
    all_dtw = all_dtw_intra + all_dtw_inter
    all_ot = all_ot_intra + all_ot_inter
    all_cos = all_cos_intra + all_cos_inter
    
    # DTW vs OT
    plt.subplot(2, 3, 1)
    plt.scatter(all_dtw, all_ot, alpha=0.6)
    plt.xlabel('DTW Distance')
    plt.ylabel('OT Distance')
    plt.title('DTW vs OT Correlation')
    corr_dtw_ot = np.corrcoef(all_dtw, all_ot)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {corr_dtw_ot:.3f}', 
             transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    # DTW vs Cosine
    plt.subplot(2, 3, 2)
    plt.scatter(all_dtw, all_cos, alpha=0.6)
    plt.xlabel('DTW Distance')
    plt.ylabel('Cosine Similarity')
    plt.title('DTW vs Cosine Correlation')
    corr_dtw_cos = np.corrcoef(all_dtw, all_cos)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {corr_dtw_cos:.3f}', 
             transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    # OT vs Cosine
    plt.subplot(2, 3, 3)
    plt.scatter(all_ot, all_cos, alpha=0.6)
    plt.xlabel('OT Distance')
    plt.ylabel('Cosine Similarity')
    plt.title('OT vs Cosine Correlation')
    corr_ot_cos = np.corrcoef(all_ot, all_cos)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {corr_ot_cos:.3f}', 
             transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    # 3D scatter plot (DTW, OT, Cosine)
    ax = plt.subplot(2, 3, 4, projection='3d')
    scatter = ax.scatter(all_dtw, all_ot, all_cos, 
                        c=['red' if i < len(all_dtw_intra) else 'blue' for i in range(len(all_dtw))],
                        alpha=0.6)
    ax.set_xlabel('DTW Distance')
    ax.set_ylabel('OT Distance')
    ax.set_zlabel('Cosine Similarity')
    ax.set_title('3D Metric Space')
    
    # Metric agreement analysis
    plt.subplot(2, 3, 5)
    # Normalize metrics for agreement analysis
    dtw_norm = (np.array(all_dtw) - np.min(all_dtw)) / (np.max(all_dtw) - np.min(all_dtw))
    ot_norm = (np.array(all_ot) - np.min(all_ot)) / (np.max(all_ot) - np.min(all_ot))
    cos_norm = 1 - np.array(all_cos)  # Invert cosine so higher = more dissimilar
    
    agreement_dtw_ot = 1 - np.abs(dtw_norm - ot_norm)
    agreement_dtw_cos = 1 - np.abs(dtw_norm - cos_norm)
    agreement_ot_cos = 1 - np.abs(ot_norm - cos_norm)
    
    agreements = [np.mean(agreement_dtw_ot), np.mean(agreement_dtw_cos), np.mean(agreement_ot_cos)]
    agreement_labels = ['DTW-OT', 'DTW-Cosine', 'OT-Cosine']
    
    plt.bar(agreement_labels, agreements)
    plt.ylabel('Agreement Score')
    plt.title('Metric Agreement Analysis')
    plt.ylim(0, 1)
    
    # Success threshold comparison
    plt.subplot(2, 3, 6)
    thresholds = np.linspace(0, 1, 100)
    
    for metric, metric_name, color in zip(['dtw', 'ot', 'cosine'], 
                                         ['DTW', 'OT', 'Cosine'], 
                                         ['red', 'blue', 'green']):
        success_rates = []
        metric_data = [v for task_metrics in intra_task_metrics.values() for v in task_metrics[metric]]
        
        if metric == 'cosine':
            # For cosine, higher is better, so use upper percentiles
            metric_thresholds = np.linspace(np.min(metric_data), np.max(metric_data), 100)
            for threshold in metric_thresholds:
                success_rate = sum(1 for v in metric_data if v >= threshold) / len(metric_data)
                success_rates.append(success_rate)
        else:
            # For DTW and OT, lower is better
            metric_thresholds = np.linspace(np.min(metric_data), np.percentile(metric_data, 95), 100)
            for threshold in metric_thresholds:
                success_rate = sum(1 for v in metric_data if v <= threshold) / len(metric_data)
                success_rates.append(success_rate)
        
        plt.plot(thresholds, success_rates, label=metric_name, color=color)
    
    plt.xlabel('Normalized Threshold')
    plt.ylabel('Success Rate')
    plt.title('Success Rate Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('metric_correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_multi_metric_separability(intra_task_metrics, inter_task_metrics):
    """
    Analyze how well DTW, OT, and Cosine Similarity can distinguish between different tasks
    """
    print("\n=== MULTI-METRIC TASK SEPARABILITY ANALYSIS ===")
    
    separability_results = {}
    metrics = ['dtw', 'ot', 'cosine']
    
    for task in intra_task_metrics.keys():
        separability_results[task] = {}
        
        for metric in metrics:
            intra_values = intra_task_metrics[task][metric]
            intra_mean = np.mean(intra_values)
            intra_std = np.std(intra_values)
            
            # Collect all inter-task values involving this task
            inter_values = []
            for task_pair, task_metrics in inter_task_metrics.items():
                if task in task_pair:
                    inter_values.extend(task_metrics[metric])
            
            if inter_values:
                inter_mean = np.mean(inter_values)
                inter_std = np.std(inter_values)
                
                # Calculate separability metrics
                if metric == 'cosine':
                    # For cosine similarity, we want high intra and low inter for good separation
                    separability_ratio = intra_mean / inter_mean if inter_mean > 0 else float('inf')
                    effect_size = (intra_mean - inter_mean) / np.sqrt((intra_std**2 + inter_std**2) / 2)
                else:
                    # For DTW and OT, we want low intra and high inter for good separation
                    separability_ratio = inter_mean / intra_mean if intra_mean > 0 else float('inf')
                    effect_size = (inter_mean - intra_mean) / np.sqrt((intra_std**2 + inter_std**2) / 2)
                
                # Statistical test
                t_stat, p_value = stats.ttest_ind(inter_values, intra_values)
                
                separability_results[task][metric] = {
                    'separability_ratio': separability_ratio,
                    'effect_size': effect_size,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'intra_mean': intra_mean,
                    'inter_mean': inter_mean
                }
        
        print(f"\n{task}:")
        for metric in metrics:
            if metric in separability_results[task]:
                result = separability_results[task][metric]
                print(f"  {metric.upper()}:")
                print(f"    Separability Ratio: {result['separability_ratio']:.3f}")
                print(f"    Effect Size: {result['effect_size']:.3f}")
                print(f"    T-test p-value: {result['p_value']:.6f}")
                print(f"    Significant: {'Yes' if result['p_value'] < 0.05 else 'No'}")
    
    return separability_results

def evaluate_multi_metric_success(intra_task_metrics, instruction_to_actions):
    """
    Evaluate trajectory success using DTW, OT, and Cosine Similarity metrics
    """
    print("\n=== MULTI-METRIC TRAJECTORY SUCCESS EVALUATION ===")
    
    success_metrics = {}
    metrics = ['dtw', 'ot', 'cosine']
    
    for task, task_metrics in intra_task_metrics.items():
        success_metrics[task] = {}
        
        for metric in metrics:
            values = task_metrics[metric]
            if not values:
                continue
            
            if metric == 'cosine':
                # For cosine similarity, higher is better
                q25 = np.percentile(values, 75)  # Top 25%
                q50 = np.percentile(values, 50)  # Top 50%
                q75 = np.percentile(values, 25)  # Top 75%
            else:
                # For DTW and OT, lower is better
                q25 = np.percentile(values, 25)
                q50 = np.percentile(values, 50)
                q75 = np.percentile(values, 75)
            
            success_metrics[task][metric] = {
                'conservative_threshold': q25,
                'moderate_threshold': q50,
                'liberal_threshold': q75,
                'distribution': {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'q25': np.percentile(values, 25),
                    'q50': np.percentile(values, 50),
                    'q75': np.percentile(values, 75)
                }
            }
        
        success_metrics[task]['total_trajectories'] = len(instruction_to_actions[task])
        
        print(f"\n{task}:")
        print(f"  Total trajectories: {len(instruction_to_actions[task])}")
        for metric in metrics:
            if metric in success_metrics[task]:
                thresholds = success_metrics[task][metric]
                print(f"  {metric.upper()} thresholds:")
                print(f"    Conservative: {thresholds['conservative_threshold']:.4f}")
                print(f"    Moderate: {thresholds['moderate_threshold']:.4f}")
                print(f"    Liberal: {thresholds['liberal_threshold']:.4f}")
    
    return success_metrics

def evaluate_generated_trajectory_multi_metric(generated_actions, reference_trajectories, task_name, success_metrics):
    """
    Evaluate a generated trajectory against reference trajectories using all metrics
    """
    if task_name not in success_metrics:
        print(f"No success metrics available for task: {task_name}")
        return None
    
    # Calculate all metrics to all reference trajectories
    dtw_distances = []
    ot_distances = []
    cos_similarities = []
    
    for ref_traj in reference_trajectories:
        # DTW
        dtw_distance, _ = fastdtw(generated_actions, ref_traj, dist=euclidean)
        dtw_distances.append(dtw_distance)
        
        # OT
        ot_distance = compute_optimal_transport_distance(generated_actions, ref_traj)
        ot_distances.append(ot_distance)
        
        # Cosine Similarity
        step_cos_sim = compute_stepwise_cosine_similarity(generated_actions, ref_traj)
        mean_cos_sim = np.mean(step_cos_sim)
        cos_similarities.append(mean_cos_sim)
    
    results = {}
    metrics_data = {
        'dtw': dtw_distances,
        'ot': ot_distances,
        'cosine': cos_similarities
    }
    
    for metric, distances in metrics_data.items():
        if metric == 'cosine':
            best_score = max(distances)  # Higher is better for cosine
            mean_score = np.mean(distances)
        else:
            best_score = min(distances)  # Lower is better for DTW and OT
            mean_score = np.mean(distances)
        
        # Evaluate success based on different thresholds
        thresholds = success_metrics[task_name][metric]
        
        if metric == 'cosine':
            conservative_success = best_score >= thresholds['conservative_threshold']
            moderate_success = best_score >= thresholds['moderate_threshold']
            liberal_success = best_score >= thresholds['liberal_threshold']
        else:
            conservative_success = best_score <= thresholds['conservative_threshold']
            moderate_success = best_score <= thresholds['moderate_threshold']
            liberal_success = best_score <= thresholds['liberal_threshold']
        
        results[metric] = {
            'best_score': best_score,
            'mean_score': mean_score,
            'conservative_success': conservative_success,
            'moderate_success': moderate_success,
            'liberal_success': liberal_success,
            'percentile': stats.percentileofscore(distances, best_score)
        }
    
    print(f"\nGenerated Trajectory Multi-Metric Evaluation for '{task_name}':")
    for metric in ['dtw', 'ot', 'cosine']:
        result = results[metric]
        print(f"  {metric.upper()}:")
        print(f"    Best score: {result['best_score']:.4f}")
        print(f"    Mean score: {result['mean_score']:.4f}")
        print(f"    Conservative success: {result['conservative_success']}")
        print(f"    Moderate success: {result['moderate_success']}")
        print(f"    Liberal success: {result['liberal_success']}")
        print(f"    Percentile: {result['percentile']:.1f}%")
    
    return results

# Main execution
if __name__ == "__main__":
    tfrecord_path = '/home/yibo/yibo/Spot_VLA/dataset/modified_libero_rlds/libero_goal/libero_goal-train.tfrecord-00000-of-00016'
    
    # Open log file
    logfile = open("libero_multi_metric_comprehensive_analysis.txt", "w")
    sys.stdout = logfile
    
    try:
        # Run comprehensive multi-metric analysis
        results = enhanced_dtw_analysis(tfrecord_path, action_dim=7, save_plots=True)
        
        # Print summary statistics
        print("\n" + "="*50)
        print("MULTI-METRIC SUMMARY STATISTICS")
        print("="*50)
        
        print(f"Total tasks analyzed: {len(results['intra_task_metrics'])}")
        print(f"Total inter-task comparisons: {len(results['inter_task_metrics'])}")
        
        # Overall separability for each metric
        for metric in ['dtw', 'ot', 'cosine']:
            separability_scores = []
            for task_results in results['task_separability'].values():
                if metric in task_results:
                    separability_scores.append(task_results[metric]['separability_ratio'])
            
            if separability_scores:
                print(f"{metric.upper()} separability ratio: {np.mean(separability_scores):.3f} ± {np.std(separability_scores):.3f}")
        
    finally:
        sys.stdout = sys.__stdout__
        logfile.close()
        print("Multi-metric analysis complete! Check:")
        print("  - 'libero_multi_metric_comprehensive_analysis.txt'")
        print("  - 'multi_metric_analysis_comprehensive.png'")
        print("  - 'metric_correlation_analysis.png'")