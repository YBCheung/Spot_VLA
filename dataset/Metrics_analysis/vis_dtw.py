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

def enhanced_dtw_analysis(tfrecord_path, action_dim=7, save_plots=True):
    """
    Comprehensive DTW analysis for trajectory classification and success evaluation
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

    return analyze_dtw_patterns(instruction_to_actions, instruction_to_episode_ids, save_plots)

def analyze_dtw_patterns(instruction_to_actions, instruction_to_episode_ids, save_plots=True):
    """
    Analyze DTW patterns for task distinction and success evaluation
    """
    results = {}
    
    # 1. Intra-task DTW analysis (same instruction)
    print("=== INTRA-TASK DTW ANALYSIS ===")
    intra_task_dtws = {}
    
    for instruction, actions_list in instruction_to_actions.items():
        if len(actions_list) < 2:
            continue
            
        dtw_distances = []
        for (i, a1), (j, a2) in combinations(enumerate(actions_list), 2):
            distance, _ = fastdtw(a1, a2, dist=euclidean)
            dtw_distances.append(distance)
        
        intra_task_dtws[instruction] = dtw_distances
        mean_dtw = np.mean(dtw_distances)
        std_dtw = np.std(dtw_distances)
        
        print(f"\n{instruction}:")
        print(f"  Trajectories: {len(actions_list)}")
        print(f"  Mean DTW: {mean_dtw:.4f} ± {std_dtw:.4f}")
        print(f"  Min DTW: {np.min(dtw_distances):.4f}")
        print(f"  Max DTW: {np.max(dtw_distances):.4f}")
    
    # 2. Inter-task DTW analysis (different instructions)
    print("\n=== INTER-TASK DTW ANALYSIS ===")
    inter_task_dtws = {}
    instructions = list(instruction_to_actions.keys())
    
    for i, instr1 in enumerate(instructions):
        for j, instr2 in enumerate(instructions[i+1:], i+1):
            dtw_distances = []
            
            # Sample pairs to avoid too many comparisons
            max_pairs = 10  # Limit pairs for efficiency
            actions1 = instruction_to_actions[instr1][:max_pairs]
            actions2 = instruction_to_actions[instr2][:max_pairs]
            
            for a1 in actions1:
                for a2 in actions2:
                    distance, _ = fastdtw(a1, a2, dist=euclidean)
                    dtw_distances.append(distance)
            
            task_pair = f"{instr1} vs {instr2}"
            inter_task_dtws[task_pair] = dtw_distances
            
            mean_dtw = np.mean(dtw_distances)
            std_dtw = np.std(dtw_distances)
            print(f"\n{task_pair}:")
            print(f"  Mean DTW: {mean_dtw:.4f} ± {std_dtw:.4f}")
    
    # 3. Create visualizations
    if save_plots:
        create_dtw_visualizations(intra_task_dtws, inter_task_dtws, instruction_to_actions)
    
    # 4. Statistical analysis for task distinguishability
    task_separability = analyze_task_separability(intra_task_dtws, inter_task_dtws)
    
    # 5. Success evaluation metrics
    success_metrics = evaluate_trajectory_success(intra_task_dtws, instruction_to_actions)
    
    results = {
        'intra_task_dtws': intra_task_dtws,
        'inter_task_dtws': inter_task_dtws,
        'task_separability': task_separability,
        'success_metrics': success_metrics
    }
    
    return results

def create_dtw_visualizations(intra_task_dtws, inter_task_dtws, instruction_to_actions):
    """
    Create comprehensive visualizations for DTW analysis
    """
    
    # 1. Box plot comparing intra-task vs inter-task DTW distributions
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Intra-task DTW distributions
    plt.subplot(2, 3, 1)
    intra_data = []
    intra_labels = []
    for task, dtws in intra_task_dtws.items():
        intra_data.extend(dtws)
        intra_labels.extend([f"Intra-{task[:20]}..."] * len(dtws))
    
    inter_data = []
    inter_labels = []
    for task_pair, dtws in inter_task_dtws.items():
        inter_data.extend(dtws)
        inter_labels.extend([f"Inter-{task_pair[:20]}..."] * len(dtws))
    
    # Combined box plot
    all_data = intra_data + inter_data
    all_labels = ['Intra-task'] * len(intra_data) + ['Inter-task'] * len(inter_data)
    
    df = pd.DataFrame({'DTW Distance': all_data, 'Type': all_labels})
    sns.boxplot(data=df, x='Type', y='DTW Distance')
    plt.title('DTW Distance Distribution: Intra-task vs Inter-task')
    plt.yscale('log')
    
    # Subplot 2: Heatmap of mean DTW distances between tasks
    plt.subplot(2, 3, 2)
    tasks = list(intra_task_dtws.keys())
    n_tasks = len(tasks)
    dtw_matrix = np.zeros((n_tasks, n_tasks))
    
    # Fill diagonal with intra-task means
    for i, task in enumerate(tasks):
        if task in intra_task_dtws:
            dtw_matrix[i, i] = np.mean(intra_task_dtws[task])
    
    # Fill off-diagonal with inter-task means
    for task_pair, dtws in inter_task_dtws.items():
        tasks_in_pair = task_pair.split(' vs ')
        if len(tasks_in_pair) == 2:
            try:
                i = tasks.index(tasks_in_pair[0])
                j = tasks.index(tasks_in_pair[1])
                mean_dtw = np.mean(dtws)
                dtw_matrix[i, j] = mean_dtw
                dtw_matrix[j, i] = mean_dtw
            except ValueError:
                continue
    
    sns.heatmap(dtw_matrix, xticklabels=[t[:15] for t in tasks], 
                yticklabels=[t[:15] for t in tasks], annot=True, fmt='.2f', cmap='viridis')
    plt.title('Mean DTW Distance Matrix Between Tasks')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    
    # Subplot 3: Distribution of trajectory lengths per task
    plt.subplot(2, 3, 3)
    traj_lengths = {}
    for task, actions_list in instruction_to_actions.items():
        lengths = [len(actions) for actions in actions_list]
        traj_lengths[task[:20]] = lengths
    
    lengths_data = []
    lengths_labels = []
    for task, lengths in traj_lengths.items():
        lengths_data.extend(lengths)
        lengths_labels.extend([task] * len(lengths))
    
    df_lengths = pd.DataFrame({'Length': lengths_data, 'Task': lengths_labels})
    sns.boxplot(data=df_lengths, y='Task', x='Length')
    plt.title('Trajectory Length Distribution by Task')
    
    # Subplot 4: DTW vs Trajectory Length Correlation
    plt.subplot(2, 3, 4)
    dtw_length_data = []
    for task, dtws in intra_task_dtws.items():
        if task in instruction_to_actions:
            actions_list = instruction_to_actions[task]
            avg_length = np.mean([len(actions) for actions in actions_list])
            for dtw_val in dtws:
                dtw_length_data.append((avg_length, dtw_val, task))
    
    if dtw_length_data:
        lengths, dtws, tasks = zip(*dtw_length_data)
        plt.scatter(lengths, dtws, alpha=0.6)
        plt.xlabel('Average Trajectory Length')
        plt.ylabel('DTW Distance')
        plt.title('DTW Distance vs Trajectory Length')
        
        # Add correlation coefficient
        corr_coef = np.corrcoef(lengths, dtws)[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {corr_coef:.3f}', 
                transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    # Subplot 5: Success threshold analysis
    plt.subplot(2, 3, 5)
    success_thresholds = np.linspace(0, np.percentile(intra_data, 95), 100)
    success_rates = []
    
    for threshold in success_thresholds:
        total_comparisons = 0
        successful_comparisons = 0
        
        for task, dtws in intra_task_dtws.items():
            total_comparisons += len(dtws)
            successful_comparisons += sum(1 for d in dtws if d <= threshold)
        
        success_rate = successful_comparisons / total_comparisons if total_comparisons > 0 else 0
        success_rates.append(success_rate)
    
    plt.plot(success_thresholds, success_rates)
    plt.xlabel('DTW Threshold')
    plt.ylabel('Success Rate')
    plt.title('Success Rate vs DTW Threshold')
    plt.grid(True, alpha=0.3)
    
    # Subplot 6: Task separability score
    plt.subplot(2, 3, 6)
    separability_scores = []
    task_names = []
    
    for task in intra_task_dtws.keys():
        intra_mean = np.mean(intra_task_dtws[task])
        
        # Find inter-task distances involving this task
        inter_distances = []
        for task_pair, dtws in inter_task_dtws.items():
            if task in task_pair:
                inter_distances.extend(dtws)
        
        if inter_distances:
            inter_mean = np.mean(inter_distances)
            # Separability score: higher is better (inter >> intra)
            separability = inter_mean / intra_mean if intra_mean > 0 else 0
            separability_scores.append(separability)
            task_names.append(task[:20])
    
    plt.barh(range(len(task_names)), separability_scores)
    plt.yticks(range(len(task_names)), task_names)
    plt.xlabel('Separability Score (Inter-DTW / Intra-DTW)')
    plt.title('Task Separability Scores')
    
    plt.tight_layout()
    plt.savefig('dtw_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_task_separability(intra_task_dtws, inter_task_dtws):
    """
    Analyze how well DTW can distinguish between different tasks
    """
    print("\n=== TASK SEPARABILITY ANALYSIS ===")
    
    # Calculate separability metrics
    separability_results = {}
    
    for task in intra_task_dtws.keys():
        intra_distances = intra_task_dtws[task]
        intra_mean = np.mean(intra_distances)
        intra_std = np.std(intra_distances)
        
        # Collect all inter-task distances involving this task
        inter_distances = []
        for task_pair, dtws in inter_task_dtws.items():
            if task in task_pair:
                inter_distances.extend(dtws)
        
        if inter_distances:
            inter_mean = np.mean(inter_distances)
            inter_std = np.std(inter_distances)
            
            # Calculate separability metrics
            separability_ratio = inter_mean / intra_mean if intra_mean > 0 else float('inf')
            effect_size = (inter_mean - intra_mean) / np.sqrt((intra_std**2 + inter_std**2) / 2)
            
            # Statistical test
            t_stat, p_value = stats.ttest_ind(inter_distances, intra_distances)
            
            separability_results[task] = {
                'separability_ratio': separability_ratio,
                'effect_size': effect_size,
                't_statistic': t_stat,
                'p_value': p_value,
                'intra_mean': intra_mean,
                'inter_mean': inter_mean
            }
            
            print(f"\n{task}:")
            print(f"  Separability Ratio: {separability_ratio:.3f}")
            print(f"  Effect Size (Cohen's d): {effect_size:.3f}")
            print(f"  T-test p-value: {p_value:.6f}")
            print(f"  Significant separation: {'Yes' if p_value < 0.05 else 'No'}")
    
    return separability_results

def evaluate_trajectory_success(intra_task_dtws, instruction_to_actions):
    """
    Evaluate trajectory success using DTW-based metrics
    """
    print("\n=== TRAJECTORY SUCCESS EVALUATION ===")
    
    success_metrics = {}
    
    for task, dtws in intra_task_dtws.items():
        if not dtws:
            continue
            
        # Define success criteria based on DTW distances
        q25 = np.percentile(dtws, 25)
        q50 = np.percentile(dtws, 50)
        q75 = np.percentile(dtws, 75)
        
        # Conservative threshold (top 25% similar trajectories)
        conservative_threshold = q25
        # Moderate threshold (top 50%)
        moderate_threshold = q50
        # Liberal threshold (top 75%)
        liberal_threshold = q75
        
        success_metrics[task] = {
            'conservative_threshold': conservative_threshold,
            'moderate_threshold': moderate_threshold,
            'liberal_threshold': liberal_threshold,
            'total_trajectories': len(instruction_to_actions[task]),
            'dtw_distribution': {
                'mean': np.mean(dtws),
                'std': np.std(dtws),
                'min': np.min(dtws),
                'max': np.max(dtws),
                'q25': q25,
                'q50': q50,
                'q75': q75
            }
        }
        
        print(f"\n{task}:")
        print(f"  Total trajectories: {len(instruction_to_actions[task])}")
        print(f"  DTW thresholds for success:")
        print(f"    Conservative (25th percentile): {conservative_threshold:.4f}")
        print(f"    Moderate (50th percentile): {moderate_threshold:.4f}")
        print(f"    Liberal (75th percentile): {liberal_threshold:.4f}")
    
    return success_metrics

def evaluate_generated_trajectory(generated_actions, reference_trajectories, task_name, success_metrics):
    """
    Evaluate a generated trajectory against reference trajectories
    """
    if task_name not in success_metrics:
        print(f"No success metrics available for task: {task_name}")
        return None
    
    # Calculate DTW distances to all reference trajectories
    dtw_distances = []
    for ref_traj in reference_trajectories:
        distance, _ = fastdtw(generated_actions, ref_traj, dist=euclidean)
        dtw_distances.append(distance)
    
    min_dtw = min(dtw_distances)
    mean_dtw = np.mean(dtw_distances)
    
    # Evaluate success based on different thresholds
    thresholds = success_metrics[task_name]
    
    evaluation = {
        'min_dtw_to_reference': min_dtw,
        'mean_dtw_to_reference': mean_dtw,
        'conservative_success': min_dtw <= thresholds['conservative_threshold'],
        'moderate_success': min_dtw <= thresholds['moderate_threshold'],
        'liberal_success': min_dtw <= thresholds['liberal_threshold'],
        'dtw_percentile': stats.percentileofscore(dtw_distances + [min_dtw], min_dtw)
    }
    
    print(f"\nGenerated Trajectory Evaluation for '{task_name}':")
    print(f"  Min DTW to reference: {min_dtw:.4f}")
    print(f"  Mean DTW to references: {mean_dtw:.4f}")
    print(f"  Conservative success: {evaluation['conservative_success']}")
    print(f"  Moderate success: {evaluation['moderate_success']}")
    print(f"  Liberal success: {evaluation['liberal_success']}")
    print(f"  DTW percentile: {evaluation['dtw_percentile']:.1f}%")
    
    return evaluation

# Main execution
if __name__ == "__main__":
    tfrecord_path = '/home/yibo/yibo/Spot_VLA/dataset/modified_libero_rlds/libero_goal/libero_goal-train.tfrecord-00000-of-00016'
    # Open log file
    logfile = open("libero_dtw_comprehensive_analysis.txt", "w")
    sys.stdout = logfile
    
    try:
        # Run comprehensive DTW analysis
        results = enhanced_dtw_analysis(tfrecord_path, action_dim=7, save_plots=True)
        
        # Print summary statistics
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        
        print(f"Total tasks analyzed: {len(results['intra_task_dtws'])}")
        print(f"Total inter-task comparisons: {len(results['inter_task_dtws'])}")
        
        # Overall separability
        all_separability = [s['separability_ratio'] for s in results['task_separability'].values()]
        if all_separability:
            print(f"Average task separability ratio: {np.mean(all_separability):.3f} ± {np.std(all_separability):.3f}")
        
    finally:
        sys.stdout = sys.__stdout__
        logfile.close()
        print("Analysis complete! Check 'libero_dtw_comprehensive_analysis.txt' and 'dtw_analysis_comprehensive.png'")