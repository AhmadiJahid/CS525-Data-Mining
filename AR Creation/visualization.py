"""
Visualization module for association rule mining.
This module contains functions to visualize transaction matrices and association rules.
"""

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import os
import numpy as np

def visualize_rules_network(rules_df, max_rules=50, min_lift=1.0, save_path='output/rule_network.png'):
    """
    Create a network visualization of association rules.
    
    Args:
        rules_df (DataFrame): The rules DataFrame from association_rules function
        max_rules (int): Maximum number of rules to include
        min_lift (float): Minimum lift to filter rules
        save_path (str): Path to save the visualization
    """
    print(f"Creating network visualization of top {max_rules} rules with min lift {min_lift}...")
    
    if rules_df.empty:
        print("No rules to visualize.")
        return
    
    # Filter rules by minimum lift
    filtered_rules = rules_df[rules_df['lift'] >= min_lift]
    
    if filtered_rules.empty:
        print(f"No rules with lift >= {min_lift}.")
        return
    
    # Take top rules by lift
    top_rules = filtered_rules.sort_values('lift', ascending=False).head(max_rules)
    
    # Create a network graph
    G = nx.DiGraph()
    
    # Process each rule
    for _, rule in top_rules.iterrows():
        antecedents = frozenset(rule['antecedents'])
        consequents = frozenset(rule['consequents'])
        
        # Add the items as nodes
        for item in antecedents:
            if item not in G:
                # Clean feature name for display
                display_name = item
                if item.startswith('Procedure_'):
                    display_name = item.replace('Procedure_', 'Proc: ')
                elif item.startswith('Diagnosis_'):
                    display_name = item.replace('Diagnosis_', 'Diag: ')
                elif item.startswith('Gender_'):
                    display_name = item.replace('Gender_', 'Gender: ')
                elif item.startswith('Age_'):
                    display_name = item.replace('Age_', 'Age: ')
                
                G.add_node(item, label=display_name, type='antecedent')
        
        for item in consequents:
            if item not in G:
                # Clean feature name for display
                display_name = item
                if item.startswith('Procedure_'):
                    display_name = item.replace('Procedure_', 'Proc: ')
                elif item.startswith('Diagnosis_'):
                    display_name = item.replace('Diagnosis_', 'Diag: ')
                elif item.startswith('Gender_'):
                    display_name = item.replace('Gender_', 'Gender: ')
                elif item.startswith('Age_'):
                    display_name = item.replace('Age_', 'Age: ')
                
                G.add_node(item, label=display_name, type='consequent')
        
        # Add edges between all antecedents and all consequents
        for a_item in antecedents:
            for c_item in consequents:
                G.add_edge(a_item, c_item, weight=rule['lift'], 
                           confidence=rule['confidence'], 
                           support=rule['support'])
    
    # Set up the plot
    plt.figure(figsize=(15, 10))
    
    # Use spring layout for better visualization
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Draw nodes
    antecedent_nodes = [n for n, attr in G.nodes(data=True) 
                       if attr.get('type', '') == 'antecedent']
    consequent_nodes = [n for n, attr in G.nodes(data=True) 
                       if attr.get('type', '') == 'consequent']
    
    nx.draw_networkx_nodes(G, pos, nodelist=antecedent_nodes, node_color='skyblue', 
                          node_size=500, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, nodelist=consequent_nodes, node_color='lightgreen', 
                          node_size=500, alpha=0.8)
    
    # Draw edges with varying width based on lift
    edges = G.edges(data=True)
    weights = [d['weight'] for _, _, d in edges]
    
    # Normalize weights for better visualization
    max_weight = max(weights) if weights else 1
    min_weight = min(weights) if weights else 0
    normalized_weights = [(w - min_weight) / (max_weight - min_weight) * 5 + 1 
                          for w in weights]
    
    nx.draw_networkx_edges(G, pos, width=normalized_weights, alpha=0.7, 
                         edge_color='gray', arrows=True, arrowsize=15)
    
    # Draw labels
    labels = {n: G.nodes[n]['label'] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, 
                           font_weight='bold')
    
    plt.axis('off')
    plt.title(f'Association Rules Network (Top {len(top_rules)} Rules by Lift)', fontsize=15)
    
    # Add a legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', 
                  markersize=10, label='Antecedents'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
                  markersize=10, label='Consequents')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Network visualization saved to {save_path}")
    plt.close()

def visualize_rule_metrics(rules_df, save_path='output/rule_metrics.png'):
    """
    Create scatter plots of rule metrics (support, confidence, lift).
    
    Args:
        rules_df (DataFrame): The rules DataFrame from association_rules function
        save_path (str): Path to save the visualization
    """
    print("Creating visualization of rule metrics...")
    
    if rules_df.empty:
        print("No rules to visualize.")
        return
    
    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Support vs Confidence scatter plot
    axs[0].scatter(rules_df['support'], rules_df['confidence'], alpha=0.5, 
                  s=rules_df['lift']*10)
    axs[0].set_xlabel('Support')
    axs[0].set_ylabel('Confidence')
    axs[0].set_title('Support vs. Confidence (size = lift)')
    axs[0].grid(True, alpha=0.3)
    
    # Support vs Lift scatter plot
    axs[1].scatter(rules_df['support'], rules_df['lift'], alpha=0.5, 
                  s=rules_df['confidence']*100)
    axs[1].set_xlabel('Support')
    axs[1].set_ylabel('Lift')
    axs[1].set_title('Support vs. Lift (size = confidence)')
    axs[1].grid(True, alpha=0.3)
    
    # Confidence vs Lift scatter plot
    axs[2].scatter(rules_df['confidence'], rules_df['lift'], alpha=0.5, 
                  s=rules_df['support']*1000)
    axs[2].set_xlabel('Confidence')
    axs[2].set_ylabel('Lift')
    axs[2].set_title('Confidence vs. Lift (size = support)')
    axs[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Rule metrics visualization saved to {save_path}")
    plt.close()

def visualize_feature_distribution(transaction_matrix, save_path='output/feature_distribution.png'):
    """
    Visualize the distribution of features in the transaction matrix.
    
    Args:
        transaction_matrix (DataFrame): The one-hot encoded transaction matrix
        save_path (str): Path to save the visualization
    """
    print("Creating visualization of feature distribution...")
    
    if transaction_matrix.empty:
        print("Empty transaction matrix. Cannot visualize feature distribution.")
        return
    
    # Calculate feature frequencies
    feature_counts = transaction_matrix.sum().sort_values(ascending=False)
    
    # Group features by type
    procedure_features = [f for f in feature_counts.index if f.startswith('Procedure_')]
    diagnosis_features = [f for f in feature_counts.index if f.startswith('Diagnosis_')]
    demographic_features = [f for f in feature_counts.index if f.startswith(('Gender_', 'Age_')) or f == 'anchor_age']
    
    # Function to plot feature distributions by type
    def plot_feature_type(features, title, ax, color):
        if not features:
            ax.text(0.5, 0.5, f"No {title} features found", 
                   horizontalalignment='center', verticalalignment='center')
            ax.set_title(title)
            return
        
        # Get counts for these features
        counts = feature_counts[features]
        
        # Take top 15 for readability
        if len(counts) > 15:
            counts = counts.head(15)
        
        # Clean labels
        labels = counts.index.map(lambda x: x.split('_', 1)[1] if '_' in x else x)
        
        # Plot horizontal bar chart
        bars = ax.barh(range(len(counts)), counts.values, color=color, alpha=0.7)
        ax.set_yticks(range(len(counts)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(title)
        ax.set_xlabel('Frequency')
        
        # Add count values to bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 1, i, f"{width}", va='center', fontsize=8)
    
    # Create figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot each feature type
    plot_feature_type(procedure_features, 'Procedures', axs[0], 'skyblue')
    plot_feature_type(diagnosis_features, 'Diagnoses', axs[1], 'lightgreen')
    plot_feature_type(demographic_features, 'Demographics', axs[2], 'salmon')
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Feature distribution visualization saved to {save_path}")
    plt.close()

def visualize_rules_summary(rules_df, save_path='output/rules_summary.png'):
    """
    Create a summary visualization of the generated rules.
    
    Args:
        rules_df (DataFrame): The rules DataFrame from association_rules function
        save_path (str): Path to save the visualization
    """
    print("Creating rules summary visualization...")
    
    if rules_df.empty:
        print("No rules to visualize.")
        return
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    
    # Distribution of support
    axs[0, 0].hist(rules_df['support'], bins=20, color='skyblue', alpha=0.7)
    axs[0, 0].set_title('Distribution of Support')
    axs[0, 0].set_xlabel('Support')
    axs[0, 0].set_ylabel('Count')
    axs[0, 0].grid(True, alpha=0.3)
    
    # Distribution of confidence
    axs[0, 1].hist(rules_df['confidence'], bins=20, color='lightgreen', alpha=0.7)
    axs[0, 1].set_title('Distribution of Confidence')
    axs[0, 1].set_xlabel('Confidence')
    axs[0, 1].set_ylabel('Count')
    axs[0, 1].grid(True, alpha=0.3)
    
    # Distribution of lift
    axs[1, 0].hist(rules_df['lift'], bins=20, color='salmon', alpha=0.7)
    axs[1, 0].set_title('Distribution of Lift')
    axs[1, 0].set_xlabel('Lift')
    axs[1, 0].set_ylabel('Count')
    axs[1, 0].grid(True, alpha=0.3)
    
    # Number of items in antecedents and consequents
    antecedent_sizes = rules_df['antecedents'].apply(len)
    consequent_sizes = rules_df['consequents'].apply(len)
    
    # Create a combined histogram
    axs[1, 1].hist([antecedent_sizes, consequent_sizes], bins=range(1, max(antecedent_sizes.max(), consequent_sizes.max()) + 2), 
                  color=['skyblue', 'salmon'], alpha=0.7, label=['Antecedents', 'Consequents'])
    axs[1, 1].set_title('Number of Items in Rule Parts')
    axs[1, 1].set_xlabel('Number of Items')
    axs[1, 1].set_ylabel('Count')
    axs[1, 1].set_xticks(range(1, max(antecedent_sizes.max(), consequent_sizes.max()) + 1))
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)
    
    # Add overall statistics as text
    stats_text = (
        f"Total Rules: {len(rules_df)}\n"
        f"Average Support: {rules_df['support'].mean():.4f}\n"
        f"Average Confidence: {rules_df['confidence'].mean():.4f}\n"
        f"Average Lift: {rules_df['lift'].mean():.4f}\n"
        f"Max Lift: {rules_df['lift'].max():.4f}\n"
        f"Min Lift: {rules_df['lift'].min():.4f}\n"
    )
    
    fig.text(0.5, 0.95, stats_text, ha='center', va='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])  # Make room for the text
    
    # Save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Rules summary visualization saved to {save_path}")
    plt.close()

def create_html_report(transactions_matrix, rules_df, procedure_rules, output_dir='output'):
    """
    Create an HTML report summarizing the association rule mining results.
    
    Args:
        transactions_matrix (DataFrame): Transaction matrix
        rules_df (DataFrame): All association rules
        procedure_rules (DataFrame): Filtered procedure rules
        output_dir (str): Directory to save the report
        
    Returns:
        str: Path to the generated HTML report
    """
    print("Creating HTML report...")
    
    if transactions_matrix.empty:
        print("Empty transaction matrix. Cannot create report.")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'association_rules_report.html')
    
    # Start building the HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Association Rule Mining Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 20px;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1, h2, h3 {{
                color: #333;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .metrics {{
                display: flex;
                justify-content: space-between;
                flex-wrap: wrap;
            }}
            .metric-box {{
                background-color: #f2f2f2;
                border-radius: 5px;
                padding: 15px;
                margin: 10px;
                min-width: 200px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #333;
            }}
            .metric-label {{
                font-size: 14px;
                color: #666;
            }}
            img {{
                max-width: 100%;
                height: auto;
                margin: 10px 0;
                border: 1px solid #ddd;
            }}
        </style>
    </head>
    <body>
        <h1>Association Rule Mining Report</h1>
        <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Dataset Overview</h2>
        <div class="metrics">
            <div class="metric-box">
                <div class="metric-value">{transactions_matrix.shape[0]:,}</div>
                <div class="metric-label">Total Transactions</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{transactions_matrix.shape[1]:,}</div>
                <div class="metric-label">Total Features</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{len([c for c in transactions_matrix.columns if c.startswith('Procedure_')]):,}</div>
                <div class="metric-label">Procedure Features</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{len([c for c in transactions_matrix.columns if c.startswith('Diagnosis_')]):,}</div>
                <div class="metric-label">Diagnosis Features</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{len([c for c in transactions_matrix.columns if c.startswith(('Gender_', 'Age_')) or c == 'anchor_age']):,}</div>
                <div class="metric-label">Demographic Features</div>
            </div>
        </div>
        
        <h2>Association Rules Summary</h2>
    """
    
    # Add rule metrics if available
    if not rules_df.empty:
        html_content += f"""
        <div class="metrics">
            <div class="metric-box">
                <div class="metric-value">{len(rules_df):,}</div>
                <div class="metric-label">Total Rules</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{rules_df['support'].mean():.4f}</div>
                <div class="metric-label">Average Support</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{rules_df['confidence'].mean():.4f}</div>
                <div class="metric-label">Average Confidence</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{rules_df['lift'].mean():.4f}</div>
                <div class="metric-label">Average Lift</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{rules_df['lift'].max():.4f}</div>
                <div class="metric-label">Maximum Lift</div>
            </div>
        </div>
        """
    else:
        html_content += "<p>No association rules were generated.</p>"
    
    # Add procedure rules section if available
    if not procedure_rules.empty:
        html_content += f"""
        <h2>Procedure Rules</h2>
        <p>Found {len(procedure_rules)} rules with procedures in the consequent.</p>
        
        <h3>Top Rules by Lift</h3>
        <table>
            <tr>
                <th>Antecedents</th>
                <th>Consequents</th>
                <th>Support</th>
                <th>Confidence</th>
                <th>Lift</th>
            </tr>
        """
        
        # Add top 10 rules by lift
        for _, rule in procedure_rules.sort_values('lift', ascending=False).head(10).iterrows():
            # Format antecedents and consequents
            antecedents_str = ', '.join([item.split('_', 1)[1] if '_' in item else item for item in rule['antecedents']])
            consequents_str = ', '.join([item.split('_', 1)[1] if '_' in item else item for item in rule['consequents']])
            
            html_content += f"""
            <tr>
                <td>{antecedents_str}</td>
                <td>{consequents_str}</td>
                <td>{rule['support']:.4f}</td>
                <td>{rule['confidence']:.4f}</td>
                <td>{rule['lift']:.4f}</td>
            </tr>
            """
        
        html_content += "</table>"
    else:
        html_content += "<h2>Procedure Rules</h2><p>No procedure rules were found.</p>"
    
    # Add visualization section
    html_content += """
    <h2>Visualizations</h2>
    <p>The following visualizations provide insight into the association rules and feature distributions:</p>
    """
    
    # Check for visualization files
    visualizations = []
    for viz_file, desc in [
        ('feature_distribution.png', 'Distribution of features in the transaction matrix'),
        ('rule_metrics.png', 'Scatter plots of rule metrics (support, confidence, lift)'),
        ('rules_summary.png', 'Summary of rule statistics'),
        ('procedure_rules_network.png', 'Network visualization of top procedure rules')
    ]:
        viz_path = os.path.join(output_dir, viz_file)
        if os.path.exists(viz_path):
            visualizations.append((viz_file, desc))
    
    if visualizations:
        for viz_file, desc in visualizations:
            html_content += f"""
            <h3>{desc}</h3>
            <img src="{viz_file}" alt="{desc}">
            """
    else:
        html_content += "<p>No visualizations available.</p>"
    
    # Finish HTML content
    html_content += """
    <h2>Conclusion</h2>
    <p>This report provides an overview of the association rules mined from the medical dataset. 
    The rules can be used to understand relationships between patient demographics, diagnoses, and procedures.</p>
    </body>
    </html>
    """
    
    # Write the HTML file
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"HTML report saved to {report_path}")
    return report_path