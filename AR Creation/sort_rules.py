import pandas as pd
import ast
import os
import re

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

# Load the diagnosis_to_proc_demo_rules.csv file
try:
    rules_df = pd.read_csv('output/diagnosis_to_proc_demo_rules.csv')
    print(f"Successfully loaded file with {len(rules_df)} rules")
    
    # Display the first few rows to understand the structure
    print("\nSample of rules:")
    print(rules_df.head(2))
    
    # Check the column structure
    print("\nColumns in the file:", rules_df.columns.tolist())
except Exception as e:
    print(f"Error loading file: {str(e)}")
    print("Trying alternative path...")
    try:
        rules_df = pd.read_csv('diagnosis_to_proc_demo_rules.csv')
        print(f"Successfully loaded file with {len(rules_df)} rules from current directory")
    except Exception as e2:
        print(f"Error loading from alternative path: {str(e2)}")
        # Create a sample dataframe for testing if all else fails
        rules_df = pd.DataFrame({
            'antecedents': ["{'Diagnosis_Sepsis'}", "{'Diagnosis_Heart failure'}"],
            'consequents': ["{'Procedure_Central venous catheter'}", "{'Gender_F'}"],
            'support': [0.01, 0.02],
            'confidence': [0.7, 0.6],
            'lift': [5.0, 3.0]
        })
        print("Created sample dataframe for testing")

# Function to extract item names from sets, frozen sets, or string representations
def extract_items(item_set):
    if isinstance(item_set, str):
        # Handle string representations - could be frozenset or various formats
        try:
            # Try to evaluate as Python expression
            items = ast.literal_eval(item_set)
            if isinstance(items, (set, frozenset)):
                return list(items)
            return [items]
        except:
            # If that fails, try regex to extract items
            matches = re.findall(r"['\"](.*?)['\"]", item_set)
            if matches:
                return matches
            # Last resort: treat the whole string as one item
            return [item_set]
    elif isinstance(item_set, (set, frozenset)):
        return list(item_set)
    else:
        return [item_set]

# Define function to check if a rule is procedure-only, demographic-only, or mixed
def categorize_rule(consequents):
    items = extract_items(consequents)
    
    # Check for procedure and demographic patterns
    procedure_items = [item for item in items if 'Procedure_' in str(item)]
    demographic_items = [item for item in items if 'Gender_' in str(item) or 'Age_' in str(item)]
    
    if procedure_items and not demographic_items:
        return 'procedure'
    elif demographic_items and not procedure_items:
        return 'demographic'
    elif procedure_items and demographic_items:
        return 'mixed'
    else:
        return 'other'
    
# Add a category column to the rules
rules_df['rule_category'] = rules_df['consequents'].apply(categorize_rule)

# Print example categorizations for debugging
print("\nExample categorizations:")
for category in ['procedure', 'demographic', 'mixed', 'other']:
    examples = rules_df[rules_df['rule_category'] == category].head(1)
    for i, row in examples.iterrows():
        print(f"Category: {category}")
        print(f"- Antecedents: {row['antecedents']}")
        print(f"- Consequents: {row['consequents']}")
        print()

# Count rules by category
category_counts = rules_df['rule_category'].value_counts()
print("\nRule categories:")
for category, count in category_counts.items():
    print(f"- {category}: {count}")

# Split into separate dataframes
procedure_rules = rules_df[rules_df['rule_category'] == 'procedure']
demographic_rules = rules_df[rules_df['rule_category'] == 'demographic']
mixed_rules = rules_df[rules_df['rule_category'] == 'mixed']
other_rules = rules_df[rules_df['rule_category'] == 'other']

# Save to separate CSV files
try:
    procedure_rules.to_csv('output/diagnosis_to_procedure_rules.csv', index=False)
    demographic_rules.to_csv('output/diagnosis_to_demographic_rules.csv', index=False)
    
    if not mixed_rules.empty:
        mixed_rules.to_csv('output/diagnosis_to_mixed_rules.csv', index=False)
    
    if not other_rules.empty:
        other_rules.to_csv('output/diagnosis_to_other_rules.csv', index=False)
    
    print("\nSaved separate rule files:")
    print(f"- diagnosis_to_procedure_rules.csv: {len(procedure_rules)} rules")
    print(f"- diagnosis_to_demographic_rules.csv: {len(demographic_rules)} rules")
    
    if not mixed_rules.empty:
        print(f"- diagnosis_to_mixed_rules.csv: {len(mixed_rules)} rules")
    
    if not other_rules.empty:
        print(f"- diagnosis_to_other_rules.csv: {len(other_rules)} rules")
except Exception as e:
    print(f"Error saving files: {str(e)}")

# Additional analysis: Display top rules by lift for each category if available
if not procedure_rules.empty:
    print("\nTop procedure rules by lift:")
    top_proc = procedure_rules.sort_values('lift', ascending=False).head(3)
    for i, row in top_proc.iterrows():
        print(f"- {row['antecedents']} → {row['consequents']} (Lift: {row['lift']:.2f}, Confidence: {row['confidence']:.2f})")

if not demographic_rules.empty:
    print("\nTop demographic rules by lift:")
    top_demo = demographic_rules.sort_values('lift', ascending=False).head(3)
    for i, row in top_demo.iterrows():
        print(f"- {row['antecedents']} → {row['consequents']} (Lift: {row['lift']:.2f}, Confidence: {row['confidence']:.2f})")

print("\nDone! You can now analyze procedure rules and demographic rules separately.")