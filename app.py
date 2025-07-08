from flask import Flask, request, jsonify
import pandas as pd
import io
import networkx as nx
from uuid import uuid4
import re

app = Flask(__name__)

def validate_data(df):
    """Validate event sequence data."""
    if not all(col in df.columns for col in ['SequenceID', 'EventType', 'Timestamp']):
        return False, "Missing required columns: SequenceID, EventType, Timestamp"
    if not df['Timestamp'].apply(lambda x: isinstance(x, (int, float))).all():
        return False, "Timestamps must be numeric"
    if not df['EventType'].apply(lambda x: isinstance(x, str) and x.isalnum()).all():
        return False, "Event types must be alphanumeric strings"
    return True, ""

def calculate_intervals(group):
    """Calculate temporal intervals between consecutive events."""
    intervals = []
    for i in range(len(group) - 1):
        e_i, t_i = group.iloc[i]['EventType'], group.iloc[i]['Timestamp']
        e_j, t_j = group.iloc[i + 1]['EventType'], group.iloc[i + 1]['Timestamp']
        delta = round(t_j - t_i, 2)
        intervals.append((e_i, e_j, [delta, delta]))
    return intervals

def vnda(sequences, constraints=None):
    """Implement Vásquez Net Discovery Algorithm (VNDA)."""
    # Initialize V-net components
    E = set(sequences['EventType'])
    INIT = set(sequences.groupby('SequenceID').first()['EventType'])
    END = set(sequences.groupby('SequenceID').last()['EventType'])
    Frec = sequences['EventType'].value_counts().to_dict()
    T = []
    R = []

    # Calculate tl_eval (maximum time length across sequences)
    time_lengths = sequences.groupby('SequenceID')['Timestamp'].apply(
        lambda g: g.max() - g.min()
    )
    tl_eval = round(time_lengths.max(), 6)

    # Calculate temporal constraints
    for seq_id, group in sequences.groupby('SequenceID'):
        T.extend(calculate_intervals(group))
    
    # Apply user-defined constraints if provided
    constraint_based_edges = set()
    if constraints is not None:
        # Create a lookup dictionary for constraints
        constraint_dict = {}
        for _, row in constraints.iterrows():
            key = (row['Event1ID'], row['Event2ID'])
            constraint_dict[key] = [row['MinTime'], row['MaxTime']]
        
        # Update T with constraint values where applicable
        updated_T = []
        for e_i, e_j, [i_minus, i_plus] in T:
            if (e_i, e_j) in constraint_dict:
                # Use constraint values and mark as constraint-based
                updated_T.append((e_i, e_j, constraint_dict[(e_i, e_j)]))
                constraint_based_edges.add((e_i, e_j))
            else:
                # Keep original calculated values
                updated_T.append((e_i, e_j, [i_minus, i_plus]))
        T = updated_T
        
        # Add any additional constraints that weren't found in the calculated intervals
        existing_pairs = {(e_i, e_j) for e_i, e_j, _ in T}
        for (e_i, e_j), [min_time, max_time] in constraint_dict.items():
            if (e_i, e_j) not in existing_pairs and e_i in E and e_j in E:
                T.append((e_i, e_j, [min_time, max_time]))
                constraint_based_edges.add((e_i, e_j))

    # Construct occurrence matrices
    matrices = []
    for seq_id, group in sequences.groupby('SequenceID'):
        matrix = {e: [0] * (len(group) + 1) for e in E}
        matrix['INIT'] = [0] * (len(group) + 1)
        matrix['END'] = [0] * (len(group) + 1)
        matrix['INIT'][0] = 1
        matrix['END'][-1] = 1
        for i, event in enumerate(group['EventType']):
            matrix[event][i + 1] = 1
        matrices.append(matrix)

    # Define logical predicates (simplified)
    R.append(f"INIT({','.join(INIT)})")
    R.append(f"END({','.join(END)})")

    # Construct graph topology
    G = nx.DiGraph()
    for e in E:
        is_init = e in INIT
        is_end = e in END
        G.add_node(e, init=is_init, end=is_end)
        
    for e_i, e_j, [i_minus, i_plus] in T:
        is_constraint_based = (e_i, e_j) in constraint_based_edges
        G.add_edge(e_i, e_j, 
                  label=f"{e_i}^1[{i_minus},{i_plus}]^1{e_j}",
                  constraint_based=is_constraint_based)

    return {
        'E': list(E),
        'T': T,
        'Frec': Frec,
        'R': R,
        'INIT': list(INIT),
        'END': list(END),
        'tl_eval': tl_eval,
        'matrices': matrices,
        'graph': G
    }

def format_textual_output(vnet):
    """Format V-net components as text."""
    lines = ["V-net Definition:"]
    lines.append(f"E = {{{', '.join(vnet['E'])}}}")
    lines.append("T = {" + ", ".join(f"{e_i}^1[{i_minus},{i_plus}]^1{e_j}" for e_i, e_j, [i_minus, i_plus] in vnet['T']) + "}")
    lines.append(f"Frec = {{{', '.join(f'{k}: {v}' for k, v in vnet['Frec'].items())}}}")
    lines.append(f"R = {{{', '.join(vnet['R'])}}}")
    lines.append(f"INIT = {{{', '.join(vnet['INIT'])}}}")
    lines.append(f"END = {{{', '.join(vnet['END'])}}}")
    lines.append(f"tl_eval = {vnet['tl_eval']}")
    for i, matrix in enumerate(vnet['matrices'], 1):
        lines.append(f"\nMATRIX {i}")
        lines.append("\t" + "\t".join([''] + list(vnet['E'])))
        for key in matrix:
            lines.append(f"\t{key}\t" + "\t".join(str(v) for v in matrix[key]))
    return "\n".join(lines)

def convert_graph_to_cytoscape(G):
    """Convert NetworkX graph to Cytoscape.js format."""
    elements = []
    for node, data in G.nodes(data=True):
        is_init = data.get('init', False)
        is_end = data.get('end', False)
        
        node_element = {
            'data': {
                'id': node,
                'init': 'true' if is_init else 'false',
                'end': 'true' if is_end else 'false'
            }
        }
        elements.append(node_element)
    for source, target, data in G.edges(data=True):
        elements.append({
            'data': {
                'source': source,
                'target': target,
                'label': data['label'],
                'constraint_based': data.get('constraint_based', False)
            }
        })
    return elements

def parse_warning_predicates(predicates_text):
    """Parse warning predicates from text input."""
    if not predicates_text or not predicates_text.strip():
        return []
    
    predicates = []
    lines = predicates_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        predicates.append(line)
    
    return predicates

def evaluate_frequency_predicate(predicate, vnet):
    """Evaluate frequency-based predicates like Frec(a)=2, Frec(b)>=1."""
    # Pattern: Frec(event)operator(value)
    pattern = r'Frec\(([a-zA-Z0-9_]+)\)\s*(>=|<=|>|<|=|!=)\s*(\d+)'
    match = re.match(pattern, predicate)
    
    if not match:
        return False, f"Invalid frequency predicate format: {predicate}"
    
    event, operator, value = match.groups()
    value = int(value)
    
    if event not in vnet['Frec']:
        actual_freq = 0
    else:
        actual_freq = vnet['Frec'][event]
    
    if operator == '=':
        result = actual_freq == value
    elif operator == '>=':
        result = actual_freq >= value
    elif operator == '<=':
        result = actual_freq <= value
    elif operator == '>':
        result = actual_freq > value
    elif operator == '<':
        result = actual_freq < value
    elif operator == '!=':
        result = actual_freq != value
    else:
        return False, f"Unsupported operator: {operator}"
    
    return result, f"Frec({event}) = {actual_freq} {operator} {value}"

def evaluate_temporal_predicate(predicate, vnet):
    """Evaluate temporal predicates like 'a->b before c'."""
    # Pattern: event1->event2 before event3
    pattern = r'([a-zA-Z0-9_]+)->([a-zA-Z0-9_]+)\s+before\s+([a-zA-Z0-9_]+)'
    match = re.match(pattern, predicate)
    
    if not match:
        return False, f"Invalid temporal predicate format: {predicate}"
    
    event1, event2, event3 = match.groups()
    
    # Check if the edge event1->event2 exists and event3 exists
    G = vnet['graph']
    
    if not G.has_edge(event1, event2):
        return False, f"Edge {event1}->{event2} not found in V-net"
    
    if event3 not in G.nodes():
        return False, f"Event {event3} not found in V-net"
    
    # For this implementation, we'll check if the transition exists
    # In a more sophisticated version, you could check actual temporal ordering
    return True, f"Transition {event1}->{event2} exists and {event3} is present"

def evaluate_logical_predicate(predicate, vnet):
    """Evaluate logical predicates with AND (∧) operations."""
    # Split by logical AND operator
    if '∧' in predicate:
        parts = predicate.split('∧')
        results = []
        details = []
        
        for part in parts:
            part = part.strip()
            if part.startswith('Frec('):
                result, detail = evaluate_frequency_predicate(part, vnet)
            elif '->' in part and 'before' in part:
                result, detail = evaluate_temporal_predicate(part, vnet)
            else:
                result, detail = False, f"Unknown predicate type: {part}"
            
            results.append(result)
            details.append(detail)
        
        # All parts must be true for AND operation
        final_result = all(results)
        return final_result, " AND ".join(details)
    
    # Single predicate
    if predicate.startswith('Frec('):
        return evaluate_frequency_predicate(predicate, vnet)
    elif '->' in predicate and 'before' in predicate:
        return evaluate_temporal_predicate(predicate, vnet)
    else:
        return False, f"Unknown predicate type: {predicate}"

def check_warning_predicates(predicates, vnet):
    """Check all warning predicates against the V-net."""
    warnings = []
    
    for predicate in predicates:
        try:
            result, detail = evaluate_logical_predicate(predicate, vnet)
            if result:
                warnings.append({
                    'predicate': predicate,
                    'matched': True,
                    'detail': detail,
                    'message': f"WARNING: Condition matched - {detail}"
                })
            else:
                # Still include non-matched predicates for debugging
                warnings.append({
                    'predicate': predicate,
                    'matched': False,
                    'detail': detail,
                    'message': f"OK: Condition not matched - {detail}"
                })
        except Exception as e:
            warnings.append({
                'predicate': predicate,
                'matched': False,
                'detail': str(e),
                'message': f"ERROR: Failed to evaluate predicate - {str(e)}"
            })
    
    return warnings

@app.route('/')
def index():
    """Serve the main page."""
    with open('index.html', 'r') as f:
        return f.read()

@app.route('/process', methods=['POST'])
def process():
    """Process event sequence input and generate V-net."""
    try:
        predicates = None
        
        if 'file' in request.files:
            # File mode
            file = request.files['file']
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.filename.endswith('.txt'):
                df = pd.read_csv(file, sep='\t')
            else:
                return jsonify({'error': 'Unsupported file format. Use .csv or .txt'}), 400
            
            # Constraints file
            constraints = None
            if 'constraints' in request.files and request.files['constraints'].filename:
                constraints = pd.read_csv(request.files['constraints'])
                if not all(col in constraints.columns for col in ['Event1ID', 'Event2ID', 'MinTime', 'MaxTime']):
                    return jsonify({'error': 'Constraints file must contain Event1ID, Event2ID, MinTime, MaxTime columns'}), 400
            
            # Predicates file
            if 'predicates' in request.files and request.files['predicates'].filename:
                predicates_file = request.files['predicates']
                predicates_text = predicates_file.read().decode('utf-8')
                predicates = parse_warning_predicates(predicates_text)
                
        else:
            # Manual mode
            data = request.json
            if data['type'] != 'manual':
                return jsonify({'error': 'Invalid input type'}), 400
            df = pd.read_csv(io.StringIO(data['data']))
            
            constraints = None
            if data.get('constraints'):
                constraints = pd.read_csv(io.StringIO(data['constraints']))
                if not all(col in constraints.columns for col in ['Event1ID', 'Event2ID', 'MinTime', 'MaxTime']):
                    return jsonify({'error': 'Constraints file must contain Event1ID, Event2ID, MinTime, MaxTime columns'}), 400
            
            # Parse predicates from manual input
            if data.get('predicates'):
                predicates = parse_warning_predicates(data['predicates'])
        
        valid, error = validate_data(df)
        if not valid:
            return jsonify({'error': error}), 400

        vnet = vnda(df, constraints=constraints)
        textual_output = format_textual_output(vnet)
        graph_output = convert_graph_to_cytoscape(vnet['graph'])
        
        # Evaluate warning predicates if provided
        warnings = []
        if predicates:
            warnings = check_warning_predicates(predicates, vnet)

        return jsonify({
            'textual': textual_output,
            'graph': graph_output,
            'warnings': warnings
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)