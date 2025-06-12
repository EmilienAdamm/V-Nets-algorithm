from flask import Flask, request, jsonify
import pandas as pd
import io
import networkx as nx
from uuid import uuid4

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

def vnda(sequences):
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
        G.add_node(e, init=(e in INIT), end=(e in END))
    for e_i, e_j, [i_minus, i_plus] in T:
        G.add_edge(e_i, e_j, label=f"{e_i}^1[{i_minus},{i_plus}]^1{e_j}")

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
        elements.append({
            'data': {
                'id': node,
                'init': data.get('init', False),
                'end': data.get('end', False)
            }
        })
    for source, target, data in G.edges(data=True):
        elements.append({
            'data': {
                'source': source,
                'target': target,
                'label': data['label']
            }
        })
    return elements

@app.route('/')
def index():
    """Serve the main page."""
    with open('index.html', 'r') as f:
        return f.read()

@app.route('/process', methods=['POST'])
def process():
    """Process event sequence input and generate V-net."""
    try:
        if 'file' in request.files:
            file = request.files['file']
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.filename.endswith('.txt'):
                df = pd.read_csv(file, sep='\t')
            else:
                return jsonify({'error': 'Unsupported file format. Use .csv or .txt'}), 400
        else:
            data = request.json
            if data['type'] != 'manual':
                return jsonify({'error': 'Invalid input type'}), 400
            df = pd.read_csv(io.StringIO(data['data']))
        
        valid, error = validate_data(df)
        if not valid:
            return jsonify({'error': error}), 400

        vnet = vnda(df)
        textual_output = format_textual_output(vnet)
        graph_output = convert_graph_to_cytoscape(vnet['graph'])

        return jsonify({
            'textual': textual_output,
            'graph': graph_output
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)