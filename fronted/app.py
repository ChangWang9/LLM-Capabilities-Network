"""
LLM Capability Network Visualization System
Nature Neuroscience Style with PyVis
"""

import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
import json
import os
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from pathlib import Path
import webbrowser
from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
import colorsys

# ==================== Data Classes ====================

@dataclass
class Paper:
    """Represents a research paper"""
    name: str
    url: Optional[str] = None
    capabilities: List[str] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []

@dataclass
class Capability:
    """Represents a capability node"""
    name: str
    count: int
    papers: List[Paper]
    color: str = "#6B46C1"
    
@dataclass
class Edge:
    """Represents an edge between capabilities"""
    source: str
    target: str
    weight: int
    papers: List[Paper]

# ==================== Network Builder ====================

class LLMCapabilityNetwork:
    """
    Main class for building and visualizing the LLM capability network
    with Nature neuroscience-inspired styling
    """
    
    # Nature Neuroscience color palette
    COLORS = {
        'high_freq': '#1FA2FF',      # Electric blue (>500)
        'mid_freq': '#12D8FA',       # Cyan (100-500)
        'low_freq': '#6B46C1',       # Neural purple (<100)
        'synapse': '#2563EB',        # Synapse blue
        'glia': '#10B981',           # Glia green
        'background': '#0F0F1E',     # Dark background
        'edge_default': 'rgba(31, 162, 255, 0.35)',
        'edge_strong': 'rgba(18, 216, 250, 0.6)',
        'edge_medium': 'rgba(14, 165, 233, 0.5)',
        'edge_weak': 'rgba(107, 70, 193, 0.25)'
    }
    
    def __init__(self, data_path: Optional[str] = None):
        """Initialize the network builder"""
        self.graph = nx.DiGraph()
        self.capabilities: Dict[str, Capability] = {}
        self.edges: List[Edge] = []
        self.papers: Dict[str, Paper] = {}
        
        if data_path:
            self.load_data(data_path)
    
    def normalize_capability_name(self, name: str) -> str:
        """Normalize capability names for consistency"""
        if not name:
            return ''
        return (name.strip().lower()
                .replace('"', '')
                .replace("'", '')
                .replace('capability', '')
                .strip())
    
    def load_data(self, file_path: str):
        """Load and process Excel data"""
        print(f"Loading data from {file_path}...")
        
        # Read Excel file
        df = pd.read_excel(file_path)
        
        # Process each row
        for idx, row in df.iterrows():
            paper_name = row.get('paper_name') or row.get('actual_file_name', f'Paper_{idx}')
            paper_url = row.get('url', None)
            
            # Parse capabilities
            capabilities_str = row.get('capability', '')
            capabilities = self._parse_capabilities(capabilities_str)
            
            # Create paper object
            paper = Paper(name=paper_name, url=paper_url, capabilities=capabilities)
            self.papers[paper_name] = paper
            
            # Update capability statistics
            for cap in capabilities:
                cap_norm = self.normalize_capability_name(cap)
                if not cap_norm:
                    continue
                    
                if cap_norm not in self.capabilities:
                    self.capabilities[cap_norm] = Capability(
                        name=cap_norm,
                        count=0,
                        papers=[],
                        color=self._get_node_color(0)
                    )
                
                self.capabilities[cap_norm].count += 1
                self.capabilities[cap_norm].papers.append(paper)
            
            # Parse edges
            edges_str = row.get('Edges', '')
            edge_pairs = self._parse_edges(edges_str)
            
            for source, target in edge_pairs:
                source_norm = self.normalize_capability_name(source)
                target_norm = self.normalize_capability_name(target)
                
                if source_norm and target_norm and source_norm != target_norm:
                    # Find or create edge
                    edge_found = False
                    for edge in self.edges:
                        if edge.source == source_norm and edge.target == target_norm:
                            edge.weight += 1
                            edge.papers.append(paper)
                            edge_found = True
                            break
                    
                    if not edge_found:
                        self.edges.append(Edge(
                            source=source_norm,
                            target=target_norm,
                            weight=1,
                            papers=[paper]
                        ))
        
        # Update colors based on final counts
        for cap in self.capabilities.values():
            cap.color = self._get_node_color(cap.count)
        
        print(f"Loaded {len(self.capabilities)} capabilities and {len(self.edges)} edges")
    
    def _parse_capabilities(self, cap_str: str) -> List[str]:
        """Parse capability string from Excel"""
        if not cap_str or pd.isna(cap_str):
            return []
        
        cap_str = str(cap_str).strip()
        
        try:
            if cap_str.startswith('[') and cap_str.endswith(']'):
                # Parse as list
                import ast
                return ast.literal_eval(cap_str)
            else:
                return [cap_str]
        except:
            return [cap_str] if cap_str else []
    
    def _parse_edges(self, edge_str: str) -> List[Tuple[str, str]]:
        """Parse edge string from Excel"""
        if not edge_str or pd.isna(edge_str):
            return []
        
        edge_str = str(edge_str).strip()
        edges = []
        
        try:
            if edge_str.startswith('[') and edge_str.endswith(']'):
                import ast
                edge_list = ast.literal_eval(edge_str)
                for edge in edge_list:
                    if isinstance(edge, list) and len(edge) == 2:
                        edges.append((edge[0], edge[1]))
            else:
                # Try simple comma split
                parts = edge_str.split(',')
                if len(parts) >= 2:
                    edges.append((parts[0].strip(), parts[1].strip()))
        except:
            pass
        
        return edges
    
    def _get_node_color(self, count: int) -> str:
        """Get node color based on frequency"""
        if count > 500:
            return self.COLORS['high_freq']
        elif count > 100:
            return self.COLORS['mid_freq']
        else:
            return self.COLORS['low_freq']
    
    def _get_edge_color(self, weight: int) -> str:
        """Get edge color based on weight"""
        if weight > 10:
            return self.COLORS['edge_strong']
        elif weight > 5:
            return self.COLORS['edge_medium']
        else:
            return self.COLORS['edge_weak']
    
    def build_network(self, min_frequency: int = 500, min_connections: int = 2) -> nx.DiGraph:
        """Build NetworkX graph with filters"""
        self.graph.clear()
        
        # Add nodes (filtered by frequency)
        for cap_name, cap in self.capabilities.items():
            if cap.count >= min_frequency:
                self.graph.add_node(
                    cap_name,
                    label=cap_name,
                    size=np.sqrt(cap.count) * 2 + 20,
                    color=cap.color,
                    title=f"{cap_name}<br>Frequency: {cap.count}<br>Papers: {len(cap.papers)}",
                    value=cap.count,
                    papers=[p.name for p in cap.papers[:5]]  # Store top 5 papers
                )
        
        # Add edges (filtered by weight)
        for edge in self.edges:
            if (edge.weight >= min_connections and 
                edge.source in self.graph.nodes and 
                edge.target in self.graph.nodes):
                self.graph.add_edge(
                    edge.source,
                    edge.target,
                    weight=edge.weight,
                    color=self._get_edge_color(edge.weight),
                    title=f"Co-occurrence: {edge.weight}",
                    papers=[p.name for p in edge.papers[:3]]
                )
        
        return self.graph
    
    def create_pyvis_network(self, output_path: str = "network.html",
                            min_frequency: int = 500, 
                            min_connections: int = 2,
                            physics_enabled: bool = True) -> Network:
        """Create interactive PyVis network with Nature neuroscience styling"""
        
        # Build the graph
        self.build_network(min_frequency, min_connections)
        
        # Initialize PyVis network with custom settings
        net = Network(
            height="100vh",
            width="100%",
            bgcolor=self.COLORS['background'],
            font_color="#E5E7EB",
            directed=True
        )
        
        # Add custom physics for neuroscience-like layout
        if physics_enabled:
            net.set_options("""
            {
                "nodes": {
                    "borderWidth": 2,
                    "borderWidthSelected": 4,
                    "font": {
                        "size": 14,
                        "strokeWidth": 3,
                        "strokeColor": "#0F0F1E"
                    },
                    "shadow": {
                        "enabled": true,
                        "color": "rgba(14, 165, 233, 0.3)",
                        "size": 10,
                        "x": 0,
                        "y": 0
                    }
                },
                "edges": {
                    "smooth": {
                        "type": "continuous",
                        "roundness": 0.5
                    },
                    "arrows": {
                        "to": {
                            "enabled": true,
                            "scaleFactor": 0.5
                        }
                    },
                    "shadow": {
                        "enabled": true,
                        "color": "rgba(37, 99, 235, 0.2)",
                        "size": 5
                    },
                    "width": 2
                },
                "physics": {
                    "enabled": true,
                    "forceAtlas2Based": {
                        "gravitationalConstant": -50,
                        "centralGravity": 0.005,
                        "springLength": 230,
                        "springConstant": 0.08,
                        "damping": 0.4,
                        "avoidOverlap": 0.5
                    },
                    "maxVelocity": 50,
                    "minVelocity": 0.1,
                    "solver": "forceAtlas2Based",
                    "stabilization": {
                        "enabled": true,
                        "iterations": 1000,
                        "updateInterval": 50
                    }
                },
                "interaction": {
                    "hover": true,
                    "tooltipDelay": 100,
                    "navigationButtons": true,
                    "keyboard": true
                }
            }
            """)
        
        # Add nodes and edges from NetworkX graph
        net.from_nx(self.graph)
        
        # Generate HTML with custom styling
        html = net.generate_html()
        
        # Inject custom CSS for Nature neuroscience style
        custom_css = """
        <style>
            body {
                background: linear-gradient(135deg, #0F0F1E 0%, #1A1A2E 50%, #16213E 100%);
                margin: 0;
                padding: 0;
            }
            
            #mynetwork {
                background: radial-gradient(ellipse at center, rgba(14, 165, 233, 0.05) 0%, transparent 70%);
            }
            
            .vis-tooltip {
                background: rgba(26, 26, 46, 0.95) !important;
                border: 1px solid rgba(255, 255, 255, 0.2) !important;
                border-radius: 8px !important;
                color: #E5E7EB !important;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
                padding: 12px !important;
                backdrop-filter: blur(10px) !important;
            }
        </style>
        """
        
        # Insert custom CSS before </head>
        html = html.replace('</head>', custom_css + '</head>')
        
        # Save the network
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return net
    
    def find_paths(self, source: str, target: str, max_paths: int = 10) -> List[List[str]]:
        """Find all paths between two capabilities"""
        source_norm = self.normalize_capability_name(source)
        target_norm = self.normalize_capability_name(target)
        
        if source_norm not in self.graph or target_norm not in self.graph:
            return []
        
        try:
            # Use NetworkX to find all simple paths
            paths = list(nx.all_simple_paths(
                self.graph, 
                source_norm, 
                target_norm, 
                cutoff=5  # Max path length
            ))
            return paths[:max_paths]
        except nx.NetworkXNoPath:
            return []
    
    def get_subgraph(self, nodes: List[str], include_neighbors: bool = True) -> nx.DiGraph:
        """Get subgraph containing specified nodes and optionally their neighbors"""
        nodes_norm = [self.normalize_capability_name(n) for n in nodes]
        
        if include_neighbors:
            # Include all neighbors
            extended_nodes = set(nodes_norm)
            for node in nodes_norm:
                if node in self.graph:
                    extended_nodes.update(self.graph.neighbors(node))
                    extended_nodes.update(self.graph.predecessors(node))
            nodes_to_include = list(extended_nodes)
        else:
            nodes_to_include = nodes_norm
        
        return self.graph.subgraph(nodes_to_include)
    
    def get_statistics(self) -> Dict:
        """Get network statistics"""
        if not self.graph:
            self.build_network()
        
        stats = {
            'total_capabilities': len(self.capabilities),
            'total_papers': len(self.papers),
            'total_edges': len(self.edges),
            'graph_nodes': self.graph.number_of_nodes(),
            'graph_edges': self.graph.number_of_edges(),
            'avg_degree': np.mean([d for n, d in self.graph.degree()]) if self.graph.number_of_nodes() > 0 else 0,
            'density': nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0,
            'connected_components': nx.number_weakly_connected_components(self.graph),
            'most_common_capabilities': sorted(
                [(cap.name, cap.count) for cap in self.capabilities.values()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }
        
        return stats

# ==================== Flask Web Application ====================

app = Flask(__name__)
CORS(app)

# Global network instance
network_instance = None

# Try to auto-load default dataset from uploads if present
try:
    default_upload = Path('uploads') / 'Edges_merged_cleaned.xlsx'
    if default_upload.exists():
        print(f"Auto-loading dataset: {default_upload}")
        network_instance = LLMCapabilityNetwork(str(default_upload))
        print("✓ Default dataset loaded for API endpoints")
except Exception as e:
    print(f"Failed to auto-load default dataset: {e}")

@app.route('/')
def index():
    """Serve the main visualization page with fallback if template is empty"""
    index_path = Path('templates/index.html')
    try:
        if index_path.exists() and index_path.stat().st_size > 0:
            return render_template_string(index_path.read_text(encoding='utf-8'))
    except Exception:
        pass

    # Fallback to root-level network.html if present
    fallback_path = Path('network.html')
    try:
        if fallback_path.exists() and fallback_path.stat().st_size > 0:
            return render_template_string(fallback_path.read_text(encoding='utf-8'))
    except Exception:
        pass

    # Final minimal placeholder
    return (
        "<html><head><meta charset='utf-8'><title>LLM Capability Network</title></head>"
        "<body style='font-family:Segoe UI,Arial,sans-serif;padding:24px;background:#0F0F1E;color:#E5E7EB'>"
        "<h2>服务已启动，但未找到可显示的页面</h2>"
        "<p>请在 <code>templates/index.html</code> 中添加前端模板，或在项目根目录放置 <code>network.html</code>。</p>"
        "</body></html>"
    )

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    global network_instance
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save uploaded file
    upload_path = Path('uploads')
    upload_path.mkdir(exist_ok=True)
    file_path = upload_path / file.filename
    file.save(str(file_path))
    
    # Process the file
    try:
        network_instance = LLMCapabilityNetwork(str(file_path))
        stats = network_instance.get_statistics()
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/network', methods=['GET'])
def get_network():
    """Get network data"""
    if not network_instance:
        return jsonify({'error': 'No data loaded'}), 400
    
    min_freq = int(request.args.get('min_frequency', 500))
    min_conn = int(request.args.get('min_connections', 2))
    
    # Build network with filters
    graph = network_instance.build_network(min_freq, min_conn)
    
    # Convert to JSON format
    nodes = []
    for node_id, attrs in graph.nodes(data=True):
        nodes.append({
            'id': node_id,
            'label': attrs.get('label', node_id),
            'size': attrs.get('size', 30),
            'color': attrs.get('color', '#6B46C1'),
            'value': attrs.get('value', 1),
            'title': attrs.get('title', ''),
            'papers': attrs.get('papers', [])
        })
    
    edges = []
    for source, target, attrs in graph.edges(data=True):
        edges.append({
            'from': source,
            'to': target,
            'weight': attrs.get('weight', 1),
            'color': attrs.get('color', 'rgba(37, 99, 235, 0.4)'),
            'title': attrs.get('title', ''),
            'papers': attrs.get('papers', [])
        })
    
    return jsonify({
        'nodes': nodes,
        'edges': edges
    })

@app.route('/api/paths', methods=['POST'])
def find_paths():
    """Find paths between two nodes"""
    if not network_instance:
        return jsonify({'error': 'No data loaded'}), 400
    
    data = request.json
    source = data.get('source')
    target = data.get('target')
    
    if not source or not target:
        return jsonify({'error': 'Source and target required'}), 400
    
    paths = network_instance.find_paths(source, target)
    
    return jsonify({
        'paths': paths,
        'count': len(paths)
    })

@app.route('/api/node/<node_id>', methods=['GET'])
def get_node_info(node_id):
    """Get detailed information about a node"""
    if not network_instance:
        return jsonify({'error': 'No data loaded'}), 400
    
    node_norm = network_instance.normalize_capability_name(node_id)
    
    if node_norm not in network_instance.capabilities:
        return jsonify({'error': 'Node not found'}), 404
    
    cap = network_instance.capabilities[node_norm]
    
    return jsonify({
        'name': cap.name,
        'count': cap.count,
        'papers': [{'name': p.name, 'url': p.url} for p in cap.papers[:20]],
        'total_papers': len(cap.papers)
    })

@app.route('/api/edge', methods=['GET'])
def get_edge_info():
    """Get detailed information about an edge"""
    if not network_instance:
        return jsonify({'error': 'No data loaded'}), 400
    
    source = request.args.get('source')
    target = request.args.get('target')
    
    if not source or not target:
        return jsonify({'error': 'Source and target required'}), 400
    
    source_norm = network_instance.normalize_capability_name(source)
    target_norm = network_instance.normalize_capability_name(target)
    
    for edge in network_instance.edges:
        if edge.source == source_norm and edge.target == target_norm:
            return jsonify({
                'source': edge.source,
                'target': edge.target,
                'weight': edge.weight,
                'papers': [{'name': p.name, 'url': p.url} for p in edge.papers[:10]]
            })
    
    return jsonify({'error': 'Edge not found'}), 404

@app.route('/api/export', methods=['GET'])
def export_network():
    """Export network as PyVis HTML"""
    if not network_instance:
        return jsonify({'error': 'No data loaded'}), 400
    
    min_freq = int(request.args.get('min_frequency', 500))
    min_conn = int(request.args.get('min_connections', 2))
    
    output_path = 'exports/network_export.html'
    Path('exports').mkdir(exist_ok=True)
    
    network_instance.create_pyvis_network(
        output_path=output_path,
        min_frequency=min_freq,
        min_connections=min_conn
    )
    
    return jsonify({
        'success': True,
        'path': output_path
    })

# ==================== Main Execution ====================

def main():
    """Main function to run the application"""
    
    # Example: Create a sample network
    print("=" * 60)
    print("LLM Capability Network Visualization")
    print("Nature Neuroscience Style")
    print("=" * 60)
    
    # Check if data file exists
    data_file = "Edges_cleaned_updated_with_datasets_final_v4.xlsx"
    
    if os.path.exists(data_file):
        print(f"\nLoading data from {data_file}...")
        network = LLMCapabilityNetwork(data_file)
        
        # Print statistics
        stats = network.get_statistics()
        print(f"\nNetwork Statistics:")
        print(f"- Total Capabilities: {stats['total_capabilities']}")
        print(f"- Total Papers: {stats['total_papers']}")
        print(f"- Total Edges: {stats['total_edges']}")
        print(f"- Graph Nodes (filtered): {stats['graph_nodes']}")
        print(f"- Graph Edges (filtered): {stats['graph_edges']}")
        print(f"- Average Degree: {stats['avg_degree']:.2f}")
        print(f"- Network Density: {stats['density']:.4f}")
        
        print(f"\nTop 5 Most Common Capabilities:")
        for cap, count in stats['most_common_capabilities'][:5]:
            print(f"  - {cap}: {count}")
        
        # Create visualization
        print(f"\nGenerating interactive visualization...")
        network.create_pyvis_network(
            output_path="llm_capability_network.html",
            min_frequency=500,
            min_connections=2
        )
        
        print(f"✓ Visualization saved to 'llm_capability_network.html'")
        
        # Open in browser
        webbrowser.open('llm_capability_network.html')
    
    # Start Flask app
    print(f"\nStarting web server...")
    print(f"Navigate to http://localhost:5000 to access the application")
    app.run(debug=True, port=5000)

if __name__ == "__main__":
    main()