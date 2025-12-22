import sys
import linecache
import re
import json
import base64
from pathlib import Path
import urllib.request


def _get_litegraph_js():
    """Fetch and cache LiteGraph.js"""
    cache_file = Path(__file__).parent / ".tmp" / "litegraph.min.js"
    if cache_file.exists():
        return cache_file.read_text(encoding='utf-8')
    
    url = "https://unpkg.com/litegraph.js@0.7.18/build/litegraph.min.js"
    with urllib.request.urlopen(url) as response:
        js = response.read().decode('utf-8')
    
    cache_file.parent.mkdir(exist_ok=True)
    cache_file.write_text(js, encoding='utf-8')
    return js


def _get_litegraph_css():
    """Fetch and cache LiteGraph.css"""
    cache_file = Path(__file__).parent / ".tmp" / "litegraph.css"
    if cache_file.exists():
        return cache_file.read_text(encoding='utf-8')
    
    url = "https://unpkg.com/litegraph.js@0.7.18/css/litegraph.css"
    with urllib.request.urlopen(url) as response:
        css = response.read().decode('utf-8')
    
    cache_file.parent.mkdir(exist_ok=True)
    cache_file.write_text(css, encoding='utf-8')
    return css


class NameAware:
    def __init__(self, name_override=None):
        self.name, self.parent = self._find_assignment_context()
        if name_override is not None:
            self.name = name_override
        self.viz_params = {}

    def _find_assignment_context(self):
        for depth in range(1, 20):
            try:
                frame = sys._getframe(depth)
            except ValueError:
                return None, None
            
            if frame.f_code.co_qualname.startswith('NameAware'):
                continue
            
            line = linecache.getline(frame.f_code.co_filename, frame.f_lineno).strip()
            
            if 'super()' in line:
                continue
            
            match = re.match(r'self\.(\w+)\s*=', line)
            if match:
                return match.group(1), frame.f_locals.get('self')
            
            match = re.match(r'(\w+)\s*=', line)
            if match:
                return match.group(1), None
        
        return None, None

    def set_image(self, name, image_path_or_array, width=None, height=None):
        """Set an image parameter from file path or numpy array"""
        if isinstance(image_path_or_array, (str, Path)):
            with open(image_path_or_array, 'rb') as f:
                data = base64.b64encode(f.read()).decode('utf-8')
            ext = Path(image_path_or_array).suffix.lower().lstrip('.')
            mime = {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'gif': 'image/gif'}.get(ext, 'image/png')
        else:
            import io
            try:
                from PIL import Image
                import numpy as np
                arr = np.asarray(image_path_or_array)
                if arr.dtype != np.uint8:
                    if arr.max() <= 1.0:
                        arr = (arr * 255).astype(np.uint8)
                    else:
                        arr = arr.astype(np.uint8)
                img = Image.fromarray(arr)
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                data = base64.b64encode(buf.getvalue()).decode('utf-8')
                mime = 'image/png'
            except ImportError:
                raise ImportError("PIL/Pillow required for numpy array images")
        
        self.viz_params[name] = {
            "type": "image",
            "value": f"data:{mime};base64,{data}",
            "width": width or 100,
            "height": height or 100
        }

    def set_float(self, name, value, min_val=0.0, max_val=1.0, step=0.01):
        """Set a float slider parameter"""
        self.viz_params[name] = {
            "type": "float",
            "value": float(value),
            "min": min_val,
            "max": max_val,
            "step": step
        }

    def set_int(self, name, value, min_val=0, max_val=100):
        """Set an integer parameter"""
        self.viz_params[name] = {
            "type": "int",
            "value": int(value),
            "min": min_val,
            "max": max_val
        }

    def set_text(self, name, value):
        """Set a text parameter"""
        self.viz_params[name] = {
            "type": "text",
            "value": str(value)
        }

    def set_bool(self, name, value):
        """Set a boolean toggle parameter"""
        self.viz_params[name] = {
            "type": "bool",
            "value": bool(value)
        }

    def set_enum(self, name, value, options):
        """Set a dropdown parameter"""
        self.viz_params[name] = {
            "type": "enum",
            "value": value,
            "options": options
        }


class Placeholder:
    def __init__(self, name):
        self.name = name
    
    def __repr__(self):
        return f"Placeholder({self.name})"


class PlaceholderTuple:
    def __init__(self, names):
        self.placeholders = tuple(Placeholder(n) for n in names)
    
    def __iter__(self):
        return iter(self.placeholders)


class GraphContext:
    def __init__(self):
        self.trace = []
        self.placeholder_to_node = {}
    
    def __call__(self, node, *args, **kwargs):
        frame = sys._getframe(1)
        line = linecache.getline(frame.f_code.co_filename, frame.f_lineno).strip()
        
        output_names = self._parse_output_names(line)
        inputs = self._build_inputs(args, kwargs)
        
        node_idx = len(self.trace)
        
        if len(output_names) == 1:
            output_placeholder = Placeholder(output_names[0])
            self.placeholder_to_node[id(output_placeholder)] = (node_idx, 0)
            output_placeholders = [output_placeholder]
            result = output_placeholder
        else:
            output_placeholders = [Placeholder(n) for n in output_names]
            for i, p in enumerate(output_placeholders):
                self.placeholder_to_node[id(p)] = (node_idx, i)
            result = PlaceholderTuple.__new__(PlaceholderTuple)
            result.placeholders = tuple(output_placeholders)
        
        self.trace.append((output_placeholders, node, inputs))
        return result
    
    def _parse_output_names(self, line):
        match = re.match(r'([\w\s,]+)\s*=\s*ctx\s*\(', line)
        if match:
            names_str = match.group(1)
            names = [n.strip() for n in names_str.split(',')]
            return names
        return [f"_anon_{len(self.trace)}"]
    
    def _build_inputs(self, args, kwargs):
        inputs = {}
        for i, arg in enumerate(args):
            inputs[f"_{i}"] = arg
        for key, value in kwargs.items():
            inputs[key] = value
        return inputs
    
    def _get_name(self, obj):
        if isinstance(obj, Placeholder):
            return obj.name
        if hasattr(obj, 'name') and obj.name:
            return obj.name
        return repr(obj)
    
    def print_graph(self):
        for output_placeholders, node, inputs in self.trace:
            node_name = node.name if hasattr(node, 'name') else type(node).__name__
            output_names = [p.name for p in output_placeholders]
            out_str = output_names[0] if len(output_names) == 1 else f"({', '.join(output_names)})"
            
            input_parts = []
            for key, value in inputs.items():
                name = self._get_name(value)
                if key.startswith('_') and key[1:].isdigit():
                    input_parts.append(name)
                else:
                    input_parts.append(f"{key}={name}")
            
            print(f"{out_str} = {node_name}({', '.join(input_parts)})")
    
    def to_html(self, filename="graph.html", title="Compute Graph"):
        nodes_data = []
        
        for idx, (output_placeholders, node, inputs) in enumerate(self.trace):
            node_name = node.name if hasattr(node, 'name') else type(node).__name__
            node_type = type(node).__name__
            
            input_slots = []
            for key in inputs.keys():
                if key.startswith('_') and key[1:].isdigit():
                    input_slots.append({"name": f"in_{key[1:]}", "type": "value"})
                else:
                    input_slots.append({"name": key, "type": "value"})
            
            output_slots = [{"name": p.name, "type": "value"} for p in output_placeholders]
            
            viz_params = getattr(node, 'viz_params', {})
            
            nodes_data.append({
                "id": idx,
                "title": node_name,
                "type": node_type,
                "inputs": input_slots,
                "outputs": output_slots,
                "viz_params": viz_params
            })
        
        connections = []
        for dst_idx, (output_placeholders, node, inputs) in enumerate(self.trace):
            for slot_idx, (key, value) in enumerate(inputs.items()):
                if isinstance(value, Placeholder) and id(value) in self.placeholder_to_node:
                    src_idx, src_slot = self.placeholder_to_node[id(value)]
                    connections.append({
                        "src_node": src_idx,
                        "src_slot": src_slot,
                        "dst_node": dst_idx,
                        "dst_slot": slot_idx
                    })
        
        html = self._generate_html(nodes_data, connections, title)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"Graph saved to {filename}")
        return filename
    
    def _generate_html(self, nodes_data, connections, title):
        nodes_json = json.dumps(nodes_data, indent=2)
        connections_json = json.dumps(connections, indent=2)
        litegraph_js = _get_litegraph_js()
        litegraph_css = _get_litegraph_css()
        
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
{litegraph_css}
    </style>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            overflow: hidden;
        }}
        #toolbar {{
            position: fixed;
            top: 0; left: 0; right: 0;
            height: 50px;
            background: #16213e;
            display: flex;
            align-items: center;
            padding: 0 15px;
            gap: 10px;
            z-index: 100;
            border-bottom: 1px solid #0f3460;
        }}
        #toolbar h1 {{
            font-size: 16px;
            font-weight: 600;
            margin-right: 20px;
            color: #e94560;
        }}
        .toolbar-group {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 0 15px;
            border-left: 1px solid #0f3460;
        }}
        button {{
            background: #0f3460;
            color: #eee;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            transition: background 0.2s;
        }}
        button:hover {{ background: #e94560; }}
        #canvas-container {{
            position: fixed;
            top: 50px; left: 0; right: 0; bottom: 0;
        }}
        #graph-canvas {{ width: 100%; height: 100%; }}
        #info {{ font-size: 12px; color: #888; font-family: monospace; }}
    </style>
</head>
<body>
    <div id="toolbar">
        <h1>{title}</h1>
        <div class="toolbar-group">
            <span id="info"></span>
        </div>
        <div class="toolbar-group">
            <button onclick="exportGraph()">Export JSON</button>
            <button onclick="autoArrange()">Auto Arrange</button>
            <button onclick="centerView()">Center View</button>
        </div>
    </div>
    
    <div id="canvas-container">
        <canvas id="graph-canvas"></canvas>
    </div>

    <script>
{litegraph_js}
    </script>
    <script>
        const nodesData = {nodes_json};
        const connectionsData = {connections_json};
        
        let graph = null;
        let canvas = null;
        let nodeInstances = [];
        
        const HORIZONTAL_SPACING = 300;
        const VERTICAL_SPACING = 150;
        
        // Image cache
        const imageCache = {{}};
        
        function loadImage(src) {{
            if (imageCache[src]) return Promise.resolve(imageCache[src]);
            return new Promise((resolve, reject) => {{
                const img = new Image();
                img.onload = () => {{
                    imageCache[src] = img;
                    resolve(img);
                }};
                img.onerror = reject;
                img.src = src;
            }});
        }}
        
        document.addEventListener('DOMContentLoaded', () => {{
            // Register custom node types
            const nodeTypes = new Set(nodesData.map(n => n.type));
            nodeTypes.forEach(typeName => {{
                function CustomNode() {{
                    this.images = {{}};
                }}
                CustomNode.title = typeName;
                CustomNode.prototype.onExecute = function() {{}};
                
                // Custom drawing for images
                CustomNode.prototype.onDrawForeground = function(ctx) {{
                    if (!this.vizParams || Object.keys(this.vizParams).length === 0) return;
                    
                    ctx.imageSmoothingEnabled = false;
                    
                    let yOffset = this.size[1] - 10;
                    
                    for (const [name, param] of Object.entries(this.vizParams)) {{
                        if (param.type === 'image') {{
                            const imgData = this.images[name];
                            if (imgData && imgData.loaded && imgData.img) {{
                                const x = (this.size[0] - imgData.width) / 2;
                                yOffset -= imgData.height + 15;
                                ctx.drawImage(imgData.img, x, yOffset, imgData.width, imgData.height);
                                
                                ctx.fillStyle = "#aaa";
                                ctx.font = "10px sans-serif";
                                ctx.textAlign = "left";
                                ctx.fillText(name, x, yOffset - 2);
                            }}
                        }} else {{
                            // Draw as text label
                            yOffset -= 16;
                            ctx.fillStyle = "#888";
                            ctx.font = "11px monospace";
                            ctx.textAlign = "left";
                            
                            let valueStr;
                            if (param.type === 'float') {{
                                valueStr = param.value.toFixed(4);
                            }} else if (param.type === 'bool') {{
                                valueStr = param.value ? 'true' : 'false';
                            }} else {{
                                valueStr = String(param.value);
                            }}
                            
                            ctx.fillText(`${{name}}: ${{valueStr}}`, 10, yOffset + 11);
                        }}
                    }}
                    
                    ctx.imageSmoothingEnabled = true;
                }};
                
                LiteGraph.registerNodeType("compute/" + typeName, CustomNode);
            }});
            
            graph = new LGraph();
            const canvasEl = document.getElementById('graph-canvas');
            canvas = new LGraphCanvas(canvasEl, graph);
            
            canvas.background_image = null;
            canvas.render_shadows = false;
            canvas.render_connection_arrows = true;
            
            function resize() {{
                canvasEl.width = window.innerWidth;
                canvasEl.height = window.innerHeight - 50;
                canvas.resize();
            }}
            window.addEventListener('resize', resize);
            resize();
            
            // Create nodes
            nodesData.forEach((nodeData, idx) => {{
                const node = LiteGraph.createNode("compute/" + nodeData.type);
                node.title = nodeData.title;
                node.inputs = [];
                node.outputs = [];
                node.images = {{}};
                node.vizParams = nodeData.viz_params || {{}};
                
                nodeData.inputs.forEach(inp => {{
                    node.addInput(inp.name, inp.type);
                }});
                
                nodeData.outputs.forEach(out => {{
                    node.addOutput(out.name, out.type);
                }});
                
                // Calculate extra height and load images
                let extraHeight = 0;
                for (const [paramName, param] of Object.entries(node.vizParams)) {{
                    if (param.type === 'image') {{
                        extraHeight += (param.height || 100) + 20;
                        node.images[paramName] = {{
                            loaded: false,
                            img: null,
                            width: param.width || 100,
                            height: param.height || 100
                        }};
                        loadImage(param.value).then(img => {{
                            node.images[paramName].loaded = true;
                            node.images[paramName].img = img;
                            node.setDirtyCanvas(true);
                        }});
                    }} else {{
                        extraHeight += 16;
                    }}
                }}
                
                node.pos = [100, 100 + idx * VERTICAL_SPACING];
                node.size = node.computeSize();
                node.size[0] = Math.max(node.size[0], 150);
                node.size[1] = Math.max(node.size[1], 60) + extraHeight;
                
                graph.add(node);
                nodeInstances.push(node);
            }});
            
            // Create connections
            connectionsData.forEach(conn => {{
                const srcNode = nodeInstances[conn.src_node];
                const dstNode = nodeInstances[conn.dst_node];
                srcNode.connect(conn.src_slot, dstNode, conn.dst_slot);
            }});
            
            autoArrange();
            setTimeout(centerView, 100);
            
            document.getElementById('info').textContent = 
                `${{nodesData.length}} nodes, ${{connectionsData.length}} connections`;
            
            graph.start();
        }});
        
        function autoArrange() {{
            const depths = new Array(nodeInstances.length).fill(0);
            
            for (let pass = 0; pass < nodeInstances.length; pass++) {{
                connectionsData.forEach(conn => {{
                    depths[conn.dst_node] = Math.max(
                        depths[conn.dst_node],
                        depths[conn.src_node] + 1
                    );
                }});
            }}
            
            const layers = {{}};
            depths.forEach((depth, idx) => {{
                if (!layers[depth]) layers[depth] = [];
                layers[depth].push(idx);
            }});
            
            Object.keys(layers).forEach(depth => {{
                const nodesInLayer = layers[depth];
                let yOffset = 100;
                
                nodesInLayer.forEach((nodeIdx) => {{
                    const node = nodeInstances[nodeIdx];
                    node.pos = [100 + parseInt(depth) * HORIZONTAL_SPACING, yOffset];
                    yOffset += node.size[1] + 20;
                }});
            }});
            
            graph.setDirtyCanvas(true, true);
        }}
        
        function centerView() {{
            canvas.centerOnGraph();
            canvas.setZoom(0.9, [canvas.canvas.width / 2, canvas.canvas.height / 2]);
        }}
        
        function exportGraph() {{
            const data = {{
                nodes: nodesData,
                connections: connectionsData,
                litegraph: graph.serialize()
            }};
            const blob = new Blob([JSON.stringify(data, null, 2)], {{ type: 'application/json' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = '{title.replace("'", "")}.json';
            a.click();
            URL.revokeObjectURL(url);
        }}
    </script>
</body>
</html>
'''

    def show(self, title="Compute Graph"):
        """Open graph directly in browser without server"""
        import webbrowser
        TMP_FOLDER = Path(__file__).parent / ".tmp"
        TMP_FOLDER.mkdir(exist_ok=True)
        
        self.to_html(TMP_FOLDER / "compute_graph.html", title=title)
        webbrowser.open('file://' + str(TMP_FOLDER / "compute_graph.html"))


class Child(NameAware):
    def __init__(self, name_override=None):
        super().__init__(name_override=name_override)


class Master(NameAware):
    def __init__(self):
        super().__init__()
        self.child_0 = Child()
        self.child_1 = Child()
        self.child_2 = Child()

        import numpy as np
        import torch

        # block = Child()
        # block.set_image("preview", "path/to/image.png", width=120, height=80)

        # Or with numpy array
        # arr = np.random.rand(64, 64, 3)
        # block.set_image("noise", arr, width=100, height=100)

        # Set some viz params
        self.child_0.set_float("learning_rate", 0.001, 0.0, 0.1, 0.001)
        self.child_0.set_int("batch_size", 32, 1, 128)
        self.child_1.set_bool("use_bias", True)
        self.child_1.set_enum("activation", "relu", ["relu", "gelu", "silu", "tanh"])
        self.child_2.set_text("name", "output_layer")
        self.child_2.set_image("noise 1", np.random.rand(64, 64, 3), width=64*2, height=64*2)
        self.child_2.set_image("noise 2", np.random.rand(64, 64, 3), width=64*2, height=64*2)
        self.child_2.set_image("noise 3", torch.rand(64, 64, 3).sqrt().numpy(), width=64*2, height=64*2)

        self.attention_blocks = []
        for i in range(4):
            block = Child(name_override=f"attention_block_{i}")
            block.set_float("dropout", 0.1, 0.0, 0.5)
            block.set_int("heads", 8, 1, 16)
            self.attention_blocks.append(block)

    def compute_graph(self, ctx, input):
        result_0, result_1 = ctx(self.child_0, input)
        b = ctx(self.child_1, a=result_0, b=result_1)
        c = ctx(self.child_2, a=b, b=result_1)
        _c = c
        for i, block in enumerate(self.attention_blocks):
            if i == len(self.attention_blocks) - 1:
                c = ctx(block, a=c, residual=_c)
            else:
                c = ctx(block, a=c)
        return c


if __name__ == "__main__":
    master = Master()
    ctx = GraphContext()
    master.compute_graph(ctx, Placeholder("input"))
    ctx.print_graph()
    ctx.show()