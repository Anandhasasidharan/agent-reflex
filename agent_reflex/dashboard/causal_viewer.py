from __future__ import annotations

from typing import Any

from agent_reflex.graph.models import CausalGraph


def build_viewer_data(graph: CausalGraph, cause_node_id: str | None = None) -> dict[str, Any]:
    nodes_data = []
    for node in graph.get_all_nodes():
        nodes_data.append({
            "id": node.node_id,
            "label": f"{node.agent_id}:{node.step_index}",
            "agent": node.agent_id,
            "action": node.otar.action,
            "is_root_cause": node.node_id == cause_node_id,
            "error": node.error_flag,
            "subtask": node.subtask_id,
        })

    edges_data = [
        {"source": e.source_id, "target": e.target_id, "type": e.edge_type}
        for e in graph.get_edges()
    ]

    return {"nodes": nodes_data, "edges": edges_data}


HTML_VIEWER_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Causal Graph</title>
<style>
  body {{ margin: 0; background: #1a1a2e; color: #eee; font-family: monospace; }}
  svg {{ width: 100%; height: 100vh; }}
  .node circle {{ stroke: #fff; stroke-width: 2px; }}
  .node text {{ fill: #fff; font-size: 12px; }}
  .link {{ stroke: #555; stroke-opacity: 0.6; fill: none; }}
  .link.data_dependency {{ stroke: #4fc3f7; stroke-dasharray: 5,5; }}
  .root-cause circle {{ fill: #e74c3c !important; }}
  .error circle {{ fill: #e67e22; }}
  .normal circle {{ fill: #2ecc71; }}
</style></head><body>
<svg id="graph"></svg>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
  const data = {graph_json};
  const width = window.innerWidth, height = window.innerHeight;
  const svg = d3.select("#graph");
  const g = svg.append("g");
  const zoom = d3.zoom().on("zoom", (e) => g.attr("transform", e.transform));
  svg.call(zoom);
  const nodes = data.nodes.map(d => ({{...d}}));
  const links = data.edges.map(d => ({{source: d.source, target: d.target, type: d.type}}));
  const simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id(d => d.id).distance(150))
    .force("charge", d3.forceManyBody().strength(-300))
    .force("center", d3.forceCenter(width/2, height/2));
  const link = g.selectAll(".link").data(links).join("line")
    .attr("class", d => `link ${{d.type}}`);
  const node = g.selectAll(".node").data(nodes).join("g").attr("class", "node")
    .call(d3.drag()
      .on("start", (e, d) => {{ if(!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }})
      .on("drag", (e, d) => {{ d.fx = e.x; d.fy = e.y; }})
      .on("end", (e, d) => {{ if(!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }}));
  node.append("circle").attr("r", 12)
    .attr("class", d => d.is_root_cause ? "root-cause" : (d.error ? "error" : "normal"));
  node.append("text").text(d => d.label).attr("dx", 16).attr("dy", 4);
  simulation.on("tick", () => {{
    link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
    node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
  }});
</script></body></html>"""
