from agent_reflex.common.types import CausalGraphNode, StepOTAR
from agent_reflex.graph.models import CausalGraph, OTARParser


def test_otar_parser_from_attributes():
    attrs = {
        "input": "What is the capital of France?",
        "agent.thought": "I need to recall geography facts",
        "agent.action": "retrieve_memory",
        "output": "Paris",
    }
    otar = OTARParser.parse(attrs)
    assert otar.observation == "What is the capital of France?"
    assert otar.thought == "I need to recall geography facts"
    assert otar.action == "retrieve_memory"
    assert otar.result == "Paris"


def test_otar_parser_from_span_events():
    events = [
        {"name": "gen_ai.completion", "attributes": {"content": "Paris is the capital"}},
        {"name": "gen_ai.prompt", "attributes": {"content": "What is the capital of France?"}},
    ]
    otar = OTARParser.from_span_events(events)
    assert otar.observation == "What is the capital of France?"
    assert otar.result == "Paris is the capital"
    assert otar.action == ""


def test_otar_parser_from_span_events_empty():
    otar = OTARParser.from_span_events([])
    assert otar.observation == ""
    assert otar.result == ""


def test_causal_graph_add_step():
    cg = CausalGraph()
    n1 = CausalGraphNode(
        node_id="step_1",
        agent_id="agent_a",
        step_index=1,
        otar=StepOTAR("q1", "thinking", "search", "answer1"),
        parent_id=None,
        subtask_id="task_1",
        execution_time_ms=100.0,
        error_flag=False,
    )
    n2 = CausalGraphNode(
        node_id="step_2",
        agent_id="agent_a",
        step_index=2,
        otar=StepOTAR("q2", "thinking", "search", "answer2"),
        parent_id="step_1",
        subtask_id="task_1",
        execution_time_ms=50.0,
        error_flag=False,
    )
    cg.add_step(n1)
    cg.add_step(n2)
    assert len(cg.get_all_nodes()) == 2
    assert cg.get_node("step_2") is not None
    children = cg.get_children("step_1")
    assert len(children) == 1
    assert children[0].node_id == "step_2"


def test_causal_graph_data_dependencies():
    cg = CausalGraph()
    cg.add_step(CausalGraphNode(
        node_id="s1", agent_id="a", step_index=1,
        otar=StepOTAR("query", "think", "search", "Paris is the capital"),
        parent_id=None, subtask_id="t1", execution_time_ms=10.0, error_flag=False,
    ))
    cg.add_step(CausalGraphNode(
        node_id="s2", agent_id="a", step_index=2,
        otar=StepOTAR("Tell me about Paris", "think", "respond", "Paris is great"),
        parent_id="s1", subtask_id="t1", execution_time_ms=10.0, error_flag=False,
    ))
    cg.infer_data_dependencies()
    edges = cg.get_edges()
    data_edges = [e for e in edges if e.edge_type == "data_dependency"]
    assert len(data_edges) >= 1


def test_causal_graph_subtask_decomposition():
    cg = CausalGraph()
    for i in range(3):
        cg.add_step(CausalGraphNode(
            node_id=f"step_{i}", agent_id="a", step_index=i,
            otar=StepOTAR("", "", "", ""),
            parent_id=None, subtask_id="sub_a", execution_time_ms=1.0, error_flag=False,
        ))
    for i in range(3, 5):
        cg.add_step(CausalGraphNode(
            node_id=f"step_{i}", agent_id="b", step_index=i,
            otar=StepOTAR("", "", "", ""),
            parent_id=None, subtask_id="sub_b", execution_time_ms=1.0, error_flag=False,
        ))
    subtasks = cg.decompose_into_subtasks()
    assert len(subtasks) == 2
    assert len(subtasks["sub_a"]) == 3
    assert len(subtasks["sub_b"]) == 2


def test_causal_graph_get_subtask_nodes():
    cg = CausalGraph()
    cg.add_step(CausalGraphNode(
        node_id="s1", agent_id="a", step_index=1,
        otar=StepOTAR("in", "think", "act", "out"),
        parent_id=None, subtask_id="sub_x", execution_time_ms=10.0, error_flag=False,
    ))
    cg.add_step(CausalGraphNode(
        node_id="s2", agent_id="a", step_index=2,
        otar=StepOTAR("in2", "think", "act", "out2"),
        parent_id=None, subtask_id="sub_y", execution_time_ms=10.0, error_flag=False,
    ))
    nodes = cg.get_subtask_nodes("sub_x")
    assert len(nodes) == 1
    assert nodes[0].node_id == "s1"
    nodes = cg.get_subtask_nodes("nonexistent")
    assert len(nodes) == 0


def test_causal_graph_get_parent_none():
    cg = CausalGraph()
    cg.add_step(CausalGraphNode(
        node_id="s1", agent_id="a", step_index=1,
        otar=StepOTAR("in", "think", "act", "out"),
        parent_id=None, subtask_id="t1", execution_time_ms=10.0, error_flag=False,
    ))
    parent = cg.get_parent("s1")
    assert parent is None


def test_causal_graph_get_parent_exists():
    cg = CausalGraph()
    cg.add_step(CausalGraphNode(
        node_id="s1", agent_id="a", step_index=1,
        otar=StepOTAR("in", "think", "act", "out"),
        parent_id=None, subtask_id="t1", execution_time_ms=10.0, error_flag=False,
    ))
    cg.add_step(CausalGraphNode(
        node_id="s2", agent_id="b", step_index=2,
        otar=StepOTAR("in2", "think", "act", "out2"),
        parent_id="s1", subtask_id="t1", execution_time_ms=20.0, error_flag=False,
    ))
    parent = cg.get_parent("s2")
    assert parent is not None
    assert parent.node_id == "s1"


def test_causal_graph_serialization():
    cg = CausalGraph()
    cg.add_step(CausalGraphNode(
        node_id="s1", agent_id="a", step_index=1,
        otar=StepOTAR("in", "think", "act", "out"),
        parent_id=None, subtask_id="t1", execution_time_ms=10.0, error_flag=False,
    ))
    json_str = cg.to_json()
    restored = CausalGraph.from_json(json_str)
    assert len(restored.get_all_nodes()) == 1
    node = restored.get_node("s1")
    assert node is not None
    assert node.otar.action == "act"


def test_causal_graph_serialization_with_edges():
    cg = CausalGraph()
    cg.add_step(CausalGraphNode(
        node_id="s1", agent_id="a", step_index=1,
        otar=StepOTAR("in", "think", "act", "out"),
        parent_id=None, subtask_id="t1", execution_time_ms=10.0, error_flag=False,
    ))
    cg.add_step(CausalGraphNode(
        node_id="s2", agent_id="b", step_index=2,
        otar=StepOTAR("in2", "think", "act", "out2"),
        parent_id="s1", subtask_id="t1", execution_time_ms=20.0, error_flag=False,
    ))
    cg.add_dependency("s1", "s2")
    json_str = cg.to_json()
    restored = CausalGraph.from_json(json_str)
    assert len(restored.get_edges()) > 0
