from agent_reflex.common.types import CausalGraphNode, StepOTAR
from agent_reflex.dashboard.causal_viewer import build_viewer_data
from agent_reflex.graph.models import CausalGraph


def test_build_viewer_data():
    cg = CausalGraph()
    cg.add_step(CausalGraphNode(
        node_id="step_1", agent_id="a", step_index=1,
        otar=StepOTAR("in", "think", "act", "out"),
        parent_id=None, subtask_id="t1", execution_time_ms=10.0, error_flag=False,
    ))
    cg.add_step(CausalGraphNode(
        node_id="step_2", agent_id="b", step_index=2,
        otar=StepOTAR("in2", "think2", "act2", "out2"),
        parent_id="step_1", subtask_id="t1", execution_time_ms=20.0, error_flag=True,
    ))
    cg.add_dependency("step_1", "step_2")

    data = build_viewer_data(cg, cause_node_id="step_2")
    assert len(data["nodes"]) == 2
    assert len(data["edges"]) == 1

    root_cause = [n for n in data["nodes"] if n["is_root_cause"]]
    assert len(root_cause) == 1
    assert root_cause[0]["id"] == "step_2"
    assert root_cause[0]["error"] is True
