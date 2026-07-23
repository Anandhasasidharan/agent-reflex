from agent_reflex.predictive.topology_scorer import PredictiveTopologyScorer


def test_single_agent():
    scorer = PredictiveTopologyScorer()
    topology = {
        "agents": [{"name": "agent_a", "tools": []}],
        "edges": [],
        "shared_contexts": 0,
    }
    scores = scorer.score_architecture(topology)
    assert len(scores) == 17
    assert all(0 <= v <= 1.0 for v in scores.values())


def test_planner_worker():
    scorer = PredictiveTopologyScorer()
    topology = {
        "agents": [
            {"name": "planner", "tools": []},
            {"name": "worker1", "tools": ["search"]},
            {"name": "worker2", "tools": ["analyze"]},
        ],
        "edges": [
            {"source": "planner", "target": "worker1"},
            {"source": "planner", "target": "worker2"},
        ],
        "shared_contexts": 1,
    }
    scores = scorer.score_architecture(topology)
    assert scores is not None
    top_mode = list(scores.keys())[0]
    assert isinstance(top_mode, str)


def test_bag_of_agents():
    scorer = PredictiveTopologyScorer()
    topology = {
        "agents": [
            {"name": "a1", "tools": ["t1"]},
            {"name": "a2", "tools": ["t2"]},
            {"name": "a3", "tools": ["t3"]},
        ],
        "edges": [
            {"source": "a1", "target": "a2"},
            {"source": "a1", "target": "a3"},
            {"source": "a2", "target": "a3"},
        ],
        "shared_contexts": 3,
    }
    scores = scorer.score_architecture(topology)
    assert scores is not None


def test_bag_of_agents_no_edges():
    scorer = PredictiveTopologyScorer()
    topology = {
        "agents": [{"name": "a1", "tools": []}, {"name": "a2", "tools": []}],
        "edges": [],
        "shared_contexts": 0,
    }
    scores = scorer.score_architecture(topology)
    assert scores is not None


def test_pipeline_topology():
    scorer = PredictiveTopologyScorer()
    topology = {
        "agents": [
            {"name": "a1", "tools": []},
            {"name": "a2", "tools": []},
            {"name": "a3", "tools": []},
            {"name": "a4", "tools": []},
        ],
        "edges": [
            {"source": "a1", "target": "a2"},
            {"source": "a3", "target": "a4"},
        ],
        "shared_contexts": 0,
    }
    scores = scorer.score_architecture(topology)
    assert scores is not None


def test_cyclic_with_root():
    scorer = PredictiveTopologyScorer()
    topology = {
        "agents": [{"name": "a1", "tools": []}, {"name": "a2", "tools": []}, {"name": "a3", "tools": []}],
        "edges": [
            {"source": "a1", "target": "a2"},
            {"source": "a2", "target": "a1"},
            {"source": "a3", "target": "a2"},
        ],
        "shared_contexts": 0,
    }
    scores = scorer.score_architecture(topology)
    assert scores is not None


def test_risk_scores_in_range():
    scorer = PredictiveTopologyScorer()
    topology = {
        "agents": [{"name": "a", "tools": ["t1", "t2"]}],
        "edges": [],
        "shared_contexts": 0,
    }
    scores = scorer.score_architecture(topology)
    for mode, score in scores.items():
        assert 0.0 <= score <= 1.0, f"{mode}: {score} out of range"
