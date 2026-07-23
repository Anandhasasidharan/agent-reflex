import pytest


def test_instrument_agent_decorator():
    from agent_reflex.instrumentation.decorators import instrument_agent

    call_count = 0

    @instrument_agent("test_agent")
    def my_agent(task: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"done: {task}"

    result = my_agent("hello")
    assert result == "done: hello"
    assert call_count == 1


def test_instrument_tool_decorator():
    from agent_reflex.instrumentation.decorators import instrument_tool

    @instrument_tool("search")
    def search_tool(query: str) -> str:
        return f"results for {query}"

    result = search_tool("test")
    assert result == "results for test"


def test_instrument_tool_exception():
    from agent_reflex.instrumentation.decorators import instrument_tool

    @instrument_tool("failing_tool")
    def bad_tool():
        raise RuntimeError("tool crash")

    with pytest.raises(RuntimeError, match="tool crash"):
        bad_tool()


def test_instrument_chat_decorator():
    from agent_reflex.instrumentation.decorators import instrument_chat

    @instrument_chat("gpt-4o")
    def call_llm(prompt: str) -> str:
        return f"response to: {prompt}"

    result = call_llm("hello")
    assert result == "response to: hello"


def test_instrument_chat_with_cache_streaming_params():
    from agent_reflex.instrumentation.decorators import instrument_chat

    @instrument_chat("gpt-4o", stream=True, cache_read_tokens=100, cache_creation_tokens=50, time_to_first_chunk_ms=200.0)
    def chat(prompt: str) -> str:
        return f"response: {prompt}"

    result = chat("hello")
    assert "hello" in result


def test_instrument_chat_exception():
    from agent_reflex.instrumentation.decorators import instrument_chat

    @instrument_chat("gpt-4o")
    def bad_chat():
        raise RuntimeError("chat crash")

    with pytest.raises(RuntimeError, match="chat crash"):
        bad_chat()


def test_instrument_agent_error_tracking():
    from agent_reflex.instrumentation.decorators import instrument_agent

    @instrument_agent("failing_agent")
    def failing_agent() -> str:
        raise ValueError("oops")

    with pytest.raises(ValueError, match="oops"):
        failing_agent()


def test_langgraph_adapter():
    from agent_reflex.instrumentation.langgraph_adapter import LangGraphAdapter

    adapter = LangGraphAdapter()
    assert adapter.get_framework_name() == "langgraph"

    class MockExecutor:
        name = "test_agent"
        nodes = []
        edges = []

        def invoke(self, inputs: dict) -> dict:
            return {"output": "done"}

    executor = MockExecutor()
    executor = adapter.instrument_agent_executor(executor)
    result = executor.invoke({"input": "hello"})
    assert result["output"] == "done"


def test_crewai_adapter():
    from agent_reflex.instrumentation.crewai_adapter import CrewAIAdapter

    adapter = CrewAIAdapter()
    assert adapter.get_framework_name() == "crewai"

    class MockCrew:
        name = "test_crew"
        agents = []
        tasks = []

        def kickoff(self, *args, **kwargs):
            return "done"

    crew = MockCrew()
    crew = adapter.instrument_agent_executor(crew)
    result = crew.kickoff()
    assert result == "done"


def test_base_adapter_abstract():
    from agent_reflex.instrumentation.base import BaseAgentAdapter

    class IncompleteAdapter(BaseAgentAdapter):
        pass

    with pytest.raises(TypeError):
        IncompleteAdapter()
