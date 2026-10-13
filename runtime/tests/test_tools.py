import asyncio

from src.runtime.tools import ImmigrationSearchInput, ImmigrationSearchOutput, ImmigrationSource, search_immigration_information


def test_immigration_tool_preserves_source_attribution() -> None:
    async def search(query: str) -> ImmigrationSearchOutput:
        assert query == "visa appointment"
        return ImmigrationSearchOutput(context="Official context", sources=[ImmigrationSource(title="HiKorea", url="https://www.hikorea.go.kr")])

    result = asyncio.run(search_immigration_information(ImmigrationSearchInput(query="visa appointment"), search))
    assert result.sources[0].url == "https://www.hikorea.go.kr"
