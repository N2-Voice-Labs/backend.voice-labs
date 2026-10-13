"""Small, source-attributed tool contract for immigration information."""

from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field


class ImmigrationSearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=500)


class ImmigrationSource(BaseModel):
    title: str
    url: str


class ImmigrationSearchOutput(BaseModel):
    context: str
    sources: list[ImmigrationSource]


SearchFunction = Callable[[str], Awaitable[ImmigrationSearchOutput]]


async def search_immigration_information(
    payload: ImmigrationSearchInput,
    search: SearchFunction,
) -> ImmigrationSearchOutput:
    """Return grounded context and sources, not a legal conclusion."""

    return await search(payload.query)
