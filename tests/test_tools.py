import pytest
from unittest.mock import patch, MagicMock
from tools import tavily_search
from schemas import TavilySearchOutput


FAKE_KEY = "tvly-fake-key-for-testing"


def _make_mock_client(results=None, raise_exc=None):
    mock_client = MagicMock()
    if raise_exc:
        mock_client.search.side_effect = raise_exc
    else:
        mock_client.search.return_value = {"results": results or []}
    return mock_client


def test_tavily_search_success():
    fake_result = {
        "title": "Paris Travel Guide",
        "url": "https://example.com/paris",
        "content": "Paris is a wonderful city with many attractions.",
        "type": "web",
    }
    mock_client = _make_mock_client([fake_result])
    with patch("tools.TAVILY_API_KEY", FAKE_KEY), patch("tools.TavilyClient", return_value=mock_client):
        out = tavily_search("Paris travel tips")

    assert out.tool_status == "ok"
    assert len(out.results) == 1
    assert out.results[0].title == "Paris Travel Guide"
    assert out.results[0].url == "https://example.com/paris"
    assert out.query == "Paris travel tips"


def test_tavily_search_api_exception():
    mock_client = _make_mock_client(raise_exc=RuntimeError("timeout"))
    with patch("tools.TAVILY_API_KEY", FAKE_KEY), patch("tools.TavilyClient", return_value=mock_client):
        out = tavily_search("Paris travel tips")

    assert out.tool_status == "error"
    assert "timeout" in out.error_message
    assert out.results == []


def test_tavily_search_no_api_key():
    with patch("tools.TAVILY_API_KEY", None):
        out = tavily_search("test query")

    assert out.tool_status == "error"
    assert "not set" in out.error_message


def test_tavily_snippet_truncated_to_500():
    fake_result = {
        "title": "Long Article",
        "url": "https://example.com",
        "content": "x" * 1000,
        "type": "web",
    }
    mock_client = _make_mock_client([fake_result])
    with patch("tools.TAVILY_API_KEY", FAKE_KEY), patch("tools.TavilyClient", return_value=mock_client):
        out = tavily_search("test")

    assert len(out.results[0].content_snippet) <= 500


def test_tavily_search_passes_topic():
    mock_client = _make_mock_client([])
    with patch("tools.TAVILY_API_KEY", FAKE_KEY), patch("tools.TavilyClient", return_value=mock_client):
        tavily_search("Tokyo food", topic="travel")

    call_kwargs = mock_client.search.call_args.kwargs
    assert call_kwargs.get("topic") == "travel"


def test_tavily_search_passes_search_depth_and_max_results():
    mock_client = _make_mock_client([])
    with patch("tools.TAVILY_API_KEY", FAKE_KEY), patch("tools.TavilyClient", return_value=mock_client):
        tavily_search("test", search_depth="advanced", max_results=3)

    call_kwargs = mock_client.search.call_args.kwargs
    assert call_kwargs["search_depth"] == "advanced"
    assert call_kwargs["max_results"] == 3


def test_tavily_search_empty_results():
    mock_client = _make_mock_client([])
    with patch("tools.TAVILY_API_KEY", FAKE_KEY), patch("tools.TavilyClient", return_value=mock_client):
        out = tavily_search("obscure query")

    assert out.tool_status == "ok"
    assert out.results == []
