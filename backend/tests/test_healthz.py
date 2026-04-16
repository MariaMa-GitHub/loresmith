import pytest


@pytest.mark.asyncio
async def test_healthz_returns_200(client):
    response = await client.get("/healthz")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_healthz_returns_ok_status(client):
    response = await client.get("/healthz")
    body = response.json()
    assert body["status"] == "ok"


@pytest.mark.asyncio
async def test_healthz_includes_version(client):
    response = await client.get("/healthz")
    body = response.json()
    assert "version" in body
