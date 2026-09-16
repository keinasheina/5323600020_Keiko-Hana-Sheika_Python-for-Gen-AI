import time
import httpx
import asyncio


async def compare_endpoints(urls: list[str]) -> list[tuple[str, int, float]]:
    """Fetch all URLs concurrently, returning (url, status_code, response_time_ms)."""

    async def fetch_one(client: httpx.AsyncClient, url: str) -> tuple[str, int, float]:
        start = time.perf_counter()
        try:
            response = await client.get(url, timeout=10.0)
            elapsed_ms = (time.perf_counter() - start) * 1000
            return (url, response.status_code, round(elapsed_ms, 2))
        except httpx.HTTPError:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return (url, 0, round(elapsed_ms, 2))

    async with httpx.AsyncClient() as client:
        tasks = [fetch_one(client, url) for url in urls]
        return await asyncio.gather(*tasks)


if __name__ == "__main__":
    test_urls = [
        "https://jsonplaceholder.typicode.com/posts/1",
        "https://jsonplaceholder.typicode.com/posts/2",
        "https://jsonplaceholder.typicode.com/posts/3",
    ]

    results = asyncio.run(compare_endpoints(test_urls))
    for url, status, latency in results:
        print(f"{url} -> status={status}, {latency}ms")