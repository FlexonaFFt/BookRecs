from __future__ import annotations

import uvicorn

from source.infrastructure.config import load_api_server_settings


def main() -> None:
    settings = load_api_server_settings()
    uvicorn.run(
        "source.interfaces.api.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
