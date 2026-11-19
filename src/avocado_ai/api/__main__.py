"""Run the API with `python -m avocado_ai.api` (or the `avocado-api` script)."""

import uvicorn

from avocado_ai.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "avocado_ai.api.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
