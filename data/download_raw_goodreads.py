import argparse
import logging
import time
import urllib.request
from pathlib import Path


BOOKS_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/"
    "byGenre/goodreads_books_young_adult.json.gz"
)
INTERACTIONS_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/"
    "byGenre/goodreads_interactions_young_adult.json.gz"
)

BOOKS_FILENAME = "books.json.gz"
INTERACTIONS_FILENAME = "interactions.json.gz"


logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# Отформатировать размер
def _fmt_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    unit = 0
    while size >= 1024 and unit < len(units) - 1:
        size /= 1024
        unit += 1
    return f"{size:.1f} {units[unit]}"


# Отформатировать секунды
def _fmt_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}ч {m}м {s}с"
    if m > 0:
        return f"{m}м {s}с"
    return f"{s}с"


# Скачать файл с прогрессом
def download_file(url: str, dst: Path, force: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        logger.info("Файл уже существует, пропускаю: %s (%s)", dst, _fmt_size(dst.stat().st_size))
        return

    logger.info("Скачивание: %s", url)
    logger.info("Сохранение в: %s", dst)

    started = time.time()
    state = {"last_log": 0.0}

    def _progress(block_count: int, block_size: int, total_size: int) -> None:
        now = time.time()
        if now - state["last_log"] < 1.0 and total_size > 0:
            return
        state["last_log"] = now

        downloaded = block_count * block_size
        if total_size > 0:
            downloaded = min(downloaded, total_size)
            elapsed = now - started
            speed = downloaded / elapsed if elapsed > 0 else 0.0
            eta = (total_size - downloaded) / speed if speed > 0 else 0.0
            logger.info(
                "Прогресс: %s / %s (%.1f%%), скорость=%s/с, осталось~%s",
                _fmt_size(downloaded),
                _fmt_size(total_size),
                downloaded * 100.0 / total_size,
                _fmt_size(int(speed)),
                _fmt_seconds(eta),
            )
        else:
            logger.info("Прогресс: %s", _fmt_size(downloaded))

    urllib.request.urlretrieve(url, dst, reporthook=_progress)
    logger.info(
        "Готово: %s (%s) за %s",
        dst,
        _fmt_size(dst.stat().st_size),
        _fmt_seconds(time.time() - started),
    )


# Скачать raw Goodreads YA в data/raw_data
def download_goodreads_raw(raw_dir: str = "data/raw_data", force: bool = False) -> None:
    raw_path = Path(raw_dir)
    download_file(BOOKS_URL, raw_path / BOOKS_FILENAME, force=force)
    download_file(INTERACTIONS_URL, raw_path / INTERACTIONS_FILENAME, force=force)
    logger.info("Все raw файлы готовы в %s", raw_path)


# CLI
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Скачать raw Goodreads YA в data/raw_data")
    parser.add_argument("--raw-dir", default="data/raw_data")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    download_goodreads_raw(raw_dir=args.raw_dir, force=args.force)
