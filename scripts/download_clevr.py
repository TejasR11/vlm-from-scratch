"""Download and extract the preprocessed 10k-example CLEVR subset used in §5.

Usage:
    uv run python scripts/download_clevr.py
"""

from __future__ import annotations

import http.cookiejar
import html
import hashlib
import re
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

CLEVR_MINI_URL = (
    "https://drive.google.com/file/d/"
    "1KsswLqfYLl1d91pg5kGUgwtPslo8njTB/view?usp=sharing"
)
GOOGLE_DRIVE_FILE_ID = "1KsswLqfYLl1d91pg5kGUgwtPslo8njTB"
ARCHIVE = Path("data/clevr_mini.zip")
ARCHIVE_SHA256 = "3d531eaea07223c7e9b08583ddd5bd8d30334545ac35b028d699cb5e9ea6b08a"
DEST = Path("data/clevr_mini")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_download_response(response) -> bool:
    content_disposition = response.headers.get("Content-Disposition", "")
    return "attachment" in content_disposition.lower()


def google_drive_download_url(file_id: str, extra_params: dict[str, str] | None = None) -> str:
    params = {"export": "download", "id": file_id}
    if extra_params:
        params.update(extra_params)
    return "https://drive.google.com/uc?" + urllib.parse.urlencode(params)


def attr_value(tag: str, name: str) -> str | None:
    match = re.search(rf'{name}="([^"]*)"', tag)
    if match:
        return html.unescape(match.group(1))
    return None


def confirm_url_from_page(
    page: str,
    cookies: http.cookiejar.CookieJar,
    file_id: str,
) -> str | None:
    form_match = re.search(r'<form[^>]+id="download-form"[^>]*>', page)
    if form_match:
        form_tag = form_match.group(0)
        action = attr_value(form_tag, "action")
        if action:
            form_end = page.find("</form>", form_match.end())
            form_html = page[form_match.start() : form_end] if form_end != -1 else page
            params = {}
            for input_tag in re.findall(r"<input\b[^>]*>", form_html):
                name = attr_value(input_tag, "name")
                value = attr_value(input_tag, "value")
                if name and value is not None:
                    params[name] = value
            return action + "?" + urllib.parse.urlencode(params)

    params = {}
    for cookie in cookies:
        if cookie.name.startswith("download_warning"):
            params["confirm"] = cookie.value

    for name in ("confirm", "uuid"):
        match = re.search(rf'name="{name}"\s+value="([^"]+)"', page)
        if match:
            params[name] = match.group(1)

    if "confirm" not in params:
        match = re.search(r"[?&]confirm=([^&\"']+)", page)
        if match:
            params["confirm"] = urllib.parse.unquote(match.group(1))
    if params:
        return google_drive_download_url(file_id, params)
    return None


def save_response(response, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    total = int(response.headers.get("Content-Length") or 0)
    downloaded = 0
    last_reported = 0

    with open(tmp, "wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if downloaded - last_reported >= 100 * 1024 * 1024:
                if total:
                    print(f"Downloaded {downloaded / 2**20:.0f}/{total / 2**20:.0f} MiB")
                else:
                    print(f"Downloaded {downloaded / 2**20:.0f} MiB")
                last_reported = downloaded

    tmp.replace(path)


def download_google_drive_file(file_id: str, dest: Path) -> None:
    cookies = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookies))
    response = opener.open(google_drive_download_url(file_id))

    if not is_download_response(response):
        page = response.read().decode("utf-8", errors="replace")
        confirm_url = confirm_url_from_page(page, cookies, file_id)
        if not confirm_url:
            raise RuntimeError(
                "Google Drive did not return a downloadable file. "
                f"Open {CLEVR_MINI_URL} and confirm the file is shared publicly."
            )
        response = opener.open(confirm_url)

    if not is_download_response(response):
        raise RuntimeError(
            "Google Drive returned an unexpected response instead of the zip file. "
            f"Try downloading {CLEVR_MINI_URL} manually and saving it as {dest}."
        )

    save_response(response, dest)


def ensure_archive() -> None:
    if ARCHIVE.exists():
        digest = sha256(ARCHIVE)
        if digest == ARCHIVE_SHA256:
            return
        print(f"Existing {ARCHIVE} has the wrong checksum; redownloading.", file=sys.stderr)
        ARCHIVE.unlink()

    print(f"Downloading CLEVR-mini from Google Drive:\n{CLEVR_MINI_URL}")
    try:
        download_google_drive_file(GOOGLE_DRIVE_FILE_ID, ARCHIVE)
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        print(
            f"Manual fallback: download {CLEVR_MINI_URL}, save it as {ARCHIVE}, "
            "then rerun this script.",
            file=sys.stderr,
        )
        sys.exit(1)


def safe_extract(zip_file: zipfile.ZipFile, path: Path) -> None:
    base = path.resolve()
    for member in zip_file.infolist():
        target = (path / member.filename).resolve()
        if not target.is_relative_to(base):
            raise RuntimeError(f"Refusing to extract unsafe zip member: {member.filename}")
    zip_file.extractall(path)


def main() -> None:
    if DEST.exists() and (DEST / "train.jsonl").exists():
        print(f"CLEVR-mini already present at {DEST}. Nothing to do.")
        return

    ensure_archive()

    digest = sha256(ARCHIVE)
    if digest != ARCHIVE_SHA256:
        print(
            f"Checksum mismatch for {ARCHIVE}: expected {ARCHIVE_SHA256}, got {digest}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Extracting to {DEST.parent}")
    with zipfile.ZipFile(ARCHIVE) as zip_file:
        safe_extract(zip_file, DEST.parent)
    print(f"Done. CLEVR-mini available at {DEST}")


if __name__ == "__main__":
    main()
