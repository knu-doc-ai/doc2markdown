import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from json import JSONDecodeError


class TableExtractor:
    """표 Markdown 추출을 별도 프로세스로 실행하는 래퍼."""

    def __init__(self):
        self.python_executable = sys.executable
        self.worker_script = Path(__file__).resolve().with_name("Table_to_markdown.py")
        self.timeout_seconds = 300

    def extract_table(self, image_path: str) -> str:
        """표 추출 워커를 별도 프로세스로 실행하고 Markdown을 반환한다."""
        resolved_image_path = str(Path(image_path).resolve())
        result_path = self._make_temp_result_path()

        try:
            worker_env = os.environ.copy()
            worker_env["PYTHONUNBUFFERED"] = "1"
            completed = subprocess.run(
                [
                    self.python_executable,
                    "-u",
                    str(self.worker_script),
                    "--extract",
                    resolved_image_path,
                    result_path,
                ],
                env=worker_env,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"table worker timed out after {self.timeout_seconds} seconds: {resolved_image_path}"
            ) from error

        result_file = Path(result_path)
        try:
            if result_file.exists() and result_file.stat().st_size > 0:
                try:
                    payload = json.loads(result_file.read_text(encoding="utf-8"))
                    if payload.get("status") == "success":
                        return str(payload.get("markdown", ""))

                    error_message = str(payload.get("error", "")).strip()
                    if error_message:
                        raise RuntimeError(error_message)
                except JSONDecodeError:
                    pass

            raise RuntimeError(f"table worker crashed (exit code {completed.returncode})")
        finally:
            if result_file.exists():
                result_file.unlink()

    def _make_temp_result_path(self) -> str:
        """워커 프로세스와 결과를 주고받을 임시 JSON 파일 경로를 만든다."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            return temp_file.name
