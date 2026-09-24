import os
import sys
import re
import traceback
import concurrent.futures
from io import StringIO
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CodeRequest(BaseModel):
    code: str

class ErrorAnalysis(BaseModel):
    error_lines: List[int]

def execute_python_code(code: str) -> dict:
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    def run():
        exec(code, {"__name__": "__main__"})

    try:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(run)
            future.result(timeout=5)
        output = sys.stdout.getvalue()
        return {"success": True, "output": output}
    except concurrent.futures.TimeoutError:
        return {"success": False, "output": "Error: code execution timed out"}
    except Exception:
        output = traceback.format_exc()
        return {"success": False, "output": output}
    finally:
        sys.stdout = old_stdout

client = OpenAI(
    api_key=os.environ.get("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openai/v1",
)

def extract_string_frame_lines(tb: str) -> List[int]:
    """
    Line numbers from '<string>' frames only - always present and always
    correct for anything raised inside exec(), regardless of any wrapper
    frames (threading, executors) surrounding it.
    """
    return [int(m) for m in re.findall(r'File "<string>", line (\d+)', tb)]

def analyze_error_with_ai(code: str, tb: str) -> List[int]:
    """
    Deterministically pick the deepest '<string>' frame - that's always
    where the real error occurred. AI is only a last-resort fallback.
    """
    string_frame_lines = extract_string_frame_lines(tb)

    if string_frame_lines:
        return [string_frame_lines[-1]]

    all_candidates = [int(m) for m in re.findall(r'line (\d+)', tb)]

    prompt = f"""Analyze this Python code and its error traceback.
Identify the single most likely line number in the CODE where the error
originated. Only choose from numbers that appear in the traceback.

CODE:
{code}

TRACEBACK:
{tb}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "error_analysis",
                "schema": {
                    "type": "object",
                    "properties": {
                        "error_lines": {
                            "type": "array",
                            "items": {"type": "integer"}
                        }
                    },
                    "required": ["error_lines"],
                    "additionalProperties": False
                }
            }
        },
    )

    result = ErrorAnalysis.model_validate_json(response.choices[0].message.content)

    if result.error_lines and all(l in all_candidates for l in result.error_lines):
        return result.error_lines

    return [all_candidates[0]] if all_candidates else []

@app.post("/code-interpreter")
def code_interpreter(request: CodeRequest):
    exec_result = execute_python_code(request.code)

    if exec_result["success"]:
        return {"error": [], "result": exec_result["output"]}

    error_lines = analyze_error_with_ai(request.code, exec_result["output"])
    return {"error": error_lines, "result": exec_result["output"]}
