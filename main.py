import os
import sys
import json
import re
import traceback
from io import StringIO
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

# CORS Enabled: Required for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Request/Response models ----------

class CodeRequest(BaseModel):
    code: str

class ErrorAnalysis(BaseModel):
    error_lines: List[int]

# ---------- Part 1: Tool Function ----------


import concurrent.futures

def execute_python_code(code: str) -> dict:
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    def run():
        exec(code, {"__name__": "__main__"})

    try:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(run)
            future.result(timeout=5)  # 5 second limit
        output = sys.stdout.getvalue()
        return {"success": True, "output": output}
    except concurrent.futures.TimeoutError:
        return {"success": False, "output": "Error: code execution timed out"}
    except Exception:
        output = traceback.format_exc()
        return {"success": False, "output": output}
    finally:
        sys.stdout = old_stdout


# ---------- Part 2: AI Error Analysis ----------

client = OpenAI(
    api_key=os.environ.get("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openai/v1",
)

def extract_traceback_line_numbers(tb: str) -> List[int]:
    """
    Extract every line number Python's own traceback reports.
    This is ground truth - Python computed these exactly, no guessing needed.
    """
    matches = re.findall(r'File "[^"]*", line (\d+)', tb)
    return [int(m) for m in matches]

def analyze_error_with_ai(code: str, tb: str) -> List[int]:
    """
    Use LLM with structured output to identify the error line number(s),
    grounded in line numbers actually extracted from the traceback.
    """
    candidate_lines = extract_traceback_line_numbers(tb)

    # If we found exactly one candidate, trust it directly - no need to
    # risk the AI mis-copying a number that's already unambiguous.
    if len(candidate_lines) == 1:
        return candidate_lines

    # If there are 0 or multiple candidates, ask the AI to pick the
    # single most relevant line, but constrain it to ONLY choose from
    # numbers we already extracted - it cannot invent a new one.
    prompt = f"""Analyze this Python code and its error traceback.

The traceback mentions these candidate line numbers (extracted directly
from the traceback text): {candidate_lines}

Identify which of these candidate line numbers best represents where the
actual error occurred (usually the deepest/last frame in the traceback).
Only return number(s) from this candidate list - do not invent new ones.

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

    # Safety net: if the AI still returns something outside our known
    # candidates, fall back to the last (deepest) candidate line instead
    # of trusting a hallucinated number.
    if candidate_lines and not all(l in candidate_lines for l in result.error_lines):
        return [candidate_lines[-1]]

    return result.error_lines if result.error_lines else candidate_lines
# ---------- Endpoint ----------

@app.post("/code-interpreter")
def code_interpreter(request: CodeRequest):
    exec_result = execute_python_code(request.code)

    if exec_result["success"]:
        return {"error": [], "result": exec_result["output"]}

    error_lines = analyze_error_with_ai(request.code, exec_result["output"])
    return {"error": error_lines, "result": exec_result["output"]}


