import os
import sys
import json
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

def analyze_error_with_ai(code: str, tb: str) -> List[int]:
    """
    Use LLM with structured output to identify error line numbers.
    """
    prompt = f"""Analyze this Python code and its error traceback.
Identify the line number(s) in the CODE where the error occurred.
Use only line numbers that actually appear in the traceback or that you can
verify by counting lines in the CODE. Do not guess or invent line numbers.

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
    return result.error_lines

# ---------- Endpoint ----------

@app.post("/code-interpreter")
def code_interpreter(request: CodeRequest):
    exec_result = execute_python_code(request.code)

    if exec_result["success"]:
        return {"error": [], "result": exec_result["output"]}

    error_lines = analyze_error_with_ai(request.code, exec_result["output"])
    return {"error": error_lines, "result": exec_result["output"]}


