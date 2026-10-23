import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# Ensure root directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.api_server import app

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*65)
    print("   BREED LENS - AI CANINE VISION RETRIEVAL SYSTEM")
    print("   * Giao dien Web: http://127.0.0.1:8000")
    print("   * API Swagger:   http://127.0.0.1:8000/docs")
    print("="*65 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)

