"""
AMMA Server Runner
Launches the FastAPI server with Uvicorn.
Access points:
- User Terminal: http://127.0.0.1:8000/
- Counsellor Clinical Command Dashboard: http://127.0.0.1:8000/counsellor
"""

import sys
import os
import uvicorn

if __name__ == "__main__":
    print("=" * 65)
    print("  AMMA: AI-Powered Mental Health Monitoring Assistance")
    print("  ML-Assisted Chat-Tone Support Review (Non-Diagnostic)")
    print("=" * 65)
    print("  * User Terminal (Motherly Amma AI): http://127.0.0.1:8000/")
    print("  * Counsellor Support Dashboard:    http://127.0.0.1:8000/counsellor")
    print("=" * 65)
    
    # Render supplies PORT at runtime; binding to 0.0.0.0 makes the service
    # reachable outside the container.  Reload is for local development only.
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)
