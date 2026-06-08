#!/usr/bin/env python
"""Entry point for running the factor performance dashboard"""

import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.app import app

if __name__ == "__main__":
    # Get configuration from environment or use defaults
    host = os.environ.get('DASH_HOST', '0.0.0.0')
    port = int(os.environ.get('DASH_PORT', 32740))
    debug = os.environ.get('DASH_DEBUG', 'False').lower() == 'true'
    
    print(f"Starting Factor Performance Dashboard...")
    print(f"Server: http://{host}:{port}")
    print(f"Debug mode: {debug}")
    
    app.run(debug=debug, host=host, port=port)

