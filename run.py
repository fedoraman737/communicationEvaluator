import os
import sys

# imports can sometims fail, so use parent directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app

if __name__ == '__main__':
    app.run(debug=True) 