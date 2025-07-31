# V-Nets Algorithm Implementation

A client-side web application that implements the Vásquez Net Discovery Algorithm (VNDA) for event sequence analysis.

## Features

- Process event sequences from CSV/TXT files or manual input
- Apply temporal constraints 
- Generate V-net visualizations with sequence tracking
- Color-coded edges showing which sequence generated each transition
- Evaluate warning predicates
- Interactive web interface
- Pure client-side processing (no server required)

## Quick Start

### Option 1: Direct Use (Recommended)

1. **Download**: Simply download the `index.html` file
2. **Open**: Double-click the file or open it in any web browser
3. **Use**: The application runs entirely in your browser - no setup required!

### Option 2: Web Hosting

1. **Upload** the `index.html` file to any web server
2. **Access** via the server URL
3. **Examples**: GitHub Pages, Netlify, Vercel, or any static hosting service

### Option 3: Local Development Server

1. **Prerequisites**: Python 3.x (any version)
2. **Run a simple server**:
   ```bash
   python -m http.server 8000
   ```
3. **Access**: Open http://localhost:8000 in your browser

## Usage

1. Upload your event sequence data (CSV/TXT format with columns: SequenceID, EventType, Timestamp)
2. Optionally upload constraints and predicates files
3. Click "Process" to generate the V-net analysis
4. View results in textual and graphical formats

## File Structure

- `index.html` - Complete web application (frontend + backend logic)
- `README.md` - Documentation
- `inputs.txt` - Sample input data
- `app.py` - Legacy Flask application (no longer needed)
- `requirements.txt` - Legacy Python dependencies (no longer needed)
- `Dockerfile` - Legacy Docker configuration (no longer needed)
- `docker-compose.yml` - Legacy Docker setup (no longer needed)