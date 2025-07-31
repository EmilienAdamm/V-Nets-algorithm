# V-Nets Algorithm Implementation

A Flask web application that implements the Vásquez Net Discovery Algorithm (VNDA) for event sequence analysis.

## Features

- Process event sequences from CSV/TXT files or manual input
- Apply temporal constraints 
- Generate V-net visualizations
- Evaluate warning predicates
- Interactive web interface

## Quick Start

### Option 1: Docker (Recommended)

1. **Prerequisites**: Docker and Docker Compose installed

2. **Run the application**:
   ```bash
   docker-compose up --build
   ```

3. **Access the app**: Open http://localhost:5000 in your browser

4. **Stop the application**:
   ```bash
   docker-compose down
   ```

### Option 2: Manual Setup

1. **Prerequisites**: Python 3.11+

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

5. **Access the app**: Open http://localhost:5000 in your browser

## Usage

1. Upload your event sequence data (CSV/TXT format with columns: SequenceID, EventType, Timestamp)
2. Optionally upload constraints and predicates files
3. Click "Process" to generate the V-net analysis
4. View results in textual and graphical formats

## File Structure

- `app.py` - Main Flask application
- `index.html` - Web interface
- `requirements.txt` - Python dependencies
- `Dockerfile` - Docker configuration
- `docker-compose.yml` - Docker Compose setup