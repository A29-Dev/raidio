🎧 Raidio – Local AI Radio Listener

Raidio is a local AI-powered radio listener that:

Streams any live radio (or microphone input),

Transcribes speech locally via Whisper,

Detects entities, people, and topics,

Displays a live transcript, map, and keyword heatmap on a dashboard.

🧩 Prerequisites
1. Core Software

Install these in order:

Software	Purpose	Download
Python 3.11+	Runs the backend (FastAPI + worker)	python.org/downloads

Node.js (LTS)	Runs the frontend dashboard (Vite + React)	nodejs.org

PostgreSQL	Stores transcriptions & metadata	postgresql.org/download

Memurai (Windows Redis alternative)	Fast in-memory message broker	memurai.com/download

FFmpeg	Captures audio streams or mic input	ffmpeg.org/download.html

Git (optional)	For cloning the repository	git-scm.com/downloads

🪟 Make sure to add Python, FFmpeg, and Node.js to your system PATH during install.

⚙️ Initial Setup
1. Clone or download the repository
git clone https://github.com/yourusername/raidio.git
cd raidio

2. Create Python virtual environments
# Backend (server)
cd live-radio
python -m venv venv-server
venv-server\Scripts\activate
pip install -r requirements.txt
deactivate

# Worker
python -m venv venv-worker
venv-worker\Scripts\activate
pip install -r requirements.txt
deactivate

3. Set up PostgreSQL

In psql:

CREATE USER radio WITH PASSWORD 'yourpassword';
CREATE DATABASE radio OWNER radio;
GRANT ALL PRIVILEGES ON DATABASE radio TO radio;


Update your .env or config.py with
DB_URL = "postgresql+asyncpg://radio:yourpassword@localhost/radio"

4. Start Memurai

From Windows Services or:

net start Memurai

▶️ Running Raidio
Step 1. Start the backend server
cd live-radio
venv-server\Scripts\activate
python server.py


Server runs on http://127.0.0.1:8000

Step 2. Start the worker
venv-worker\Scripts\activate
python worker.py


Choose a cartridge (radio stream or mic input).
Example: local_mic.py or abc_perth.py.

The worker connects to the server, Memurai, and PostgreSQL, and begins transcribing.

Step 3. Launch the web dashboard
cd live-dashboard
npm install
npm run dev


Then open your browser at:
👉 http://localhost:5173

🧠 Cartridges

Cartridges define per-station logic and tuning:

STATION_ID = "abc_perth"
DISPLAY_NAME = "ABC Perth"
STREAM_URL = "https://live-radio02.mediahubaustralia.com/2RNW/mp3"

SETTINGS = {
  "chunk_seconds": 10,
  "whisper_model": "base.en",
  "whisper_device": "cpu"
}

VU = {"enabled": True, "target_dbfs": -20.0}
AWARE_RULES = {"track_words": ["budget", "government", "Perth"]}


To add a new one: create a file under /cartridges/.

💡 Common Issues
Issue	Cause / Fix
InvalidPasswordError	Check PostgreSQL username/password
relation \"chunks\" does not exist	Run DB migrations (init_db.sql)
cudnn_ops64_9.dll missing	CUDA not properly installed for GPU Whisper
audio invalid data	Missing -f dshow for mic input on Windows
🧭 Directory Overview
raidio/
├─ live-radio/
│  ├─ server.py            # FastAPI backend
│  ├─ worker.py            # Main AI worker
│  ├─ cartridges/          # Configs per radio station
│  └─ requirements.txt
│
├─ live-dashboard/
│  ├─ src/                 # React frontend
│  ├─ index.html
│  ├─ package.json
│  └─ vite.config.js
│
└─ README.md

🚀 Summary

To run everything:

# 1. Start database + Memurai
net start Memurai

# 2. Run backend
cd live-radio && venv-server\Scripts\activate && python server.py

# 3. Run worker
venv-worker\Scripts\activate && python worker.py

# 4. Run dashboard
cd live-dashboard && npm run dev (coming soon)


You now have a full local AI radio intelligence system — running speech-to-text, entity recognition, and live web visualization all on your machine.