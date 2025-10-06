
# RAIDIO - Live Audio At a Glance

RADIO utilizes a local ai to interpret live audio feeds.

## Features
### Implemented
✅ Live transcript of audio stream via www, local files, mic input or system output

✅ VU meter for audio

✅ Source specific paramaters via cartridges

### Planned
🧠 Web front end for audio stream interpretation, including:
- Location data
- People and POI's
- Heatmap of common words
- Differentiation of speakers
- Music detection



## How to Run
**1. Start Backend API**

```cd \live-radio```

```venv-server\Scripts\activate```

```python server.py```

**2. Start the Worker**

```cd \live-radio```

```venv-worker\Scripts\activate```

```python worker.py```


