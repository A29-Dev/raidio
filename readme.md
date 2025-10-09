
# RAIDIO - Live Audio At a Glance

RADIO is an app that shows relevant information in a dashboard captured from an audio source

Using a proprietary **Cartridge** and **Expansion** system, RAIDIO can understand the context of the media you're listening to and enhance your audio experience.

<img width="1914" height="1034" alt="image" src="https://github.com/user-attachments/assets/e088a326-29ed-4222-a135-af56f6445f64" />


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

```venv-web\Scripts\activate```

```python3 manage.py runserver <port>```

## Requirements

Please ensure you have the following set up on your dev environment. This project is developed on Windows, however all Requirements have alternatives on Mac and Linux.

- Install all python packages in *requirements.txt*
- Run ```python -m spacy download en_core_web_sm```
- Install FFmpeg and FFplay and set them in your PATH files
- Install memurai
- Install Postgress 

## Disclaimer
This is a personal project and is not currently in development for commercial purposes. No data is permanently downloaded on the host system while running. 

All AI processing in RAIDIO is done locally, the only connectivity to the internet is when you choose to stream audio content from an online source. 

