We pass input - list of pairs, where each pair is a reference image + text preferences. The text preferences include room type, budget, and style preset. Output 1..N generated images with metadata we returns notes and estiamted costs.

## Tasks
- [ ] System Diagram
- [ ] Data Preparation
    - [ ] Eval strategy requirements
    - [ ] Define requirements
    - [ ] Prompts generation prompt
    - [ ] Collect images
- [ ] Jupyter Development
- [ ] API Development

To run:
```
uv run uvicorn src.main:app --host localhost --port 5000
```
## Data requirements

5 cases of Prompt + Reference Image

Run:

```
uv run uvicorn src.main:app --port 5000 --host localhost
```

or 

```
bash run.sh
```


Build Docker Container
```
docker compose build
```

Run Docker Compose

```
docker compose run 
```