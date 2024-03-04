# kalapa_vmqa_solution originally viethq18's

## Getting started
```
git clone https://github.com/tnt/kalapa_med_mcqa.git
```
### Installing
`pip install -r requirements.txt`

### Download embedding model me5 from Huggingface and convert to onnx
`git clone https://huggingface.co/intfloat/multilingual-e5-small`
`python convert_onnx.py`

### Embed Medical Corpus into Vector Storage
`python embed_corpus.py`

### Run local
`python main.py`
