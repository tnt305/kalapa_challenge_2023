# [KALAPA CHALLENGE _ Vietnamese Medical Question Answering](https://challenge.kalapa.vn/portal/vietnamese-medical-question-answering/leaderboard)

<p align="center">
 <img src="fig/visual.png" width="800" height="400">
</p>

## Public Score

In the initial stage of the Challenge, a custom scoring metric was employed for evaluation purposes. Upon conclusion of the Challenge, my achieved score stood at 74.04 out of 85.79, positioning me at rank 17 relative to the highest-scoring team. In addition, my accuracy surpassed that of several other teams, equating to a ranking around 9th or 10th place with a score of 0.67 out of 0.7879, compared to the highest accuracy team.

| **Rank**            |        **Team name**          |             **Score**              |           **Accuracy**       |    
|:-------------------:|:-----------------------------:|:----------------------------:|:---------------------------------:|
| **17** |**Thiên Đặng_AIO(me)** |   **0.7404**   | **0.6667** |


## Getting started
```
git clone https://github.com/tnt305/kalapa_challenge_2023.git
```
### Installing
`pip install -r requirements.txt`

### Download embedding model me5 from Huggingface and convert to onnx
`git clone https://huggingface.co/intfloat/multilingual-e5-small`
`python convert_onnx.py`

### Embed Medical Corpus into Vector Storage
`python embed_corpus.py`

### Run
`python main.py`
