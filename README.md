# [KALAPA CHALLENGE _ Vietnamese Medical Question Answering](https://challenge.kalapa.vn/portal/vietnamese-medical-question-answering/leaderboard)

<p align="center">
 <img src="fig/visual.png" width="800">
</p>

## Public Score
The evaluation metric used in the first stage is custom score. At the end of the Challenge, i got **74.04**/85.79 compared to the highest score team and got rank 17, in addition my accuracy is better than some other teams, equvilent to rank 9 or 10 with **0.67**/0.7879 compared to the highest accuracy team 

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
