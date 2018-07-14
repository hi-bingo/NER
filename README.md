## Chinese Name Entity Recognition
Use Bi-LSTM and CRF

### Requirements
- keras
- keras_contrib
- numpy
- sklearn
- sklearn_crfsuite

###Usage

#### Prepare data
- word2vec:	https://github.com/hankcs/HanLP/wiki/word2vec
- train_data and test_data: in data folder, also can download from http://www.pudn.com/Download/item/id/2435241.html
- modify setttings.py for your file path

#### Process data
```
python main.py -m preprocess
```
#### Train data
```
python main.py -m train
```
#### Evaluate
```
python main.py -m eval
```
#### Show NER demo
In this mode, you can input one sentence from command line, then the program ouput all entities.
```
python main.py -m demo
```

###Result
|         | precision   |  recall  | f1-score|
| --------   | -----:  | -----:  | :----:  |
| B-LOC      | 0.935   |   0.896     |0.915|
| I-LOC        |   0.929   |   0.878  |0.903|
| B-ORG       |    0.870   |  0.834  |0.852|
| I-ORG      | 0.905   |   0.887|0.896|
| B-PER       |   0.917   |  0.791  |0.849|
| I-PER      |    0.898   |  0.842  |0.869|
| avg      |    0.912   |  0.865  |0.887
