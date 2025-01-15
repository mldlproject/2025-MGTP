# 2025-MGTP
### Requirements 

```
PyTorch >= 2.0.0
torch-gemetric >= 1.3.2
rdkit
scikit-learn
```

### Usage example

**BBB Dataset**
```sh
cd ./dataset/raw/cleaned_BBBP.csv
```

**Encode data**
```sh
cd ./loader.py
```

**Masking data**
```sh
cd ./util.py
```
**Training MGT-BERT**
```sh
python ./training_mgt_bert.py
```

**Extract Features**
```sh
python ./extract_feature.py
```

