# Structure-Infused Copy Mechanisms for Abstractive Summarization

We provide the source code for the paper **"[Structure-Infused Copy Mechanisms for Abstractive Summarization](http://www.cs.ucf.edu/~feiliu/papers/COLING2018_StructSumm.pdf)"**, accepted at COLING'18. If you find the code useful, please cite the following paper. 

    @inproceedings{song-zhao-liu:2018,
     Author = {Kaiqiang Song and Lin Zhao and Fei Liu},
     Title = {Structure-Infused Copy Mechanisms for Abstractive Summarization},
     Booktitle = {Proceedings of the 27th International Conference on Computational Linguistics (COLING)},
     Year = {2018}}


## Goal

* Our system seeks to re-write a lengthy sentence, often the 1st sentence of a news article, to a concise, title-like summary. The average input and output lengths are 31 words and 8 words, respectively. 

* The code takes as input a text file with one sentence per line. It generates a text file in the same directory as the output, ended with "**.result.summary**", where each source sentence is replaced by a title-like summary.

## Dependencies

The code is written in Python (v2.7) and Theano (v1.0.1). We suggest the following environment:

* A Linux machine (Ubuntu) with GPU (Cuda 8.0)
* [Python (v2.7)](https://www.anaconda.com/download/)
* [Theano (v1.0.1)](http://deeplearning.net/software/theano/install_ubuntu.html)
* [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP)
* [Pyrouge](https://pypi.org/project/pyrouge/)

To install [Python (v2.7)](https://www.anaconda.com/download/), run the command:
```
$ wget https://repo.continuum.io/archive/Anaconda2-5.0.1-Linux-x86_64.sh
$ bash Anaconda2-5.0.1-Linux-x86_64.sh
$ source ~/.bashrc
```

To install [Theano](http://deeplearning.net/software/theano/) and its dependencies, run the below command (you may want to add `export MKL_THREADING_LAYER=GNU` to "~/.bashrc" for future use).
```
$ conda install numpy scipy mkl nose sphinx pydot-ng
$ conda install theano pygpu
$ export MKL_THREADING_LAYER=GNU
```

To download the [Stanford CoreNLP toolkit](https://stanfordnlp.github.io/CoreNLP) and use it as a server, run the command below. The CoreNLP toolkit helps derive structure information (part-of-speech tags, dependency parse trees) from source sentences.
```
$ wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip
$ unzip stanford-corenlp-full-2018-02-27.zip
$ cd stanford-corenlp-full-2018-02-27
$ nohup java -mx16g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000 &
$ cd -
```
To install [Pyrouge](https://pypi.org/project/pyrouge/), run the command below. Pyrouge is a Python wrapper for the ROUGE toolkit, an automatic metric used for summary evaluation.  
```
$ pip install pyrouge
```

## I Want to Generate Summaries..

1. Clone this repo. Download this [TAR](https://drive.google.com/file/d/1dvauV4X6r8oYhohMhdjZr_bKv8XfxFO1/view?usp=sharing) file (`model_coling18.tar.gz`) containing vocabulary files and pretrained models. Move the TAR file to folder "struct_infused_summ" and uncompress.
    ```
    $ git clone https://github.com/KaiQiangSong/struct_infused_summ/
    $ mv model_coling18.tar.gz struct_infused_summ
    $ cd struct_infused_summ
    $ tar -xvzf model_coling18.tar.gz
    $ rm model_coling18.tar.gz
    ```

2. Extract structural features from a list of input files. The file `./test_data/test_filelist.txt` contains absolute (or relative) paths to individual files (test_000.txt and test_001.txt are **toy** files). Each file contains a number of source sentences, one sentence per line. Then, execute the command:
    ```
    $ python toolkit.py -f ./test_data/test_filelist.txt
    ```

3. Generate the model configuration file in the `./settings/` folder.
    ```
    $ python genTestDataSettings.py ./test_data/test_filelist.txt ./settings/my_test_settings
    ```

    After that, you need to modify the "dataset" field of the `options_loader.py` file to point it to the new settings file: `'dataset':'settings/my_test_settings.json'`.

4. Run the testing script. The summary files, located in the same directory as the input, are ended with "**.result.summary**". 
    ```
    $ python generate.py
    ```

    `struct_edge` is the default model. It corresponds to the "2way+relation" architecture described in the paper. You can modify the file `generate.py` (Line 144-145) by globally replacing `struct_edge` with `struct_node` to enable the "2way+word" architecture.
    
    In the generated summaries, the "\<unk\>" symbol represents an unknown word; it is often a source word not included in the input vocabulary (containing 70K words). "#.##" represents numbers of the form "3.14".

## I Want to Train the Model..

1. Create a folder to save the model files. `./model/struct_node` is for the "2way+word" architecture and `./model/struct_edge` for the "2way+relation" architecture. 
    ```
    $ mkdir -p ./model/struct_node ./model/struct_edge
    ```
2. Extract structural features from the input files. `source_file.txt` and `summary_file.txt` in the `./train_data/` folder are **toy** files containing source and summary sentences, one sentence per line. Often, tens of thousands of (source, sentence) pairs are required for training. 
    ```
    $ python toolkit.py ./train_data/source_file.txt
    $ python toolkit.py ./train_data/summary_file.txt
    ```
    
    Adjust file names using below commands. `.Ndocument`, `.dfeature`, and `Nsummary` respectively contain the source sentences, structural features of source sentences, and summary sentences.
    ```
    $ cd ./train_data/
    $ mv source_file.txt.Ndocument train.Ndocument
    $ mv source_file.txt.feature train.dfeature
    $ mv summary_file.txt.Ndocument train.Nsummary
    $ cd -
    ```    
    
3. Repeat the previous step for validation data, which are used for early stopping. `./valid_data` contain **toy** files.
    ```
    $ python toolkit.py ./valid_data/source_file.txt
    $ python toolkit.py ./valid_data/summary_file.txt
    $ cd ./valid_data/
    $ mv source_file.txt.Ndocument valid.Ndocument
    $ mv source_file.txt.feature valid.dfeature
    $ mv summary_file.txt.Ndocument valid.Nsummary
    $ cd -
    ```

4. Generate the model configuration file in the `./settings/` folder.
    ```
    $ python genTrainDataSettings.py ./train_data/train ./valid_data/valid ./settings/my_train_settings
    ```
    
    After that, you need to modify the "dataset" field of the `options_loader.py` file to point to the new settings file: `'dataset':'settings/my_train_settings.json'`.

5. Download the [GloVe embeddings](http://nlp.stanford.edu/data/glove.6B.zip) and uncompress. 
    ```
    $ wget http://nlp.stanford.edu/data/glove.6B.zip
    $ unzip glove.6B.zip
    $ rm glove.6B.zip
    ```
    Modify the "vocab_emb_init_path" field in the file `./settings/vocabulary.json` from `"vocab_emb_init_path": "../../vocab/glove.6B.100d.txt"` to `"vocab_emb_init_path": "glove.6B.100d.txt"`.
    
6. Create a vocabulary file from `./train_data/train.Ndocument` and `./train_data/train.Nsummary`. Words appearing less than 5 times are excluded.
    ```
    $ python get_vocab.py my_vocab
    ```
    
7. Modify the path to the vocabulary file in `train.py` from `Vocab_Giga = loadFromPKL('../../dataset/gigaword_eng_5/giga_new.Vocab')` to `Vocab_Giga = loadFromPKL('my_vocab.Vocab')`.

8. To train the model, run the below command. 
    ```
    $ THEANO_FLAGS='floatX=float32' python train.py
    ```
    
    Two files, `model_best.npz` and `options_best.json`, will be saved in the `./model/struct_edge/` folder. "2way+relation" is the default architecture. It uses the settings file `./settings/network_struct_edge.json`. 
    
    You can modify the 'network' field of the `options_loader.py` from `'settings/network_struct_edge.json'` to `'./settings/network_struct_node.json'` to train the "2way+word" architecture.
    
    You can disable early stopping by setting `"sample":false` in `./setttings/earlyStop.json`. The training will stop when it reaches the maximum number of epoches (30 epoches). This can be modified by changing the `"max_epochs"` field in `./settings/training.json`.

## I Want to Apply the Coverage Mechanism in a 2nd Training Stage..

1. You will switch to the file `train_2.py`. Modify the path to the vocabulary file in `train_2.py` from `Vocab_Giga = loadFromPKL('../../dataset/gigaword_eng_5/giga_new.Vocab')` to `Vocab_Giga = loadFromPKL('my_vocab.Vocab')` to point it to your vocabulary file.

2. Run the below command to perform the 2nd-stage training. Two files `./model/struct_edge/model_check2_best.npz` and `./model/struct_edge/options_check2_best.json` will be generated, containing the best model parameters and system configurations for the "2way+relation" architecture.
    ```
    $ python train_2.py
    ```

## License

This project is licensed under the BSD License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

We grateful acknowledge the work of Kelvin Xu whose [code](https://github.com/kelvinxu/arctic-captions/) in part inspired this project.


