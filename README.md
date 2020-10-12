# FitBERT

FitBert ((F)ill (i)n (t)he blanks, (BERT)) is a library for using [BERT](https://arxiv.org/abs/1810.04805) to fill in the blank(s) in a section of text from a list of options. Here is the envisioned usecase for FitBert:

1. A service (statistical model or something simpler) suggests replacements/corrections for a segment of text
2. That service is specialized to a domain, and isn't good at the big picture, e.g. grammar
3. That service passes the segment of text, with the words to be replaced identified, and the list of suggestions
4. FitBert _crushes_ all but the best suggestion :muscle:

[Blog post walkthrough](https://medium.com/@samhavens/introducing-fitbert-4b047af860fd)

## Instructions

Run the command in the terminal to try the ensemble of BERT and XLNet.

```python
pip install -r requirements.txt
python run.py
```


## References

The original source code of FitBert is available here:

```bibtext
@misc{havens2019fitbert,
    title  = {Use BERT to Fill in the Blanks},
    author = {Sam Havens and Aneta Stal},
    url    = {https://github.com/Qordobacode/fitbert},
    year   = {2019}
}
```

The ensemble implementation has been taken from this [paper](https://www.aclweb.org/anthology/D19-6009.pdf).

```bibtext
@inproceedings{sharma-roychowdhury-2019-iit,
    title = "{IIT}-{KGP} at {COIN} 2019: Using pre-trained Language Models for modeling Machine Comprehension",
    author = "Sharma, Prakhar  and
      Roychowdhury, Sumegh",
    booktitle = "Proceedings of the First Workshop on Commonsense Inference in Natural Language Processing",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-6009",
    doi = "10.18653/v1/D19-6009",
    pages = "80--84",
    abstract = "In this paper, we describe our system for COIN 2019 Shared Task 1: Commonsense Inference in Everyday Narrations. We show the power of leveraging state-of-the-art pre-trained language models such as BERT(Bidirectional Encoder Representations from Transformers) and XLNet over other Commonsense Knowledge Base Resources such as ConceptNet and NELL for modeling machine comprehension. We used an ensemble of BERT-Large and XLNet-Large. Experimental results show that our model give substantial improvements over the baseline and other systems incorporating knowledge bases. We bagged 2nd position on the final test set leaderboard with an accuracy of 90.5{\%}",
}
```
