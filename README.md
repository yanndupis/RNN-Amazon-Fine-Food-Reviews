# Project: Amazon Fine Food Reviews Sentiment Analysis with Recurrent Neural Network


### Project Overview

With the emergence of both social media and online reviews, sentiment analysis became an important area of research in Machine Learning. Sentiment analysis can help improve products, marketing, [clinical medicine](https://www.ncbi.nlm.nih.gov/pubmed/25982909) or even how cities  [deliver services](http://thegovlab.org/smart-cities-smart-citizens/). Sentiment analysis is a very interesting challenge to work on because the computer needs to interact with human language (natural language processing) and be able to identify and extract subjective information.

Several algorithms like Naive Bayes, Support Vector Machine and Maximum Entropy are commonly used for sentiment analysis. However with the recent development of [recurrent neural networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/), Deep Learning has started to outperform all the other methods due to its ability to build a representation of whole sentences based on the sentence structure. Several amazing research papers have already been published on this topic by Stanford (e.g. [Sentiment Analysis](https://nlp.stanford.edu/sentiment/])) and OpenAi (e.g. [unsupervised sentiment neuron](https://blog.openai.com/unsupervised-sentiment-neuron/#sentimentneuron])).

For this project, we will perform sentiment analysis on the [Amazon Fine Food Review](https://www.kaggle.com/snap/amazon-fine-food-reviews) dataset from Kaggle.

### Analysis
You can find the final project report [here](https://github.com/yanndupis/RNN-Amazon-Fine-Food-Reviews/blob/master/Report/Fine_Food_Review_Sentiment_Analysis_Report.pdf) and also the [final Jupyter Notebook](https://github.com/yanndupis/RNN-Amazon-Fine-Food-Reviews/blob/master/Amazon%20Fine%20Food%20Reviews.ipynb).

### Install

This project requires **Python 3.5** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [tensorflow](https://www.tensorflow.org/)
- [seaborn](https://seaborn.pydata.org/)
- [string](https://docs.python.org/3.5/library/string.html)
- [collections](https://docs.python.org/3.3/library/collections.html)
- [wordcloud](https://github.com/amueller/word_cloud)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

We recommend to install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project.

### Code

The code is provided in the `Amazon Fine Food Reviews.ipynb` notebook file. You also need the `Reviews.csv` dataset file to run the Notebook. The dataset can be found [here](https://www.kaggle.com/snap/amazon-fine-food-reviews) on Kaggle.

### Run

To quickly train the model, we highly recommend to run the Notebook with GPUs using FloydHub](https://www.floydhub.com) or AWS or another platform.

If you don't have access to GPUs, you can always run this Notebook on your local machine but it might take 6/8 hours to train the model.

In a terminal or command window, navigate to the top-level project directory `Amazone_fine_food/` (that contains this README) and run one of the following commands:

```bash
ipython notebook Amazon\ Fine\ Food\ Reviews.ipynb
```  
or
```bash
jupyter notebook Amazon\ Fine\ Food\ Reviews.ipynb
```

This will open the iPython Notebook software and project file in your browser.

### Data

The dataset contains 568,454 food reviews Amazon users left from October 1999 to October 2012.).

**Features**
- `Id`
- `ProductId`: unique identifier for the product
- `UserId`: unique identifier for the user
- `ProfileName`
- `HelpfulnessNumerator`: number of users who found the review helpful
- `HelpfulnessDenominator`: number of users who indicated whether they found the review helpful
Score - rating between 1 and 5
- `Time`: timestamp for the review
- `Summary`: brief summary of the review
- `Text`: text of the review

**Target Variable**
- `Score`: rating between 1 and 5
