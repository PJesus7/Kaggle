#Machine Learning advices
[Kaggle's Kazanova advices](http://blog.hackerearth.com/winning-tips-machine-learning-competitions-kazanova-current-kaggle-3)
[toc]

## What are the steps you follow for solving a ML problem? Please describe from scratch.

Following are the steps I undertake while solving any ML problem:

- Understand the data - After you download the data, start exploring features. Look at data types. Check variable classes. Create some univariate - bivariate plots to understand the nature of variables.
- Understand the metric to optimize - Every problem comes with a unique evaluation metric. It's imperative for you to understand it, specially how does it change with target variable.
- Decide cross validation strategy - To avoid overfitting, make sure you've set up a cross validation strategy in early stages. A nice CV strategy will help you get reliable score on leaderboard.
- Start hyper parameter tuning - Once CV is at place, try improving model's accuracy using hyper parameter tuning. It further includes the following steps:

	- Data transformations - It involve steps like scaling, removing outliers, treating null values,  transform categorical variables, do feature selections, create interactions etc.
	- Choosing algorithms and tuning their hyper parameters - Try multiple algorithms to understand how model performance changes.
	- Saving results - From all the models trained above, make sure you save their predictions. They will be useful for ensembling.
	- Combining models - At last, ensemble the models, possibly on multiple levels. Make sure the models are correlated for best results.
 

##What are the model selection and data manipulation techniques you follow to solve a problem?

Generally, I try (almost) everything for most problems.  In principle for:

- Time series: I use GARCH, ARCH, regression, ARIMA models etc.
- Image classification: I use deep learning (convolutional nets) in python.
- Sound Classification : Common neural networks
- High cardinality categorical (like text data): I use linear models, FTRL, Vowpal wabbit, LibFFM, libFM, SVD etc.

For everything else,I use Gradient boosting machines (like  XGBoost and LightGBM) and deep learning (like keras, Lasagne, caffe, Cxxnet). I decide what model to keep/drop in Meta modelling with feature selection techniques. Some of the feature selection techniques I use includes:

- Forward (cv or not) - Start from null model. Add one feature at a time and check CV accuracy. If it improves keep the variable, else discard.
- Backward (cv or not) - Start from full model and remove variables one by one. If CV accuracy improves by removing any variable, discard it.
- Mixed (or stepwise) - Use a mix of above to techniques.
- Permutations
- Using feature importance - Use random forest, gbm, xgboost feature selection feature.
- Apply some stats’ logic such as chi-square test, anova.

Data manipulation could be different for every problem :

- Time series : You can calculate moving averages, derivatives. Remove outliers.
- Text : Useful techniques are TF-IDF, countvectorizers, word2vec, SVD (dimensionality reduction). Stemming, spell checking, sparse matrices, likelihood encoding, one hot encoding (or dummies), hashing.
- Image classification: Here you can do scaling, resizing, removing noise (smoothening), annotating, etc.
- Sounds : Calculate Furrier Transforms , MFCC (Mel frequency cepstral coefficients), Low pass filters, etc.
- Everything else : Univariate feature transformations (like log +1 for numerical data), feature selections, treating null values, removing outliers, converting categorical variables to numeric.
 

## Can you elaborate cross validation strategy?

Cross validation means that from my main set, I create RANDOMLY 2 sets. I built (train) my algorithm with the first one (let’s call it training set) and score the other (let’s call it validation set). I repeat this process multiple times and always check how my model performs on the test set in respect to the metric I want to optimize.

The process may look like:

- For 10 (you choose how many X) times
- Split the set in training (50%-90% of the original data) and validation (50%-10%  of the original data)
- Then fit the algorithm on the training set
- Score the validation set.
- Save the result of that scoring in respect to the chosen metric.
- Calculate the average of these 10 (X) times. That how much you expect this score in real life and is generally a good estimate.
- Remember to use a SEED to be able to replicate these X splits

Other things to consider is KFold and stratified KFold ([Read here](https://www.cs.cmu.edu/~schneide/tut5/node42.html)). When testing with time sensitive data, make sure you have past and future data on both train and test sets.

 
##Can you please explain some techniques used for cross validation?

- Kfold
- Stratified Kfold
- Random X% split
- Time based split
- For large data, just one validation set could suffice (like 20% of the data – you don’t need to do multiple times).
 

## Which are the most useful python libraries for a data scientist ?

Below are some libraries which I find most useful in solving problems:

- Data Manipulation
	- Numpy
	- Scipy
	- Pandas
- Data Visualization
	- Matplotlib
- Machine Learning / Deep Learning
	- Xgboost
	- Keras
	- Nolearn
	- Gensim
	- Scikit image
- Natural Language Processing
	- NLTK
 

## What are useful ML techniques / strategies to impute missing values or predict categorical label when all the variables are categorical in nature.

Imputing missing values is a critical step. Sometimes you may find a trend in missing values. Below are some techniques I use:

- Use mean, mode, median for imputation
- Use a value outside the range of the normal values for a variable. like -1 ,or -9999 etc.
- Replace with a likelihood – e.g. something that relates to the target variable.
- Replace with something which makes sense. For example: sometimes null may mean zero
	- Try to predict missing values based on subsets of know values
	- You may consider removing rows with many null values
 

## Do you use high performing machine like GPU. or for example do you do thing like grid search for parameters for random forest(say), which takes lot of time, so which machine do you use?

I use GPUs for every deep learning training model. I have to state that for deep learning GPU is a MUST. Training neural nets on CPUs takes ages, while a mediocre GPU can make a simple nn (e.g deep learning) 50-70 times faster. I don’t like grid search. I do this fairly manually. I think in the beginning it might be slow, but after a while you can get to decent solutions with the first set of parameters! That is because you can sort of learn which parameters are best for each problem and you get to know the algorithms better this way.

 
## How do people built around 80+ models is it by changing the hyper parameter tuning ?

It takes time. Some people do it differently. I have some sets of params that worked in the past and I initialize with these values and then I start adjusting them based on the problem at hand. Obviously you need to forcefully explore more areas (of hyper params in order to know how they work) and enrich this bank of past successful hyper parameter combinations for each model. You should consider what others are doing too. There is NO only 1 optimal set of hyper params. It is possible you get a similar score with a completely different set of params than the one you have.

## How does one improve their kaggle rank? Sometimes I feel hopeless while working on any competition.

It's not an overnight process. Improvement on kaggle or anywhere happens with time. There are no shortcuts. You need to just keep doing things. Below are some of the my recommendations:

- Play in ‘knowledge’ competitions
- See what the others are doing in kernels or in past competitions look for the ‘winning solution sections’
- Create a code bank
- Play … a lot!
 

## Can you tell us about some useful tools used in machine learning ?

Below is the list of my favourite tools:

- [Liblinear](http://www.csie.ntu.edu.tw/~cjlin/liblinear/) for linear models
- [LibSvm](http://www.csie.ntu.edu.tw/~cjlin/libsvm/) for Support Vector machines
- [Scikit](http://scikit-learn.org/stable/) Learn for all machine learning models
- [Xgboost](https://github.com/tqchen/xgboost) for fast scalable gradient boosting
- [LightGBM](https://github.com/Microsoft/LightGBM)
- [Vowpal Wabbit](http://hunch.net/~vw/) for fast memory efficient linear models
- [encog](http://www.heatonresearch.com/encog) for neural nets
- H2O in R for many models
- [LibFm](http://www.libfm.org/)
- [LibFFM](https://www.csie.ntu.edu.tw/~cjlin/libffm/)
- [Weka](http://www.cs.waikato.ac.nz/ml/weka/) in Java (has everything)
- [Graphchi](https://github.com/GraphChi) for factorizations
- [GraphLab](https://dato.com/products/create/open_source.html) for lots of stuff
- [Cxxnet](https://github.com/antinucleon/cxxnet) One of the best implementation of convolutional neural nets out there. Difficult to install and requires GPU with NVDIA Graphics card.
- [RankLib](http://people.cs.umass.edu/~vdang/ranklib.html) The best library out there made in java suited for ranking algorithms (e.g. rank products for customers) that supports optimization fucntions like NDCG.
- [Keras](http://keras.io/) and [Lasagne](https://github.com/Lasagne/Lasagne) for neural nets. This assumes you have [Theano](http://deeplearning.net/software/theano/) or [Tensorflow](https://www.tensorflow.org/).
 

## What techniques perform best on large data sets on Kaggle and in general ? How to tackle memory issues ?

Big data sets with high cardinality can be tackled well with linear models. Consider sparse models. Tools like vowpal wabbit. FTRL , libfm, libffm, liblinear are good tools matrices in python (things like csr matrices). Consider ensembling (like combining) models trained on smaller parts of the data.


## Do you agree with the statement that in general feature engineering (so exploring and recombining predictors) is more efficient than improving predictive models to increase accuracy?

In principle – Yes. I think model diversity is better than having a few really strong models. But it depends on the problem.

 
## Are the skills required to get to the leaderboard top on Kaggle also those you need as a data scientist?

There is some percentage of overlap especially when it comes to making predictive models, working with data through python/R and creating reports and visualizations.  What Kaggle does not offer (but you can get some idea) is:

- How to translate a business question to a modelling (possibly supervised) problem
- How to monitor models past their deployment
- How to explain (many times) difficult concepts to stake holders.
- I think there is always room for a good kaggler in the industry world. It is just that data science can have many possible routes. It may be for example that not everyone tends to be entrepreneurial in their work or gets to be very client facing, but rather solving very particular (technical) tasks.
 

## Which machine learning concepts are must to have to perform well in a kaggle competition?

- Data interrogation/exploration
- Data transformation – pre-processing
- Hands on knowledge of tools
- Familiarity with metrics and optimization
- Cross Validation
- Model Tuning
- Ensembling
 

## How to use ensemble modelling in R and Python to increase the accuracy of prediction. Please quote some real life examples?

You can see my [github script](https://github.com/kaz-Anova/ensemble_amazon) as I explain different Machine leaning methods based on a Kaggle competition. Also, check this [ensembling guide](http://mlwave.com/kaggle-ensembling-guide/).
 

## Which are the best machine learning techniques for imbalanced data?

I don’t do a special treatment here. I know people find that strange. This comes down to optimizing the right metric (for me). It is tough to explain in a few lines. There are many techniques for sampling, but I never had to use.  Some people are using [Smote](https://www.jair.org/media/953/live-953-2037-jair.pdf). I don’t see value in trying to change the principal distribution of your target variable. You just end up with augmented or altered principal odds. If you really want a cut-off to decide on whether you should act or not – you may set it based on the principal odds.

I may not be the best person to answer this. I personally have never found it (significantly) useful to change the distribution of the target variable or the perception of the odds in the target variable. It may just be that other algorithms are better than others when dealing with this task (for example tree-based ones should be able to handle this).
 
 
## What is Feature Engineering? 

In short, feature engineering can be understood as:

- Feature transformation (e.g. converting numerical or categorical variables to other types)
- Feature selections
- Exploiting feature interactions (like should I combine variable A with variable B?)
- Treating null values
- Treating outliers
 

## Can you share your previous solutions?

See some with code and some without (just general approach).

https://www.kaggle.com/c/malware-classification/discussion/13863
http://blog.kaggle.com/2015/05/11/microsoft-malware-winners-interview-2nd-place-gert-marios-aka-kazanova/
https://github.com/kaz-Anova/ensemble_amazon
http://blog.kaggle.com/2015/12/03/dato-winners-interview-1st-place-mad-professors/
http://blog.kaggle.com/2016/04/08/homesite-quote-conversion-winners-write-up-1st-place-kazanova-faron-clobber/
https://mlwave.com/how-we-won-3rd-prize-in-crowdanalytix-copd-competition/
http://blog.kaggle.com/2016/08/31/avito-duplicate-ads-detection-winners-interview-2nd-place-team-the-quants-mikel-peter-marios-sonny/
http://blog.kaggle.com/2016/12/15/bosch-production-line-performance-competition-winners-interview-3rd-place-team-data-property-avengers-darragh-marios-mathias-stanislav/
 
## What is your opinion about using Weka and/or R vs Python for learning machine learning?

I like Weka. It has a good documentation– especially if you want to learn the algorithms. However I have to admit that it is not as efficient as some of the R and Python implementations. It has good coverage though. Weka has some good visualizations too – especially for some tree-based algorithms. I would probably suggest you to focus on R and Python at first unless your background is strictly in Java.

