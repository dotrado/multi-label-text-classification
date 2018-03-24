# Description

This project is for multi-label text classification. The dataset is reuter-21578.

# To run the program:

Type in command line, in this case, the default data path is data.
```bash
python3 main.py
```

Type in command line, data_path is self-defined data path.
```
python3 main.py data_path
```

# View output

Output contains output files and print messages.

## Output files

It will output following files. Other information will be print to System.out.

1. TOPIC_lilsst.csv:

   This file contains a list of TOPIC words. We have remove these words from bag of terms (features).

   Short view:

   ```bash
   castorseed,potato,barley,soybean,gas,crude,nickel,coconut,nkr,platinum,citruspulp,yen,cotton,dfl,copper,fishmeal,dmk,hog,jobs,lead,rubber,interest,cornglutenfeed,cruzado,inventories,grain,sugar,oat,ship,palmkernel,alum,reserves,...
   ```

2. feature_vector_1.csv:

   This file contains one derivative of selected feature vector which has a cardinality of 125.

   Short view (unordered, because it is stored in dict):

   ```bash
   profit,exploration,quarter,earnings,rules,revs,buffer,unemployment,shipping,note,ounce,mths,index,february,consumer,...
   ```

3. feature_vector_2.csv：

   This file contains one derivative of selected feature vector which has a cardinality of 270.

   Short view (unordered, because it is stored in dict):

   ```bash
   loss,revs,profit,company,shares,note,year,dollar,tonnes,rate,bank,corp,unemployment,march,deficit,barrels,dlrs,rates,reuter,buffer,agriculture,growth,money,record,acquisition,japan,icco,dividend,beef,aluminium,soybeans,offer,account,stake,billion,...
   ```

4. KNN_predict_class_labels_125_feature_vector.txt

   The file stores predicted labels of knn classifier for feature vector with 125 cardinality.

   Short view (left is true labels, right is predicted labels):

   ```
   True labels -> Predicted labels
   {'gas', 'gnp'} -> ['interest']
   {'ship', 'acq'} -> ['ship']
   {'crude'} -> ['crude']
   ```

5. KNN_predict_class_labels_270_feature_vector.txt

   The file stores predicted labels of knn classifier for feature vector with 270 cardinality.

   ```
   True labels -> Predicted labels
   {'gas', 'gnp'} -> ['gnp']
   {'ship', 'acq'} -> ['ship']
   {'crude'} -> ['crude']
   ```

6. Naive_predict_class_labels_125_feature_vector.txt

   The file stores predicted labels of naive classifier for feature vector with 125 cardinality.

   ```
   True labels -> Predicted labels
   {'gas', 'gnp'} -> ['interest']
   {'ship', 'acq'} -> ['acq']
   {'crude'} -> ['crude']
   ```

7. Naive_predict_class_labels_270_feature_vector.txt

   The file stores predicted labels of naive classifier for feature vector with 270 cardinality.

   ```
   True labels -> Predicted labels
   {'gas', 'gnp'} -> ['earn']
   {'ship', 'acq'} -> ['acq']
   {'crude'} -> ['crude']
   ```

8. Termination_messages.txt

   The accuracy, offline and online efficiency data is list here. You can also see it when program is done.

   ```
   ========== Termination message ==========
   Mission completed.
   We select knn classifier and naive classifer.

   For feature vector with 125 cardinality:

   The accuracy of knn classifier is 0.7861500248385495.
   The offline efficient cost of knn classifier is 2.3052525520324707 s.
   The online efficient cost of knn classifier is 322.94284439086914 s.

   The accuracy of naive classifier is 0.7159463487332339.
   The offline efficient cost of naive classifier is 4.73245096206665 s.
   The online efficient cost of naive classifier is 139.92550945281982 s.

   For feature vector with 270 cardinality:

   The accuracy of knn classifier is 0.8128763040238451.
   The offline efficient cost of knn classifier is 326.36808919906616 s.
   The online efficient cost of knn classifier is 470.3035988807678 s.

   The accuracy of naive classifier is 0.7347242921013413.
   The offline efficient cost of naive classifier is 7.906782388687134 s.
   The online efficient cost of naive classifier is 274.57206416130066 s.
   ```

## Print message

print message will show the rate of progress. See details by running the program.

Termination messages is given.

# Program File Structure

`main.py` is program controller.

`data_structure.py` in `/data_structure`defines document objects and static statistic data we would use in building models and predicting class labels.

`preprocess.py` in `/data_preprocess` module is to read data from reuter-21578, parser text data, translate them into list of document objects which has the class labels and feature vector. Then, tokenize the words and construct a list of class labels and bag of terms. 

`metric.py` in `/metric defines` importance metric for feature selection.

`classifier.py` in `/classifier` defines two classifiers: knn classifier and Naive Bayes classifier. Include fit(build) and predict methods.

`mymethods.py` stores some self-defined methods.

# Workflow

## Data pre-processing

### Construct document object

I use regular expression to extract the content of each article and construct each news article as an object.

### Tokenize words - Construct bag of words

I use regular expression to find all string which only contains [a-z.-]. Convert text to a list of words. 

Now, I don't lemmatize or stem the words.

### Construct bag of features

#### Sort words

In this procedure, I combine chi square, term frequency and information entropy to do feature selection.

The formula is `ichi[term][class] = chi_2[term][class] * tf_class[term][class] * entropy[term][class] * beta[term][class]` , and I use `ichi[term] = argmax{ichi[term][class]}` as metric of term importance. 

- `chi_2[term][class]` is chi square test for term in a class.
- `tf_class[term][class]` is `term frequency / documents num` in a class.
- `entropy[term][class]` is information entropy. `p = tf(t, dj) / tf(t, ci)`

Sort the words by ichi value and we get a list of words in decreasing order of importance for classification.

Select top K words as feature vector. Get bag of features.

```
feature vector = [term 1, term 2, term 3, ... , term n]
```

### Compute feature vector

Compute term frequency in each document and use tf-idf to construct feature vector of document.

```
[tf_idf_1, tf_idf_2, tf_idf_3, ... , tf_idf_4]
```

In this project, I design two derivative of selected feature vector. One feature vector has cardinality of 125 and another feature vector has cardinality of 270.

## Classification

 I select two classifiers in this lab: knn classifier and Naive Bayes classifier.

### knn classifier

I use Euclidean distance to find k (k = 5) neighbors. They will vote for predicted labels. For knn classifier, I will generate one or multiple labels for each test documents. 

### naive bayes classifier

Generate one label for each test documents.

## Accuracy

For feature vector with cardinality of 125:

- The accuracy of knn classifier is 0.792.
- The accuracy of naive bayes classifier is 0.716.

For feature vector with cardinality of 270:

- The accuracy of knn classifier is 0.814.
- The accuracy of naive bayes classifier is 0.735.

Use following method to measure the accuracy of two classifiers.

- First, extract the topics of test documents as a list named Y\_original.

- For knn classifier, I will generate one or multiple labels for each test documents. Compare each predicted label with original class labels(topics), if a predicted label appears in original class labels, we collect it as a true label. Use

  ![](https://i.loli.net/2018/03/24/5ab5a81164d08.png)

  to measure the accuracy for a test document. Then the accuracy for the classifier is

  ![](https://i.loli.net/2018/03/24/5ab5a8a574325.png)

- For naive bayes classifier, I will generate one label for each test documents. Compare each predicted label with original class labels(topics), if a predicted label appears in original class labels, we collect it as a true label. Use

  ![](https://i.loli.net/2018/03/24/5ab5a8cb7188d.png)

  to measure the accuracy for a test document. We can see the `accuracy_i` is 0 or 1.

  Then the accuracy for the classifier is

  ![](https://i.loli.net/2018/03/24/5ab5a8e9eeeb9.png)



## Time Complexity

### The offline efficiency cost (time to build model)

For feature vector with cardinality of 125:

- The offline efficient cost of knn classifier is 0.69 s.
- The offline efficient cost of naive classifier is 1.41 s.

For feature vector with cardinality of 270:

- The offline efficient cost of knn classifier is 1.085 s.
- The offline efficient cost of naive classifier is 2.374 s.

### The online efficiency cost (time to classify)

For feature vector with cardinality of 125:

- The online efficient cost of knn classifier is 69.97 s.

- The online efficient cost of naive classifier is 30.056 s.

For feature vector with cardinality of 270:

- The online efficient cost of knn classifier is 122.966 s.
- The online efficient cost of naive classifier is 60.379 s.


Well, knn is really lazy and slow...