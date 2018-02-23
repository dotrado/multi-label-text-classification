# Description

preprocess.py is to read data from reuter-21578, parser the data and translate them into list of documents which has the class labels and feature vector.

Reuters-21578是文本分析任务中比较重要的数据集，其包含一系列SGML格式的文件，需要我们进行预处理。

preprocess.py的工作就是从reuters-21578中读取数据，将文本转化成分词向量，将TOPICS和PLACES中的内容转化为class labels，类标签将用于之后的多标签文本分类。

# How to tokenize words

I use regular expression to find all string which only contains [a-z.-]. Now, I don't lemmatize or stem the words.

# How to filter words

I choose documetn frequency to filter more important words of which df is between 3 and 0.9 * all_documents.

# To run the program:

Type in command line, in this case, the default data path is data.
```bash
python3 preprocess.py
```

Type in command line, data_path is self-defined data path.
```
python3 preprocess.py data_path
```
    
# View output

It will output two files: dataset.csv, vocabulary.csv

1. dataset.csv:
    
    Feature vectors and class labels are stored in dataset.csv. The first row is

    ```bash
    document_id - [class labels] - (term, tf)
    ```

    which is the structure of stored data. Followings are list of documents.

    ```bash
    document 0
    class labels:
    canada
    feature vector:
    "(inco,10)","(sees,1)","(major,2)","(impact,3)","(removal,3)","(toronto,2)","(march,1)","(expect,1)","(earlier,1)","(reported,1)","(jones,2)","(industrial,1)","(index,5)","(make,2)","(company,1)","(stock,5)","(individuals,1)","(institutions,1)","(shares,1)","(industrials,1)","(spokesman,1)","(cherney,1)","(reply,1)","(query,1)","(closed,1)","(lower,2)","(active,1)","(trading,1)","(exchange,1)","(wall,1)","(street,1)","(journal,1)","(selects,1)","(dropped,1)","(representative,1)","(market,1)","(non-communist,1)","(world,1)","(largest,1)","(nickel,3)","(producer,1)","(member,1)","(replacing,1)","(owens-illinois,1)","(coca-cola,1)","(boeing,1)","(effective,1)","(tomorrow,1)","(analyst,1)","(ilmar,1)","(martens,2)","(walwyn,1)","(stodgell,1)","(cochran,1)","(murray,1)","(spark,1)","(short-term,1)","(selling,1)","(pressure,1)","(investors,1)","(suddenly,1)","(eliminate,1)","(investment,1)","(added,1)","(move,1)","(long-term,1)","(struggled,1)","(recent,1)","(years,1)","(sharply,1)","(prices,1)","(earnings,1)","(fell,1)","(dlrs,2)","(previous,1)","(year,1)","(reuter,1)"
    ```

    The structure of document data:
    1. The first row means the id of this document.
    2. The second and third row is class label part. Class labels part will list class labels which come from TOPICS and PLACES.
    3. The fouth row and fifth row is feature vector. Feature vector is a list of string. The structure of each element of the vector is (term, frequency).

2. vocabulary.csv:
    The data in vocabulary.csv is in the structure (term, index) which means the index of term in the term-vector. The point here is that the feature vector of all documents can construct a sparse matrix. Most of values of feature vector of a document are 0. Therefore, I want to store the mapping of term to index in vocabulary and then I only need to store the terms of which the value is not 0 in the document object for saving storage.

    ```
    inco,0
    sees,1
    major,2
    impact,3
    removal,4
    ...
    ```


