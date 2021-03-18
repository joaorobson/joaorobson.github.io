---
date: 2020-05-04T10:42:42-42:00
featured_image: "/images/metrics.jpg"
tags: ["metrics", "machine learning", "classification"]
title: "Essential metrics for classification tasks"
---


## Introduction

A machine learning classification problem refers to the task of predicting the class of a sample given its features. Usually, these problems are divided into three types: binary, multiclass, and multilabel classification.

The difference is simple. When the problem is formed by only two classes, for instance, predict if the genre of a book is fiction or non-fiction or if a news is fake or not, it is called binary classification. Otherwise, when the number of possible classes is three or more, the problem could be multiclass or multilabel.

Multiclass tasks happen when each sample can be assigned only to one class. One example of that is the classic machine learning problem of labeling the digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). This problem involves predicting a number from 0 to 9 based on a handwritten digit image, that is, from 10 possible classes, one could assign only one of these to a sample.

The generalization of this situation, when there are no constraints to the number of classes that could be given to a sample, in other words, if one could tag a sample with one, three or one hundred classes (or whatever the total number of distinct classes inherent to the problem is), is known as multilabel classification. It happens, for instance, in the classification of a band's genre, that could include more than one single tag to each artist.


## In practice, what is the difference between them?

Basically, what differentiates the classification categories is how the data itself is represented. 

(The algorithms used to make predictions using the data, as well as how their outputs are produced, tend to vary too, but this explanation is a little bit complex. A good starting point to understand these details is [this page](https://scikit-learn.org/stable/modules/multiclass.html) from the documentaion of sklearn, a widely used tool in ML world, where there is a wonderful overview of strategies and models used on multiclass and multilabel classification.)

For **binary problems**, the data is usually represented by "1's" and "0's", "1" used for the samples of what is known as the positive class and "0's" for the negative class, like this: 

{{< rawhtml >}}

    <p align="center">
      [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]
    </p>

{{< /rawhtml >}}

It is important to say that these values could be anything ("Yes" and "No", "Good" and "Bad", "Blue" and "Red", etc.), as long as they are distinct.

For **multiclass problems**, each label is represented by a distinct value too, usually ranging from 0 to n-1, n being the number of distinct classes. A possible representation for the data in a problem with 5 classes is this vector:

{{< rawhtml >}}

    <p align="center">
      [2, 3, 2, 1, 0, 4, 4, 0, 1, 2, 4, 4, 2, 1, 1, 3, 2, 2, 4]
    </p>
{{< /rawhtml >}}

In **multilabel tasks**, things become a little more complex. In these problems, there are two strategies usually used: **Binary Relevance** and **Label powerset**. 

In the **Binary Relevance** technique, the data is formed by a vector with n positions (n being the number of distinct classes). For each position, one of two values could be given: one pointing the occurence of the class represented by that index in that sample and another indicating absence. Usually, these values are "1" and "0". So, imagining that there are five classes in our data ("0", "1", "2", "3" and "4") and 10 samples, we could have:
 

{{< rawhtml >}}

    <p align="center">
      [[0, 1, 0, 1, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 1, 1, 0, 0], </br>
       [1, 0, 0, 0, 0], [1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 0, 1, 0, 1], [1, 1, 1, 0, 1]]
    </p>

{{< /rawhtml >}}

The first item in the list, for instance, indicates that the the labels present in the first sample are "1" and "3".

The **Label powerset** option refers to the strategy of transforming the set of label combinations present in the data represented by the binary revelance method into a set of distinct labels, i.e., transform a multilabel problem into a multiclass problem. Taking the list above as an example, we would have the following transformation:


{{< rawhtml >}}
    <p align="center">
      [0, 1, 0, 1, 0] -> 0 </br>
      [1, 0, 0, 0, 0] -> 1 </br>
      [1, 0, 0, 0, 1] -> 2 </br>
      [1, 0, 1, 0, 1] -> 3 </br>
      [1, 1, 0, 0, 0] -> 4 </br>
      [1, 1, 1, 0, 0] -> 5 </br>
      [1, 1, 1, 0, 1] -> 6 </br>
      [1, 1, 1, 1, 1] -> 7 
    </p>
{{< /rawhtml >}}

And the resulting data would be:

{{< rawhtml >}}

    <p align="center">
      [0, 4, 7, 2, 5, 1, 5, 4, 3, 6]
    </p>


{{< /rawhtml >}}

Another important difference between these three types of classification is the way that the quality of the predictions is measured. Each one takes a slightly distinct approach to quantify how well the model is performing. Let's take a look at that.

### Binary classification evaluation

One of the most basic ways to measure how well a model predicts the classes of new samples is by using a diagram known as the confusion matrix. It is used to visualize how the predicted and actual classes are related. So, using the fiction/non-fiction example mentioned early, we could have a matrix that looks like this:


{{< rawhtml >}}
<img src="/images/metrics_classif/conf_matrix.png"  width="80%" style="max-width: 70%;margin: 0% 15%" />
{{< /rawhtml >}}


The rows group the number of predicted samples in each class. The columns account for the actual number of samples presented in each class. But how to make conclusions out of that?

First of all, it is important to grasp the concept behind true/false positives and true/false negatives.

True positives refer to the condition where the model correctly predicts the positive class, which in our example is "Fiction". Equivalently, a true negative occurs when the model correctly classifies a sample as being from the negative class ("Non-fiction").

The two remaining terms refer to situations when the model makes a mistake. So, a false positive (also known as type I error) happens when the sample belongs to the negative class but it is predicted as being from the positive one. So, when the label "Fiction" is assigned to a fiction book, a false positive happens. The opposite, i.e., a false negative (or type II error), occurs when a positive class sample (a fiction book) is incorrectly labeled as a negative class (non-fiction).

With this in mind, let's come back to the example. We have 100 books divided like this:

* 43 fiction books;
* 57 non-fiction books.

The model predicted the majority of the books in the right class, but some were misclassified. Summing up, there are 42 true positives, 10 false positives, 1 false negative, and 47 true negatives. 

Ok. But how to interpret this result? Is it good or bad? We need some metrics to make an evaluation.



#### Understanding accuracy, precision, and recall

In classification tasks, the most common metrics are accuracy, precision, recall, and f1-score. They provide a numerical value, ranging from 0 to 1,
that evaluate the relationship between right and wrong predictions. Here are their formulas:

$$ Accuracy = \frac{ TP + TN }{ TP + TN + FP + FN }$$
$$ Precision = \frac{ TP }{ TP + FP }$$
$$ Recall = \frac{ TP }{ TP + FN }$$
$$ F\_{1} = \frac{ 2 \cdot Precision \cdot Recall }{ Precision + Recall } $$


Each one uses a different approach to analyze the obtained results, pondering the right and wrong predictions in a slightly different way.

**Accuracy** represents how many predictions from the total were right:

$$ Accuracy = \frac{ 42 + 47 }{42 + 47 + 10 + 1} = \frac{89}{100} = 89 \\% $$


**Precision** measures the percentage of correctly positive samples classifications in relation to all the samples that were classified as being from the positive class. In our example:

$$ Precision = \frac{ 42 }{42 + 10} = \frac{42}{52} \approx 81 \\% $$


**Recall** follows a similar approach to precision, but instead of using the total of samples labeled with the positive class to calculate the percentage, it returns the percentage of correctly classified positive samples in relation to the total number of actual positive samples. Here it is its value:

$$ Recall = \frac{ 42 }{42 + 1} = \frac{42}{43} \approx 98 \\% $$


The last presented formula, known as **F1 score**, evaluates the [harmonic mean](https://en.wikipedia.org/wiki/Harmonic_mean) of precision and recall. For this example, its value is:


$$ F\_{1} = \frac{ 2 \cdot 0.81 \cdot 0.98 }{0.81 + 0.98} = \frac{1.58}{1.79} \approx 88 \\% $$


Ok. Now we have 4 values and all results look great, right? 

They are all close to 100%. Accuracy, for instance, tells us that of 100 predicted classes, almost 90% were correct predictions. 

Apparently, regardless of the problem someone is tackling, these numbers would seem enough to tell that our model is close to the best possible. Is this really true? Let's see.

Imagine that instead of classifying books, the model was predicting if the type of a patient's cancer is benign or malignant. Keeping the numbers unchanged, of 43 people that have benign cancer, only 1 would be misclassified. That's not such a huge problem. The test could be repeated to double-check on that. The big issue is with false positives. Of 57 people with a form of cancer that is malignant, 10 (approximately 20%) would receive a result saying that they have a disease that is less aggressive. This is bad. Imagine how many of these people would not repeat the exam and have serious consequences for that.

> A raw number by itself is not enough to define how good a prediction is. The context of the problem has huge importance to determine the metrics to be used and what is "good" or "bad".

Sometimes, more false negatives than false positives, as the benign/malignant cancer example, and consequently a recall lower than precision, is preferable than the contrary. On other occasions, achieve an accuracy of 70% (which is better than 50/50 guess) would be excellent. Each problem is unique and the role of a Machine Learning enthusiast or professional is to understand the metrics and use the most appropiate in each context.

### Multiclass classification evaluation


Imagine that one decides that instead of classifying the books in fiction or non-fiction, there will be a classification based on the number of stars that the book received on a online store. So, supposing that there are five classes (1 to 5 stars), one possible confusion matrix for this problem would look like that:

{{< rawhtml >}}
<img src="/images/metrics_classif/conf_matrix2.png"  width="50%" style="max-width: 50%;margin: 0% 25%" />
{{< /rawhtml >}}


Repair that it keeps having the 100 books, but there are 5 columns and 5 rows. Moreover, books are classified according to their number of stars now. How the true/false positives and negatives will be measure now without "positive" or "negative" classes? Let's check it out.

Accuracy is the easiest one to calculate. As said before, it is represented by the division of the total number of right predictions by the total number of samples. So, looking at the matrix above, all the numbers on the diagonal with darker blue squares account for the right predictions because they represent the situations where the model assigned the correct class to that sample.
For this problem, the accuracy is:

$$ Accuracy = \frac{ 15 + 14 + 10 + 12 + 17 }{100} = \frac{68}{100} = 68 \\% $$

Ok. But how to calculate the values of precision and recall? They depend on the number of false positives/negatives and consequently on the existence of positive/negative classes.

A multiclass problem could be treated as an extension of the binary classification. We only need to consider the problem as a ["collection of binary problems, one for each class"](https://scikit-learn.org/stable/modules/model_evaluation.html#from-binary-to-multiclass-and-multilabel), i.e., identify if a book has 1 to 5 stars is the same as finding out if it should have 1 star or not. If yes, obiviously it has 1 star and the problem is solved. If not, it should have 2 stars or more. For 2 stars, the same. Either the book has 2 starts or it belongs to one of the other subsequent classes (3, 4 or 5). If it has not 2, it will have 3 or more. And so on. In this way, we split the task of classifying a sample in one of 5 classes into 5 binary problems and we are able to calculate the metrics for each class. Then, we only need to combine or average the separated metrics into one, producing the final result.

So, let's consider the samples with 1 star, for example. We will treat the other classes as "2 or more". Here how the confusion matrix would look like:

{{< rawhtml >}}
<img src="/images/metrics_classif/2_more.png"  width="70%" style="max-width: 70%;margin: 0% 15%" />
{{< /rawhtml >}}

Now, calculating precision, recall, and f1-score for the class "1 star", we have:

$$ Precision = \frac{ 15 }{ 15 + 3} = \frac{15}{18} \approx 83 \\% $$
$$ Recall = \frac{ 15 }{15 + 5} = \frac{15}{20} = 75 \\% $$
$$ F\_{1} = \frac{ 2 \cdot 0.83 \cdot 0.75 }{ 0.83 + 0.75}  \approx 79 \\% $$

Repeating this process with each one of the remaining classes, there will be a total of 5 values for each one of these metrics, one for each star, as shown below:

$$ Precision\_{1star} = \frac{ 15 }{ 18 }  \approx 83 \\% \quad\quad 
Recall\_{1star} = \frac{ 15 }{ 20 }  = 75 \\% \quad\quad
F\_{11star} = \frac{ 2 \cdot 0.83 \cdot 0.75 }{ 0.83 + 0.75}  \approx 79 \\% $$

$$ Precision\_{2stars} = \frac{ 14 }{ 24 }  \approx 58 \\% \quad\quad 
Recall\_{2stars} = \frac{ 14 }{ 16 }  \approx 87 \\% \quad\quad
F\_{12stars} = \frac{ 2 \cdot 0.58 \cdot 0.87 }{ 0.58 + 0.87}  \approx 70 \\% $$

$$ Precision\_{3stars} = \frac{ 10 }{ 14 }  \approx 71 \\% \quad\quad 
Recall\_{3stars} = \frac{ 10 }{ 24 }  \approx 42 \\% \quad\quad
F\_{13stars} = \frac{ 2 \cdot 0.71 \cdot 0.42 }{ 0.71 + 0.42}  \approx 53 \\% $$


$$ Precision\_{4stars} = \frac{ 12 }{ 22 }  \approx 54 \\% \quad\quad 
Recall\_{4stars} = \frac{ 12 }{ 12 }  = 100 \\% \quad\quad
F\_{14stars} = \frac{ 2 \cdot 0.54 \cdot 1 }{ 0.54 + 1}  \approx 70 \\% $$


$$ Precision\_{5stars} = \frac{ 17 }{ 22 }  \approx 77 \\% \quad\quad 
Recall\_{5stars} = \frac{ 17 }{ 28 }  \approx 60 \\% \quad\quad
F\_{15stars} = \frac{ 2 \cdot 0.77 \cdot 0.6 }{ 0.77 + 0.6}  \approx 67 \\% $$

## Calculating precision and recall using sets intersection

It is interesting to show that precision and recall can also be calculated using sets intersection. Consider some vectors representing actual and predicted classes as two sets with the labels:


{{< rawhtml >}}
    <p align="center">
      Actual &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-> [1, 2, 0, 3, 4, 1, 1, 1, 0, 0, 3, 3, 2, 2, 4, 4, 0, 1, 2] </br>
      Predicted -> [1, 0, 0, 3, 1, 1, 0, 1, 1, 1, 1, 3, 0, 2, 1, 4, 0, 0, 2]
    </p>

{{< /rawhtml >}}

Precision and recall can be calculated for each class like this:

$$ Precision = \frac{|A \cap B|}{|B|} $$
$$ Recall = \frac{|A \cap B|}{|A|} $$

**A** stores all the values from one specific class occurring in the actual values vector.

**B** stores all the values from one specific class occurring in the predicted values vector.


So, for class "1", for instance:


{{< rawhtml >}}

    <p align="center">
      Actual &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;->
        [<span style="color: green;font-weight:bold">1</span>, 
        2, 
        0, 
        3, 
        4, 
        <span style="color: green;font-weight:bold">1</span>, 
        <span style="color: green;font-weight:bold">1</span>, 
        <span style="color: green;font-weight:bold">1</span>, 
        0, 
        0, 
        3, 
        3,
        2, 
        2, 
        4, 
        4, 
        0, 
        <span style="color: green;font-weight:bold">1</span>, 
        2] </br>
      Predicted ->
        [<span style="color: red;font-weight:bold">1</span>, 
        0, 
        0, 
        3, 
        <span style="color: red;font-weight:bold">1</span>, 
        <span style="color: red;font-weight:bold">1</span>, 
        0, 
        <span style="color: red;font-weight:bold">1</span>, 
        <span style="color: red;font-weight:bold">1</span>, 
        <span style="color: red;font-weight:bold">1</span>, 
        <span style="color: red;font-weight:bold">1</span>, 
        3,
        0, 
        2, 
        <span style="color: red;font-weight:bold">1</span>, 
        4, 
        0, 
        3, 
        2] </br>
    </p>
    <p align="center">
      A -> [1, 1, 1, 1, 1] </br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;B -> [1, 1, 1, 1, 1, 1, 1, 1]
    </p>

{{< /rawhtml >}}
So, the intersection of **A** and **B** represents the number of predictions of class "1" that are correct, i.e., true positives.

The number of elements in **A** represents the number of samples of the positive class, in this case, the class "1".

The number of elements in **B** represents the number of predictions assigned with the positive class label ("1", in this case).

With that in mind, it is clear why the formulas above allow us calculate precision and recall. We are dividing the number of true positives by the number of true positives plus the number of false positives (samples that were assigned "1" but are from another class) i.e., precision, and dividing the true positives by the number of samples that are actually from class "1", i.e., the recall.


-----------------------------

Now, there are 15 values representing the performance of the predictions. It's time to combine them to produce one single value for each one of the metrics. These strategies use the sets calculation above.

## Averaging precision, recall and f1-score

As shown above, in classification tasks with more than 2 classes, each metric will have one value per class. But real-world problems can exceed dozens of classes easily and understanding a model performance with hundred of values will become increasingly difficult. Because of that, some strategies to combine the results for each class exist. Multiclass problems usually use three ways to make this combination. They are very common (scikit-learn support them, for instance) and simple to be understood. Let's check them. 

### Micro average

Of the 3 strategies, the only one that does not use a summation is the micro average. Its calculation basically uses the concept of sets intersection presented previously to determine the value of each metric. According to scikit-learn, "micro-averaging may be preferred in multilabel settings, including multiclass classification where a majority class is to be ignored." So, the definition is basically:

|micro|

For our example, the vectors with actual and predicted labels look like this:

{{< rawhtml >}}

    <p align="center">
      Actual &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-> 
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, </br>
       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, </br>
       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
       3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, </br>
       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
       4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, </br>
       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
       5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5] </br>
      Predicted &nbsp;-> [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 5, </br>
       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 3, 3, 3, 3, </br>
       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      3, 3, 3, 3, 3, 3, 1, 2, 2, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, </br>
       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, </br>
       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    </p>
{{< /rawhtml >}}

The intersection (Actual ∩ Predicted) is represented by equal elements at the same position in both vectors. The result is:

{{< rawhtml >}}

    <p align="center">
      Actual ∩ Predicted -> 
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]</br>
    </p>

{{< /rawhtml >}}

So, the metrics using micro-average are:

$$ Precision\_{Micro} = \frac{ |Actual \cap Predicted|}{ |Predicted| } = \frac{68}{100} = 68\\% $$ 
$$ Recall\_{Micro} = \frac{ |Actual \cap Predicted|}{ |Actual| } = \frac{68}{100} = 68\\% $$ 
$$ F\_{1Micro} = \frac{ 2 \cdot 0.68 \cdot 0.68 }{ 0.68 + 0.68} = \frac{0.924}{1.36} \approx 68 \\% $$


 

### Macro average

The second way to combine the metrics is by "macro-averaging" them. It works simply calculating the mean of the values giving the same weight to all the classes. According to the [sklearn documentation](https://scikit-learn.org/stable/modules/model_evaluation.html), in problems where there are classes with a low frequency, the use of the macro average strategy may highlight their performance. However, attach the same level of importance to all classes is often wrong. The final result could be such that the low performance of infrequent classes might be "over-emphasized". 

Although there are issues using macro-average in situations where the data is unevenly distributed, this strategy could be used when the difference among the number of samples for each class is not so big.

Below, the macro-average formula:

$$
Macro \\: average = \frac{1}{|L|} \sum\_{l \in L} M(y\_{l}, \hat{y\_{l}})
$$

Where:

* *L* is the set of labels;
* *M* is the metric;
* *y* is the set of true labels;
* *ŷ* is the set of predicted labels;
* {{< rawhtml >}} <i>y<sub>l</sub></i> {{< /rawhtml >}} is the subset of *y* with label *l*;
* {{< rawhtml >}} <i>ŷ<sub>l</sub></i> {{< /rawhtml >}} is the subset of *ŷ* with label *l*.


For our example, macro-averaging the metrics will provide the following results:

$$ Precision\_{Macro} = \frac{ 0.83 + 0.58 + 0.71 + 0.54 + 0.77 }{ 5 } \approx 69\\% $$ 
$$ Recall\_{Macro} = \frac{ 0.75 + 0.87 + 0.42 + 1 + 0.6}{ 5 } \approx 73\\% $$ 
$$ F\_{1Macro} = \frac{ 0.79 + 0.7 + 0.53 + 0.7 + 0.67 }{ 5 }  \approx 68 \\% $$


### Weighted average

The weighted average combines the metrics in such a way that the frequency of each class in the true samples is taken into account, i.e., the metric score for each class is weighted according to its presence in the true labels vector. Using this technique, classes that have more samples have a greater influence in the final result than classes with few samples. So, the result is calculated proportionally to the distribution of the classes.

The formula for this strategy is:

$$
Weighted \\: average = \frac{1}{\sum\_{l \in L} |y\_{l}|} \sum\_{l \in L} |y\_{l}| M(y\_{l}, \hat{y\_{l}})
$$

Where:

* *L* is the set of labels;
* *M* is the metric;
* *y* is the set of true labels;
* *ŷ* is the set of predicted labels;
* {{< rawhtml >}} <i>y<sub>l</sub></i> {{< /rawhtml >}} is the subset of *y* with label *l*;
* {{< rawhtml >}} <i>ŷ<sub>l</sub></i> {{< /rawhtml >}} is the subset of *ŷ* with label *l*.

For our example, combine the obtained metrics using weighted average results in:

$$ Precision\_{Weighted} = \frac{1}{ 100 } \cdot (20 \cdot 0.83 + 16 \cdot 0.58 + 24 \cdot 0.71 + 12 \cdot 0.54 + 28 \cdot 0.77) \approx 71\\% $$ 
$$ Recall\_{Weighted} = \frac{1}{ 100 } \cdot (20 \cdot 0.75 + 16 \cdot 0.87 + 24 \cdot 0.42 + 12 \cdot 1 + 28 \cdot 0.6) \approx 68\\% $$ 
$$ F\_{1Weighted} = \frac{1}{ 100 } \cdot (20 \cdot 0.79 + 16 \cdot 0.7 + 24 \cdot 0.53 + 12 \cdot 0.7 + 28 \cdot 0.67) \approx 68\\% $$ 


## Multilabel classification evaluation

The last type of classification task is called multilabel, where each sample could be associated with more than one label. One example of multilabel classification is musical genres tagging in platforms such as [last.fm](https://www.last.fm).

As shown before, there are two usual forms to make multilabel classification: Binary relevance and Label powerset. Let's focus on evaluating a problem that uses the first option because is the most common way to deal with a multilabel task and it provides an efficient and flexible way to train and test the model.

So, imagine that one want to create a model that tags automatically the genre of some bands. There are uncountable musical genres. For simplicity, let's use only five genres (Eletronic, Folk, Instrumental, Rap and Rock). Based on the tags provided by last.fm, the dataset would look like that:

{{< rawhtml >}}
<style>
table, th, td {
	width: 100%;
  border: 1px solid #bebebe;
  border-collapse: collapse;
}
th, td {
  padding: 15px;
}
</style>

{{< /rawhtml >}}

| Band/Artist        | Eletronic  | Folk  | Instrumental | Rap | Rock |
|--------------------|------------|-------|--------------|-----|:----:|
| Bon Iver           |     1      |   1   |      0       | 0   |    0 |
| Beastie Boys       |     0      |   0   |      1       | 1   |    1 |
| Linkin Park        |     1      |   0   |      1       | 1   |    1 |
| twenty one pilots  |    1       |   0   |    0         |  1  |  1   |
| ...                |    ...     |  ...  |    ...       | ... |  ... |

So, the model output for each sample will be composed of five values, one for each genre we are predicting. But how to evaluate this result? Let's check out some methods.

### Evalutating results from multilabel models

#### Samples average

This method is another way to average the three metrics shown before (precision, recall, and f1-score), but it is destined to multilabel tasks only. It works calculating the metric for each pair of actual and predicted values for each sample in the test data, resulting in a unique averaged value. The formula is:

$$
Samples\\: average = \frac{1}{|S|} \sum\_{s \in S} M(y\_{s}, \hat{y\_{s}})
$$

Where:

* *S* is the set of samples;
* *M* is the metric;
* {{< rawhtml >}} <i>y<sub>s</sub></i> {{< /rawhtml >}} is the sth sample of *y*;
* {{< rawhtml >}} <i>ŷ<sub>s</sub></i> {{< /rawhtml >}} is the sth sample of *ŷ*.

So, for the four bands above, a model could output something like this:

| Band/Artist        | Eletronic  | Folk  | Instrumental | Rap | Rock |
|--------------------|------------|-------|--------------|-----|------|
| Bon Iver           |     1      |   0   |      0       | 1   |    0 |
| Beastie Boys       |     0      |  1    |      1       | 0   |    1 |
| Linkin Park        |     1      |   0   |      1       | 1   |    1 |
| twenty one pilots  |    1       |   1   |    0         |  1  |  1   |

And for each sample, precision, recall, and f1-score are calculated. So, for the first sample ("Bon Iver"), the metrics are:


$$ Precision\_{Bon\\:Iver} = \frac{TP}{TP + FP} = \frac{|\\{1,1,0,0,0\\} \cap \\{1,0,0,1,0\\}|}{|\text{n for each n} \in \\{ 1, 0, 0, 1, 0 \\} \text{where n = 1}|} =\frac{1}{2} = 50 \\% $$
$$ Recall\_{Bon\\:Iver} = \frac{TP}{TP + FN} = \frac{|\\{1,1,0,0,0\\} \cap \\{1,0,0,1,0\\}|}{|\text{n for each n} \in \\{ 1, 1, 0, 0, 0 \\} \text{where n = 1}|} =\frac{1}{2} = 50 \\% $$
$$ F\_{1Bon\\:Iver} = \frac{2 \cdot 0.5 \cdot 0.5}{0.5 + 0.5} = 50 \\% $$

Calculating the metrics for the other three artists and combining them:


$$ Precision\_{Samples} = \frac{1}{4} (0.5 + 0.667 + 1 + 0.75) \approx 73 \\% $$
$$ Recall\_{Samples} = \frac{1}{4} (0.5 + 0.667 + 1 + 1) \approx 79 \\% $$
$$ F\_{1Samples} = \frac{2 \cdot 0.73 \cdot 0.79 }{0.73 + 0.79} \approx 76 \\% $$

#### Hamming Loss

Hamming Loss is another strategy to evaluate multilabel models' results. It is defined as the fraction of the labels that are
wrongly predicted. So, it basically checks how many values in the predictions vector are different from the actual values.

The formula is:

$$
Hamming\\:Loss = \frac{1}{n\_{labels}} \sum\_{j = 0}^{n\_{labels} - 1} 1, \text{if} \\: y\_{j} \neq \hat{y\_{j}} 
$$

For our example, the actual and predicted vectors are like this:


{{< rawhtml >}}
    <p align="center">
      Actual &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;->
        [<span style="color: green;font-weight:bold">1</span>, 
        <span style="color: red;font-weight:bold">1</span>, 
        <span style="color: green;font-weight:bold">0</span>, 
        <span style="color: red;font-weight:bold">0</span>, 
        <span style="color: green;font-weight:bold">0</span>, 
        <span style="color: green;font-weight:bold">0</span>, 
        <span style="color: red;font-weight:bold">0</span>, 
        <span style="color: green;font-weight:bold">1</span>, 
        <span style="color: red;font-weight:bold">1</span>, 
        <span style="color: green;font-weight:bold">1</span>, 
        <span style="color: green;font-weight:bold">1</span>, 
        <span style="color: green;font-weight:bold">0</span>, 
        <span style="color: green;font-weight:bold">1</span>, 
        <span style="color: green;font-weight:bold">1</span>, 
        <span style="color: green;font-weight:bold">1</span>, 
        <span style="color: green;font-weight:bold">1</span>, 
        <span style="color: red;font-weight:bold">0</span>, 
        <span style="color: green;font-weight:bold">0</span>, 
        <span style="color: green;font-weight:bold">1</span>, 
        <span style="color: green;font-weight:bold">1</span>
        ] </br>
      Predicted ->
        [<span style="color: green;font-weight:bold">1</span>, 
        <span style="color: red;font-weight:bold">0</span>, 
        <span style="color: green;font-weight:bold">0</span>, 
        <span style="color: red;font-weight:bold">1</span>, 
        <span style="color: green;font-weight:bold">0</span>, 
        <span style="color: green;font-weight:bold">0</span>, 
        <span style="color: red;font-weight:bold">1</span>, 
        <span style="color: green;font-weight:bold">1</span>, 
        <span style="color: red;font-weight:bold">0</span>, 
        <span style="color: green;font-weight:bold">1</span>, 
        <span style="color: green;font-weight:bold">1</span>, 
        <span style="color: green;font-weight:bold">0</span>, 
        <span style="color: green;font-weight:bold">1</span>, 
        <span style="color: green;font-weight:bold">1</span>, 
        <span style="color: green;font-weight:bold">1</span>, 
        <span style="color: green;font-weight:bold">1</span>, 
        <span style="color: red;font-weight:bold">1</span>, 
        <span style="color: green;font-weight:bold">0</span>, 
        <span style="color: green;font-weight:bold">1</span>, 
        <span style="color: green;font-weight:bold">1</span>
        ] </br>
    </p>

{{< /rawhtml >}}
And the Hamming Loss:

$$ Hamming \\: Loss = \frac{5}{20} = 0.25 = 25 \\% $$

# Conclusion

This article provided an introduction to some essential concepts involving the basics of evaluating
classification problems. Unfortunately, not every single existing metric was shown here.
There are dozen of metrics that help us to understand and analize the performance of a machine learning
algorithm. But understanding deeply the difference among the three types of classification, the meaning
of True/False Positives/Negatives and how they can be related is already a huge step to be able to grasp more advanced topics.

I hope that the three main metrics used in classification (precision, recall, and f1-score) are clearer to you after reading the article.
And as always, any kind of feedback is welcome.

# References and Further Readings

* [scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html) (Documentation)
* [Classification: True vs. False and Positive vs. Negative](https://developers.google.com/machine-learning/crash-course/classification/true-false-positive-negative
) (Google ML Crash Course)
* [Machine Learning Fundamentals: The Confusion Matrix](https://www.youtube.com/watch?v=Kdsp6soqA7o) (YouTube video)
* [Deep dive into multi-label classification..!](https://towardsdatascience.com/journey-to-the-center-of-multi-label-classification-384c40229bff) (Medium article)
