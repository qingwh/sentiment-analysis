# Sentiment analysis for IPhone and Galaxy
Helio Is now working with a government health agency to create a suite of smart phone medical apps for use by aiding workers in developing countries. The government agency required that app suite to be bundled with one model of smart phone. Helio had created a short list of five devices that are all capable of executing the app suite’s functions. To help Helio narrow their list down to one device, we have been asked to examine the prevalence of positive and negative attitudes toward these devices on the web.

I used the AWS Elastic Map Reduce (EMR) platform to run a series of Hadoop Streaming jobs and collected large amounts of smart phone-related web pages from 400 to 500 archive files 
from a massive repository of web data called the Common Crawl. This give us around 20,000 web pages that contain references to smart phones. I compiled the data into large matrix files of 20 thousand instances.


I also developed five different classifiers, namely C5.0, random forest, KKNN and support vector machines, GBM, and identified most optimal model to predict sentiment for IPhone and Galaxy. I applied various feature engineering methods (e.g. RFE, PCA, Near Zero Variance, etc.), compared their performances and obtained the best method to conduct feature selection.

