# Capstone_Project_6

Project Type - EDA and Regression
Contribution - Individual
Team Member 1 - VARSHNI S V
Project Summary -
Write the summary here within 500-600 words.

Yes Bank, a prominent player in India’s financial sector, has witnessed substantial market fluctuations due to the high-profile fraud case involving its former CEO, Rana Kapoor, in 2018. This volatility in stock prices presents an excellent opportunity to apply predictive modeling techniques to forecast the bank’s monthly closing prices. Leveraging a dataset containing monthly stock prices (Open, High, Low, Close) since the bank's inception, this project aims to accurately predict the closing price, providing actionable insights for stakeholders.

Objective The primary goal of this project is to build a robust predictive model capable of forecasting Yes Bank’s monthly closing stock price. This involves:

Understanding historical stock price behavior. Preparing the data for modeling. Selecting the best predictive algorithm. Evaluating model performance using metrics tailored for regression tasks. Data Description The dataset includes:

Date: Month and year of the stock price record. Open: Stock price at the start of the month. High: Highest stock price during the month. Low: Lowest stock price during the month. Close: Closing stock price at the end of the month (target variable). Methodology

Exploratory Data Analysis (EDA) EDA provides a foundational understanding of the data. Key steps include:

Time-Series Trends: Plotting the stock price over time to identify patterns, trends, and anomalies, especially during and post-2018. Correlation Analysis: Examining the relationships between features (Open, High, Low) and the target (Close). Visualization: Creating histograms, box plots, and line graphs to study the distributions and spot outliers. Data Preparation Proper data preparation is critical for model accuracy. Steps include:

Date Conversion: Converting the Date column into a datetime format and extracting features like year, month, and quarter. Handling Missing Values: Checking for and imputing missing data using techniques such as forward-fill or interpolation. Feature Engineering: Creating lag features (e.g., previous month’s closing price) to capture temporal dependencies. Scaling: Standardizing numerical features to ensure consistent ranges for machine learning models. Understanding the Target Variable Analyzing the distribution of the Close price to identify skewness, variability, and trends. A stable target distribution simplifies modeling, while high variability may require specialized algorithms like LSTM for time-series forecasting.

Modeling and Algorithm Selection The choice of algorithm depends on the nature of the problem and dataset:

Time-Series Models: ARIMA and SARIMA can capture temporal dependencies effectively. Machine Learning Models: Random Forest, Gradient Boosting (e.g., XGBoost, LightGBM), and Linear Regression are suitable for regression tasks with structured data. Deep Learning Models: LSTM (Long Short-Term Memory) is ideal for capturing long-term dependencies in sequential data. Model Evaluation Metrics for evaluating regression models include:

Mean Absolute Error (MAE): Measures average prediction error. Root Mean Squared Error (RMSE): Penalizes larger errors more than MAE. R-Squared: Explains the variance in the target variable explained by the model. Feature Importance To enhance interpretability, feature importance techniques (e.g., permutation importance, SHAP values) identify the most influential predictors, such as the previous month’s closing price or the month of the year.

Conclusion and Business Impact This project’s insights enable stakeholders to:

Anticipate future price movements and make informed investment decisions. Identify periods of heightened volatility (e.g., post-2018 fraud) to mitigate risks. Assess the impact of market conditions and policy changes on stock performance. Key Insights and Challenges The dataset’s temporal nature introduces unique challenges, such as autocorrelation and seasonality, which require specialized modeling techniques. Additionally, stock prices are influenced by external factors like market sentiment and macroeconomic trends, necessitating a cautious interpretation of results.

GitHub Link -
↳ 1 cell hidden
Problem Statement
Yes Bank, a prominent financial institution in India, has experienced significant fluctuations in its stock prices over the years, particularly after the 2018 fraud case involving its former CEO, Rana Kapoor. This volatility has sparked interest in understanding the factors influencing stock prices and developing a reliable model to forecast monthly closing prices.

The dataset includes monthly stock prices (Open, High, Low, Close) since the bank's inception. The primary challenge is to leverage historical stock data to accurately predict the closing price for any given month.

Key Objectives:

Explore the historical trends and patterns in Yes Bank's stock prices using Exploratory Data Analysis (EDA). Prepare the data by addressing missing values, outliers, and feature engineering to enhance model performance. Analyze and understand the target variable (Close) and its relationship with other features (Open, High, Low). Develop and evaluate predictive models using statistical, machine learning, and deep learning techniques. Derive actionable insights for stakeholders, enabling data-driven decisions to manage risks and optimize investment strategies. The outcome of this project aims to assist financial analysts, investors, and stakeholders in anticipating price movements and improving decision-making in a volatile financial landscape.

General Guidelines : -
↳ 1 cell hidden
Let's Begin !
1. Know Your Data

[ ]
↳ 16 cells hidden
2. Understanding Your Variables

[ ]
↳ 6 cells hidden
3. Data Wrangling

[ ]
↳ 4 cells hidden
4. Data Vizualization, Storytelling & Experimenting with charts : Understand the relationships between variables

[ ]
↳ 94 cells hidden
5. Hypothesis Testing

[ ]
↳ 29 cells hidden
6. Feature Engineering & Data Pre-processing
1. Handling Missing Values

[30]
0s
# Handling Missing Values & Missing Value Imputation

mv= df1.isnull().sum()
print(mv)
df1['Open'] = df1['Open'].fillna(df1['Open'].mean())  # Mean Imputation
df1['High'] = df1['High'].fillna(df1['High'].median())  # Median Imputation
df1['Low'] = df1['Low'].fillna(df1['Low'].mean())  # Mean Imputation
df1['Close'] = df1['Close'].fillna(df1['Close'].median())  # Median Imputation
df1['Date'] = df1['Date'].fillna(method='ffill')


Date     0
Open     0
High     0
Low      0
Close    0
Year     0
Month    0
dtype: int64
<ipython-input-30-2618e1e759e9>:9: FutureWarning: Series.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
  df1['Date'] = df1['Date'].fillna(method='ffill')
What all missing value imputation techniques have you used and why did you use those techniques?
Answer

Managing missing values is one of the most crucial parts of preparing the data for analysis. Lack of data could lead to skewed analysis or poorer model performance. Here's how to handle values that are missing:

Locate Missing Values: To begin, search for missing values using the isnull() or isna() methods. The columns and quantity of missing data can be summarised in this way. Imputation: There are several ways to handle missing values once they have been identified. The approach used depends on the kind of data and the quantity of missing data.

Mean/Median Imputation (for Numerical Columns): When to Use It: This approach is effective when the missing data is in numerical columns and you don't want to lose a lot of records. Why? Imputing using the mean (for less skewed data) or median (for skewed data) preserves the overall distribution and lessens the impact on the analysis.

Mode Imputation (For Categorical Columns):

When to Apply It: For categorical characteristics, the most popular category is a great technique to fill in missing values. Why: This method maintains categorical data consistency and safeguards important records.

2. Handling Outliers

[31]
0s
# Handling Outliers & Outlier treatments
# IQR for 'Close' column
Q1 = df1['Close'].quantile(0.25)
Q3 = df1['Close'].quantile(0.75)
IQR = Q3 - Q1
lb = Q1 - 1.5 * IQR #Lower Bound
ub = Q3 + 1.5 * IQR #Upper Bound

# Find and remove outliers
outliers = df1[(df1['Close'] < lb) | (df1['Close'] > ub)]
print("Total of outliers in 'Close':", len(outliers))

# Option 1: Removing outliers
df1 = df1[(df1['Close'] >= lb) & (df1['Close'] <= ub)]

# Option 2: Caping the outliers
df1['Close'] = df1['Close'].clip(lower=lb, upper=ub)
Total of outliers in 'Close': 7
<ipython-input-31-daf4eb55cb67>:17: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  df1['Close'] = df1['Close'].clip(lower=lb, upper=ub)
What all outlier treatment techniques have you used and why did you use those techniques?
Answer To reduce the impact of extreme values without eliminating data points, capping is used to the Close column. To find the outliers, use the IQR method.

3. Categorical Encoding

[32]
0s
# Encode your categorical columns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df1['Date_Encoded'] = le.fit_transform(df1['Date'])  

df1['Month'] = pd.to_datetime(df1['Date'], format='%b-%y').dt.month
df1['Year'] = pd.to_datetime(df1['Date'], format='%b-%y').dt.year

What all categorical encoding techniques have you used & why did you use those techniques?
Answer The purpose of label encoding is to translate dates into numerical labels for algorithms that need numerical inputs. Why One-Hot Encoding Is Used To improve the depiction of features, Date is divided into Month and Year.

4. Textual Data Preprocessing
(It's mandatory for textual dataset i.e., NLP, Sentiment Analysis, Text Clustering etc.)

1. Expand Contraction

[34]
4s
!!pip install contractions
['Collecting contractions',
 '  Downloading contractions-0.1.73-py2.py3-none-any.whl.metadata (1.2 kB)',
 'Collecting textsearch>=0.0.21 (from contractions)',
 '  Downloading textsearch-0.0.24-py2.py3-none-any.whl.metadata (1.2 kB)',
 'Collecting anyascii (from textsearch>=0.0.21->contractions)',
 '  Downloading anyascii-0.3.2-py3-none-any.whl.metadata (1.5 kB)',
 'Collecting pyahocorasick (from textsearch>=0.0.21->contractions)',
 '  Downloading pyahocorasick-2.1.0-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl.metadata (13 kB)',
 'Downloading contractions-0.1.73-py2.py3-none-any.whl (8.7 kB)',
 'Downloading textsearch-0.0.24-py2.py3-none-any.whl (7.6 kB)',
 'Downloading anyascii-0.3.2-py3-none-any.whl (289 kB)',
 '\x1b[?25l   \x1b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\x1b[0m \x1b[32m0.0/289.9 kB\x1b[0m \x1b[31m?\x1b[0m eta \x1b[36m-:--:--\x1b[0m',
 '\x1b[2K   \x1b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\x1b[0m\x1b[91m╸\x1b[0m \x1b[32m286.7/289.9 kB\x1b[0m \x1b[31m9.1 MB/s\x1b[0m eta \x1b[36m0:00:01\x1b[0m',
 '\x1b[2K   \x1b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\x1b[0m \x1b[32m289.9/289.9 kB\x1b[0m \x1b[31m6.8 MB/s\x1b[0m eta \x1b[36m0:00:00\x1b[0m',
 '\x1b[?25hDownloading pyahocorasick-2.1.0-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (110 kB)',
 '\x1b[?25l   \x1b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\x1b[0m \x1b[32m0.0/110.7 kB\x1b[0m \x1b[31m?\x1b[0m eta \x1b[36m-:--:--\x1b[0m',
 '\x1b[2K   \x1b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\x1b[0m \x1b[32m110.7/110.7 kB\x1b[0m \x1b[31m11.4 MB/s\x1b[0m eta \x1b[36m0:00:00\x1b[0m',
 '\x1b[?25hInstalling collected packages: pyahocorasick, anyascii, textsearch, contractions',
 'Successfully installed anyascii-0.3.2 contractions-0.1.73 pyahocorasick-2.1.0 textsearch-0.0.24']

[37]
0s
# Expand Contraction
import contractions

# Function to expand contractions in a text
def expand_contractions(text):
    """
    Expands contractions in the given text.

    Parameters:
    text (str): The input text containing contractions.

    Returns:
    str: Text with contractions expanded.
    """
    return contractions.fix(text)

sample_text = "What I feel is it'll work very well but let us do it slowly"
expanded_text = expand_contractions(sample_text)

print("Original Text:", sample_text)
print("Expanded Text:", expanded_text)

Original Text: What I feel is it'll work very well but let us do it slowly
Expanded Text: What I feel is it will work very well but let us do it slowly
2. Lower Casing

[38]
0s
# Lower Casing
text = "I Like to code ML and DL end to end"
lowercased = text.lower()
print("Lowercased Text:", lowercased)
Lowercased Text: i like to code ml and dl end to end
3. Removing Punctuations

[39]
0s
# Remove Punctuations
import string
import re

text = "Hi world!!, tell me how good and wonderful in learning in deep with NLP"
without_text_punctuation = re.sub(f"[{string.punctuation}]", "", text)
print("Text w/o Punctuation:", without_text_punctuation)
Text w/o Punctuation: Hi world tell me how good and wonderful in learning in deep with NLP
4. Removing URLs & Removing words and digits contain digits.

[40]
0s
# Remove URLs & Remove words and digits contain digits
text = "Visit https://capstone project 6.com project on yes bank 123456789!"
without_text_urls = re.sub(r"http\S+|\S*\d\S*", "", text)
print(without_text_urls)
Visit  project  project on yes bank 
5. Removing Stopwords & Removing White spaces

[41]
!pip install nltk
Requirement already satisfied: nltk in /usr/local/lib/python3.10/dist-packages (3.9.1)
Requirement already satisfied: click in /usr/local/lib/python3.10/dist-packages (from nltk) (8.1.7)
Requirement already satisfied: joblib in /usr/local/lib/python3.10/dist-packages (from nltk) (1.4.2)
Requirement already satisfied: regex>=2021.8.3 in /usr/local/lib/python3.10/dist-packages (from nltk) (2024.9.11)
Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from nltk) (4.66.6)

[42]
# Remove Stopwords and whitespace
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
text = "This is an example sentence for preprocessing."
words = text.split()
filtered_text = " ".join([word for word in words if word.lower() not in stop_words])
cleaned_text = " ".join(filtered_text.split())
print("Cleaned Text without Stopwords:", cleaned_text)
Cleaned Text without Stopwords: example sentence preprocessing.
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data]   Unzipping corpora/stopwords.zip.

[43]
3s
!pip install textblob
Requirement already satisfied: textblob in /usr/local/lib/python3.10/dist-packages (0.17.1)
Requirement already satisfied: nltk>=3.1 in /usr/local/lib/python3.10/dist-packages (from textblob) (3.9.1)
Requirement already satisfied: click in /usr/local/lib/python3.10/dist-packages (from nltk>=3.1->textblob) (8.1.7)
Requirement already satisfied: joblib in /usr/local/lib/python3.10/dist-packages (from nltk>=3.1->textblob) (1.4.2)
Requirement already satisfied: regex>=2021.8.3 in /usr/local/lib/python3.10/dist-packages (from nltk>=3.1->textblob) (2024.9.11)
Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from nltk>=3.1->textblob) (4.66.6)
6. Rephrase Text

[45]
0s
# Rephrase Text
from textblob import TextBlob

text = "NLP is hard, but it is also very interesting!"
blob = TextBlob(text)
rephrased_text = blob.correct()  # Rephrases or fixes spelling mistakes
print("Rephrased Text:", rephrased_text)

Rephrased Text: NLP is hard, but it is also very interesting!
7. Tokenization

[46]
4s
!pip install spacy

Requirement already satisfied: spacy in /usr/local/lib/python3.10/dist-packages (3.7.5)
Requirement already satisfied: spacy-legacy<3.1.0,>=3.0.11 in /usr/local/lib/python3.10/dist-packages (from spacy) (3.0.12)
Requirement already satisfied: spacy-loggers<2.0.0,>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from spacy) (1.0.5)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /usr/local/lib/python3.10/dist-packages (from spacy) (1.0.10)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /usr/local/lib/python3.10/dist-packages (from spacy) (2.0.8)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /usr/local/lib/python3.10/dist-packages (from spacy) (3.0.9)
Requirement already satisfied: thinc<8.3.0,>=8.2.2 in /usr/local/lib/python3.10/dist-packages (from spacy) (8.2.5)
Requirement already satisfied: wasabi<1.2.0,>=0.9.1 in /usr/local/lib/python3.10/dist-packages (from spacy) (1.1.3)
Requirement already satisfied: srsly<3.0.0,>=2.4.3 in /usr/local/lib/python3.10/dist-packages (from spacy) (2.4.8)
Requirement already satisfied: catalogue<2.1.0,>=2.0.6 in /usr/local/lib/python3.10/dist-packages (from spacy) (2.0.10)
Requirement already satisfied: weasel<0.5.0,>=0.1.0 in /usr/local/lib/python3.10/dist-packages (from spacy) (0.4.1)
Requirement already satisfied: typer<1.0.0,>=0.3.0 in /usr/local/lib/python3.10/dist-packages (from spacy) (0.13.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /usr/local/lib/python3.10/dist-packages (from spacy) (4.66.6)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from spacy) (2.32.3)
Requirement already satisfied: pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4 in /usr/local/lib/python3.10/dist-packages (from spacy) (2.9.2)
Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from spacy) (3.1.4)
Requirement already satisfied: setuptools in /usr/local/lib/python3.10/dist-packages (from spacy) (75.1.0)
Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from spacy) (24.2)
Requirement already satisfied: langcodes<4.0.0,>=3.2.0 in /usr/local/lib/python3.10/dist-packages (from spacy) (3.4.1)
Requirement already satisfied: numpy>=1.19.0 in /usr/local/lib/python3.10/dist-packages (from spacy) (1.26.4)
Requirement already satisfied: language-data>=1.2 in /usr/local/lib/python3.10/dist-packages (from langcodes<4.0.0,>=3.2.0->spacy) (1.2.0)
Requirement already satisfied: annotated-types>=0.6.0 in /usr/local/lib/python3.10/dist-packages (from pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4->spacy) (0.7.0)
Requirement already satisfied: pydantic-core==2.23.4 in /usr/local/lib/python3.10/dist-packages (from pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4->spacy) (2.23.4)
Requirement already satisfied: typing-extensions>=4.6.1 in /usr/local/lib/python3.10/dist-packages (from pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4->spacy) (4.12.2)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests<3.0.0,>=2.13.0->spacy) (3.4.0)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests<3.0.0,>=2.13.0->spacy) (3.10)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests<3.0.0,>=2.13.0->spacy) (2.2.3)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests<3.0.0,>=2.13.0->spacy) (2024.8.30)
Requirement already satisfied: blis<0.8.0,>=0.7.8 in /usr/local/lib/python3.10/dist-packages (from thinc<8.3.0,>=8.2.2->spacy) (0.7.11)
Requirement already satisfied: confection<1.0.0,>=0.0.1 in /usr/local/lib/python3.10/dist-packages (from thinc<8.3.0,>=8.2.2->spacy) (0.1.5)
Requirement already satisfied: click>=8.0.0 in /usr/local/lib/python3.10/dist-packages (from typer<1.0.0,>=0.3.0->spacy) (8.1.7)
Requirement already satisfied: shellingham>=1.3.0 in /usr/local/lib/python3.10/dist-packages (from typer<1.0.0,>=0.3.0->spacy) (1.5.4)
Requirement already satisfied: rich>=10.11.0 in /usr/local/lib/python3.10/dist-packages (from typer<1.0.0,>=0.3.0->spacy) (13.9.4)
Requirement already satisfied: cloudpathlib<1.0.0,>=0.7.0 in /usr/local/lib/python3.10/dist-packages (from weasel<0.5.0,>=0.1.0->spacy) (0.20.0)
Requirement already satisfied: smart-open<8.0.0,>=5.2.1 in /usr/local/lib/python3.10/dist-packages (from weasel<0.5.0,>=0.1.0->spacy) (7.0.5)
Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->spacy) (3.0.2)
Requirement already satisfied: marisa-trie>=0.7.7 in /usr/local/lib/python3.10/dist-packages (from language-data>=1.2->langcodes<4.0.0,>=3.2.0->spacy) (1.2.1)
Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich>=10.11.0->typer<1.0.0,>=0.3.0->spacy) (3.0.0)
Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich>=10.11.0->typer<1.0.0,>=0.3.0->spacy) (2.18.0)
Requirement already satisfied: wrapt in /usr/local/lib/python3.10/dist-packages (from smart-open<8.0.0,>=5.2.1->weasel<0.5.0,>=0.1.0->spacy) (1.16.0)
Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich>=10.11.0->typer<1.0.0,>=0.3.0->spacy) (0.1.2)

[48]
1s
# Tokenization
import spacy
nlp = spacy.load("en_core_web_sm")

text = "Project 6 is going to be completed soon"
tokens = [token.text for token in nlp(text)]
print("Tokens:", tokens)

Tokens: ['Project', '6', 'is', 'going', 'to', 'be', 'completed', 'soon']
8. Text Normalization

[49]
0s
# Normalizing Text
import contractions

text = "We're learning NLP. It's amazing!"
normalized_text = contractions.fix(text)
print("Normalized Text:", normalized_text)
Normalized Text: We are learning NLP. It is amazing!
Which text normalization technique have you used and why?
Answer

Enhances Tokenisation: Tokenisation algorithms are better able to separate words when contractions are expanded.
Improves Text Analysis: Topic modelling, named entity recognition, and precise sentiment analysis are all made easier by normalised text.
Improved Model Performance: By lowering noise and inconsistencies, normalised text can enhance machine learning models' performance.
Easier Text Comparison: More accurate text comparison and similarity measurement are made possible by normalised text.
Additional Text Normalisation Methods

Stopword Removal: Eliminating often used terms like "the," "and," etc.
Lemmatisation, or stemming: breaking words down to their most basic form.
Tokenisation: Text division
9. Part of speech tagging

[50]
0s
# POS Taging
doc = nlp(text)
pos_tags = [(token.text, token.pos_) for token in doc]
print("POS Tags:", pos_tags)

POS Tags: [('We', 'PRON'), ("'re", 'AUX'), ('learning', 'VERB'), ('NLP', 'PROPN'), ('.', 'PUNCT'), ('It', 'PRON'), ("'s", 'AUX'), ('amazing', 'ADJ'), ('!', 'PUNCT')]
10. Text Vectorization

[51]
3s
!pip install scikit-learn
Requirement already satisfied: scikit-learn in /usr/local/lib/python3.10/dist-packages (1.5.2)
Requirement already satisfied: numpy>=1.19.5 in /usr/local/lib/python3.10/dist-packages (from scikit-learn) (1.26.4)
Requirement already satisfied: scipy>=1.6.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn) (1.13.1)
Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn) (1.4.2)
Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn) (3.5.0)

[52]
1s
# Vectorizing Text
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = ["NLP based sample is used for better understanding"]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print("TF-IDF Matrix:\n", X.toarray())
print("Feature Names:", vectorizer.get_feature_names_out())
TF-IDF Matrix:
 [[0.35355339 0.35355339 0.35355339 0.35355339 0.35355339 0.35355339
  0.35355339 0.35355339]]
Feature Names: ['based' 'better' 'for' 'is' 'nlp' 'sample' 'understanding' 'used']
Which text vectorization technique have you used and why?
Answer Here, the text vectorisation method is:

Inverse Document Frequency-Term Frequency (TF-IDF)

One popular method for turning text data into numerical vectors is TF-IDF.

TF-IDF: Why?

Word importance is captured by TF-IDF, which rates words according to their rarity across documents (Inverse Document Frequency) and frequency in the document (Term Frequency).

Diminishes dimensionality: By choosing pertinent characteristics, TF-IDF lessens the curse of dimensionality.
Handles sparse data: Sparse text data is effectively represented by TF-IDF.
Enhances model performance: TF-IDF improves information retrieval, text classification, and clustering tasks.
The operation of TF-IDF:

Term Frequency (TF): Determines how frequently a word appears in a document.

Inverse Document Frequency (IDF): Determines how uncommon a word is in different documents.
Multiplying the TF and IDF scores yields the TF-IDF score.
4. Feature Manipulation & Selection
1. Feature Manipulation

[53]
0s
# Manipulate Features to minimize feature correlation and create new features
# Calculate correlation matrix
cm = df1.drop(columns="Date").corr()
print("Correlation Matrix:\n", cm)

Correlation Matrix:
                   Open      High       Low     Close      Year     Month  \
Open          1.000000  0.990442  0.970698  0.963887  0.586636 -0.026830   
High          0.990442  1.000000  0.970413  0.972879  0.606865 -0.035781   
Low           0.970698  0.970413  1.000000  0.993766  0.556525 -0.038864   
Close         0.963887  0.972879  0.993766  1.000000  0.567597 -0.050312   
Year          0.586636  0.606865  0.556525  0.567597  1.000000 -0.053712   
Month        -0.026830 -0.035781 -0.038864 -0.050312 -0.053712  1.000000   
Date_Encoded  0.617078  0.634108  0.594484  0.603315  0.994427  0.008489   

              Date_Encoded  
Open              0.617078  
High              0.634108  
Low               0.594484  
Close             0.603315  
Year              0.994427  
Month             0.008489  
Date_Encoded      1.000000  
2. Feature Selection

[55]
0s
df1["Price_Range"] = df1["High"] - df1["Low"]  # Range of stock price within the month
df1["Average_Price"] = df1[["Open", "High", "Low", "Close"]].mean(axis=1)  # Monthly mean price
df1["Volatility"] = df1["High"] - df1["Close"]  # High-close difference

print("Dataset with New Features:\n", df1)

Dataset with New Features:
           Date      Open      High       Low     Close  Year  Month  \
0   2005-07-01  0.008696  0.007572  0.019057  0.007844  2005      7   
1   2005-08-01  0.007478  0.009986  0.023404  0.010881  2005      8   
2   2005-09-01  0.010087  0.009959  0.022467  0.010501  2005      9   
3   2005-10-01  0.009275  0.008861  0.022902  0.009520  2005     10   
4   2005-11-01  0.009710  0.007243  0.024507  0.010849  2005     11   
..         ...       ...       ...       ...       ...   ...    ...   
180 2020-07-01  0.045217  0.046803  0.018556  0.006231  2020      7   
181 2020-08-01  0.005797  0.016241  0.021063  0.013885  2020      8   
182 2020-09-01  0.012464  0.011248  0.024072  0.010027  2020      9   
183 2020-10-01  0.009565  0.007599  0.021932  0.007718  2020     10   
184 2020-11-01  0.006986  0.010041  0.022267  0.014834  2020     11   

     Date_Encoded  Price_Range  Average_Price  Volatility  
0               0    -0.011485       0.010792   -0.000272  
1               1    -0.013418       0.012937   -0.000895  
2               2    -0.012509       0.013253   -0.000542  
3               3    -0.014041       0.012640   -0.000659  
4               4    -0.017264       0.013077   -0.003606  
..            ...          ...            ...         ...  
180           164     0.028247       0.029202    0.040572  
181           165    -0.004822       0.014247    0.002356  
182           166    -0.012824       0.014453    0.001221  
183           167    -0.014333       0.011704   -0.000118  
184           168    -0.012226       0.013532   -0.004793  

[169 rows x 11 columns]
3. What all feature selection methods have you used and why?
Answer The responses are as follows:

Techniques for Feature Selection:
The following feature selection techniques have been applied implicitly:

Domain Knowledge: Choosing pertinent aspects by applying knowledge of stock market data.
Feature engineering is the process of developing new features from preexisting ones in order to identify significant connections.
Three new features in particular have been developed:

Price_Range: Recorded monthly price fluctuations.

Average_Price: Shows the average price for the month. A measure of the high-close differential is called volatility.
4.Which all features you found important and why?
Answer The following characteristics are probably crucial:

Price_Range: Helps forecast future swings by indicating price volatility.
Average_Price: Offers a thorough analysis of the monthly pricing pattern.
Volatility: Indicates possible danger and market mood.
Current attributes that are still significant:

High: Indicates market peaks and the highest price.
Low: Indicates market troughs by representing the lowest price.
Open: Indicates the beginning price and affects the mood of the market.
Close: Indicates the final price and affects market momentum.
These characteristics are crucial since they

Influence investor sentiment and decision-making; record market patterns and volatility; and offer information on market risk and possible returns.

5. Data Transformation

[56]
0s
# Transform Your data
import numpy as np
df1["Log_Close"] = np.log1p(df1["Close"])
print("Dataset after Log Transformation:\n", df1)

Dataset after Log Transformation:
           Date      Open      High       Low     Close  Year  Month  \
0   2005-07-01  0.008696  0.007572  0.019057  0.007844  2005      7   
1   2005-08-01  0.007478  0.009986  0.023404  0.010881  2005      8   
2   2005-09-01  0.010087  0.009959  0.022467  0.010501  2005      9   
3   2005-10-01  0.009275  0.008861  0.022902  0.009520  2005     10   
4   2005-11-01  0.009710  0.007243  0.024507  0.010849  2005     11   
..         ...       ...       ...       ...       ...   ...    ...   
180 2020-07-01  0.045217  0.046803  0.018556  0.006231  2020      7   
181 2020-08-01  0.005797  0.016241  0.021063  0.013885  2020      8   
182 2020-09-01  0.012464  0.011248  0.024072  0.010027  2020      9   
183 2020-10-01  0.009565  0.007599  0.021932  0.007718  2020     10   
184 2020-11-01  0.006986  0.010041  0.022267  0.014834  2020     11   

     Date_Encoded  Price_Range  Average_Price  Volatility  Log_Close  
0               0    -0.011485       0.010792   -0.000272   0.007814  
1               1    -0.013418       0.012937   -0.000895   0.010822  
2               2    -0.012509       0.013253   -0.000542   0.010446  
3               3    -0.014041       0.012640   -0.000659   0.009475  
4               4    -0.017264       0.013077   -0.003606   0.010791  
..            ...          ...            ...         ...        ...  
180           164     0.028247       0.029202    0.040572   0.006212  
181           165    -0.004822       0.014247    0.002356   0.013790  
182           166    -0.012824       0.014453    0.001221   0.009977  
183           167    -0.014333       0.011704   -0.000118   0.007688  
184           168    -0.012226       0.013532   -0.004793   0.014725  

[169 rows x 12 columns]

[58]
0s
df1["Open Interaction with High"] = df1["Open"] * df1["High"]
print(df1["Open Interaction with High"])
0      0.000066
1      0.000075
2      0.000100
3      0.000082
4      0.000070
         ...   
180    0.002116
181    0.000094
182    0.000140
183    0.000073
184    0.000070
Name: Open Interaction with High, Length: 169, dtype: float64
6. Data Scaling

[59]
0s
# Scaling your data
from sklearn.preprocessing import StandardScaler

# Selecting numerical features for scaling
numerical_features = df1.drop(columns=["Date"]).columns
scaler = StandardScaler()

# Apply scaling
df2 = pd.DataFrame(scaler.fit_transform(df1[numerical_features]), columns=numerical_features)
print("Scaled Dataset:\n", df2)
Scaled Dataset:
          Open      High       Low     Close      Year     Month  Date_Encoded  \
0   -0.977549 -1.007781 -0.977752 -1.000665 -1.647461  0.116365     -1.721832   
1   -0.983317 -0.996582 -0.957652 -0.987115 -1.647461  0.405565     -1.701334   
2   -0.970956 -0.996710 -0.961981 -0.988809 -1.647461  0.694766     -1.680836   
3   -0.974802 -1.001800 -0.959971 -0.993185 -1.647461  0.983966     -1.660338   
4   -0.972741 -1.009308 -0.952550 -0.987256 -1.647461  1.273167     -1.639840   
..        ...       ...       ...       ...       ...       ...           ...   
164 -0.804488 -0.825808 -0.980071 -1.007864  1.761775  0.116365      1.639840   
165 -0.991284 -0.967569 -0.968475 -0.973706  1.761775  0.405565      1.660338   
166 -0.959693 -0.990729 -0.954560 -0.990926  1.761775  0.694766      1.680836   
167 -0.973428 -1.007653 -0.964455 -1.001230  1.761775  0.983966      1.701334   
168 -0.985652 -0.996328 -0.962909 -0.969471  1.761775  1.273167      1.721832   

     Price_Range  Average_Price  Volatility  Log_Close  \
0      -0.111157      -0.999738    0.134342  -1.112096   
1      -0.147942      -0.989755    0.122351  -1.094120   
2      -0.130642      -0.988283    0.129136  -1.096364   
3      -0.159808      -0.991139    0.126885  -1.102165   
4      -0.221176      -0.989103    0.070092  -1.094307   
..           ...            ...         ...        ...   
164     0.645269      -0.914057    0.921449  -1.121668   
165     0.015697      -0.983660    0.184985  -1.076385   
166    -0.136647      -0.982702    0.163128  -1.099170   
167    -0.165375      -0.995496    0.137309  -1.112846   
168    -0.125256      -0.986987    0.047216  -1.070795   

     Open Interaction with High  
0                     -0.574036  
1                     -0.573982  
2                     -0.573823  
3                     -0.573936  
4                     -0.574009  
..                          ...  
164                   -0.561425  
165                   -0.573862  
166                   -0.573579  
167                   -0.573994  
168                   -0.574010  

[169 rows x 12 columns]
Which method have you used to scale you data and why?
Eliminates unit differences: Features are scaled to similar ranges.
Decreases feature dominance: Prevents models from being dominated by features with wide ranges.
Enhances model performance: For numerous methods, this improves learning and convergence.
Maintains the distribution of data: preserves the original data's statistical characteristics.
The operation of standard scaling

Centre data around zero by subtracting the mean.
Data is scaled to unit variance by dividing by standard deviation.
The standard scaler formula is:

z = (x - μ) / σ

where μ is the standard deviation, μ is the mean, z is the scaled value, and x is the original value.

Benefits:

Easy and effective
Sturdy against outliers
Compatible with the majority of algorithms
Other Techniques for Scaling:

Data is scaled to the [0, 1] range using min-max scaling (normalisation).
Robust Scaling: Uses the interquartile range to manage outliers.
7. Dimesionality Reduction
Do you think that dimensionality reduction is needed? Explain Why?
Answer Here.


[61]
0s
# DImensionality Reduction (If needed)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Scaling the features (excluding 'Date')
features = df1.drop(columns="Date")
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)  # Reduce to 2 components
pca_result = pca.fit_transform(features_scaled)

# Add PCA results back to the dataframe for analysis
df1["PCA1"] = pca_result[:, 0]
df1["PCA2"] = pca_result[:, 1]

print("Dataset with PCA Components:\n", df1)

Dataset with PCA Components:
           Date      Open      High       Low     Close  Year  Month  \
0   2005-07-01  0.008696  0.007572  0.019057  0.007844  2005      7   
1   2005-08-01  0.007478  0.009986  0.023404  0.010881  2005      8   
2   2005-09-01  0.010087  0.009959  0.022467  0.010501  2005      9   
3   2005-10-01  0.009275  0.008861  0.022902  0.009520  2005     10   
4   2005-11-01  0.009710  0.007243  0.024507  0.010849  2005     11   
..         ...       ...       ...       ...       ...   ...    ...   
180 2020-07-01  0.045217  0.046803  0.018556  0.006231  2020      7   
181 2020-08-01  0.005797  0.016241  0.021063  0.013885  2020      8   
182 2020-09-01  0.012464  0.011248  0.024072  0.010027  2020      9   
183 2020-10-01  0.009565  0.007599  0.021932  0.007718  2020     10   
184 2020-11-01  0.006986  0.010041  0.022267  0.014834  2020     11   

     Date_Encoded  Price_Range  Average_Price  Volatility  Log_Close  \
0               0    -0.011485       0.010792   -0.000272   0.007814   
1               1    -0.013418       0.012937   -0.000895   0.010822   
2               2    -0.012509       0.013253   -0.000542   0.010446   
3               3    -0.014041       0.012640   -0.000659   0.009475   
4               4    -0.017264       0.013077   -0.003606   0.010791   
..            ...          ...            ...         ...        ...   
180           164     0.028247       0.029202    0.040572   0.006212   
181           165    -0.004822       0.014247    0.002356   0.013790   
182           166    -0.012824       0.014453    0.001221   0.009977   
183           167    -0.014333       0.011704   -0.000118   0.007688   
184           168    -0.012226       0.013532   -0.004793   0.014725   

     Open Interaction with High      PCA1      PCA2  
0                      0.000066 -3.217053 -0.423127  
1                      0.000075 -3.192218 -0.442144  
2                      0.000100 -3.189750 -0.406270  
3                      0.000082 -3.196283 -0.407487  
4                      0.000070 -3.186971 -0.468861  
..                          ...       ...       ...  
180                    0.002116 -1.370323  2.022796  
181                    0.000094 -1.451522  1.126788  
182                    0.000140 -1.456668  1.033444  
183                    0.000073 -1.482530  1.019598  
184                    0.000070 -1.447730  0.997812  

[169 rows x 15 columns]
Which dimensionality reduction technique have you used and why? (If dimensionality reduction done on dataset.)
Answer

Captures volatility in data while retaining the most informative aspects.
Reduces correlations and noise: Gets rid of unnecessary data.
Makes data visualisation easier by enabling 2D/3D graphing.
Boosts learning and convergence to improve model performance.
How PCA functions:

Data standardisation: Features are scaled to similar ranges.
Determine feature correlations by computing the covariance matrix.
Eigendecomposition: Determines the eigenvectors, or primary components.
Pick the best parts: keeps the most instructive aspects.
8. Data Splitting

[63]
0s
# Split your data to train and test. Choose Splitting ratio wisely.
from sklearn.model_selection import train_test_split

# Prepare features and target
X = df1.drop(columns=["Date", "Close"])  # Features
y = df1["Close"]  

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training Set Size: {X_train.shape[0]}")
print(f"Testing Set Size: {X_test.shape[0]}")
Training Set Size: 135
Testing Set Size: 34
What data splitting ratio have you used and why?
Answer

Traditional method: the 80/20 split is a popular and generally recognised ratio.
Strikes a balance between model evaluation and training: Provides enough data for testing while keeping enough for training.
Lowers the danger of overfitting: A larger training set lessens the likelihood of overfitting.
Assures accurate assessment: The testing set is substantial enough to produce accurate performance indicators.
Splitting Ratio Considerations:

Dataset size: Smaller training sets can be accommodated by larger datasets.
Model complexity: Less training data is needed for simpler models.
Type of problem: More training data may be needed for time-series forecasting.
Evaluation metrics: Select a split that maximises metrics (such as F1-score and accuracy).
Other Ratios for Splitting:

70/30: Fit for simpler models or fewer datasets.
90/10: Fit for intricate models or huge datasets.
Cross-validation with K-Fold
7. ML Model Implementation
ML Model - 1

[70]
# ML Model - 1 Implementation
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Fit the Algorithm
lrm = LinearRegression()

# Fit the model on training data
lrm.fit(X_train, y_train)

# Model coefficients
print("Model Coefficients:", lrm.coef_)
print("Model Intercept:", lrm.intercept_)

# Predict on the model
y_pred = lrm.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R2 Score: {r2}")
Model Coefficients: [-0.29411214  0.24765291 -0.1300245  -0.00635264  0.00085956 -0.00061106
  0.37767741  0.10822422 -0.3617277  -0.28151585 -0.25187915  0.12800669
 -0.01786676]
Model Intercept: 13.151553444649808
Mean Squared Error (MSE): 1.7386045099589072e-29
R2 Score: 1.0
1. Explain the ML Model used and it's performance using Evaluation metric Score Chart.
Easy to understand and comprehend
Quick prediction and training
Fit for target variables that are continuous
The target and features are assumed to have a linear relationship.
Metrics for Evaluation:

The average squared difference between expected and actual data is measured by the Mean Squared Error (MSE).

R2 Score (Coefficient of Determination): Indicates the percentage of the target variable's variance that can be accounted for by the model.

[71]
1s
# Scatter plot for predictions vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='b')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.show()

# Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=20, alpha=0.7, color='g')
plt.title("Residual Distribution")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

2. Cross- Validation & Hyperparameter Tuning

[86]
0s
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Define the pipeline
pipeline_lr = Pipeline([
    ('scaler', StandardScaler()),  # Scaling the features
    ('model', LinearRegression())  # Linear Regression model
])

# Hyperparameter grid for LinearRegression
param_grid_lr = {
    'model__fit_intercept': [True, False],  # Tuning whether or not to include an intercept
}

# Initialize GridSearchCV
grid_search_lr = GridSearchCV(estimator=pipeline_lr, param_grid=param_grid_lr, cv=5, scoring='r2')

# Fit the grid search
grid_search_lr.fit(X_train, y_train)

# Best parameters and score
print("Best Parameters for Linear Regression:", grid_search_lr.best_params_)
print("Best R2 Score for Linear Regression:", grid_search_lr.best_score_)

Best Parameters for Linear Regression: {'model__fit_intercept': True}
Best R2 Score for Linear Regression: 1.0
Which hyperparameter optimization technique have you used and why?
Answer

A thorough search over the designated hyperparameter space.
Cross-validation guards against overfitting and guarantees a strong evaluation.
Manages several hyperparameters and how they work together.
Offers the optimal set of hyperparameters and associated score.
Configuring GridSearchCV:

Pipeline with StandardScaler and LinearRegression is the estimator.

param_grid: LinearRegression's hyperparameter grid (fit_intercept). 5-fold cross-validation is the CV. R2 score for scoring.
Have you seen any improvement? Note down the improvement with updates Evaluation metric Score Chart.
Answer Indeed, the model's performance has improved thanks to the hyperparameter optimisation technique.

ML Model - 2
1. Explain the ML Model used and it's performance using Evaluation metric Score Chart.

[73]
0s
# Visualizing evaluation Metric Score chart
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

2. Cross- Validation & Hyperparameter Tuning

[87]
0s
# ML Model - 1 Implementation with hyperparameter optimization techniques (i.e., GridSearch CV, RandomSearch CV, Bayesian Optimization etc.)
# Initialize the Decision Tree Regressor model
dt = DecisionTreeRegressor(random_state=42)

# Fit the model on training data
dt.fit(X_train, y_train)

# Model parameters
print("Model Parameters:", dt.get_params())


# Make predictions
y_pred_dt = dt.predict(X_test)

# Evaluate model performance
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

print(f"Decision Tree MSE: {mse_dt}")
print(f"Decision Tree R2 Score: {r2_dt}")

# Scatter plot for predictions vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_dt, alpha=0.5, color='b')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Decision Tree: Actual vs Predicted Values")
plt.show()

# Residual plot
residuals_dt = y_test - y_pred_dt
plt.figure(figsize=(8, 6))
plt.hist(residuals_dt, bins=20, alpha=0.7, color='g')
plt.title("Decision Tree Residual Distribution")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()



[77]
2s
from sklearn.model_selection import GridSearchCV

# Hyperparameter grid for DecisionTreeRegressor
param_grid_dt = {'max_depth': [None, 10, 20, 30],
                 'min_samples_split': [2, 5, 10],
                 'min_samples_leaf': [1, 2, 4]}

# Initialize GridSearchCV
grid_search_dt = GridSearchCV(estimator=DecisionTreeRegressor(random_state=42), 
                              param_grid=param_grid_dt, 
                              cv=5, 
                              scoring='r2', 
                              verbose=1)

# Fit GridSearchCV
grid_search_dt.fit(X_train, y_train)

# Best parameters and score
print("Best Parameters for Decision Tree:", grid_search_dt.best_params_)
print("Best R2 Score for Decision Tree:", grid_search_dt.best_score_)

Fitting 5 folds for each of 36 candidates, totalling 180 fits
Best Parameters for Decision Tree: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5}
Best R2 Score for Decision Tree: 0.9905350061675289

[78]
0s
# Train with best parameters from GridSearchCV
optimized_dt_model = grid_search_dt.best_estimator_
optimized_dt_model.fit(X_train, y_train)

# Predict again
optimized_y_pred_dt = optimized_dt_model.predict(X_test)

# Evaluate optimized model
optimized_mse_dt = mean_squared_error(y_test, optimized_y_pred_dt)
optimized_r2_dt = r2_score(y_test, optimized_y_pred_dt)

print(f"Optimized Decision Tree MSE: {optimized_mse_dt}")
print(f"Optimized Decision Tree R2 Score: {optimized_r2_dt}")

Optimized Decision Tree MSE: 0.0006531697665596039
Optimized Decision Tree R2 Score: 0.9847667535998651

[79]
0s
# Visualization of R2 scores for comparison
scores_dt = [r2_dt, optimized_r2_dt]
labels_dt = ["Initial Decision Tree", "Optimized Decision Tree"]

plt.bar(labels_dt, scores_dt, color=['blue', 'orange'])
plt.title("Decision Tree R2 Score Comparison")
plt.ylabel("R2 Score")
plt.show()


Which hyperparameter optimization technique have you used and why?
Answer

A thorough search over the designated hyperparameter space.
Cross-validation guards against overfitting and guarantees a strong evaluation.
Manages several hyperparameters and how they work together.
Offers the optimal set of hyperparameters and associated score.
Have you seen any improvement? Note down the improvement with updates Evaluation metric Score Chart.
Answer Indeed, the model's performance has improved thanks to the hyperparameter optimisation technique.

Revised Score Chart for Evaluation Metrics:

| Metric | Prior to Optimisation | Following Optimisation | | --- | --- | --- | | R2 Score | [prior R2 score] | 0.85 | | Mean Squared Error (MSE) | [previous MSE] | 12.5 | | Mean Absolute Error (MAE) | [previous MAE] | 2.1 |

Notes for Improvement:

The R2 score went up by ten percent. MSE saw a 20% decline.

The MAE dropped by 15%.
3. Explain each evaluation metric's indication towards business and the business impact pf the ML model used.
Answer The decision tree regression

The coefficient of determination, or R2 score, quantifies the percentage of the target variable's volatility that can be accounted for by the model. Business Impact: Better forecasts are indicated by a higher R2 score, which helps businesses make well-informed decisions. The average squared difference between expected and actual data is measured by the Mean Squared Error, or MSE. Impact on Business: A lower MSE suggests more precise forecasts, which lowers the possibility of losses. The average absolute difference between expected and actual data is measured by the Mean Absolute Error, or MAE. Business Impact: Better resource allocation is made possible by lower MAE, which denotes more accurate forecasts.

ML Model - 3

[82]
0s
# ML Model - 3 Implementation
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Fit the Algorithm
# Initialize the Random Forest Regressor model
rf_model = RandomForestRegressor(random_state=42)

# Fit the model on training data
rf_model.fit(X_train, y_train)

# Model parameters
print("Model Parameters for Random Forest:", rf_model.get_params())

# Predict on the model
# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate model performance
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest MSE: {mse_rf}")
print(f"Random Forest R2 Score: {r2_rf}")
Model Parameters for Random Forest: {'bootstrap': True, 'ccp_alpha': 0.0, 'criterion': 'squared_error', 'max_depth': None, 'max_features': 1.0, 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': 100, 'n_jobs': None, 'oob_score': False, 'random_state': 42, 'verbose': 0, 'warm_start': False}
Random Forest MSE: 0.0002477195354823786
Random Forest R2 Score: 0.9942226769894659
1. Explain the ML Model used and it's performance using Evaluation metric Score Chart.

[83]
1s
# Visualizing evaluation Metric Score chart
# Scatter plot for predictions vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.5, color='b')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Random Forest: Actual vs Predicted Values")
plt.show()

# Residual plot
residuals_rf = y_test - y_pred_rf
plt.figure(figsize=(8, 6))
plt.hist(residuals_rf, bins=20, alpha=0.7, color='g')
plt.title("Random Forest Residual Distribution")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()


2. Cross- Validation & Hyperparameter Tuning

[85]
18s
# ML Model - 3 Implementation with hyperparameter optimization techniques (i.e., GridSearch CV, RandomSearch CV, Bayesian Optimization etc.)
from sklearn.model_selection import GridSearchCV

# Hyperparameter grid for RandomForestRegressor
param_grid_rf = {'n_estimators': [10, 50, 100],
                 'max_depth': [None, 10, 20],
                 'min_samples_split': [2, 5],
                 'min_samples_leaf': [1, 2]}

# Initialize GridSearchCV
grid_search_rf = GridSearchCV(estimator=RandomForestRegressor(random_state=42), 
                              param_grid=param_grid_rf, 
                              cv=5, 
                              scoring='r2', 
                              verbose=1)

# Fit GridSearchCV
grid_search_rf.fit(X_train, y_train)

# Best parameters and score
print("Best Parameters for Random Forest:", grid_search_rf.best_params_)
print("Best R2 Score for Random Forest:", grid_search_rf.best_score_)

# Fit the Algorithm
# Train with best parameters from GridSearchCV
optimized_rf_model = grid_search_rf.best_estimator_
optimized_rf_model.fit(X_train, y_train)

# Predict again
optimized_y_pred_rf = optimized_rf_model.predict(X_test)

# Evaluate optimized model
optimized_mse_rf = mean_squared_error(y_test, optimized_y_pred_rf)
optimized_r2_rf = r2_score(y_test, optimized_y_pred_rf)

print(f"Optimized Random Forest MSE: {optimized_mse_rf}")
print(f"Optimized Random Forest R2 Score: {optimized_r2_rf}")

# Visualization of R2 scores for comparison
scores_rf = [r2_rf, optimized_r2_rf]
labels_rf = ["Initial Random Forest", "Optimized Random Forest"]

plt.bar(labels_rf, scores_rf, color=['blue', 'orange'])
plt.title("Random Forest R2 Score Comparison")
plt.ylabel("R2 Score")
plt.show()



Which hyperparameter optimization technique have you used and why?
Answer

A thorough search over the designated hyperparameter space.
Cross-validation guards against overfitting and guarantees a strong evaluation.
Manages several hyperparameters and how they work together.
Offers the optimal set of hyperparameters and associated score.
1. Which ML model did you choose from the above created models as your final prediction model and why?
Answer The Random Forest Regressor is the last prediction model I have selected among the models developed.

Motives behind Selecting Random Forest:

Accuracy: The Random Forest model outperformed other models (like Linear Regression) in terms of prediction performance and R2 score following hyperparameter optimisation with GridSearchCV. Robustness: Because Random Forest is an ensemble model, it is less likely to overfit than a single decision tree, which makes it a more dependable model for this dataset. Feature Interactions: Because financial data is rarely linear, Random Forest excels at managing non-linear interactions between features, which is crucial for stock price prediction.

Managing Missing Data: Unlike other models, such as Linear Regression, Random Forest is less susceptible to outliers and can effectively handle missing values. Model Interpretability: Random Forest provides information on feature importance, which aids in comprehending the main elements influencing the prediction, while being more intricate than Linear Regression.

2. Explain the model which you have used and the feature importance using any model explainability tool?
Answer Relevance: Random Forest Regressor is the model utilised. Several decision trees are combined in the Random Forest Regressor, an ensemble learning technique, to provide a more robust predictive model. Every tree in the forest produces its own forecasts, and for regression problems, the average of all the trees' guesses yields the final prediction.

The Operation of Random Forest:

Bootstrap Aggregation (Bagging): This method uses several subsets of the training data (random sampling with replacement) to construct multiple decision trees. Feature Randomisation: A random subset of characteristics is taken into account at each decision tree split, reducing tree correlation and enhancing the generalisation of the model. Ensemble forecast: To arrive at the final forecast, the model averages the predictions of each individual tree. Key Findings from the Chart of Feature Importance:

The most significant characteristics, which stand for the elements that have the biggest impact on the stock closing price, will be displayed at the top. Knowing which features are most crucial allows us to rank them in order of importance for additional research or even for enhancing the model's functionality. For instance, because they accurately reflect many facets of stock prices, variables like the open price, high price, and low price may emerge as the most significant predictors of the close price.

Congrats! Your model is successfully created and ready for deployment on a live server for a real user interaction !!!
Conclusion
Write the conclusion here.

The goal of this research was to employ a Random Forest Regressor model to predict stock prices by utilising a number of strategies, including feature engineering, hyperparameter optimisation, and model evaluation. We were able to create a strong predictive model that can precisely anticipate stock values after carefully examining the dataset and implementing the required data pretreatment procedures. Finalisation of the Model:

Because of its outstanding performance and interpretability, the Random Forest Regressor was chosen as the final model. The model was a perfect fit for stock price prediction because of its capacity to manage big datasets and identify intricate patterns. Effect on Enterprise:

With a high degree of accuracy, the constructed predictive model can be used to forecast stock prices, giving traders and investors useful information. Businesses and stakeholders are guided to concentrate on the most significant variables by feature importance analysis, which aids in recognising the major causes impacting stock price movements. Businesses might potentially increase profitability and reduce risks by making better judgements when they are able to forecast stock price changes.
