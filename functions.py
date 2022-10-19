# basic datascience imports
import pandas as pd
import numpy as np

# regular expressions import
import re

# visualization imports
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# bi- and n- gram import
import nltk

# modeling imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# shhhhhhh
import warnings
warnings.filterwarnings("ignore")


def clean(text: str) -> list:
    'A simple function to cleanup text data'
    wnl = nltk.stem.WordNetLemmatizer()
    stopwords = set(nltk.corpus.stopwords.words('english'))
    text = (text.encode('ascii', 'ignore')
             .decode('utf-8', 'ignore')
             .lower())
    words = re.sub(r'[^\w\s]', '', text)#😉 # tokenization
    words = re.sub(r'\w{15,}','',words).split()
    return [wnl.lemmatize(word) for word in words if word not in stopwords]


def readme_length(df):
    '''Creates a column with the length (by # of characters) of the readme_contents column'''
    l = []
    for val in df.readme_contents.index:
        l.append(
            {
                'length': len(df.iloc[val].readme_contents),
            }
                )
    l = pd.DataFrame(l)
    df = df.join(l, how='right')
    return df

def clean_readme(df):
    '''Creates a new column that encodes, decodes, lemmatizes and removes 
        words greater than 15 characters and puts them into a list'''
    df['clean'] = df.readme_contents.apply(clean)
    return df


def count(df):
    '''Plots a barplot of the distributions of character lengths and word counts within the repository readmes'''
    plt.figure(figsize=(16,9))
    plt.bar(range(len(df.word_count)), sorted(df.word_count), color='black')
    plt.bar(range(len(df.length)), sorted(df.length), color='lightsteelblue', alpha=0.5)
    plt.ylabel('Count')
    plt.xlim(0,100)
    plt.grid()
    plt.title('Distribution of Character Length and word Count per Repository')
    plt.legend(['Word Count', 'Length'])
    plt.show()


def word_count(df):
    '''Creates a column with number of words in the clean column'''
    w = []
    for val in df.readme_contents.index:
        w.append(
            {
                'word_count': len(df.clean.iloc[val]),
            }
                )
    w = pd.DataFrame(w)
    df = df.join(w, how='right')
    return df


def combined_words(df):
    '''Joins the list of words in the clean column to one string'''
    df['words'] = df.clean.apply(' '.join)
    return df


def train_count(df, X_train):
    '''
    Plots a barplot of the distributions of character lengths and word counts for the Train repository readmes
    '''
    plt.figure(figsize=(16,9))
    plt.bar(range(len(df.word_count.iloc[X_train.index.values])), sorted(df.word_count.iloc[X_train.index.values]), color='black')
    plt.bar(range(len(df.length.iloc[X_train.index.values])), sorted(df.length.iloc[X_train.index.values]), color='lightsteelblue', alpha=0.5)
    plt.ylabel('Count')
    plt.xlim(0,80)
    plt.grid()
    plt.title('Distribution of Character Length and word Count per Repository')
    plt.legend(['Word Count', 'Length'])
    plt.show()


def word_split(df):
    '''
    Takes in a DataFrame
    Splits words by Scripting Language, rejoining them.
    Returns the words ready for use in all the other functions requiring the words broken.
    '''
    for lang in df.language.unique():
        if lang == "TypeScript":
            ts_words = df[df.language == lang].words
        elif lang == "resources":
            resource_words = df[df.language == lang].words
        elif lang == "Python":
            py_words = df[df.language == lang].words
        elif lang == "JavaScript":
            js_words = df[df.language == lang].words
        elif lang == "C++":
            cplus2_words = df[df.language == lang].words
        elif lang == "Shell":
            shell_words = df[df.language == lang].words
        # elif lang == "Dart":
            # dart_words = df[df.language == lang].words
        elif lang == "C":
            c_words = df[df.language == lang].words
        elif lang == "Java":
            java_words = df[df.language == lang].words
        elif lang == "Markdown":
            md_words = df[df.language == lang].words
        elif lang == "Go":
            go_words = df[df.language == lang].words
        elif lang == "Rust":
            rust_words = df[df.language == lang].words
        elif lang == "C#":
            csharp_words = df[df.language == lang].words
        elif lang == "Vue":
            vue_words = df[df.language == lang].words
        elif lang == "Vim Script":
            vim_words = df[df.language == lang].words
        elif lang == "PHP":
            php_words = df[df.language == lang].words
        elif lang == "Clojure":
            clojure_words = df[df.language == lang].words
        # elif lang == "HTML":
        #     html_words = df[df.language == lang].words
    ts_words = ' '.join(ts_words)
    resource_words = ' '.join(resource_words)
    py_words = ' '.join(py_words)
    js_words = ' '.join(js_words)
    cplus2_words = ' '.join(cplus2_words)
    shell_words = ' '.join(shell_words)
    # dart_words = ' '.join(dart_words)
    c_words = ' '.join(c_words)
    java_words = ' '.join(java_words)
    md_words = ' '.join(md_words)
    go_words = ' '.join(go_words)
    rust_words = ' '.join(rust_words)
    csharp_words = ' '.join(csharp_words)
    vue_words = ' '.join(vue_words)
    vim_words = ' '.join(vim_words)
    php_words = ' '.join(php_words)
    clojure_words = ' '.join(clojure_words)
    # html_words = ' '.join(html_words)
    all_words = df.words
    all_words = ' '.join(all_words)
    return ts_words, resource_words, py_words, js_words, cplus2_words, shell_words, c_words, java_words, md_words, go_words, rust_words, csharp_words, vue_words, vim_words, php_words, clojure_words, all_words

def word_freqs(df):
    '''
    Takes in a DataFrame
    Runs through the scripting languages, sorts the words by scripting language, joins the words, and creates a value count series then creates a DataFrame with columns for each language and counts for each word.
    Saves the Word Count DataFrame as a CSV.
    Prints the Language, Word Count for that Language, and Percentage of Total words of all Documents.
    Returns the Word Count DataFrame.
    '''
    ts_words, resource_words, py_words, js_words, cplus2_words, shell_words, c_words, java_words, md_words, go_words, rust_words, csharp_words, vue_words, vim_words, php_words, clojure_words, all_words = word_split(df)
    
    all_words_freq = pd.Series(all_words.split()).value_counts()
    ts_freq = pd.Series(ts_words.split()).value_counts()
    resource_freq = pd.Series(resource_words.split()).value_counts()
    py_freq = pd.Series(py_words.split()).value_counts()
    js_freq = pd.Series(js_words.split()).value_counts()
    cplus2_freq = pd.Series(cplus2_words.split()).value_counts()
    shell_freq = pd.Series(shell_words.split()).value_counts()
    # dart_freq = pd.Series(dart_words.split()).value_counts()
    c_freq = pd.Series(c_words.split()).value_counts()
    java_freq = pd.Series(java_words.split()).value_counts()
    md_freq = pd.Series(md_words.split()).value_counts()
    go_freq = pd.Series(go_words.split()).value_counts()
    rust_freq = pd.Series(rust_words.split()).value_counts()
    csharp_freq = pd.Series(csharp_words.split()).value_counts()
    vue_freq = pd.Series(vue_words.split()).value_counts()
    vim_freq = pd.Series(vim_words.split()).value_counts()
    php_freq = pd.Series(php_words.split()).value_counts()
    clojure_freq = pd.Series(clojure_words.split()).value_counts()
    # html_freq = pd.Series(html_words.split()).value_counts()
    word_counts = pd.concat([ts_freq, resource_freq, py_freq, js_freq, cplus2_freq, shell_freq, c_freq, java_freq, md_freq, go_freq, rust_freq, csharp_freq, vue_freq, vim_freq, php_freq, clojure_freq, all_words_freq],axis=1).fillna(0).astype(int)
    word_counts.columns = "typescript", "resource", "python", "javascript", "c++", "shell", "c", "java", "markdown", "go", "rust", "c#", "vue", "vim", "php", "clojure", "all"  
    word_counts.to_csv("word_counts.csv")
    for col in word_counts.columns:
        print(f'''{col} : 
        total word count = {word_counts[col].sum()} | percent of documents = {round(word_counts[col].sum() / word_counts['all'].sum(), 3)}''')

    return word_counts

def bigram_charts(df):
    '''
    Takes in a DataFrame,
    Runs through scripting languages, sorts into words, then turns those words in bigrams
    Creates a Plot of Each Language's Most used Words and Bigrams.
    '''
    ts_words, resource_words, py_words, js_words, cplus2_words, shell_words, c_words, java_words, md_words, go_words, rust_words, csharp_words, vue_words, vim_words, php_words, clojure_words, all_words = word_split(df)
    print("Normal Word Use by Language:")
    # pd.Series(ts_words.split()).value_counts().head(20).plot.barh()
    # plt.title("TypeScript Word Use")
    # plt.show()
    pd.Series(resource_words.split()).value_counts().head(20).plot.barh()
    plt.title("Resource Word Use")
    plt.show()
    pd.Series(py_words.split()).value_counts().head(20).plot.barh()
    plt.title("Python Word Use")
    plt.show()
    pd.Series(js_words.split()).value_counts().head(20).plot.barh()
    plt.title("JavaScript Word Use")
    plt.show()
    # pd.Series(cplus2_words.split()).value_counts().head(20).plot.barh()
    # plt.title("C++ Word Use")
    # plt.show()
    # pd.Series(shell_words.split()).value_counts().head(20).plot.barh()
    # plt.title("Shell Word Use")
    # plt.show()
    # pd.Series(dart_words.split()).value_counts().head(20).plot.barh()
    # plt.title("Dart Word Use")
    # plt.show()
    # pd.Series(c_words.split()).value_counts().head(20).plot.barh()
    # plt.title("C Word Use")
    # plt.show()
    # pd.Series(java_words.split()).value_counts().head(20).plot.barh()
    # plt.title("Java Word Use")
    # plt.show()
    # pd.Series(md_words.split()).value_counts().head(20).plot.barh()
    # plt.title("Markdown Word Use")
    # plt.show()
    # pd.Series(go_words.split()).value_counts().head(20).plot.barh()
    # plt.title("Go Word Use")
    # plt.show()
    # pd.Series(rust_words.split()).value_counts().head(20).plot.barh()
    # plt.title("Rust Word Use")
    # plt.show()
    # pd.Series(csharp_words.split()).value_counts().head(20).plot.barh()
    # plt.title("C# Word Use")
    # plt.show()
    # pd.Series(vue_words.split()).value_counts().head(20).plot.barh()
    # plt.title("Vue Word Use")
    # plt.show()
    # pd.Series(vim_words.split()).value_counts().head(20).plot.barh()
    # plt.title("Vim Word Use")
    # plt.show()
    # pd.Series(php_words.split()).value_counts().head(20).plot.barh()
    # plt.title("PHP Word Use")
    # plt.show()
    # pd.Series(clojure_words.split()).value_counts().head(20).plot.barh()
    # plt.title("Clojure Word Use")
    # plt.show()
    # pd.Series(html_words.split()).value_counts().head(20).plot.barh()
    # plt.title("HTML Word Use")
    # plt.show()
    pd.Series(all_words.split()).value_counts().head(20).plot.barh()
    plt.title("All Word Use")
    plt.show()
    print(" ")
    print("Bigram Word Use by Language")
    # pd.Series(nltk.bigrams(ts_words.split())).value_counts().head(20).plot.barh()
    # plt.title("TypeScript Bigram Word Use")
    # plt.show()
    pd.Series(nltk.bigrams(resource_words.split())).value_counts().head(20).plot.barh()
    plt.title("Resource Bigram Word Use")
    plt.show()
    pd.Series(nltk.bigrams(py_words.split())).value_counts().head(20).plot.barh()
    plt.title("Python Bigram Word Use")
    plt.show()
    pd.Series(nltk.bigrams(js_words.split())).value_counts().head(20).plot.barh()
    plt.title("JavaScript Bigram Word Use")
    plt.show()
    # pd.Series(nltk.bigrams(cplus2_words.split())).value_counts().head(20).plot.barh()
    # plt.title("C++ Bigram Word Use")
    # plt.show()
    # pd.Series(nltk.bigrams(shell_words.split())).value_counts().head(20).plot.barh()
    # plt.title("Shell Bigram Word Use")
    # plt.show()
    # pd.Series(nltk.bigrams(dart_words.split())).value_counts().head(20).plot.barh()
    # plt.title("Dart Bigram Word Use")
    # plt.show()
    # pd.Series(nltk.bigrams(c_words.split())).value_counts().head(20).plot.barh()
    # plt.title("C Bigram Word Use")
    # plt.show()
    # pd.Series(nltk.bigrams(java_words.split())).value_counts().head(20).plot.barh()
    # plt.title("Java Bigram Word Use")
    # plt.show()
    # pd.Series(nltk.bigrams(md_words.split())).value_counts().head(20).plot.barh()
    # plt.title("Markdown Bigram Word Use")
    # plt.show()
    # pd.Series(nltk.bigrams(go_words.split())).value_counts().head(20).plot.barh()
    # plt.title("Go Bigram Word Use")
    # plt.show()
    # pd.Series(nltk.bigrams(rust_words.split())).value_counts().head(20).plot.barh()
    # plt.title("Rust Bigram Word Use")
    # plt.show()
    # pd.Series(nltk.bigrams(csharp_words.split())).value_counts().head(20).plot.barh()
    # plt.title("C# Bigram Word Use")
    # plt.show()
    # pd.Series(nltk.bigrams(vue_words.split())).value_counts().head(20).plot.barh()
    # plt.title("Vue Bigram Word Use")
    # plt.show()
    # pd.Series(nltk.bigrams(vim_words.split())).value_counts().head(20).plot.barh()
    # plt.title("Vim Bigram Word Use")
    # plt.show()
    # pd.Series(nltk.bigrams(php_words.split())).value_counts().head(20).plot.barh()
    # plt.title("PHP Bigram Word Use")
    # plt.show()
    # pd.Series(nltk.bigrams(clojure_words.split())).value_counts().head(20).plot.barh()
    # plt.title("Clojure Bigram Word Use")
    plt.show()
    # pd.Series(nltk.bigrams(html_words.split())).value_counts().head(20).plot.barh()
    # plt.title("HTML Bigram Word Use")
    # plt.show()
    pd.Series(nltk.bigrams(all_words.split())).value_counts().head(20).plot.barh()
    plt.title("All Bigram Word Use")
    plt.show()


def wordcloud_img(df):
    '''
    Takes in a DataFrame,
    Runs through scripting languages, sorts into words, then turns those words in bigrams
    Creates a Word Cloud Image of Each Language's Most used Bigrams as well as the Most Used words and bigrams among all languages.
    '''
    ts_words, resource_words, py_words, js_words, cplus2_words, shell_words, c_words, java_words, md_words, go_words, rust_words, csharp_words, vue_words, vim_words, php_words, clojure_words, all_words = word_split(df)

    print("Word Clouds of Most Used Word by Language")
    # plt.figure(figsize=(14, 7))
    # img = WordCloud(background_color='white').generate(' '.join(pd.Series(ts_words.split())))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title("TypeScript Word Use")
    # plt.show()
    plt.figure(figsize=(14, 7))
    img = WordCloud(background_color='white').generate(' '.join(pd.Series(resource_words.split())))
    plt.imshow(img)
    plt.axis('off')
    plt.title("Resource Word Use")
    plt.show()
    plt.figure(figsize=(14, 7))
    img = WordCloud(background_color='white').generate(' '.join(pd.Series(py_words.split())))
    plt.imshow(img)
    plt.axis('off')
    plt.title("Python Word Use")
    plt.show()
    plt.figure(figsize=(14, 7))
    img = WordCloud(background_color='white').generate(' '.join(pd.Series(js_words.split())))
    plt.imshow(img)
    plt.axis('off')
    plt.title("JavaScript Word Use")
    plt.show()
    # plt.figure(figsize=(14, 7))
    # img = WordCloud(background_color='white').generate(' '.join(pd.Series(cplus2_words.split())))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title("C++ Word Use")
    # plt.show()
    # plt.figure(figsize=(14, 7))
    # img = WordCloud(background_color='white').generate(' '.join(pd.Series(shell_words.split())))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title("Shell Word Use")
    # plt.show()
    # plt.figure(figsize=(14, 7))
    # img = WordCloud(background_color='white').generate(' '.join(pd.Series(dart_words.split())))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title("Dart Word Use")
    # plt.show()
    # plt.figure(figsize=(14, 7))
    # img = WordCloud(background_color='white').generate(' '.join(pd.Series(c_words.split())))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title("C Word Use")
    # plt.show()
    # plt.figure(figsize=(14, 7))
    # img = WordCloud(background_color='white').generate(' '.join(pd.Series(java_words.split())))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title("Java Word Use")
    # plt.show()
    # plt.figure(figsize=(14, 7))
    # img = WordCloud(background_color='white').generate(' '.join(pd.Series(md_words.split())))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title("Markdown Word Use")
    # plt.show()
    # plt.figure(figsize=(14, 7))
    # img = WordCloud(background_color='white').generate(' '.join(pd.Series(go_words.split())))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title("Go Word Use")
    # plt.show()
    # plt.figure(figsize=(14, 7))
    # img = WordCloud(background_color='white').generate(' '.join(pd.Series(rust_words.split())))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title("Rust Word Use")
    # plt.show()
    # plt.figure(figsize=(14, 7))
    # img = WordCloud(background_color='white').generate(' '.join(pd.Series(csharp_words.split())))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title("C# Word Use")
    # plt.show()
    # plt.figure(figsize=(14, 7))
    # img = WordCloud(background_color='white').generate(' '.join(pd.Series(vue_words.split())))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title("Vue Word Use")
    # plt.show()
    # plt.figure(figsize=(14, 7))
    # img = WordCloud(background_color='white').generate(' '.join(pd.Series(vim_words.split())))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title("Vim Word Use")
    # plt.show()
    # plt.figure(figsize=(14, 7))
    # img = WordCloud(background_color='white').generate(' '.join(pd.Series(php_words.split())))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title("PHP Word Use")
    # plt.show()
    # plt.figure(figsize=(14, 7))
    # img = WordCloud(background_color='white').generate(' '.join(pd.Series(clojure_words.split())))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title("Clojure Word Use")
    # plt.show()
    # plt.figure(figsize=(14, 7))
    # img = WordCloud(background_color='white').generate(' '.join(pd.Series(html_words.split())))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title("HTML Word Use")
    # plt.show()
    plt.figure(figsize=(14, 7))
    img = WordCloud(background_color='white').generate(' '.join(pd.Series(all_words.split())))
    plt.imshow(img)
    plt.axis('off')
    plt.title("All Word Use")
    plt.show()

    print(" ")
    print("Word Clouds of Most Used Bigrams by Language")
    # plt.figure(figsize=(14, 7))
    # img = WordCloud(background_color='white').generate(' '.join(pd.Series(nltk.bigrams(ts_words.split())).apply('_'.join)))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title("TypeScript Word Use (Bigrams)")
    # plt.show()
    plt.figure(figsize=(14, 7))
    img = WordCloud(background_color='white').generate(' '.join(pd.Series(nltk.bigrams(resource_words.split())).apply('_'.join)))
    plt.imshow(img)
    plt.axis('off')
    plt.title("Resource Word Use (Bigrams)")
    plt.show()
    plt.figure(figsize=(14, 7))
    img = WordCloud(background_color='white').generate(' '.join(pd.Series(nltk.bigrams(py_words.split())).apply('_'.join)))
    plt.imshow(img)
    plt.axis('off')
    plt.title("Python Word Use (Bigrams)")
    plt.show()
    plt.figure(figsize=(14, 7))
    img = WordCloud(background_color='white').generate(' '.join(pd.Series(nltk.bigrams(js_words.split())).apply('_'.join)))
    plt.imshow(img)
    plt.axis('off')
    plt.title("JavaScript Word Use (Bigrams)")
    plt.show()
    # plt.figure(figsize=(14, 7))
    # img = WordCloud(background_color='white').generate(' '.join(pd.Series(nltk.bigrams(cplus2_words.split())).apply('_'.join)))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title("C++ Word Use (Bigrams)")
    # plt.show()
    # plt.figure(figsize=(14, 7))
    # img = WordCloud(background_color='white').generate(' '.join(pd.Series(nltk.bigrams(shell_words.split())).apply('_'.join)))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title("Shell Word Use (Bigrams)")
    # plt.show()
    # plt.figure(figsize=(14, 7))
    # img = WordCloud(background_color='white').generate(' '.join(pd.Series(nltk.bigrams(dart_words.split())).apply('_'.join)))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title("Dart Word Use (Bigrams)")
    # plt.show()
    # plt.figure(figsize=(14, 7))
    # img = WordCloud(background_color='white').generate(' '.join(pd.Series(nltk.bigrams(c_words.split())).apply('_'.join)))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title("C Word Use (Bigrams)")
    # plt.show()
    # plt.figure(figsize=(14, 7))
    # img = WordCloud(background_color='white').generate(' '.join(pd.Series(nltk.bigrams(java_words.split())).apply('_'.join)))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title("Java Word Use (Bigrams)")
    # plt.show()
    # plt.figure(figsize=(14, 7))
    # img = WordCloud(background_color='white').generate(' '.join(pd.Series(nltk.bigrams(md_words.split())).apply('_'.join)))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title("Markdown Word Use (Bigrams)")
    # plt.show()
    # plt.figure(figsize=(14, 7))
    # img = WordCloud(background_color='white').generate(' '.join(pd.Series(nltk.bigrams(go_words.split())).apply('_'.join)))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title("Go Word Use (Bigrams)")
    # plt.show()
    # plt.figure(figsize=(14, 7))
    # img = WordCloud(background_color='white').generate(' '.join(pd.Series(nltk.bigrams(rust_words.split())).apply('_'.join)))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title("Rust Word Use (Bigrams)")
    # plt.show()
    # plt.figure(figsize=(14, 7))
    # img = WordCloud(background_color='white').generate(' '.join(pd.Series(nltk.bigrams(csharp_words.split())).apply('_'.join)))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title("C# Word Use (Bigrams)")
    # plt.show()
    # plt.figure(figsize=(14, 7))
    # img = WordCloud(background_color='white').generate(' '.join(pd.Series(nltk.bigrams(vue_words.split())).apply('_'.join)))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title("Vue Word Use (Bigrams)")
    # plt.show()
    # plt.figure(figsize=(14, 7))
    # img = WordCloud(background_color='white').generate(' '.join(pd.Series(nltk.bigrams(vim_words.split())).apply('_'.join)))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title("Vim Word Use (Bigrams)")
    # plt.show()
    # plt.figure(figsize=(14, 7))
    # img = WordCloud(background_color='white').generate(' '.join(pd.Series(nltk.bigrams(php_words.split())).apply('_'.join)))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title("PHP Word Use (Bigrams)")
    # plt.show()
    # plt.figure(figsize=(14, 7))
    # img = WordCloud(background_color='white').generate(' '.join(pd.Series(nltk.bigrams(clojure_words.split())).apply('_'.join)))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title("Clojure Word Use (Bigrams)")
    # plt.show()
    # plt.figure(figsize=(14, 7))
    # img = WordCloud(background_color='white').generate(' '.join(pd.Series(nltk.bigrams(html_words.split())).apply('_'.join)))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title("HTML Word Use (Bigrams)")
    # plt.show()
    plt.figure(figsize=(14, 7))
    img = WordCloud(background_color='white').generate(' '.join(pd.Series(nltk.bigrams(all_words.split())).apply('_'.join)))
    plt.imshow(img)
    plt.axis('off')
    plt.title("All Word Use (Bigrams)")
    plt.show()


def split(df):
    '''
    Takes in a DataFrame.
    Creates X using the 'words' column, and y using the 'language' column.
    Splits X and y into X_train, X_test, y_train, y_test (test_size= 0.2 and random_state= 536).
    Returns X_train, X_test, y_train, y_test for use in the modeling functions.
    '''
    X = df.words
    y = df.language
    X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.2, random_state= 536)
    return X_train, X_test, y_train, y_test

def modeling_tree(df):
    X_train, X_test, y_train, y_test = split(df)
    cv = CountVectorizer()
    X_bow = cv.fit_transform(X_train)
    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(X_bow, y_train)
    
    print(f"Accuracy of Decision Tree (Using Basic words and max_depth 3) on Train: {tree.score(X_bow, y_train)}")
    print(f'''Feature Importance:
{pd.Series(dict(zip(cv.get_feature_names_out(), tree.feature_importances_))).sort_values().tail()}''')

def modeling_bi_tri_tree(df):
    '''
    Takes in a DataFrame.
    Uses split() to create X and y Trains and Tests.
    Uses CountVectorizer to create 1 - 3 ngrams for modelling fitting X_train on it, then runs it through a max_depth 5 Decision Tree.
    Prints the Accuracy of the Decision Tree on Train.
    '''
    X_train, X_test, y_train, y_test = split(df)
    cv = CountVectorizer(ngram_range=(1,3), analyzer='word')
    X_bow = cv.fit_transform(X_train)
    tree = DecisionTreeClassifier(max_depth=5)
    tree.fit(X_bow, y_train)
    print(f" Accuracy of Decision Tree (Using 1 - 3 ngrams and max_depth 5) on Train: {tree.score(X_bow, y_train)}")
    print(f'''Feature Importance:
{pd.Series(dict(zip(cv.get_feature_names_out(), tree.feature_importances_))).sort_values().tail()}''')


def modeling_rf(df):
    '''
    Takes in a DataFrame.
    Uses split() to create X and y Trains and Tests.
    Uses CountVectorizer to create 1 - 3 ngrams for modelling fitting X_train on it, then runs it through a max_depth 4 Random Forest.
    Prints the Accuracy of the Random Forest on Train and the Classification Report.
    '''
    X_train, X_test, y_train, y_test = split(df)
    cv = CountVectorizer(ngram_range=(1,3), analyzer='word')
    X_bow = cv.fit_transform(X_train)

    rf = RandomForestClassifier(bootstrap=True, 
                            class_weight=None, 
                            criterion='gini',
                            min_samples_leaf=3,
                            n_estimators=1000,
                            max_depth=4, 
                            random_state=536)
    rf.fit(X_bow, y_train)
    y_pred = rf.predict(X_bow)
    y_pred_proba = rf.predict_proba(X_bow)
    print('Accuracy of Random Forest classifier on training set: {:.2f}'
        .format(rf.score(X_bow, y_train)))
    print(classification_report(y_train, y_pred))


def tree_test(df):
    '''
    Takes in a DataFrame.
    Uses split() to create X and y Trains and Tests.
    Uses CountVectorizer to create 1 - 3 ngrams for modelling fitting X_test on it, then runs it through a max_depth 5 Decision Tree.
    Prints the Accuracy of the Decision Tree on Test.
    '''
    X_train, X_test, y_train, y_test = split(df)
    cv = CountVectorizer(ngram_range=(1,3), analyzer='word')
    X_bow = cv.fit_transform(X_train)
    tree = DecisionTreeClassifier(max_depth=5)
    tree.fit(X_bow, y_train)
    X_test_bow = cv.transform(X_test)
    print(f"Accuracy of Decision Tree on Test: {tree.score(X_test_bow, y_test)}")
    print(f'''Feature Importance:
{pd.Series(dict(zip(cv.get_feature_names_out(), tree.feature_importances_))).sort_values().tail()}''')