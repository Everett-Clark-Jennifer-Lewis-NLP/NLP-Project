# NLP Project

by: Jennifer Lewis and Everett Clark

<p>
  <a href="https://github.com/JenniferMLewis" target="_blank">
    <img alt="Jennifer" src="https://img.shields.io/github/followers/JenniferMLewis?label=Follow_Jennifer&style=social" />
  </a>
  <a href="https://github.com/etclark3" target="_blank">
    <img alt="Everett" src="https://img.shields.io/github/followers/etclark3?label=Follow_Everett&style=social" />
  </a>
</p>

The Google Slides can be found [here](https://docs.google.com/presentation/d/1ocxJPW4Q79RHn_NCmicGK0cq04c-JrwR7pw3hyGUMH0/edit?usp=sharing) 

[[Project Description](#project_description)]
[[Project Planning](#planning)]
[[Key Findings](#findings)]
[[Data Dictionary](#dictionary)]
[[Data Acquire and Prep](#wrangle)]
[[Data Exploration](#explore)]
[[Modeling](#modeling)]
[[Conclusion](#conclusion)]
___
## <a name="project_description"></a>Project Description:

### We acquired 100 READMEs from [Github's Most Starred Repos](https://github.com/search?l=%3Fspoken_language_code%3Den&p=11&q=stars%3A%3E10000&ref=advsearch&type=Repositories) (10/17/22), encoded and decoded the text, and created n-grams to predict the programming language used within the repo.

### Baseline prediction, predicting Javascript, was 25%. Our best model predicted the programming language at 60%.

***
## <a name="planning"></a>Project Planning: 

![Planning Pipeline](https://user-images.githubusercontent.com/98612085/196799270-7deb0ff8-078b-4b4b-9b9f-c96972429204.png)

        
### Hypothesis: 
### "Resources" would have longer READMEs than other languages


### Target variable: 
### Programming Language used

***
## <a name="findings"></a>Key Findings:
### 1. The best model performance was the ---
### 2. Modeling with Bi-grams and Tri-Grams increased model efficiency

***
## <a name="dictionary"></a>Data Dictionary  
<img width="505" alt="image" src="https://user-images.githubusercontent.com/98612085/196540690-c960f8aa-0907-4295-aa52-5e426709ea77.png">

***

## <a name="wrangle"></a>Acquisition and Preparation
After acquiring the repos we needed from Github's most-starred, we ran acquire.py and generated json data, which was then converted into a Dataframe.

<img width="575" alt="Screen Shot 2022-10-19 at 3 44 17 PM" src="https://user-images.githubusercontent.com/98612085/196800238-ca353653-5d11-467d-b883-04b3b8408dc4.png">

### Wrangle steps: 
Using a variety of functions from functions.py, we created four additional columns, resulting in:

<img width="903" alt="Screen Shot 2022-10-18 at 4 30 07 PM" src="https://user-images.githubusercontent.com/98612085/196800313-81acc5c6-56ab-4396-b137-40f97afed3b4.png">

*********************
## <a name="explore"></a>Data Exploration:
- Python files used for exploration:
    - functions.py

### Takeaways from Exploration:

***

## <a name="model"></a>Modeling:

## Modeling:

### Baseline
- Baseline Results: 25% (Using Javascript as the most common occurrence
    
    
### Model 1: ---
- Model 1 results:


### Model 2 : ---
- Model 2 results:


### Model 3 : ---
- Model 3 results:


## Selecting the Best Model:

| Model | Description| Validation/Out of Sample Accuracy |
| ---- | ----| ----|
| Baseline | JS as the most common prediction| 25% |
| Decision Tree** | using only single words| 46% |
| Decision Tree | using single words, bi- and tri- grams| 62% |
| Random Forest | using bi- and tri- grams| 53% |  

- The first model (Decision Tree**) had the best performance on test at 60%


## Testing the Model

- Model Testing Results: 

***

## <a name="conclusion"></a>Conclusion:

## Recommendations:
### 1. Reduce language variation within the corpus to improve accuracy
### 2. Narrow down words more specifically within the corpus
### 3. Further exploration of word combinations and their relationship to a programming language
[[Back to top](#top)]
