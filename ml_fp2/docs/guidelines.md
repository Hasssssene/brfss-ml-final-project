# **Title**

Behavioral Risk Factor Surveillance System (BRFSS)  
---

# **1\. Context**

BRFSS is the nation’s premier system of health-related telephone surveys that collect state data about U.S. residents regarding their health-related risk behaviors, chronic health conditions, and use of preventive services. BRFSS collects data in all 50 states as well as the District of Columbia and three U.S. territories. BRFSS completes more than 400,000 adult interviews each year, making it the largest continuously conducted health survey system in the world. So this dataset can be used as multiple objectives. A classic one is to predict the presence of diabetes (DIABETE4), heart disease (CVDINFR4), or general health status (**GENHLTH**) based on hundreds of variables related to diet, exercise, smoking, demographics, and more.

To learn more about the data see the [official page](https://www.cdc.gov/brfss/annual_data/annual_2021.html).

Complete description about each column of the CSV file can be found in the code book ([official site](https://www.cdc.gov/brfss/annual_data/2021/pdf/codebook21_llcp-v2-508.pdf), a local copy in [UA server](https://www.dlsi.ua.es/~juanra/UA/datasets/LLCP2021_codebook_llcp-v2-508.pdf))

**Learning Objectives:**

* Manage and preprocess a complex, real-world clinical dataset with numerous features and missing values.  
* Frame a multi-class problem as a more tractable binary classification task.  
* Apply and compare a wide range of supervised classification models for tabular data.  
* Implement robust evaluation strategies, including cross-validation and statistical testing.  
* Develop skills in feature engineering, feature selection, and model tuning to maximize predictive performance.

---

# **2\. The Dataset**

The project uses the "Lifestyle and Health Classification Dataset", a rich collection of behavioral data.

* **Source:** You can download the dataset from a [UA server](https://www.dlsi.ua.es/~juanra/UA/datasets/LLCP2021.csv.tar.gz)  
* **Content:**  
  * **Instances:** \~430.000  
  * **Features:** \>300 (person features) explained in the codebook  
* **Target Variable (`GENHLTH`):** There is a **multiclass classification problem** with classes {1: Excellent, 2 Very good, 3: Good, 4: Fair, 5: Poor}.  
* **Important Notes:** This is a challenging dataset. You will need to handle:  
  * **Numerous Features:** Many are categorical and require you to revise the encoding.  
  * **Missing Values:** Many columns contain missing or unknown values, in categorical variables are coded with 7 or 9 (consult codebook).  
  * **Class Imbalance:** Frequency of categories in the target variable are imbalanced.  
  * **High correlation variables**: There are some variables directly related to the target. You need to identify and remove from the dataset before train models.

---

# **3\. Tasks & Technical Requirements**

1. **Implementation:** The project must be implemented in **Python**.  
2. **Libraries:** You will primarily use `scikit-learn`, `pandas`, and `numpy` but you can use others. Visualization libraries like `matplotlib` or `seaborn` are also highly recommended.  
3. **Environment:** You can use **Google Colab** for development, but any local or cloud-based Python IDE is acceptable.  
4. **Development:** The project can be completed **individually** or in **pairs**.  
5. **Report:** You must write a technical report (`report.pdf`) documenting your process, experiments, results, and conclusions.  
6. **Code:** Your code should be clean, well-commented, and reproducible.

---

# **4\. Submission & Deadlines**

* **Deliverables:**  
  1. A **technical report** in PDF format (`report.pdf`). The report must include, as its final section, a statement titled **“Ethics and Use of Generative AI”**, following the guidelines provided below.  
  2. A **Python code file**. Submit your code as a Python file (`.py`), such as `code.py`.

     **⚠️Important:** Only pure Python code files (`.py`) will be evaluated. If you submit the code as a Jupyter Notebook (`.ipynb`), it will **not** be evaluated. If you use Google Colab, you can use the appropriate menu option to download or export your notebook as a Python (`.py`) file.  
  3. When submitting your work through the form, you must provide, in the **General observations** field, the URL to a screencast video (no longer than 10 minutes), recorded individually or in pairs, presenting the development process and results of your final project. The screencast may be presented in either English or Spanish.  
       
* **Ethics and Use of Generative AI:** The final section of the technical report must clearly state whether the submitted work is entirely your own, whether you received assistance from another person, and whether you used any generative AI tool (e.g., ChatGPT, Microsoft Copilot, Google Gemini, Claude, or similar tools).  
    
  If you received assistance from another person, briefly describe the nature of that assistance and indicate which parts of the work were affected.  
    
  If you used a generative AI tool at any stage of the project, you must disclose its use and briefly describe:  
  1. the tool used (e.g., ChatGPT, Copilot, Gemini, etc.);  
  2. the purpose for which it was used (e.g., to understand a Machine Learning concept, generate or improve code, debug an error, assist with data analysis, or improve the wording of the report); and  
  3. how the generated output was used in the final work.

  For example:


  > **Use of Generative AI:** ChatGPT was used as a support tool to debug a cross-validation execution and to clarify the interpretation of certain evaluation metrics. All suggestions provided by the generative AI were personally reviewed and validated before being adapted and incorporated into the code. I assume responsibility for and authorship of this work.


  If no assistance from other people was received and no generative AI tools were used, explicitly state this. For example:


  > **Statement:** This work is entirely my own. I did not receive assistance from other people and did not use any generative AI tools in the preparation of this work.


  The use of generative AI or external assistance must not be concealed. Failure to provide an accurate statement regarding the use of generative AI or assistance from others may result in a grade of zero and may be subject to the applicable disciplinary regulations of the University of Alicante.

* **Method:** Submit the deliverables by completing [**this form**](https://docs.google.com/forms/d/e/1FAIpQLSdb7OUZ8iqyMVpYI1CNdbrCKDj0crp28NygUjL8Xmc8GggQYg/viewform). The form will also ask for a self-assessment of your grade and a brief justification.

* **Deadlines:**  
  * C2 Call (December): **November 8, 2026, by 11:59 PM**.  
  * C4 Call (July): **June 18, 2026, by 11:59 PM**.

---

# **5\. Evaluation Criteria**

The final grade is not based solely on achieving the highest possible metric. Significant weight is given to the quality of your experimentation, the justification for your choices, and the depth of your analysis. A project with a slightly lower result but an outstanding, well-reasoned report will receive a higher grade than a project with a high and a superficial or copy & paste IA generative report.

## **Basic Level (up to 5 points): Foundational Analysis**

* **Data Cleaning & Preprocessing:** Perform essential data wrangling: handle missing values, correct data types if it is necessary, and apply one-hot (no ordinal) or less label encoding (ordinal) to categorical features.  
* **Exploratory Data Analysis (EDA):** Analyze the data to understand its structure. This must include:  
  * Analysis of the target variable to quantify the class imbalance.  
  * A baseline `ROC-AUC` score from a `DummyClassifier`.  
* **Feature selection**: This dataset has high dimensionality, so we need to apply a basic feature selection algorithm to test classifiers in a reasonable time cost.   
* **Modeling:** Apply and analyze at least two different `scikit-learn` classifiers (e.g., Logistic Regression, Decision Tree, etc.).  
* **Parameter Tuning:** Manually adjust the hyperparameters of your models to improve the `ROC-AUC` score.  
* **Validation:** Use a stratified **10-fold cross-validation** for all experiments.  
* **Report:** A clear report explaining the preprocessing steps, EDA findings, models, and results.

## **Intermediate Level (up to 3 points): Deeper Experimentation**

* **Expanded Modeling:** Apply and compare a wider range of classifiers (at least 5 total), including ensemble methods like `RandomForestClassifier` or `GradientBoostingClassifier`.  
* **Imbalance Handling:** Use and compare at least one explicit technique to address class imbalance in the minority class, the majority class, or both.  
* **Feature Engineering & Selection:**  
  * Use dimensionality reduction techniques (e.g., PCA).  
  * Apply advanced feature selection algorithms to identify the most predictive variables.  
  * You must formulate a hypothesis about the group of variables you have selected. In your report, you should try to justify why this selection was made and compare its results with the model that uses all the variables.  
* **Statistical Comparison:** Use a hypothesis test (e.g., Wilcoxon signed-rank test) to compare your best-performing models (10 results for a stratified 10-fold cross-validation) and determine if the difference in their `ROC-AUC` scores is statistically significant.  
  **Note:** It is very important that the partitions are exactly the same in all cases to perform this statistical test coherently. This can be achieved by using the `random_state` parameter and setting it to a fixed integer value in all experiments.  
* **Error Analysis:** Identify at least 3 instances that your best model consistently misclassifies. Analyze their features and provide a well-reasoned hypothesis for why the model is failing on these specific cases. Include visualizations to support your analysis.  
* **XAI (eXplainable AI)**: Use explainability techniques in AI to estimate the most important features in the ensemble or in individual examples. Libraries such as LIME or SHAP can be used.   
* **Report:** A detailed report with summary tables and plots comparing all models and techniques used.

## **Advanced Level (up to 2 points): State-of-the-Art Techniques**

* **Automated Hyperparameter Optimization:** Use an automated search strategy (e.g., `GridSearchCV`, `RandomizedSearchCV` or others more advanced algorithms) to find the optimal parameters for your best models. For a top grade, research and apply **nested cross-validation** to get an *unbiased* estimate of performance.  
* **Advanced Models:** Use a high-performance gradient boosting library not in `scikit-learn`, or state-of-the art deep neural networks on tabular data and compare its performance.  
* **Advanced Problem Framing or Feature Engineering:** Demonstrate originality by either:  
  * Tackling the original **three-class classification problem** (evaluating with appropriate multi-class metrics like macro-averaged F1-score or AUC).  
  * Performing sophisticated feature engineering (e.g., creating interaction terms, binning numerical features based on clinical knowledge).  
* **Outstanding Report:** The final report must be exceptional: professionally structured, clearly written, and culminating in a comprehensive summary that justifies the final recommended model based on all experiments conducted.

