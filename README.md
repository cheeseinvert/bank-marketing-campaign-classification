# Bank Marketing Campaign Classifier Comparison
A machine learning analysis to predict telemarketing campaign success and improve efficiency using the CRISP-DM framework.

**Jupyter Notebook:** https://github.com/cheeseinvert/bank-marketing-campaign-classification/blob/main/prompt_III.ipynb

## What is the problem? 
A Portuguese banking institution conducts direct marketing campaigns via telephone to sell long-term deposit products. With only an 11.3% success rate, 88.7% of calls do not result in subscriptions, creating inefficiencies:
- **High operational costs** - agents spend time on low-probability leads
- **Customer dissatisfaction** - repeated unwanted calls damage bank relationships  
- **Inefficient resource allocation** - no systematic way to prioritize which clients to contact

My goal is to build and compare multiple classification models (K-Nearest Neighbors, Logistic Regression, Support Vector Machines, and Decision Trees) that predict whether a client will subscribe to a term deposit. As a result of my analysis, I will provide clear recommendations to the bank on how to optimize their telemarketing campaigns to reduce costs while maintaining conversion rates.

## What is the data?
The dataset contains information on 41,188 phone contacts from 17 marketing campaigns conducted between May 2008 and November 2010. After cleaning (removing 'unknown' values and the imbalanced 'default' column), the dataset contains the following features:

| Feature Category | Features | Description |
| --- | --- | --- |
| **Client Demographics** | age, job, marital, education | Personal information about the client |
| **Financial Products** | housing, loan | Whether client has housing loan or personal loan |
| **Campaign Contact** | month, day_of_week, duration, campaign, pdays, previous, poutcome | Details about current and previous campaign contacts |
| **Economic Indicators** | emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed | Quarterly and monthly economic context indicators |
| **Target Variable** | y | Binary outcome: did client subscribe to term deposit? (yes/no) |

**Key Data Characteristics:**
- **Highly imbalanced:** 88.7% "No" (36,548), 11.3% "Yes" (4,640)
- **Cleaned dataset:** Dropped 'default' column (only 3 positive cases) and 'contact' column (deemed not predictive)
- **Encoding strategy:** One-hot encoding for nominal categories, ordinal encoding for education/month/day_of_week, standardized numerical features

## What are the findings?

### Model Performance
After comparing four classification models with hyperparameter tuning via GridSearchCV (5-fold cross-validation), the results show:

| Model | Train Time (s) | Train Acc | Test Acc | Precision | Recall | F1-Score |
| --- | --- | --- | --- | --- | --- | --- |
| KNN (k=7) | 18.11 | 0.9238 | 0.9061 | 0.6218 | 0.4014 | 0.4879 |
| Logistic Regression (C=0.1) | 3.56 | 0.9093 | 0.9116 | 0.6686 | 0.4096 | 0.5080 |
| SVC (rbf kernel) | 170.09 | 0.9096 | 0.9070 | 0.6822 | 0.3099 | 0.4262 |
| **Decision Tree (depth=5)** | **1.79** | **0.9168** | **0.9161** | **0.6346** | **0.5810** | **0.6066*** |

***Best performing model based on F1-Score**

**Winner: Decision Tree**
- **Best F1-Score (0.6066)** - Optimal balance between precision and recall
- **Fastest training time (1.79s)** - Highly efficient for deployment
- **Highest recall (0.5810)** - Captures 58% of actual subscribers
- **Strong precision (0.6346)** - 63% of predicted subscribers are correct

**Why F1-Score?** With 88.7% class imbalance, accuracy is misleading (a model predicting all "No" would achieve 88.7% accuracy!). F1-Score balances precision (minimizing wasted calls) and recall (finding actual subscribers), making it ideal for this business context.

### Model Insights

**Decision Tree Advantages:**
- **Interpretability** - Easy to explain decision rules to non-technical stakeholders
- **Speed** - 95x faster than SVC, enables real-time scoring
- **Balance** - Best trade-off between finding subscribers (recall=58%) and avoiding false positives (precision=63%)

**Logistic Regression (Runner-up, F1=0.5080):**
- Higher precision (67%) but lower recall (41%)
- Better when minimizing wasted calls is priority
- Provides coefficient interpretation for feature importance

### Key Drivers of Subscription Success
Based on the Decision Tree model structure and feature importance:

1. **Call Duration** - Longest conversations are strongest predictor of success (consistently mentioned in research literature)
2. **Economic Indicators** - Euribor rate and employment variation significantly impact decisions  
3. **Campaign Timing** - Specific months show higher success rates (likely end-of-quarter effects)
4. **Previous Campaign History** - Number of previous contacts affects probability
5. **Client Demographics** - Age, job type, and education level influence decisions

### Business Impact
Using the Decision Tree model (F1=0.6066, Recall=0.5810), the bank can:

**Scenario Analysis:**
- **Current approach:** Contact all 10,000 clients → 1,130 subscriptions (11.3% rate)
- **Model approach:** Contact top 5,000 clients → ~656 subscriptions (58.1% of potential)
  
**Benefits:**
- **50% reduction in calls** (5,000 instead of 10,000)
- **Capture 58% of subscribers** with half the effort
- **Precision of 63%** means 1 in 1.6 calls succeeds (vs 1 in 9 currently)
- **5.6x improvement** in call efficiency (63%/11.3%)

## What do I recommend?

### Immediate Actions (Next Quarter)
1. **Deploy the Decision Tree model** to score all clients in the database
   - Priority 1: Top 30% (highest scores) - intensive outreach
   - Priority 2: Next 20% (medium scores) - standard contact
   - Priority 3: Bottom 50% (low scores) - skip or minimal contact

2. **Conduct A/B testing** to validate real-world performance
   - Group A: Model-selected top 50% clients  
   - Group B: Random 50% (control)
   - Measure: conversion rate, cost per acquisition, customer satisfaction

3. **Focus on call quality over quantity**
   - Train agents on engagement techniques (duration is key predictor)
   - Provide scripts that extend conversations naturally
   - Reward conversion rate, not call volume

4. **Optimize campaign timing**
   - Analyze which specific months/days have highest success in model
   - Schedule intensive campaigns during peak periods
   - Reduce activity during low-probability windows

### Strategic Recommendations

**Customer Segmentation Strategy:**
1. **High-value targets** (Model score >0.7):
   - Contact up to 3 times per campaign
   - Assign to most experienced agents
   - Offer best interest rates

2. **Medium-value targets** (Model score 0.4-0.7):
   - Contact once per campaign
   - Standard agent assignment
   - Standard offers

3. **Low-value targets** (Model score <0.4):
   - Skip phone contact
   - Consider email/SMS alternatives
   - Re-score quarterly for status changes

**Operational Changes:**
1. **Limit re-contact frequency** - Avoid contacting same client more than 3 times per year
2. **Monitor economic indicators** - Adjust campaign intensity based on Euribor rates
3. **Weekly model scoring** - Re-score database as new data becomes available
4. **Quarterly retraining** - Update model every 3 months to maintain accuracy

### Future Improvements

**Model Enhancements:**
1. **Feature engineering** - Create interaction terms (age × education, duration × poutcome)
2. **Collect additional data** - Client income, account balance, investment history
3. **Remove 'duration'** - Since duration isn't known before call, build production model without it

---

## Technical Details

**Model Hyperparameters:**
- Decision Tree: max_depth=5 (prevents overfitting)
- Logistic Regression: C=0.1 (strong regularization)
- KNN: n_neighbors=7
- SVC: rbf kernel

**Evaluation Approach:**
- 80/20 train-test split (stratified by target)
- 5-fold cross-validation for hyperparameter tuning
- Primary metric: F1-Score (handles class imbalance)
- GridSearchCV for systematic hyperparameter optimization

**Data Preprocessing:**
- Dropped 'default' column (only 3 positives) and 'contact' column
- Removed all 'unknown' values (10.7% data loss)
- One-hot encoding: job, marital, housing, loan, poutcome
- Ordinal encoding: education (7 levels), month (12 levels), day_of_week (5 levels)
- StandardScaler: all 10 numerical features
