
## Churn Around, Bright Eyes: Creating a Mrediction Model for Churn Detection in Telco Communications

### Goals:

    Identify potential drivers of churn
    Build a model to predict future rates of churn
    Make suggestions to combat the current churn projections based on my findings

### ACQUIRE

    This data was acquired from the telco.csv file via the CodeUp database in March 2023
    Size of data before cleaning: 7043 rows, 48 columns
    Each observation represents the data for one (1) telco customer
    Each column represents the features available to all telco customers

### PREPARE

During my preparation, I used the following functions to clean my data. This cleaning process includes dropping null values, reassigning numerical value to binary variables, converting monetary values to floats, and splitting the data for train/test/validation.
Functions called:

    get_telco_data():

This function will check whether or not there is a csv for the telco data saved locally. If no such csv exists locally, this function will pull the telco data from Codeup's MySQL database and return as a dataframe based on the credentials provided in the env.py file in use

    prep_telco():

This function presp data from the telco csv (acquired via the get_telco_data() function in acquire_copy and preps it for future use.

    tvt_split():

20% test, 80% train_validate, then of the 80% train_validate: 25% validate, 75% train. Final split will be 20/20/60
Additional measures

    Split monthly charges into 3 categories
    Split churn into 2 categories
    I opted to include customers who don't use the internet services or recieve technical support. Seeing the low churn rate for non-internet customers reaffirms my belief that internet service is a high-impact indicator of churn.

### EXPLORATION

Questions that need answers:

    Do total monthly charges contribute to increasing churn rates?

    Do customers with families churn more frequently than customers without?

    How does internet service play into churn?

    Is tech support a significant factor in churn?

    Can multiple lines indicate a likeliness to churn?

Question 1: Churn Down for What?
On average, are customers who churn paying a higher monthly bill than customers who stay?

The first thing I'm interested in looking at is how the monthly charges of churned customers compares with the monthly charges of current customers. I believe that higher monthly costs may contribute to higher customer churn rates.
Hypotheses Testing

Q1 Hypothesis: The average monthly charges for a churned customer are greater than the average monthly charges for a current customer.

Q1 Null Hypothesis: The average monthly charges for a churned customer are less than or equal to the average monthly charges for a current customer.
Test Type: T-Test

My hypotheses calls for a two-tailed t-test. The hypothesis asks us to consider a sample group from a larger overall group (2 sample) and posits that there will be a significant difference (greater than) between them.
Results

    p-value < a
    We can reject the null hypothesis, which suggests that customers who churned have a higher average monthly cost than customers who still subscribe.

Question 2: Hotline Bling
Is there a relationship between how many lines a customer has on their phone plan and likeliness to churn?
Hypothesis Testing

Q2 Hypthesis: Customers who have multiple lines are more likely to churn than customers who do not have multiple lines.

Q2 Null Hypothesis: Customers who have multiple lines are not more likely to churn than customers who do not have multiple lines
Test Type: T-Test

    Variables measured: churn, multiple_lines
    Values returned: t-value, p-value

Results

    p-value < a
    We can reject the null hypothesis, which suggests that customers who churned have a higher average monthly cost than customers who still subscribe.

Question 3: What are we? Churn vs Relationships

Are customers with relationships to others less likely to churn than customers who do not have relationships? I believe that customers who share accounts and services amongst friends/family/significant others are less likely to churn because more people rely on those services.
Hypotheses Testing

Q3 Hypothesis: Customers with dependents or significant others are less likely to churn than customers who are single

Q3 Null Hypothesis: Customers with dependents or significant others are as no more or less likely to churn than customers who are single
Test Type: T-Test

    Variables measured: partner, dependents, churn
    Values returned: t-value, p-value

Results

    p-value < a
    We can reject the null hypothesis, which suggests that customers who churned have a higher average monthly cost than customers who still subscribe.

Question 4: To Churn or not to Churn?
Do customers churn at a higher rate if they've experienced numerous technical issues within their tenure window?
Hypotheses Testing:

Hypothesis: Customers who experience documented technical issues with their internet service are more likely to churn than customers who don't rely heavily on tech support.

Null Hypothesis: Customers who experience documented technical issues with their internet service are less likely or just as likely to churn as customers who don't rely heavily on tech support.
Test Type: Chi-square

    Variables measured: internet_service_type, churn, tenure, tech_support
    Values returned: chi-sq, degrees of freedom, p-value, expected frequencies

Results:

    p < a
    We reject the null hypothesis that customers who experience documented technical issues with their internet service are less likely or just as likely to churn as customers who don't rely heavily on tech support.
    Customers who churned reported more technical issues leading up to their departure for telco.

Question 5: Daily Dose of Fiber
What about customers with fiber optic? Is there a relationship between fiber optic use and stability (based on tech support) that may lead customers to churn?
Hypotheses Testing:

Hypothesis: Customers who experience documented technical issues with their internet service are more likely to churn than customers who don't rely heavily on tech support. Null Hypothesis: Customers who experience documented technical issues with their internet service are less likely or just as likely to churn as customers who don't rely heavily on tech support.
Test Type: Chi-square

Variables measured: internet_service_type, churn, tenure, tech_support Values returned: chi-sq, degrees of freedom, p-value, expected frequencies
Results

It appears that customers who've churned also relied more heavily on tech support- especially amongst fiber optic internet users. This could suggest some frustrations amongst users with the reliability and accessability of telco's internet connectivity.

 

### MODELING
Model Prep
Baseline

    My baseline prediction for this dataset is that customers do not churn
    Baseline accuracy: 73.47%

Features Kept

    churn
    tenure
    monthly_charges
    dependents
    partner
    internet_service_type
    multiple lines
    tech_support

Models
Model 1: DecisionTree

    max_depth=5
    Test Accuracy: 78.35%
    Train Accuracy: 79.20%

Model 2: Random Forest

    max_depth= 5
    min_sample_leaf= 3
    Val Accuracy: 79.63%
    Train Accuracy: 79.73%

Model 3: KNeighbor

    n_neighbors = 15
    Test Accuracy: 79.49%
    Train Accuracy: 80.24%

Model 4: Decision Tree (Train/val)

    max_depth= 5
    Val Accuracy: 79.30%
    Train Accuracy: 80.02%

Model 5: Random Forest (Train/val)

    max_depth= 5
    min_sample_leaf= 6
    Val Accuracy: 79.66%
    Train Accuracy: 79.76%

Model 6: KNeighbor (Train/val)

    n_neighbors = 24
    Val Accuracy: 78.77%
    Train Accuracy: 80.07%

Observations:

    All of my models are performing at at least 78% on my train data.

    I ran 2 Decision Tree models: 1 for train/test, 1 for train/val. On both sets, the max depth of 5 produced the best results

    My random forest models also both used the same max_depths for my top performing models. My train/val sets produced slightly higher results and had 6 sample leaves rather than 3.

    Both my KNN train/test and train/val models had the highest overall train accuracy scores

    Model 5:Random Forest returned the highest validation accuracy score of my 3 train/val sets. It also had the least amount of variance in my train/val sets, as well as the highest recall score of 93%. I'll continue with this model.

### IN SUMMATION

Telco is only retaining about 73% of it's customer base as of today. The prediction model that I've created can detect which customers are likely to churn with nearly 80% accuracy based on the aforementioned features for modeling.
Takeaways

Customers don't like change. Enter fiber optic. It's the new frontier in web connectivity, but we need to prioritize making sure that our customers are getting the tech support they need to switch over from DSL. Tech support reliance is a fair indicator of churn. We can see an uptick of reliance on tech support right before customers leave. Families of all kinds are less likely to churn. Month-to-month customers- which make up the majority of Telco's customer base- are much more likely to churn because of their contract type.
Suggestions

    I would suggest targeting units with family plans/ multi-line packages. Add limited/short time technical support as a bonus feature. I think there's an interesting relationship between multiple lines and monthly charges, so my future exploration would look into that. I would also bring data in from competitors offering similar services across multiple lines to run a cost/benefits analysis.

    Fiber optic is still fairly new on the market, so reliability may be work looking into. I would suggest adding location as a metric to be considered so that we can analyze whether or not there are trends of reliability issues in certain regions.

    Fiber optic service satisfaction should be measured more closely and frequently. Sending a quick monthly survey to assess customer satisfaction could be beneficial in understanding this metric further.

    Incentivise month-to-month customers to transition into year long contracts. Offering free tech support and/or a rate cap for their first year may make this doable.

