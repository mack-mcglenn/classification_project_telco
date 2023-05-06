{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "098711c6",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split\n",
    "from scipy import stats\n",
    "import acquire_copy as acq\n",
    "import prepare_functions as prep\n",
    "import explore_functions as ex\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0442de38",
   "metadata": {},
   "source": [
    "# Churn Around, Bright Eyes: Creating a prediction model for churn in Telco Communications\n",
    "\n",
    "Goals:\n",
    "\n",
    "   - Identify potential drivers of churn\n",
    "   - Build a model to predict future rates of churn\n",
    "   - Make suggestions to combat the current churn projections based on my findings\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "528e1881",
   "metadata": {},
   "source": [
    "# ACQUIRE\n",
    "\n",
    "- This data was acquired from the telco.csv file via the CodeUp database in March 2023\n",
    "- Size of data before cleaning: 7043 rows, 48 columns\n",
    "- Each observation represents the data for one (1) telco customer\n",
    "- Each column represents the features available to all telco customers"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "176c8c2c",
   "metadata": {},
   "source": [
    "# PREPARE\n",
    "\n",
    "During my preparation, I used the following functions to clean my data. This cleaning process includes dropping null values, reassigning numerical value to binary variables, converting monetary values to floats, and splitting the data for train/test/validation.\n",
    "\n",
    "#### Functions called:\n",
    "\n",
    "- get_telco_data():\n",
    "\n",
    "This function will check whether or not there is a csv for the telco data saved locally. If no such csv exists locally, this function will pull the telco data from Codeup's MySQL database and return as a dataframe based on the credentials provided in the env.py file in use\n",
    "\n",
    "- prep_telco():\n",
    "\n",
    "This function presp data from the telco csv (acquired via the get_telco_data() function in acquire_copy and preps it for future use.\n",
    "\n",
    "- tvt_split():\n",
    "\n",
    "20% test, 80% train_validate, then of the 80% train_validate: 25% validate, 75% train. Final split will be 20/20/60\n",
    "\n",
    "#### Additional measures\n",
    "\n",
    "- Split monthly charges into 3 categories\n",
    "- Split churn into 2 categories\n",
    "- I opted to include customers who don't use the internet services or recieve technical support. Seeing the low churn rate for non-internet customers reaffirms my belief that internet service is a high-impact indicator of churn.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4fee365e",
   "metadata": {},
   "source": [
    "## EXPLORATION\n",
    "\n",
    "Questions that need answers:\n",
    "\n",
    "- Do total monthly charges contribute to increasing churn rates?\n",
    "\n",
    "- Do customers with families churn more frequently than customers without?\n",
    "\n",
    "- How does internet service play into churn?\n",
    "\n",
    "- Is tech support a significant factor in churn? \n",
    "\n",
    "- Can multiple lines indicate a likeliness to churn?"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "667ebe1d",
   "metadata": {},
   "source": [
    "## Question 1: Churn Down for What?\n",
    "## On average, are customers who churn paying a higher monthly bill than customers who stay?\n",
    "The first thing I'm interested in looking at is how the monthly charges of churned customers compares with the monthly charges of current customers. I believe that higher monthly costs may contribute to higher customer churn rates."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "50ba536f",
   "metadata": {},
   "source": [
    "### Hypotheses Testing\n",
    "\n",
    "Q1 Hypothesis: The average monthly charges for a churned customer are greater than the average monthly charges for a current customer. \n",
    "\n",
    "Q1 Null Hypothesis: The average monthly charges for a churned customer are less than or equal to the average monthly charges for a current customer. \n",
    "\n",
    "\n",
    "### Test Type: T-Test\n",
    "\n",
    "My hypotheses calls for a two-tailed t-test. The hypothesis asks us to consider a sample group from a larger overall group (2 sample) and posits that there will be a significant difference (*greater than*) between them.\n",
    "\n",
    "### Results\n",
    "\n",
    "1. p-value < a\n",
    "2. We can reject the null hypothesis, which suggests that customers who churned have a higher average monthly cost than customers who still subscribe.\n",
    " \n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8db4597a",
   "metadata": {},
   "source": [
    "## Question 2: Hotline Bling\n",
    "## Is there a relationship between how many lines a customer has on their phone plan and likeliness to churn?\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6cfb9c8e",
   "metadata": {},
   "source": [
    "### Hypothesis Testing\n",
    "Q2 Hypthesis: Customers who have multiple lines are more likely to churn than customers who do not have multiple lines.\n",
    "\n",
    "Q2 Null Hypothesis: Customers who have multiple lines are not more likely to churn than customers who do not have multiple lines\n",
    "\n",
    "### Test Type: T-Test\n",
    "\n",
    "- Variables measured: churn, multiple_lines\n",
    "- Values returned: t-value, p-value\n",
    "\n",
    "### Results\n",
    "- p-value < a\n",
    "- We can reject the null hypothesis, which suggests that customers who churned have a higher average monthly cost than customers who still subscribe."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e953211a",
   "metadata": {},
   "source": [
    "## Question 3: What are we? Churn vs Relationships\n",
    "\n",
    "Are customers with relationships to others less likely to churn than customers who do not have relationships? I believe that customers who share accounts and services amongst friends/family/significant others are less likely to churn because more people rely on those services.\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8091d78b",
   "metadata": {},
   "source": [
    "### Hypotheses Testing\n",
    "\n",
    "Q3 Hypothesis: Customers with dependents or significant others are less likely to churn than customers who are single \n",
    "\n",
    "Q3 Null Hypothesis: Customers with dependents or significant others are as no more or less likely to churn than customers who are single \n",
    "\n",
    "### Test Type: T-Test\n",
    "\n",
    "- Variables measured: partner, dependents, churn \n",
    "- Values returned: t-value, p-value\n",
    "\n",
    "### Results\n",
    "\n",
    "1. p-value < a\n",
    "2. We can reject the null hypothesis, which suggests that customers who churned have a higher average monthly cost than customers who still subscribe."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "45cf31d0",
   "metadata": {},
   "source": [
    "## Question 4: To Churn or not to Churn?\n",
    "## Do customers churn at a higher rate if they've experienced numerous technical issues within their tenure window?\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5805ef32",
   "metadata": {},
   "source": [
    "### Hypotheses Testing:\n",
    "\n",
    "Hypothesis: Customers who experience documented technical issues with their internet service are more likely to churn than customers who don't rely heavily on tech support.\n",
    "\n",
    "Null Hypothesis: Customers who experience documented technical issues with their internet service are less likely or just as likely to churn as customers who don't rely heavily on tech support.\n",
    "\n",
    "### Test Type: Chi-square\n",
    "- Variables measured: internet_service_type, churn, tenure, tech_support\n",
    "- Values returned: chi-sq, degrees of freedom, p-value, expected frequencies\n",
    "\n",
    "### Results:\n",
    "- p < a\n",
    "- We reject the null hypothesis that customers who experience documented technical issues with their internet service are less likely or just as likely to churn as customers who don't rely heavily on tech support.\n",
    "- Customers who churned reported more technical issues leading up to their departure for telco."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b97b2875",
   "metadata": {},
   "source": [
    "##  Question 5: Daily Dose of Fiber\n",
    "\n",
    "### What about customers with fiber optic? Is there a relationship between fiber optic use and stability (based on tech support) that may lead customers to churn?"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "343fc5d6",
   "metadata": {},
   "source": [
    "### Hypotheses Testing:\n",
    "\n",
    "Hypothesis: Customers who experience documented technical issues with their internet service are more likely to churn than customers who don't rely heavily on tech support.\n",
    "Null Hypothesis: Customers who experience documented technical issues with their internet service are less likely or just as likely to churn as customers who don't rely heavily on tech support.\n",
    "### Test Type: Chi-square\n",
    "Variables measured: internet_service_type, churn, tenure, tech_support\n",
    "Values returned: chi-sq, degrees of freedom, p-value, expected frequencies\n",
    "\n",
    "### Results\n",
    "It appears that customers who've churned also relied more heavily on tech support- especially amongst fiber optic internet users. This could suggest some frustrations amongst users with the reliability and accessability of telco's internet connectivity. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fef19382",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "ae4d7a8a",
   "metadata": {},
   "source": [
    "# MODELING\n",
    "\n"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "d7ad543b",
   "metadata": {},
   "source": [
    "## Model Prep\n",
    "\n",
    " #### Baseline\n",
    "- My baseline prediction for this dataset is that customers do not churn\n",
    "- Baseline accuracy: 73.47%\n",
    "\n",
    "#### Features Kept\n",
    "\n",
    "- churn                \n",
    "- tenure              \n",
    "- monthly_charges      \n",
    "- dependents           \n",
    "- partner\n",
    "- internet_service_type\n",
    "- multiple lines\n",
    "- tech_support \n",
    "\n",
    "## Models\n",
    "\n",
    "#### Model 1: DecisionTree\n",
    "- max_depth=5\n",
    "- Test Accuracy: 78.35%\n",
    "- Train Accuracy: 79.20%\n",
    "\n",
    "#### Model 2: Random Forest\n",
    "- max_depth= 5\n",
    "- min_sample_leaf= 3\n",
    "- Val Accuracy: 79.63%\n",
    "- Train Accuracy: 79.73%\n",
    "\n",
    "#### Model 3: KNeighbor\n",
    "- n_neighbors = 15\n",
    "- Test Accuracy: 79.49%\n",
    "- Train Accuracy: 80.24%\n",
    "\n",
    "#### Model 4: Decision Tree (Train/val)\n",
    "- max_depth= 5\n",
    "- Val Accuracy: 79.30%\n",
    "- Train Accuracy: 80.02%\n",
    "\n",
    "#### Model 5: Random Forest (Train/val)\n",
    "- max_depth= 5\n",
    "- min_sample_leaf= 6\n",
    "- Val Accuracy: 79.66%\n",
    "- Train Accuracy: 79.76%\n",
    "\n",
    "#### Model 6: KNeighbor (Train/val)\n",
    "- n_neighbors = 24\n",
    "- Val Accuracy: 78.77%\n",
    "- Train Accuracy: 80.07%\n",
    "\n",
    "\n",
    "#### Observations:\n",
    "- All of my models are performing at at least 78% on my train data.\n",
    "\n",
    "- I ran 2 Decision Tree models: 1 for train/test, 1 for train/val. On both sets, the max depth of 5 produced the best results\n",
    "\n",
    "- My random forest models also both used the same max_depths for my top performing models. My train/val sets produced slightly higher results and had 6 sample leaves rather than 3.\n",
    "\n",
    "- Both my KNN train/test and train/val models had the highest overall train accuracy scores\n",
    "\n",
    "- Model 5:Random Forest returned the highest validation accuracy score of my 3 train/val sets. It also had the least amount of variance in my train/val sets, as well as the highest recall score of 93%. I'll continue with this model.\n",
    "\n",
    "\n",
    " \n",
    "\n"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "e84bd59d",
   "metadata": {},
   "source": [
    "# IN SUMMATION\n",
    "\n",
    "## Conclusion\n",
    "\n",
    "Telco is only retaining about 73% of it's customer base as of today.\n",
    "The prediction model that I've created can detect which customers are likely to churn with nearly 80% accuracy based on the aforementioned features for modeling.\n",
    "\n",
    "## Takeaways\n",
    "\n",
    "Customers don't like change. Enter fiber optic. It's the new frontier in web connectivity, but we need to prioritize making sure that our customers are getting the tech support they need to switch over from DSL. Tech support reliance is a fair indicator of churn. We can see an uptick of reliance on tech support right before customers leave. Families of all kinds are less likely to churn. Month-to-month customers- which make up the majority of Telco's customer base- are much more likely to churn because of their contract type. \n",
    "\n",
    "## Suggestions\n",
    "\n",
    "- I would suggest targeting units with family plans/ multi-line packages. Add limited/short time technical support as a bonus feature. I think there's an interesting relationship between multiple lines and monthly charges, so my future exploration would look into that. I would also bring data in from competitors offering similar services across multiple lines to run a cost/benefits analysis.\n",
    "\n",
    "\n",
    "- Fiber optic is still fairly new on the market, so reliability may be work looking into. I would suggest adding location as a metric to be considered so that we can analyze whether or not there are trends of reliability issues in certain regions.\n",
    "\n",
    "\n",
    "- Fiber optic service satisfaction should be measured more closely and frequently. Sending a quick monthly survey to assess customer satisfaction could be beneficial in understanding this metric further.\n",
    "\n",
    "- Incentivise month-to-month customers to transition into year long contracts. Offering free tech support and/or a rate cap for their first year may make this doable.\n",
    "\n",
    "\n",
    "\n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  },
  "vscode": {
   "interpreter": {
    "hash": "71bfaea8ea00cb3be56244b8032df9c26926b69327c94985b8bc8641baaa839e"
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 
}
