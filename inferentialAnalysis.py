import numpy as np
import pandas as pd
import scipy

import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Extract the data. Return both the raw data and dataframe
def generateDataset(filename):
    data = pd.read_csv(filename)
    df = data[0:]
    df = df.dropna()
    return data, df

# Run a t-test
def runTTest(ivA, ivB, dv):
    ttest = scipy.stats.ttest_ind(ivA[dv], ivB[dv]) #runs an independent sample's t-test to compare condition A and condition B
    print(ttest)
    
# Run ANOVA - adjust weights of model to minimize the sum squared error
def runAnova(data, formula):
    model = ols(formula, data).fit()
    aov_table = sm.stats.anova_lm(model, typ = 2)
    print(aov_table)
    
# Run the analysis
rawData, df = generateDataset('simpsons_paradox.csv')

print("Does gender correlate with admissions?")
men = df[(df['Gender'] == 'Male')]
women = df[(df['Gender'] == 'Female')]
runTTest(men, women, "Admitted")

print("Does department correlate with admissions?")
simpleFormula = "Admitted ~ C(Department)"
runAnova(rawData, simpleFormula)

print("Do gender and department correlate with admissions?")
moreComplex = "Admitted ~ C(Department) + C(Gender)"
runAnova(rawData, moreComplex)
         
# Monday Problems - worked with Taylor and Marissa

#1a) Independent - Year, categorical
#   Dependent - GPA, continuous
#   Stat test - T-test

#1b) Independent - snowfall, continuous
#   Dependent - time, continuous
#   Stat test - generalized regression

#1c) Independent - season, categorical
#   Dependent - hikers, continuous
#   Stat test - t-test

#1d) Independent - home state, categorical
#   Dependent - degree level, categorical
#   Stat test - chi-squared test

# Problem 2:
