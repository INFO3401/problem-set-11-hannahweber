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
         
# Monday Problems - worked with Taylor and Marissa and Jacob (11.12)

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
# I was able to create the columns but I kept getting an error when I tried to use them in the correlation formulas. However, I did this question last and was able to play with the data enough and I don't think it is biased. There is definitely a significant difference between the male and female numbers across the board but I don't think that the school is biased to either men or women. Because there are way more men than women their numbers of acceptance and rejection are significantly higher but when looking at the percentages they typically are admitted less and rejected more. I think that the school could focus on increasing their recruitment of women but I don't think they are biased in their admissions. 

df["percentRejected"] = (df["Rejected"] / (df["Rejected"] + df["Admitted"]))
df["percentAdmitted"] = (df["Admitted"] / (df["Rejected"] + df["Admitted"]))

print("Do gender and department correlate with percentRejected?")
rejectFormula = "percentRejected ~ C(Department) + C(Gender)"
runAnova(rawData, rejectFormula)

print("Do gender and department correlate with percentAccepted?")
acceptFormula = "percentAccepted ~ C(Department) + C(Gender)"
runAnova(rawData, acceptFormula)

# Monday Problems - worked with Taylor, Marissa, Jack, Jacob (11.26)

# Problem 3:
# There was a text error where there was a space after the word "male" and so the code was only running on some of the male data, rather than all of the male data. To fix this issue you would either have to correct the csv or add the 'male ' to the code. 
# For the question "does department correlate with admission", the results stayed the same. The output of the T-Test's statistic and p-value changed drastically once the error was fixed. The gender and department correlation with admissions also experienced a big change with the data correction. It is important to find these issues so that the statistical analysis is accurate.

# Problem 4:
# The Simpson's Paradox is when a trend occurs in different groups of data but then disappears when looking at the data combined. This is seen with the admission and rejection data for each department. When combined there is a constant decrease in admission rates looking at the departments in consecutive order and the rejection rates vary a lot between departments. When you break these departments down into male and female groups you can see the drastic difference between the overall male to female rates and the admission and rejection trends. The male rates are the biggest influence on the overall trends and each department follows the same trends as the overall data. As for the female rates, both the admission and rejection trends vary throughout departments and don't follow a similar pattern as the male and overall trends. The two charts are included in the GitHub.