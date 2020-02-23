
import pandas as pd
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
"""
df = pd.read_csv('EmployeePerformance.csv')

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder() 
df['EmpNumber'] = lb.fit_transform(df['EmpNumber'])
df['Gender'] = lb.fit_transform(df['Gender'])
df['EducationBackground'] = lb.fit_transform(df['EducationBackground'])
df['MaritalStatus'] = lb.fit_transform(df['MaritalStatus'])
df['EmpDepartment'] = lb.fit_transform(df['EmpDepartment'])
df['EmpJobRole'] = lb.fit_transform(df['EmpJobRole'])
df['BusinessTravelFrequency'] = lb.fit_transform(df['BusinessTravelFrequency'])
df['OverTime'] = lb.fit_transform(df['OverTime'])
df['Attrition'] = lb.fit_transform(df['Attrition'])

print(df.head())

#Save your results
df.to_csv('EmployeePerformancee.csv')

"""
df = pd.read_csv('EmployeePerformancee.csv')
print(df.head(5))
X = df[['EmpNumber','Age','Gender','EducationBackground','MaritalStatus','EmpDepartment','EmpJobRole','BusinessTravelFrequency','DistanceFromHome','EmpEducationLevel','EmpEnvironmentSatisfaction','EmpHourlyRate','EmpJobInvolvement','EmpJobLevel','EmpJobSatisfaction','NumCompaniesWorked','OverTime','EmpLastSalaryHikePercent','EmpRelationshipSatisfaction','TotalWorkExperienceInYears','TrainingTimesLastYear','EmpWorkLifeBalance','ExperienceYearsAtThisCompany','ExperienceYearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager','PerformanceRating']].values
y = df['Attrition'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.25, random_state=111)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(max_depth=10,n_estimators=10)
rfc.fit(X_train,y_train)
y_pred=rfc.predict(X_test)

print(y_test)
print("****")
print(y_pred)
print("==accuracy==")
print('Accuracy: ',metrics.accuracy_score(y_test,y_pred))

plt.show(sns.countplot(x='EmpDepartment',data=df))

"""
myinput = { 'EmpNumber':[1111],
            'Age':[22],
            'Gender':[33],
            'EducationBackground':[44],
            'MaritalStatus':[12],
            'EmpDepartment':[55],
            'EmpJobRole':[66],
            'BusinessTravelFrequency':[77],
            'DistanceFromHome':[88],
            'EmpEducationLevel':[99],
            'EmpEnvironmentSatisfaction':[13],
            'EmpHourlyRate':[14],
            'EmpJobInvolvement':[100],
            'EmpJobLevel':[111],
            'EmpJobSatisfaction':[222],
            'NumCompaniesWorked':[333],
            'OverTime':[444],
            'EmpLastSalaryHikePercent':[555],
            'EmpRelationshipSatisfaction':[123],
            'TotalWorkExperienceInYears':[124],
            'TrainingTimesLastYear':[125],
            'YearsSinceLastPromotion':[126],
            'YearsWithCurrManager':[127],
            'EmpWorkLifeBalance':[666],
            'ExperienceYearsAtThisCompany':[777],
            'ExperienceYearsInCurrentRole':[888],
            'PerformanceRating':[999]
            }

myinput['EmpNumber'][0] = float(input('Enter Emp number'))
myinput['Age'][0] = float(input('Enter Age'))
myinput['EmpDepartment'][0] = float(input('Enter emp department'))
myinput['Gender'][0] = float(input('Enter Gender'))
myinput['EducationBackground'][0] = float(input('Enter EducationBackground'))
myinput['MaritalStatus'][0] = float(input('Enter MaritalStatus'))
myinput['EmpJobRole'][0] = float(input('Enter EmpJobRole'))                          
myinput['BusinessTravelFrequency'][0] = float(input('Enter BusinessTravelFrequency'))
myinput['DistanceFromHome'][0] = float(input('Enter DistanceFromHome'))                         
myinput['EmpEducationLevel'][0] = float(input('Enter EmpEducationLevel'))
myinput['EmpEnvironmentSatisfaction'][0] = float(input('Enter EmpEnvironmentSatisfaction'))
myinput['EmpHourlyRate'][0] = float(input('Enter EmpHourlyRate'))
myinput['EmpJobInvolvement'][0] = float(input('Enter EmpJobInvolvement'))
myinput['EmpJobLevel'][0] = float(input('Enter EmpJobLevel'))
myinput['EmpJobSatisfaction'][0] = float(input('Enter EmpJobSatisfaction'))
myinput['NumCompaniesWorked'][0] = float(input('Enter NumCompaniesWorked'))
myinput['OverTime'][0] = float(input("Enter OverTime"))
myinput['EmpLastSalaryHikePercent'][0]  = float(input('Enter EmpLastSalaryHikePercent')),
myinput['EmpRelationshipSatisfaction'][0] = float(input('Enter EmpRelationshipSatisfaction')),
myinput['TotalWorkExperienceInYears'][0] = float(input('Enter TotalWorkExperienceInYears'))
myinput['TrainingTimesLastYear'][0] = float(input('Enter TrainingTimesLastYear'))
myinput['YearsSinceLastPromotion'][0] = float(input('Enter YearsSinceLastPromotion'))
myinput['YearsWithCurrManager'][0] = float(input('enter YearsWithCurrManager'))
myinput['EmpWorkLifeBalance'][0] = float(input('Enter EmpWorkLifeBalance'))
myinput['ExperienceYearsAtThisCompany'][0] = float(input('Enter ExperienceYearsAtThisCompany'))
myinput['ExperienceYearsInCurrentRole'][0] = float(input('Enter ExperienceYearsInCurrentRole'))                          
myinput['PerformanceRating'][0] = float(input('enter performance rating'))


df2 = pd.DataFrame(myinput,columns=['EmpDepartment','PreformanceRating'])

y_pred = rfc.predict(df2)

print(df2)
"""

