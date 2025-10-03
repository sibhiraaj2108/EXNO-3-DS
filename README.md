## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```

<img width="579" height="489" alt="image" src="https://github.com/user-attachments/assets/963d79d1-9127-4c12-b25d-91166bb7c777" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

<img width="245" height="227" alt="image" src="https://github.com/user-attachments/assets/b6a73c9d-2fc0-4ee3-b243-c792ae8bb71a" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```

<img width="529" height="435" alt="image" src="https://github.com/user-attachments/assets/b3c68e59-1658-441b-878b-5eaabe6a85ba" />

```
 le=LabelEncoder()
 dfc=df.copy()
 dfc['ord_2']=le.fit_transform(dfc['ord_2'])
 dfc
```

<img width="554" height="432" alt="image" src="https://github.com/user-attachments/assets/e8334c2d-3ff9-4883-9e0f-13f4d7581cc1" />

```
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

ohe = OneHotEncoder(sparse_output=False)  # use sparse_output instead of sparse
df2 = df.copy()
enc = pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]), 
                   columns=ohe.get_feature_names_out(["nom_0"]))
df2 = pd.concat([df2, enc], axis=1)
print(df2)
```

<img width="772" height="248" alt="image" src="https://github.com/user-attachments/assets/a7072870-cb2a-4f72-80fd-ce6e33f65e6f" />

```
pd.get_dummies(df2,columns=["nom_0"])
```

<img width="1100" height="429" alt="image" src="https://github.com/user-attachments/assets/cc2d1304-8dda-40c9-b5e1-7b4c5b29ad0d" />

```
pip install --upgrade category_encoders
```
```
 from category_encoders import BinaryEncoder
 df=pd.read_csv("data.csv")
```

<img width="652" height="430" alt="image" src="https://github.com/user-attachments/assets/cd213119-23a1-42d0-bfad-ed63b6647524" />

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```
<img width="667" height="440" alt="image" src="https://github.com/user-attachments/assets/09cfaa42-2dfd-42c8-a9e6-567ec5591303" />

```
dfb=pd.concat([df,nd],axis=1)
dfb
```

<img width="907" height="432" alt="image" src="https://github.com/user-attachments/assets/b6b056db-22d0-4570-a7b2-19f4bd272571" />

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```

<img width="757" height="429" alt="image" src="https://github.com/user-attachments/assets/598f55c1-9154-4daa-b9da-f364d626737e" />

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```

<img width="1071" height="503" alt="image" src="https://github.com/user-attachments/assets/6835793f-0612-414f-a869-4986661e9d05" />

```
 df.skew()
```

<img width="406" height="246" alt="image" src="https://github.com/user-attachments/assets/b1d77560-bb91-4b97-bc5a-f22e03b73a2a" />

```
np.log(df["Highly Positive Skew"])
```

<img width="424" height="546" alt="image" src="https://github.com/user-attachments/assets/bc642037-b5c8-4a4e-ac51-7769f2265061" />

```
np.reciprocal(df["Moderate Positive Skew"])
```

<img width="512" height="541" alt="image" src="https://github.com/user-attachments/assets/bb17700c-955c-4a1e-823d-73a1e00623e9" />

```
np.sqrt(df["Highly Positive Skew"])
```

<img width="504" height="544" alt="image" src="https://github.com/user-attachments/assets/58cc0c41-a012-4ba0-b94d-569199a21058" />

```
 np.square(df["Highly Positive Skew"])
```

<img width="427" height="542" alt="image" src="https://github.com/user-attachments/assets/ac1d50d0-2e7d-442e-b16f-0ac91ea749c7" />

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

<img width="1324" height="555" alt="image" src="https://github.com/user-attachments/assets/243571be-a087-4ea5-a937-3335cafba213" />

```
df.skew()
```

<img width="531" height="287" alt="image" src="https://github.com/user-attachments/assets/84e4d340-5841-45c0-aefb-c235029fe4d5" />

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
<img width="542" height="325" alt="image" src="https://github.com/user-attachments/assets/637bc713-3040-4f87-918b-74f4f0205b38" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate  Negative Skew"]])
df
```

<img width="1784" height="578" alt="image" src="https://github.com/user-attachments/assets/69b71bb9-75f3-4e8f-9b9d-b7b25084923c" />

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```


<img width="880" height="532" alt="image" src="https://github.com/user-attachments/assets/2dd135a7-41e0-410f-a45e-8191a5c449d0" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```


<img width="810" height="536" alt="image" src="https://github.com/user-attachments/assets/4095e180-b8e9-4d0c-a769-3bc2ce8150be" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```


<img width="819" height="528" alt="image" src="https://github.com/user-attachments/assets/83028284-006e-46d5-94e9-cb8c8b7cab8e" />

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```


<img width="789" height="538" alt="image" src="https://github.com/user-attachments/assets/11ff129c-98a8-42c4-8cbe-47f31d49c3c8" />

```
dt=pd.read_csv("titanic_dataset.csv")
dt
```

<img width="935" height="327" alt="image" src="https://github.com/user-attachments/assets/afb55c48-5086-4a73-bc17-6543504e8a97" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
 dt["Age_1"]=qt.fit_transform(dt[["Age"]])
 sm.qqplot(dt['Age'],line='45') 
 plt.show()
```


<img width="755" height="547" alt="image" src="https://github.com/user-attachments/assets/8b92f95e-b1c7-443e-8f71-a0c8d8a75f3e" />

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```


<img width="733" height="533" alt="image" src="https://github.com/user-attachments/assets/2c391942-308e-469e-b9da-e51029577e73" />

# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully
       
