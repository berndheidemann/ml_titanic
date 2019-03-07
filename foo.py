Jupyter Notebook
firstTry
Last Checkpoint: vor ein paar Sekunden
(autosaved)
Current Kernel Logo
Python 3 
File
Edit
View
Insert
Cell
Kernel
Widgets
Help

%load_ext autoreload
%autoreload 2
%matplotlib inline
 numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics
import pandas as pd
import os
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
import numpy as np
os.listdir("./")
['.idea',
 '.ipynb_checkpoints',
 'firstTry.ipynb',
 'foo.py',
 'gender_submission.csv',
 'test.csv',
 'train.csv',
 'venv']
df_raw = pd.read_csv("train.csv")
df_raw.head(5)
​
PassengerId	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked
0	1	0	3	Braund, Mr. Owen Harris	male	22.0	1	0	A/5 21171	7.2500	NaN	S
1	2	1	1	Cumings, Mrs. John Bradley (Florence Briggs Th...	female	38.0	1	0	PC 17599	71.2833	C85	C
2	3	1	3	Heikkinen, Miss. Laina	female	26.0	0	0	STON/O2. 3101282	7.9250	NaN	S
3	4	1	1	Futrelle, Mrs. Jacques Heath (Lily May Peel)	female	35.0	1	0	113803	53.1000	C123	S
4	5	0	3	Allen, Mr. William Henry	male	35.0	0	0	373450	8.0500	NaN	S
m = RandomForestRegressor(n_jobs=-1)
Survived
m.fit(df_raw.drop('Survived', axis=1), df_raw.Survived)
C:\Users\User\Envs\mnist\lib\site-packages\sklearn\ensemble\forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-20-98a7c6b79d7e> in <module>()
----> 1 m.fit(df_raw.drop('Survived', axis=1), df_raw.Survived)

C:\Users\User\Envs\mnist\lib\site-packages\sklearn\ensemble\forest.py in fit(self, X, y, sample_weight)
    248 
    249         # Validate or convert input data
--> 250         X = check_array(X, accept_sparse="csc", dtype=DTYPE)
    251         y = check_array(y, accept_sparse='csc', ensure_2d=False, dtype=None)
    252         if sample_weight is not None:

C:\Users\User\Envs\mnist\lib\site-packages\sklearn\utils\validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, warn_on_dtype, estimator)
    525             try:
    526                 warnings.simplefilter('error', ComplexWarning)
--> 527                 array = np.asarray(array, dtype=dtype, order=order)
    528             except ComplexWarning:
    529                 raise ValueError("Complex data not supported\n"

C:\Users\User\Envs\mnist\lib\site-packages\numpy\core\numeric.py in asarray(a, dtype, order)
    536 
    537     """
--> 538     return array(a, dtype, copy=False, order=order)
    539 
    540 

ValueError: could not convert string to float: 'Braund, Mr. Owen Harris'

The above code will result in an error. There was a value inside the dataset “Conventional”, and it did not know how to create a model using that String. We have to pass numbers to most machine learning models and certainly to random forests. So step 1 is to convert everything into numbers.

This dataset contains a mix of continuous and categorical variables.

continuous — numbers where the meaning is numeric such as price. categorical — either numbers where the meaning is not continuous like zip code or string such as “large”, “medium”, “small”

for n,c in df_raw.items():
        if is_string_dtype(c): df_raw[n] = c.astype('category').cat.as_ordered()
for n,c in df_raw.items():
        if is_string_dtype(c): df_raw[n] = c.astype('category').cat.as_ordered()
​
Change any columns of strings in a panda's dataframe to a column of categorical values. This applies the changes inplace.

df_raw.Ticket.cat.categories
Index(['110152', '110413', '110465', '110564', '110813', '111240', '111320',
       '111361', '111369', '111426',
       ...
       'STON/O2. 3101290', 'SW/PP 751', 'W./C. 14258', 'W./C. 14263',
       'W./C. 6607', 'W./C. 6608', 'W./C. 6609', 'W.E.P. 5734', 'W/C 14208',
       'WE/P 5735'],
      dtype='object', length=681)
df_raw.isnull().sum().sort_index()
Age            177
Cabin          687
Embarked         2
Fare             0
Name             0
Parch            0
PassengerId      0
Pclass           0
Sex              0
SibSp            0
Survived         0
Ticket           0
dtype: int64
Cells without a value

def proc_df(df, y_fld=None, skip_flds=None, ignore_flds=None, do_scale=False, na_dict=None,
            preproc_fn=None, max_n_cat=None, subset=None, mapper=None):
    if not ignore_flds: ignore_flds=[]
    if not skip_flds: skip_flds=[]
    if subset: df = get_sample(df,subset)
    else: df = df.copy()
    ignored_flds = df.loc[:, ignore_flds]
    df.drop(ignore_flds, axis=1, inplace=True)
    if preproc_fn: preproc_fn(df)
    if y_fld is None: y = None
    else:
        if not is_numeric_dtype(df[y_fld]): df[y_fld] = pd.Categorical(df[y_fld]).codes
        y = df[y_fld].values
        skip_flds += [y_fld]
    df.drop(skip_flds, axis=1, inplace=True)
​
    if na_dict is None: na_dict = {}
    else: na_dict = na_dict.copy()
    na_dict_initial = na_dict.copy()
    for n,c in df.items(): na_dict = fix_missing(df, c, n, na_dict)
    if len(na_dict_initial.keys()) > 0:
        df.drop([a + '_na' for a in list(set(na_dict.keys()) - set(na_dict_initial.keys()))], axis=1, inplace=True)
    if do_scale: mapper = scale_vars(df, mapper)
    for n,c in df.items(): numericalize(df, c, n, max_n_cat)
    df = pd.get_dummies(df, dummy_na=True)
    df = pd.concat([ignored_flds, df], axis=1)
    res = [df, y, na_dict]
    if do_scale: res = res + [mapper]
    return res
def fix_missing(df, col, name, na_dict):
    if is_numeric_dtype(col):
        if pd.isnull(col).sum() or (name in na_dict):
            df[name+'_na'] = pd.isnull(col)
            filler = na_dict[name] if name in na_dict else col.median()
            df[name] = col.fillna(filler)
            na_dict[name] = filler
    return na_dict
def fix_missing(df, col, name, na_dict):
    if is_numeric_dtype(col):
        if pd.isnull(col).sum() or (name in na_dict):
            df[name+'_na'] = pd.isnull(col)
            filler = na_dict[name] if name in na_dict else col.median()
            df[name] = col.fillna(filler)
            na_dict[name] = filler
    return na_dict
def numericalize(df, col, name, max_n_cat):
    if not is_numeric_dtype(col) and ( max_n_cat is None or len(col.cat.categories)>max_n_cat):
        df[name] = pd.Categorical(col).codes+1
df, y, nas = proc_df(df_raw, 'Survived')
df, y, nas = proc_df(df_raw, 'Survived')
5
df.head(5)
PassengerId	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked	Age_na
0	1	3	109	2	22.0	1	0	524	7.2500	0	3	False
1	2	1	191	1	38.0	1	0	597	71.2833	82	1	False
2	3	3	354	1	26.0	0	0	670	7.9250	0	3	False
3	4	1	273	1	35.0	1	0	50	53.1000	56	3	False
4	5	3	16	2	35.0	0	0	473	8.0500	0	3	False
m = RandomForestRegressor(n_jobs=-1)
m.fit(df, y)
m.score(df,y)
C:\Users\User\Envs\mnist\lib\site-packages\sklearn\ensemble\forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
0.9045685936151854
df_raw_test = pd.read_csv("test.csv")
5
df_raw_test.head(5)
PassengerId	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked
0	892	3	Kelly, Mr. James	male	34.5	0	0	330911	7.8292	NaN	Q
1	893	3	Wilkes, Mrs. James (Ellen Needs)	female	47.0	1	0	363272	7.0000	NaN	S
2	894	2	Myles, Mr. Thomas Francis	male	62.0	0	0	240276	9.6875	NaN	Q
3	895	3	Wirz, Mr. Albert	male	27.0	0	0	315154	8.6625	NaN	S
4	896	3	Hirvonen, Mrs. Alexander (Helga E Lindqvist)	female	22.0	1	1	3101298	12.2875	NaN	S
for n,c in df_raw.items():
        if is_string_dtype(c): df_raw[n] = c.astype('category').cat.as_ordered()
df_test, y_test, nas_test = proc_df(df_raw_test)
)
df_test.head(5)
PassengerId	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked	Age_na	Fare_na
0	892	3	207	2	34.5	0	0	153	7.8292	0	2	False	False
1	893	3	404	1	47.0	1	0	222	7.0000	0	3	False	False
2	894	2	270	2	62.0	0	0	74	9.6875	0	2	False	False
3	895	3	409	2	27.0	0	0	148	8.6625	0	3	False	False
4	896	3	179	1	22.0	1	1	139	12.2875	0	3	False	False
, axis=1
df_test=df_test.drop("Fare_na", axis=1)
Fare_na needs to be dropped because it doesnt exist in train
preds = m.predict(df_test)
submission=np.zeros((len(preds), 2), dtype=int)
for i in range(len(preds) ):
    submission[i][0] = df_test.iloc[i]["PassengerId"]
    submission[i][1] = preds[i]> 0.5 if 1 else 0 
submission
array([[ 892,    0],
       [ 893,    0],
       [ 894,    0],
       [ 895,    0],
       [ 896,    1],
       [ 897,    0],
       [ 898,    0],
       [ 899,    1],
       [ 900,    1],
       [ 901,    0],
       [ 902,    0],
       [ 903,    0],
       [ 904,    1],
       [ 905,    0],
       [ 906,    1],
       [ 907,    1],
       [ 908,    0],
       [ 909,    1],
       [ 910,    0],
       [ 911,    0],
       [ 912,    0],
       [ 913,    1],
       [ 914,    1],
       [ 915,    0],
       [ 916,    1],
       [ 917,    0],
       [ 918,    1],
       [ 919,    0],
       [ 920,    1],
       [ 921,    0],
       [ 922,    0],
       [ 923,    0],
       [ 924,    0],
       [ 925,    0],
       [ 926,    0],
       [ 927,    0],
       [ 928,    1],
       [ 929,    1],
       [ 930,    0],
       [ 931,    1],
       [ 932,    0],
       [ 933,    0],
       [ 934,    0],
       [ 935,    1],
       [ 936,    1],
       [ 937,    0],
       [ 938,    0],
       [ 939,    0],
       [ 940,    1],
       [ 941,    0],
       [ 942,    0],
       [ 943,    0],
       [ 944,    1],
       [ 945,    1],
       [ 946,    0],
       [ 947,    0],
       [ 948,    0],
       [ 949,    0],
       [ 950,    0],
       [ 951,    1],
       [ 952,    0],
       [ 953,    0],
       [ 954,    0],
       [ 955,    1],
       [ 956,    1],
       [ 957,    1],
       [ 958,    1],
       [ 959,    0],
       [ 960,    1],
       [ 961,    1],
       [ 962,    1],
       [ 963,    0],
       [ 964,    1],
       [ 965,    0],
       [ 966,    1],
       [ 967,    1],
       [ 968,    0],
       [ 969,    1],
       [ 970,    0],
       [ 971,    0],
       [ 972,    1],
       [ 973,    0],
       [ 974,    0],
       [ 975,    0],
       [ 976,    0],
       [ 977,    0],
       [ 978,    1],
       [ 979,    0],
       [ 980,    0],
       [ 981,    1],
       [ 982,    0],
       [ 983,    0],
       [ 984,    1],
       [ 985,    0],
       [ 986,    1],
       [ 987,    0],
       [ 988,    1],
       [ 989,    0],
       [ 990,    0],
       [ 991,    0],
       [ 992,    1],
       [ 993,    0],
       [ 994,    0],
       [ 995,    0],
       [ 996,    1],
       [ 997,    0],
       [ 998,    0],
       [ 999,    0],
       [1000,    0],
       [1001,    0],
       [1002,    0],
       [1003,    1],
       [1004,    1],
       [1005,    1],
       [1006,    1],
       [1007,    0],
       [1008,    0],
       [1009,    1],
       [1010,    1],
       [1011,    1],
       [1012,    1],
       [1013,    0],
       [1014,    1],
       [1015,    0],
       [1016,    0],
       [1017,    0],
       [1018,    0],
       [1019,    1],
       [1020,    0],
       [1021,    0],
       [1022,    0],
       [1023,    0],
       [1024,    0],
       [1025,    0],
       [1026,    0],
       [1027,    0],
       [1028,    0],
       [1029,    0],
       [1030,    0],
       [1031,    0],
       [1032,    0],
       [1033,    1],
       [1034,    0],
       [1035,    0],
       [1036,    0],
       [1037,    0],
       [1038,    1],
       [1039,    0],
       [1040,    1],
       [1041,    0],
       [1042,    1],
       [1043,    0],
       [1044,    0],
       [1045,    0],
       [1046,    0],
       [1047,    0],
       [1048,    1],
       [1049,    1],
       [1050,    1],
       [1051,    0],
       [1052,    1],
       [1053,    1],
       [1054,    1],
       [1055,    0],
       [1056,    0],
       [1057,    1],
       [1058,    0],
       [1059,    0],
       [1060,    1],
       [1061,    0],
       [1062,    0],
       [1063,    0],
       [1064,    0],
       [1065,    0],
       [1066,    0],
       [1067,    1],
       [1068,    1],
       [1069,    0],
       [1070,    1],
       [1071,    1],
       [1072,    0],
       [1073,    0],
       [1074,    1],
       [1075,    0],
       [1076,    1],
       [1077,    0],
       [1078,    1],
       [1079,    0],
       [1080,    0],
       [1081,    0],
       [1082,    0],
       [1083,    1],
       [1084,    0],
       [1085,    0],
       [1086,    1],
       [1087,    0],
       [1088,    1],
       [1089,    1],
       [1090,    0],
       [1091,    0],
       [1092,    0],
       [1093,    1],
       [1094,    0],
       [1095,    1],
       [1096,    0],
       [1097,    0],
       [1098,    0],
       [1099,    0],
       [1100,    1],
       [1101,    0],
       [1102,    0],
       [1103,    0],
       [1104,    0],
       [1105,    1],
       [1106,    0],
       [1107,    1],
       [1108,    1],
       [1109,    0],
       [1110,    1],
       [1111,    0],
       [1112,    1],
       [1113,    0],
       [1114,    1],
       [1115,    0],
       [1116,    1],
       [1117,    1],
       [1118,    0],
       [1119,    1],
       [1120,    0],
       [1121,    0],
       [1122,    0],
       [1123,    1],
       [1124,    0],
       [1125,    0],
       [1126,    0],
       [1127,    0],
       [1128,    0],
       [1129,    1],
       [1130,    1],
       [1131,    1],
       [1132,    1],
       [1133,    1],
       [1134,    1],
       [1135,    0],
       [1136,    0],
       [1137,    0],
       [1138,    1],
       [1139,    0],
       [1140,    1],
       [1141,    1],
       [1142,    1],
       [1143,    0],
       [1144,    1],
       [1145,    0],
       [1146,    0],
       [1147,    0],
       [1148,    0],
       [1149,    0],
       [1150,    1],
       [1151,    0],
       [1152,    0],
       [1153,    0],
       [1154,    1],
       [1155,    1],
       [1156,    0],
       [1157,    0],
       [1158,    1],
       [1159,    0],
       [1160,    0],
       [1161,    0],
       [1162,    1],
       [1163,    0],
       [1164,    1],
       [1165,    0],
       [1166,    0],
       [1167,    1],
       [1168,    0],
       [1169,    0],
       [1170,    0],
       [1171,    0],
       [1172,    1],
       [1173,    1],
       [1174,    0],
       [1175,    1],
       [1176,    1],
       [1177,    0],
       [1178,    0],
       [1179,    1],
       [1180,    0],
       [1181,    0],
       [1182,    0],
       [1183,    0],
       [1184,    0],
       [1185,    0],
       [1186,    0],
       [1187,    0],
       [1188,    1],
       [1189,    0],
       [1190,    0],
       [1191,    0],
       [1192,    0],
       [1193,    0],
       [1194,    0],
       [1195,    0],
       [1196,    1],
       [1197,    1],
       [1198,    1],
       [1199,    1],
       [1200,    0],
       [1201,    0],
       [1202,    0],
       [1203,    0],
       [1204,    0],
       [1205,    0],
       [1206,    1],
       [1207,    1],
       [1208,    1],
       [1209,    0],
       [1210,    0],
       [1211,    0],
       [1212,    0],
       [1213,    0],
       [1214,    0],
       [1215,    0],
       [1216,    1],
       [1217,    0],
       [1218,    1],
       [1219,    0],
       [1220,    0],
       [1221,    0],
       [1222,    1],
       [1223,    0],
       [1224,    0],
       [1225,    1],
       [1226,    0],
       [1227,    0],
       [1228,    0],
       [1229,    0],
       [1230,    0],
       [1231,    1],
       [1232,    0],
       [1233,    0],
       [1234,    0],
       [1235,    1],
       [1236,    0],
       [1237,    1],
       [1238,    0],
       [1239,    0],
       [1240,    0],
       [1241,    1],
       [1242,    1],
       [1243,    0],
       [1244,    0],
       [1245,    1],
       [1246,    1],
       [1247,    0],
       [1248,    1],
       [1249,    0],
       [1250,    0],
       [1251,    0],
       [1252,    0],
       [1253,    1],
       [1254,    1],
       [1255,    0],
       [1256,    1],
       [1257,    0],
       [1258,    0],
       [1259,    1],
       [1260,    1],
       [1261,    0],
       [1262,    0],
       [1263,    1],
       [1264,    0],
       [1265,    0],
       [1266,    1],
       [1267,    1],
       [1268,    1],
       [1269,    0],
       [1270,    0],
       [1271,    0],
       [1272,    0],
       [1273,    0],
       [1274,    0],
       [1275,    0],
       [1276,    0],
       [1277,    1],
       [1278,    0],
       [1279,    0],
       [1280,    0],
       [1281,    0],
       [1282,    0],
       [1283,    1],
       [1284,    0],
       [1285,    0],
       [1286,    0],
       [1287,    1],
       [1288,    0],
       [1289,    1],
       [1290,    0],
       [1291,    0],
       [1292,    1],
       [1293,    0],
       [1294,    1],
       [1295,    1],
       [1296,    1],
       [1297,    0],
       [1298,    0],
       [1299,    0],
       [1300,    1],
       [1301,    1],
       [1302,    0],
       [1303,    1],
       [1304,    1],
       [1305,    0],
       [1306,    1],
       [1307,    0],
       [1308,    0],
       [1309,    0]])
 ,
np.savetxt("submission.csv", submission.astype(int), fmt='%i', delimiter=",", header="PassengerId,Survived", comments='')
?np.savetxt
​
×
Drag and Drop
The image will be downloaded by Fatkun