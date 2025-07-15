import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score,ShuffleSplit,GridSearchCV
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.tree import DecisionTreeRegressor
import pickle
import json

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

df = pd.read_csv('20_DuAn/Bengaluru_House_Data.csv')

# print(df.head(5))
# print(df.shape)
# print(df.columns)
# print(df['area_type'].unique())
# print(df['area_type'].value_counts())

df1 = df.drop(['area_type','availability','society','balcony'],axis='columns')

# print(df1.isnull().sum())

df2 = df1.dropna().copy()
# print(df2.isnull().sum())


# ---------------------Feature Engineering--------------
# Add new feature(integer) for bhk (Bedrooms Hall Kitchen)


#  trích xuất số phòng ngủ (BHK) : ví dụ 2 BHK ==> 2
df2['bhk'] = df2['size'].apply(lambda x: int(x.split(' ')[0]))
# print(df2.bhk.unique())


# Explore total_sqft feature
# để kiểm tra xem một giá trị x có thể chuyển thành float được hay không
def is_float(x):
    try: 
        float(x)
    except:
        return False
    return True

# in ra không phải số hợp lệ
# print(df2[~df2['total_sqft'].apply(is_float)].head(10))


# lấy giá trị trung bình nếu ví dụ(2100-2850,.....) và xóa các giá trị khác không như(24.46Sq,.....)
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        # trường hợp dạng '2100 - 2850'
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        # trường hợp bình thường '1200.0'
        return float(x)
    except:
        # trường hợp không thể chuyển
        return None
    
df3 =df2.copy()
df3['total_sqft'] = df3['total_sqft'].apply(convert_sqft_to_num)

# loại bỏ dòng NAN ở total_sqft
# df3 = df3[df3['total_sqft'].notnull()]
# print(df3.loc[30])



# ---------Add new feature called price per square feet-------
df4 = df3.copy()
# giá trên mỗi mét vuông(Nhân * 100000 để đổi giá từ lakh sang đơn vị INR.)
df4['price_per_sqft'] = df4['price']*100000/df4['total_sqft']
# print(df4.head(10))


#  một bảng |thống kê| mô tả (descriptive statistics) cho cột price_per_sqft
df4_stats = df4['price_per_sqft'].describe()
# df4.to_csv("20_DuAn/bhp.csv",index=False)
# print(df4_stats)


df4['location'] = df4['location'].apply(lambda x: x.strip())
location_stats = df4['location'].value_counts(ascending=False)
# print(location_stats)
# print(location_stats.values.sum())
# print(len(location_stats[location_stats<=10]))



# Gom nhóm location_stats<=10 sẽ giúp mô hình tổng quát hóa tốt hơn và tránh overfitting vào những địa điểm ít dữ liệu.
location_stats_less_than_10 = location_stats[location_stats<=10]
df4['location'] = df4['location'].apply(lambda x: 'other' if x in location_stats_less_than_10 else x ) 
# print(len(df4['location'].unique()))
# print(df4.head(10))




# ---------------Outlier Removal Using Business Logic

# print(df4[df4['total_sqft']/df4['bhk']<300].head(10))

# loại bỏ dưới 300 bằng toán tử ( ~ ) 
df5 = df4[~(df4['total_sqft']/df4['bhk']<300)]

# print(df5['price_per_sqft'].describe())
# Ở đây chúng ta thấy giá tối thiểu cho mỗi sqft là 267 rs/sqft trong khi giá tối đa là 12000000, điều này cho thấy sự thay đổi lớn về giá bất động sản. Chúng ta nên loại bỏ các giá trị ngoại lệ theo vị trí bằng cách sử dụng giá trị trung bình và một độ lệch chuẩn


# Tránh việc bị ảnh hưởng bởi các giá trị cực cao hoặc cực thấp của price_per_sqft trong từng khu vực riêng biệt.
def remove_pps_outliers(df_t):
    df_out = pd.DataFrame()
    for key,subdf in df_t.groupby('location'):
        m = np.mean(subdf['price_per_sqft'])
        st = np.std(subdf['price_per_sqft'])
        reduced_df = subdf[(subdf['price_per_sqft']>(m-st))&(subdf['price_per_sqft']<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out

df6 = remove_pps_outliers(df5)
# print(df6.shape)



# kiểm tra xem giá bất động sản 2 BHK và 3 BHK ở một vị trí nhất định trông như thế nào
def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    plt.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    
# plot_scatter_chart(df6,"Rajaji Nagar")
# plot_scatter_chart(df6,"Hebbal")
# plt.show()



# Chúng ta cũng nên loại bỏ các bất động sản có cùng vị trí, giá của (ví dụ) căn hộ 3 phòng ngủ thấp hơn căn hộ 2 phòng ngủ (có cùng diện tích ft vuông).
# loại bỏ những căn hộ BHK lớn hơn nhưng có price_per_sqft thấp hơn trung bình của BHK nhỏ hơn tại cùng một location
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df7 = remove_bhk_outliers(df6)
# df8 = df7.copy()
# print(df7.shape)


# Dựa trên biểu đồ trên, chúng ta có thể thấy rằng các điểm dữ liệu được đánh dấu màu đỏ bên dưới là các giá trị ngoại lệ và chúng đang bị xóa do hàm remove_bhk_outliers
# plot_scatter_chart(df7,"Hebbal")
# plot_scatter_chart(df6,"Rajaji Nagar")
# plt.show()


# vẽ biểu đồ histogram cho cột price_per_sqft
# plt.rcParams["figure.figsize"] = (20,10)
# plt.hist(df7.price_per_sqft,rwidth=0.8)
# plt.xlabel("Price Per Square Feet")
# plt.ylabel("Count")
# plt.show()



# plt.hist(df7.bath,rwidth=0.8)
# plt.xlabel("Number of bathrooms")
# plt.ylabel("Count")
# plt.show()




# thường mỗi phòng ngủ sẽ có 1 phòng tắm: ta sẽ lấy điều kiện (số phòng tắm < số phòng ngủ + 2) nếu số phòng ngủ nhiều hơn số phòng tắm 1 có thể chấp nhận được
df8 = df7[df7.bath<df7.bhk+2]
# print(df8.head())



# -------------------------------------TRAIN----------

df9 = df8.drop(['size','price_per_sqft'],axis='columns')

# Use One Hot Encoding For Location
dummies = pd.get_dummies(df9.location,dtype=int)
# print(dummies.head())


# mã hóa nóng sẽ xóa 1 cột tránh over fitting
df10 = pd.concat([df9,dummies.drop('other',axis='columns')],axis='columns')

df11 = df10.drop('location',axis='columns')

# print(df11.shape)

x = df11.drop(['price'],axis='columns')
y = df11.price




# -------------------------------------------------------------------

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=10)
lr = LinearRegression()
lr.fit(X_train,y_train)
# print(lr.score(X_test,y_test))



# Use K Fold cross validation to measure accuracy of our LinearRegression model
# ShuffleSplit: tạo ra 5 lần chia dữ liệu ngẫu nhiên (random splits)
cv = ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
# print(cross_val_score(LinearRegression(),x,y,cv=cv))




# find best model using GridSearchCV
# viết hàm tạo gridsearchcv

def find_best_model_using_gridsearchcv(x, y):
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                'fit_intercept': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['squared_error', 'friedman_mse'],  # ✅ Đổi 'mse' -> 'squared_error'
                'splitter': ['best', 'random']
            }
        }
    }

    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(x, y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])

# print(find_best_model_using_gridsearchcv(x,y))


# Kiểm tra mô hình cho một vài thuộc tính
def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(x.columns==location)[0][0]

    x_input = np.zeros(len(x.columns))
    x_input[0] = sqft
    x_input[1] = bath
    x_input[2] = bhk
    if loc_index >= 0:
        x_input[loc_index] = 1

    return lr.predict([x_input])[0]

# print(predict_price('1st Phase JP Nagar',1000, 2, 2))
# print(predict_price('1st Phase JP Nagar',1000, 3, 3))


# Export the tested model to a pickle file
# lưu vào file mô hình đã huấn luyện 
with open('20_DuAn/banglore_home_price_model.pickle','wb') as f:
    pickle.dump(lr,f)
 
 

# để lưu tên các cột (columns) của dataframe x vào một file columns.json
# Đây là bước quan trọng nếu bạn muốn triển khai model sau này (dùng dự đoán trong ứng dụng web hoặc API)
columns = {
    'data_colums' : [col.lower() for col in x.columns]
    }
with open("20_DuAN/columns.json","w") as f:
    f.write(json.dumps(columns))