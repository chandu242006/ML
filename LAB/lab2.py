import pandas as p
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns


def load_data(file_path):
    data = p.read_excel(file_path, sheet_name="Purchase data")
    price=p.read_excel(file_path, sheet_name="IRCTC Stock Price")
    thyroid=p.read_excel(file_path,sheet_name="thyroid0387_UCI")
    return data,price,thyroid


    #-------------------------------------------------------1A
def create_matrix(data):
    x = data[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
    y = data[["Payment (Rs)"]].values
    return x, y

def calculate_rank(x):
    return np.linalg.matrix_rank(x)

def pseudo_inverse(x):
    return np.linalg.pinv(x)    

def calculate_cost(x_inverse, y):
    return x_inverse @ y


#-------------------------------------------------------3A
def numpy_mean(price):
    return np.mean(price)

def numpy_variance(price):
    return np.var(price)



def my_mean(price):
    total = 0
    for value in price:
        total += value
    return total / len(price)

def my_variance(price):
    mean = my_mean(price)
    total = 0
    for value in price:
        total += (value - mean) ** 2
    return total / len(price)


def wed_mean(data):
    wed=data[data["Day"]=="Wed"]
    return np.mean(wed["Price"])

def april_mean(data):
    april=data[data["Month"]=="Apr"]
    return np.mean(april["Price"])

def prob_loss(data):
    loss=data[data["Chg%"]<0]
    return len(loss)/len(data)

def prob_profit_wed(data):
    profit=data[(data["Day"]=="Wed")&(data["Chg%"]>0)]
    return len(profit)/len(data)

def cond_prob(data):
    wednesday=data[data["Day"]=="Wed"]
    profit=wednesday[wednesday["Chg%"]>0]
    return len(profit)/len(wednesday)

def scatter_plot(data): 
    plt.figure(figsize=(8,5))
    plt.scatter(data["Day"], data["Chg%"])
    plt.title("Chg% vs Day")
    plt.xlabel("Day")
    plt.ylabel("Chg %")
    plt.grid(True)
    plt.show()

#-----------------------------------------------4A
def attribute_info(data):
    print(data.dtypes)
    print(data.head())

def encode_data(data,ordinal_cols,nominal_cols):
    encoded=data.copy()
    encoded=encoded.replace({"t":1,"f":0,"?":np.nan})
    for col in ordinal_cols:
        encoded[col]=encoded[col].astype("category").cat.codes
    encoded=p.get_dummies(encoded,columns=nominal_cols)
    return encoded

def numeric_range(data):
    numeric_cols=data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        print(col,data[col].min(),data[col].max())

def missing_values(data):
    print(data.isnull().sum())

def mean_variance(data):
    numeric_cols=data.select_dtypes(include=[np.number]).columns
    print(data[numeric_cols].mean())
    print(data[numeric_cols].var())

def detect_outliers(data):
    numeric_cols=data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        q1=data[col].quantile(0.25)
        q3=data[col].quantile(0.75)
        iqr=q3-q1
        lower=q1-1.5*iqr
        upper=q3+1.5*iqr
        outliers=data[(data[col]<lower)|(data[col]>upper)]
        print(col,len(outliers))
 
#-------------------------------------------------------------------------5A
def get_binary_columns(data):
    cols=[]
    for col in data.columns:
        values=set(data[col].dropna().unique())
        if values.issubset({0,1}):
            cols.append(col)
    return cols

def jc_smc(v1,v2):
    f11=f10=f01=f00=0
    for a,b in zip(v1,v2):
        if a==1 and b==1:
            f11+=1
        elif a==1 and b==0:
            f10+=1
        elif a==0 and b==1:
            f01+=1
        elif a==0 and b==0:
            f00+=1
    if (f11+f10+f01)==0:
        jc=0
    else:
        jc=f11/(f11+f10+f01)
    if (f11+f10+f01+f00)==0:
        smc=0
    else:
        smc=(f11+f00)/(f11+f10+f01+f00)
    return jc,smc
 
#-----------------------------------------------------------------6A
def cosine_similarity(v1,v2):
    v1=np.array(v1,dtype=float)
    v2=np.array(v2,dtype=float)
    dot=np.dot(v1,v2)
    norm1=np.linalg.norm(v1)
    norm2=np.linalg.norm(v2)
    if norm1==0 or norm2==0:
        return 0
    return dot/(norm1*norm2)
 
 
#-------------------------------------------------------7A
def heatmap_plot(data,n=20):
    subset=data.select_dtypes(include=[np.number]).fillna(0).iloc[:n]
    binary_cols=get_binary_columns(subset)

    if not binary_cols:
        print("No binary columns found")
        return

    jc_matrix=np.zeros((n,n))
    smc_matrix=np.zeros((n,n))
    cos_matrix=np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            v1=subset.iloc[i][binary_cols].astype(int)
            v2=subset.iloc[j][binary_cols].astype(int)
            jc_matrix[i][j],smc_matrix[i][j]=jc_smc(v1,v2)
            cos_matrix[i][j]=cosine_similarity(subset.iloc[i],subset.iloc[j])

    plt.figure(figsize=(18,5))

    plt.subplot(131)
    sns.heatmap(jc_matrix)
    plt.title("JC")

    plt.subplot(132)
    sns.heatmap(smc_matrix)
    plt.title("SMC")

    plt.subplot(133)
    sns.heatmap(cos_matrix)
    plt.title("Cosine")

    plt.show()
 
#-------------------------------------------------------8A
def impute_data(data):
    imputed = data.copy()
    numeric_cols = imputed.select_dtypes(include=[np.number]).columns
    categorical_cols = imputed.select_dtypes(include=[object]).columns
 
    for col in numeric_cols:
        q1 = imputed[col].quantile(0.25)
        q3 = imputed[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        has_outliers = ((imputed[col] < lower) | (imputed[col] > upper)).any()
        if has_outliers:
            imputed[col] = imputed[col].fillna(imputed[col].median())
        else:
            imputed[col] = imputed[col].fillna(imputed[col].mean())
 
    for col in categorical_cols:
        imputed[col] = imputed[col].fillna(imputed[col].mode()[0])
 
    return imputed
 
 
#-----------------------------------------------9A
def normalize_data(data):
    normalized = data.copy()
    numeric_cols = normalized.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        q1 = normalized[col].quantile(0.25)
        q3 = normalized[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        has_outliers = ((normalized[col] < lower) | (normalized[col] > upper)).any()
        if has_outliers:
            mean = normalized[col].mean()
            std = normalized[col].std()
            normalized[col] = (normalized[col] - mean) / std
        else:
            min_val = normalized[col].min()
            max_val = normalized[col].max()
            normalized[col] = (normalized[col] - min_val) / (max_val - min_val)
    return normalized

def main():

    file_path = r"C:\Users\CHANDRA SEKHAR\OneDrive\Documents\MY_SUBJECTS\ML\LAB\Lab Session Data (1).xlsx"
    data,price,thyroid = load_data(file_path)

    #1
    x, y = create_matrix(data)
    rank = calculate_rank(x)
    x_inverse = pseudo_inverse(x)
    product_cost = calculate_cost(x_inverse, y)
    print("Matrix:")
    print(x)
    print("Output:")
    print(y)
    print("Rank:",rank)
    print("Cost of Each Product")
    print("Candies:",product_cost[0][0])
    print("Mangoes:",product_cost[1][0])
    print("Milk Packets:",product_cost[2][0])

    #3
    prices=price["Price"]
    print(f"mean:",numpy_mean(prices))
    print(f"var:",numpy_variance(prices))
    print(f"my_mean:",my_mean(prices))
    print(f"my_var",my_variance(prices))
    print(f"wednesday_mean:",wed_mean(price))
    print(f"aprial mean :",april_mean(price))
    print(f"probability loss:",prob_loss(price))
    print(f"pron wed profit :",prob_profit_wed(price))
    print(f" cond prob :",cond_prob(price))

    scatter_plot(price)


    #4
    attribute_info(thyroid)
 
    ordinal_cols = []
    nominal_cols = [c for c in thyroid.columns if thyroid[c].dtype == object and thyroid[c].nunique() <= 10]
    encoded_thyroid = encode_data(thyroid, ordinal_cols, nominal_cols)
 
    numeric_range(encoded_thyroid)
    missing_values(encoded_thyroid)
    mean_variance(encoded_thyroid)
    detect_outliers(encoded_thyroid)
 
    #8
    imputed_thyroid = impute_data(encoded_thyroid)
    imputed_encoded = encode_data(imputed_thyroid, ordinal_cols, nominal_cols)
 
    #5
    binary_cols = get_binary_columns(imputed_encoded)
    v1 = imputed_encoded.iloc[0][binary_cols].values
    v2 = imputed_encoded.iloc[1][binary_cols].values
    jc, smc = jc_smc(v1, v2)
    print("JC:", jc)
    print("SMC:", smc)
 
    #6
    numeric_vec1 = imputed_encoded.select_dtypes(include=[np.number]).fillna(0).iloc[0].values
    numeric_vec2 = imputed_encoded.select_dtypes(include=[np.number]).fillna(0).iloc[1].values
    cos_sim = cosine_similarity(numeric_vec1, numeric_vec2)
    print("Cosine Similarity:", cos_sim)
 
    #7
    heatmap_plot(imputed_encoded, n=20)
 
    #9
    normalized_thyroid = normalize_data(imputed_thyroid)
    print(normalized_thyroid.head())


if __name__ == "__main__":
    main()