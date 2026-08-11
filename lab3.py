from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import scipy.spatial.distance
import matplotlib as plt

#A2
def label_encode(input_data,column_name):
    label_encoder=LabelEncoder()
    encoded_dataframe=input_data.copy()
    encoded_dataframe[column_name]=label_encoder.fit_transform(
        input_data[column_name].astype(str)
    )
    return encoded_dataframe,label_encoder
#A2
def one_hot_encode(input_data, column_name):
    encoded_dataframe = input_data.copy()
    one_hot_columns = pd.get_dummies(
        encoded_dataframe[column_name], 
        prefix=column_name, 
        drop_first=False
    )
    encoded_dataframe = pd.concat(
        [encoded_dataframe.drop(column_name, axis=1), one_hot_columns], 
        axis=1
    )
    return encoded_dataframe
#A3
def encodedataset(data):
    new = data.drop(columns=["Dt_Customer"])
    new, edumap = label_encode(new, "Education")
    new = one_hot_encode(new, "Marital_Status")
    # collect the one-hot column names it created, for A5-A8 reference
    maritalcols = [c for c in new.columns if c.startswith("Marital_Status_")]
    return new, edumap, maritalcols

#A4
def minkowsi(a,b,p):
    total=0
    for i in range(len(a)):
        total+=abs(a[i]-b[i])**p
    return total**(1/p)

#A5
def distance(a,b,maxp):
    plist=[i for i in range(1,maxp+1)]
    dlist=[minkowsi(a,b,p) for p in plist]
    return plist,dlist

#A6
def compare(a,b,maxp):
    rows=[]
    for i in range(1,maxp+1):
        mine=minkowsi(a,b,i)
        pkg=scipy.spatial.distance.minkowski(a,b,i)
        rows.append([i,mine,pkg,abs(mine-pkg)])
    return rows

#A7
def dotproduct(a,b):
    total=0
    for i in range(len(a)):
        total+=a[i]*b[i]
    return total
    
def norm(a):
    return dotproduct(a,a)**0.5             

#A8
def mean(data):
    return sum(data)/len(data)

def variance(data):
    m=mean(data)
    return sum((x-m)**2 for x in data)/len(data)

def std(data):
    return variance(data)**0.5

def matrixstats(matrix):
    means=[]
    variances=[]
    stds=[]
    for col in range(len(matrix[0])):
        column=[row[col] for row in matrix]
        means.append(mean(column))
        variances.append(variance(column))
        stds.append(std(column))
    return means,variances,stds

#A9
def comparestats(matrix):
    mymeans,myvars,mystds=matrixstats(matrix)
    npmeans=np.mean(matrix,axis=0)          
    npstds=np.std(matrix,axis=0)
    rows=[]
    for i in range(len(mymeans)):
        rows.append([mymeans[i],npmeans[i],abs(mymeans[i]-npmeans[i]),
                     mystds[i],npstds[i],abs(mystds[i]-npstds[i])])
    return rows

def main():

    sample_df = pd.DataFrame({
        'Product': ['Apple', 'Banana', 'Apple', 'Orange', 'Banana'],
        'Color': ['Red', 'Yellow', 'Red', 'Orange', 'Yellow'],
        'Price': [100, 50, 100, 75, 50]
    })

    print("Original Data")
    print(sample_df)

    label_df, encoder = label_encode(sample_df, "Product")
    print("\nLabel Encoding")
    print(label_df)

    one_hot_df = one_hot_encode(sample_df, "Color")
    print("\nOne Hot Encoding")
    print(one_hot_df)

    data = pd.read_excel(r"C:\Users\CHANDRA SEKHAR\OneDrive\Documents\MY_SUBJECTS\ML\LAB\Lab Session Data (1).xlsx",sheet_name="marketing_campaign")
    encoded, edumap, maritalcols = encodedataset(data)

    print("\nEncoded Dataset")
    print(encoded.head())

    v1 = [1, 2, 3]
    v2 = [4, 6, 8]

    print("\nA4")
    print("p=1 :", minkowsi(v1, v2, 1))
    print("p=2 :", minkowsi(v1, v2, 2))
    print("p=3 :", minkowsi(v1, v2, 3))

    vectors = encoded.fillna(0).astype(float)

    veca = list(vectors.iloc[0])
    vecb = list(vectors.iloc[1])

    ##A5
    plist, dlist = distance(veca, vecb, 10)
    for p, d in zip(plist, dlist):
        print("p =", p, "distance =", d)

    ##A6

    for row in compare(veca, vecb, 10):
        print("p =", row[0],"\n" "mine =", row[1],"\n""scipy =", row[2])

    ##A7
    print("Dot Product :", dotproduct(veca, vecb))
    print("Numpy Dot   :", np.dot(veca, vecb))
    print("Norm A      :", norm(veca))
    print("Norm B      :", norm(vecb))

    ##A8
    matrix = vectors.values.tolist()
    means, variances, stds = matrixstats(matrix)

    for name, m, v, s in zip(vectors.columns, means, variances, stds):
        print(name, "Mean =", m, "Variance =", v, "Std =", s)

    ##A9
    for name,row in zip(vectors.columns,comparestats(matrix)):
            print(name,"| mean mine=",row[0],"numpy=",row[1],"diff=",row[2],"| std mine=",row[3],"numpy=",row[4],"diff=",row[5])


if __name__ == "__main__":
    main()
