import random
import statistics

def one():
    NAME=input("enter:")
    count=0
    for i in NAME:
        if i.lower() in "aeiou":
            count+=1
    print(count)



def two():
    print("matrixa:")
    a=[[int(input()),int(input())],[int(input()), int(input())]]
    print("matrixb:")
    b=[[int(input()),int(input())],[int(input()), int(input())]]

    print(a[0][0]*b[0][0]+a[0][1]*b[1][0],a[0][0]*b[0][1]+a[0][1]*b[1][1])

    print(a[1][0]*b[0][0]+a[1][1]*b[1][0],a[1][0]*b[0][1]+a[1][1]*b[1][1])



def three():
    a=[]
    b=[]
    print("enter list 1:")
    for i in range(3):
        a.append(int(input()))

    print("enter list 2:")
    for i in range(3):
        b.append(int(input()))
    count=0

    for i in a:
        if i in b:
            count+=1

    print("Common elements:", count)


def four():
    a=[]

    for i in range(2):
        row=[]
        for j in range(2):
            row.append(int(input()))
        a.append(row)

    print("Trans:")

    for i in range(2):
        for j in range(2):
            print(a[j][i],end=" ")
        print()





def five():
    a=[]

    for i in range(100):
        a.append(random.randint(100,150))
    print("Mean =",statistics.mean(a))
    print("Median =",statistics.median(a))
    print("Mode =",statistics.mode(a))


one()
two()
three()
four()
five()