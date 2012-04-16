from random import uniform
from math import log
import matplotlib.pyplot as plt 
from pprint import pprint

N = 2000
L = 10

def unif(n):
    vec = []
    while len(vec) < n:
        vec.append(uniform(-1,1))
    return vec

def generate_class(label=1):
    X = []
    while len(X) < N:
        x,y = unif(2)
        if label ==1 and x**2 + y**2 >= 1: continue
        if label ==-1:
            x *= 2; y*= 2
            if x**2 + y**2 < 1: continue
        X.append([x,y])
    return X

def generate_classifiers():
    C = []
    while len(C) < L:
        a,b,c = unif(3)
        C.append([a,b,c])
    return C

def is_prediction_hit(classifier, data, label):
    a,b,c = classifier
    x, y = data
    if linear(a,b,c,x,y) > 0:
        pred = 1
    else:
        pred = -1
    return pred == label
    

def linear(a,b,c, x, y):
    return a*x+b*y+c

def draw_data(X1, X2):
    for (x,y) in X1:
        plt.plot(x,y, 'b.')
    for (x,y) in X2:
        plt.plot(x,y, 'rx')
    
    plt.axis('equal')
    
    
def draw_linear_classifier(a,b,c):
    x1 = 0; x2 = 2;
    y1 = -(a*x1+c)/b
    y2 = -(a*x2+c)/b
    plt.plot((x1,y1), (x2,y2), 'y-')
    plt.ylim(ymax = 2, ymin = -2)
    plt.xlim(xmax = 2, xmin = -2)

def fill_scout():
    pred = {}
    for j, classifier in enumerate(C):
        pred[j] = []
        for i,data in enumerate(X1):
            pred[j].append(is_prediction_hit(classifier, data, 1))
        for i,data in enumerate(X2):
            pred[j].append(is_prediction_hit(classifier, data, -1))
    return pred

def report_stats(pred):
    for j in range(L):
        a=len([c for c in pred[j] if c])
        b=len([c for c in pred[j] if not c])
        print j, a, b,
        if a == b: print C[j],
        print 

def get_draftee(pred):
    min_We = 1e100
    min_W = 1e100
    min_classifier = -1
    for j in range(L):
        if j in in_committee: continue
        We = 0
        W = 0
        for i,prediction in enumerate(pred[j]):
            W += w[i]
            if not prediction:
                We += w[i]
        if We < min_We:
            min_classifier = j
            min_We = We
            min_W = W
            
    em = float(min_We)/min_W
    print min_We, min_W
    alpha[min_classifier] = .5*log((1-em)/em)
    in_committee.add(min_classifier)
    
    for i in range(2*N):
        if not pred[min_classifier]:
            w[i] *= ((1-em)/em)**.5
        else:
            w[i] *= (em/(1-em))**.5 

def test(X, y):
    cnt = 0
    hat = []
    for i,data in enumerate(X):
        pred = 0
        for j, classifier in enumerate(C):
            if is_prediction_hit(classifier, data, y):
                pred += alpha[j]
            else:
                pred += -alpha[j]
        if (pred > 0 and y > 0) or (pred < 0 and y < 0): 
            cnt += 1
        if pred > 0: hat.append(1)
        else: hat.append(-1)
        
    return cnt, hat
        
def testAll(X1, X2):

    c1,hat1= test(X1, 1)
    c2,hat2= test(X2, -1)
    print float(c1+c2)/2/N
    
    for i,(x,y) in enumerate(X1):
        label = 'b.' if hat1[i] == 1 else 'rx'
        plt.plot(x,y, label)
    
    for i,(x,y) in enumerate(X2):
        label = 'b.' if hat2[i] == 1 else 'rx'
        plt.plot(x,y, label)
    plt.axis('equal')
    plt.show()


X1 = generate_class(1)
X2 = generate_class(-1)

T1 = generate_class(1)
T2 = generate_class(-1)

C = generate_classifiers()

pred = fill_scout()
#report_stats(pred)

#draw_data(X1,X2)
#for cl in C[:L]:
#    a,b,c = cl
#    draw_linear_classifier(a,b,c)
#plt.show()

w = [1.0]*(2*N)
alpha = [0]*L
in_committee = set()

for m in range(L):
    print m
    get_draftee(pred)
    
#pprint (alpha)
#print '-'*79

testAll(X1,X2)
testAll(T1,T2)

