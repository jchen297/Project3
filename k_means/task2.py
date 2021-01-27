import numpy as np
import matplotlib.pyplot as plt

################################# Part 1-3 ##################################################
# function for plot (x,y) label
def anotation(x,y):
    label = f"({round(x,1)},{round(y,1)})"
    plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,-12), # distance from text to points (x,y)
                 ha='center')

# function for plot
def plot_(X,mu,class_vector=0):
    color = ['r','g','b']
    xs,ys = X
    # Ploting
    fig = plt.figure(figsize=(10,6))
    for i in range(len(xs)):
        x,y = xs[i],ys[i]
        if type(class_vector) == int:
            plt.plot( x,y,marker = '^', markersize=9,markeredgewidth=1,markeredgecolor='k',markerfacecolor='none')
        else:
            plt.plot( x,y,marker = '^', markersize=9,markeredgewidth=1,markeredgecolor=color[class_vector[i]],markerfacecolor=color[class_vector[i]])
        # plt.scatter(x,y,facecolors=facecolor,c = color[class_vector[i]],s = 50)
        anotation(x,y)

    for i in range(3):
        x,y = mu[i]
        plt.scatter(x,y,c=color[i],marker='o',s = 150)
        anotation(x,y)

# function for calculate Euclidean distance
def eu_dis(p1, p2):
    return np.linalg.norm(p1 - p2)

# function to find minimum distance within a row in a (N,3) array
def min_dis(arr):
    return np.where(arr == np.amin(arr))[0][0]

# function to generate random centroid
def random_center(xs, ys, k):
    mu = np.zeros((k, 2))
    for i in range(k):
        xi = np.random.randint(1, 10)
        yi = np.random.randint(1, 10)
        mu[i] = [xs[xi], ys[yi]]
    return mu

# k mean function
def k_means(k, X, iterr, mu=0):
    # get x coor and y coor
    xs, ys = X
    N = len(xs)
    # if no initial center
    if type(mu) == int:
        mu = random_center(xs, ys, k)

    iterr_count = 0
    while iterr_count < iterr:
        # Store old mu
        old_mu = mu
        # compute distance between points and centers
        dis = np.zeros((k, N))
        for i in range(k):
            cen = np.tile(mu[i], [N, 1])
            dis[i] = list(map(eu_dis, X.T, cen))
        # Get classification vector
        class_vector = np.array(list(map(min_dis, dis.T)))

        # recompute mu
        mu = np.zeros((k, 2))
        for i in range(k):
            x = xs[class_vector == i]
            y = ys[class_vector == i]
            mu[i] = [x.mean(), y.mean()]
        iterr_count += 1

        # if mu is not changing stop
        if np.array_equal(mu, old_mu):
            break
    return class_vector, mu


X = np.array([[5.9, 3.2],[4.6, 2.9],[6.2, 2.8],[4.7, 3.2],[5.5, 4.2],[5.0, 3.0],[4.9, 3.1],[6.7, 3.1],[5.1, 3.8],[6.0, 3.0]]).T
mu1 = np.array([[6.2, 3.2],[6.6, 3.7],[6.5, 3.0]])

############### Q 1 & 2
k = 3
iterr = 1
# perform k means, and get class_vector and new centroid
class_vector, mu = k_means(k, X, iterr, mu1)

# plot class_vector with old centroid and save pic
plot_(X, mu1, class_vector)
plt.savefig('task2_iter1_a.png')

# plot new centroid and save pic
plot_(X, mu)
plt.savefig('task2_iter1_b.png')

############### Q 3
k = 3
iterr = 1
# store the centroid mu from first iter
mu1 = mu
# second iter
class_vector, mu = k_means(k, X, iterr, mu)

# plot class_vector with old centroid and save pic
plot_(X, mu1, class_vector)
plt.savefig('task2_iter2_a.png')

# plot new centroid and save pic
plot_(X, mu)
plt.savefig('task2_iter2_b.png')

################################# Part 4 ##################################################
import time


# function to generate random centroid
def random_center_img(k):
    mu = np.zeros((k,3))
    for i in range(k):
        r = np.random.randint(1,255)
        g = np.random.randint(1,255)
        b = np.random.randint(1,255)
        mu[i] = [r,g,b]
    return mu

# # function to compare the mu with updated mu
# def compare_mus(mu,old_mu,k):
#     diff = []
#     for i in range(k):
#         old = np.tile(old_mu[i],[k,1])
#         diff.append(min(list(map(eu_dis,mu,old))))
#     return np.mean(diff)

# function to apply new centroid to the whole image
def change_color(rgbs,class_vector,mu):
    for k in range(len(mu)):
        rgbs[class_vector == k] = mu[k]
    # print(mu)
    return rgbs

def color_quantization_k_means(k,img,iterr,mu=0):
    print(f'----------------------- When k={k} ---------------------------')
    # image shape
    r,h,d = img.shape

    # there are N pixels
    N = r*h

    # convert img to 1D
    rgbs = img.reshape(N,3).copy()

    # # initial cluster center
    # # in order to make the algorithm faster, I set initial centroid manually
    if type(mu) == int:
        xs, ys, zs = rgbs.T
        mu = random_center_img(xs,ys,zs,N,k)

    times = []
    iterr_count = 0
    # print(f'----------- k-means color quantization k = {k} -----------')
    while iterr_count < iterr:
        # print(f'{iterr_count} iteration start')
        start_time = time.time()
        # Store old mu
        old_mu = mu
        # compute distance between points and centers
        dis = np.zeros((k,N))
        for i in range(k):
            cen = np.tile(mu[i],[N,1])
            dis[i] = list(map(eu_dis,rgbs,cen))
        # Get classification vector
        class_vector = np.array(list(map(min_dis,dis.T)))

        # recompute mu
        mu = np.zeros((k,d))
        for i in range(k):
            RGB = rgbs[class_vector==i]
            R,G,B = RGB.T
            if R.size == 0 or G.size == 0 or B.size == 0:
                mu[i] = old_mu[i]
            else:
                mu[i] = [int(R.mean()),int(G.mean()),int(B.mean())]
        # record time
        end_time = time.time()
        dif = round(end_time-start_time,3)
        times.append(dif)
        print('updated µ: ',mu)
        # mu_change = compare_mus(mu,old_mu,k)
        # print('µ change: µ - old_µ = ', mu_change)
        # print(f'iteration end -- Run time: {dif}s \n')
        iterr_count+=1
        # # if mu is not changeing stop
        # if mu_change < 1.5*k:
        #     break

    print(f'----------- Total Running for k={k}: {sum(times)}s -----------')
    return change_color(rgbs,class_vector,mu).reshape(r,h,d)

import cv2

# read image
img = cv2.imread('Project3_clustering/baboon.png')

# set initial centroid to make the algorithm faster
kcen = [np.array([[ 69, 108, 216],[174, 173, 149],[ 78,  98,  92]], dtype=np.float),
        np.array([[ 60,  90, 227],[130, 164, 168],[100, 121, 111],[214, 185, 143],[ 56,  71,  69]], dtype=np.float),
        np.array([[222, 180, 118],[156, 174, 172],[115, 146, 138],[ 50,  63,  62],[102, 104, 221],[ 78, 161, 184],[ 46,  79, 233],[ 80, 105, 100],[138, 128, 100],[218, 195, 165]], dtype=np.float),
        np.array([[ 65,  81,  74],[131, 159, 147],[ 87, 103,  90],[130, 125, 101],[ 87, 174, 194],[228, 186, 123],[145, 166, 191],[105, 107, 223],[ 60,  81, 187],[ 38,  66,  75],[ 63,  84, 237],[181, 152, 110],[174, 183, 175],[221, 197, 165],[ 48,  55,  53],[ 59, 106, 119],[ 31,  33,  38],[ 97, 132, 126],[ 31,  74, 240],[ 60, 139, 170]], dtype=np.float)]
ks = [3,5,10,20]


print(f'----------------------------------------- Color Quantization  Start -----------------------------------------\n')
start_time = time.time()
for i in range(4):
    new_img = color_quantization_k_means(ks[i],img,1,kcen[i])
    cv2.imwrite(f'task2_baboon_{ks[i]}.jpg',new_img)
    print(f'/task2_baboon_{ks[i]}.jpg saved at current directory\n')
end_time = time.time()
dif = round(end_time-start_time,3)
print(f'---------------------------------All Process Finished!! -- Run time: {dif}s ---------------------------------')