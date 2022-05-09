import numpy as np
import matplotlib.pyplot as plt
import util

def discrimAnalysis(x, y):
    """
    Estimate the parameters in LDA/QDA and visualize the LDA/QDA models
    
    Inputs
    ------
    x: a N-by-2 2D array contains the height/weight data of the N samples
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of five elments: mu_male,mu_female,cov,cov_male,cov_female
    in which mu_male, mu_female are mean vectors (as 1D arrays)
             cov, cov_male, cov_female are covariance matrices (as 2D arrays)
    Besides producing the five outputs, you need also to plot 1 figure for LDA 
    and 1 figure for QDA in this function         
    """
    ### TODO: Write your code here
    
    total_people = len(y)
    number_of_Male = (y == 1).sum()
    number_of_Female = (y == 2).sum()

    #print(total_people,"|",number_of_Male,"|",number_of_Female)

    M_m = 0
    Var_m = 0
    height_m = []
    weight_m = []
    #=======================    calculate mean and variance for Male    =======================
    for i in range(total_people):
        yi = y[i]
        xi = x[i]
        if(yi == 1):
            M_m += xi
            height_m.append(xi[0])
            weight_m.append(xi[1])
    
    M_m = M_m/number_of_Male
    print("M_m",M_m)

    for i in range(total_people):
        yi = y[i]
        xi = x[i] 
        if(yi == 1):
            Var_m += np.outer(xi - M_m, np.transpose(xi - M_m))
            #print(Var_m)
    Var_m = Var_m/number_of_Male
    print("Var_m",Var_m)



    M_f = 0
    Var_f = 0
    height_f = []
    weight_f = []    
    #=======================    calculate mean and variance for FeMale    =======================
    for i in range(total_people):
        yi = y[i]
        xi = x[i]
        if(yi == 2):
            M_f += xi
            height_f.append(xi[0])
            weight_f.append(xi[1])
    #print(M_m)
    M_f = M_f/number_of_Female
    print("M_f",M_f)

    for i in range(total_people):
        yi = y[i]
        xi = x[i] 
        if(yi == 2):
            Var_f += np.outer(xi - M_f, np.transpose(xi - M_f))
            #print(Var_f)
    Var_f = Var_f/number_of_Female

    print("Var_f",Var_f)

    #=======================    calculate common variance    =======================
    Var_common = 0
    for i in range(total_people):
        yi = y[i]
        xi = x[i] 
        if(yi == 2):
            Var_common += np.outer(xi - M_f, np.transpose(xi - M_f))
        else:
            Var_common += np.outer(xi - M_m, np.transpose(xi - M_m))

    Var_common = Var_common/total_people
    print("Var_common",Var_common)
    
    #================================ Plot ================================
    # =================== LDA =============================================

    axes = plt.gca()
    axes.set_xlim([50, 80])
    axes.set_ylim([80, 280])

    # scattered data points
    plt.scatter(height_m, weight_m, color = 'blue')
    plt.scatter(height_f, weight_f, color = 'red')    

    # plot contour   
    X, Y = np.meshgrid(np.arange(50, 80, 1), np.arange(80, 280, 1))
    X_flat = X.flatten().reshape(-1, 1)
    Y_flat = Y.flatten().reshape(-1, 1)
    X_set = np.concatenate((X_flat, Y_flat), axis = 1)
    Z_m = util.density_Gaussian(np.transpose(M_m), Var_common, X_set).reshape((200, 30))
    Z_f = util.density_Gaussian(np.transpose(M_f), Var_common, X_set).reshape((200, 30))

    plt.contour(X, Y, Z_m, colors = 'blue')
    plt.contour(X, Y, Z_f, colors = 'red')
    
    # Plot boundary
    Inv_Var = np.linalg.inv(Var_common)
    IDA_m = np.matmul(np.matmul(np.transpose(M_m),Inv_Var),np.transpose(X_set)) -  1/2 * np.matmul(np.matmul(np.transpose(M_m), Inv_Var), M_m)
    IDA_f = np.matmul(np.matmul(np.transpose(M_f),Inv_Var),np.transpose(X_set)) -  1/2 * np.matmul(np.matmul(np.transpose(M_f), Inv_Var), M_f)

    Z_bound = (IDA_m - IDA_f).reshape((200, 30))
    plt.contour(X, Y, Z_bound, 0)    

    plt.xlabel('height')
    plt.ylabel('weight')
    plt.title('LDA')
    plt.show()

    # =================== QDA =============================================

    axes = plt.gca()
    axes.set_xlim([50, 80])
    axes.set_ylim([80, 280])

    # scattered data points
    plt.scatter(height_m, weight_m, color = 'blue')
    plt.scatter(height_f, weight_f, color = 'red')    

    # plot contour   
    X, Y = np.meshgrid(np.arange(50, 80, 1), np.arange(80, 280, 1))
    X_flat = X.flatten().reshape(-1, 1)
    Y_flat = Y.flatten().reshape(-1, 1)
    X_set = np.concatenate((X_flat, Y_flat), axis = 1)
    Z_m = util.density_Gaussian(np.transpose(M_m), Var_m, X_set).reshape((200, 30))
    Z_f = util.density_Gaussian(np.transpose(M_f), Var_f, X_set).reshape((200, 30))

    plt.contour(X, Y, Z_m, colors = 'blue')
    plt.contour(X, Y, Z_f, colors = 'red')
    
    # Plot boundary
    Inv_Var_m = np.linalg.inv(Var_m)
    Inv_Var_f = np.linalg.inv(Var_f)
    det_Var_m = np.linalg.det(Var_m)
    det_Var_f = np.linalg.det(Var_f)

    QDA_m = np.zeros((X_set.shape[0], 1))
    QDA_f = np.zeros((X_set.shape[0], 1))

    for i in range(X_set.shape[0]):
        QDA_m[i] = -1/2*np.log(det_Var_m) -  1/2 * np.matmul(np.matmul(np.transpose(X_set[i] - M_m), Inv_Var_m), (X_set[i] - M_m))
        QDA_f[i] = -1/2*np.log(det_Var_f) -  1/2 * np.matmul(np.matmul(np.transpose(X_set[i] - M_f), Inv_Var_f), (X_set[i] - M_f))

    Z_bound = (QDA_m - QDA_f).reshape((200, 30))
    plt.contour(X, Y, Z_bound, 0)    

    plt.xlabel('height')
    plt.ylabel('weight')
    plt.title('QDA')
    plt.show()
    
    return (M_m,M_f,Var_common,Var_m,Var_f)
    

def misRate(mu_male,mu_female,cov,cov_male,cov_female,x,y):
    """
    Use LDA/QDA on the testing set and compute the misclassification rate
    
    Inputs
    ------
    mu_male,mu_female,cov,cov_male,mu_female: parameters from discrimAnalysis
    
    x: a N-by-2 2D array contains the height/weight data of the N samples  
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of two elements: (mis rate in LDA, mis rate in QDA )
    """
    ### TODO: Write your code here
    N = len(y)
    LDA_miss_count = 0
    QDA_miss_count = 0

    M_m = mu_male
    M_f = mu_female
    Var_common = cov
    Var_m = cov_male
    Var_f = cov_female


    for i in range(N):
        xi = x[i]
        yi = y[i]

        #LDA Test
        Inv_Var = np.linalg.inv(Var_common)
        IDA_m = np.matmul(np.matmul(np.transpose(M_m),Inv_Var),np.transpose(xi)) -  1/2 * np.matmul(np.matmul(np.transpose(M_m), Inv_Var), M_m)
        IDA_f = np.matmul(np.matmul(np.transpose(M_f),Inv_Var),np.transpose(xi)) -  1/2 * np.matmul(np.matmul(np.transpose(M_f), Inv_Var), M_f)

        LDA_y = 2
        if(IDA_m > IDA_f):
            LDA_y = 1
        if(LDA_y != yi):
            LDA_miss_count += 1

        #QDA Test
        Inv_Var_m = np.linalg.inv(Var_m)
        Inv_Var_f = np.linalg.inv(Var_f)
        det_Var_m = np.linalg.det(Var_m)
        det_Var_f = np.linalg.det(Var_f)

        QDA_m = -1/2*np.log(det_Var_m) -  1/2 * np.matmul(np.matmul(np.transpose(xi - M_m), Inv_Var_m), (xi - M_m))
        QDA_f = -1/2*np.log(det_Var_f) -  1/2 * np.matmul(np.matmul(np.transpose(xi - M_f), Inv_Var_f), (xi - M_f))

        QDA_y = 2
        if(QDA_m > QDA_f):
            QDA_y = 1
        if(QDA_y != yi):
            QDA_miss_count += 1

    mis_lda = LDA_miss_count/N
    mis_qda = QDA_miss_count/N
    
    return (mis_lda, mis_qda)


if __name__ == '__main__':
    
    # load training data and testing data
    x_train, y_train = util.get_data_in_file('trainHeightWeight.txt')
    x_test, y_test = util.get_data_in_file('testHeightWeight.txt')
    
    # parameter estimation and visualization in LDA/QDA
    mu_male,mu_female,cov,cov_male,cov_female = discrimAnalysis(x_train,y_train)
    
    # misclassification rate computation
    mis_LDA,mis_QDA = misRate(mu_male,mu_female,cov,cov_male,cov_female,x_test,y_test)

    print("mis_LDA",mis_LDA)
    print("mis_QDA",mis_QDA)
    

    
    
    

    
