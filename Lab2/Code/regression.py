import numpy as np
import matplotlib.pyplot as plt
import util

def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)
    
    Inputs:
    ------
    beta: hyperparameter in the proir distribution
    
    Outputs: None
    -----
    """
    #=========================================================
    # plot contour   
    Mean = [0, 0]
    Var = [[beta, 0], 
           [0, beta]]

    # plot the gausian contour
    X, Y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    X_flat = X.flatten().reshape(-1, 1)
    #print("X_flat",np.shape(X_flat))
    Y_flat = Y.flatten().reshape(-1, 1)
    X_set = np.concatenate((X_flat, Y_flat), axis = 1)
    #print("X_set",np.shape(X_set))
    Z = util.density_Gaussian(Mean,Var, X_set).reshape((100, 100))
    #print("Z",np.shape(Z))

    plt.contour(X, Y, Z, colors = 'blue')
    plt.xlabel('a0')
    plt.ylabel('a1')
    plt.title('priorDistribution')
    plt.show()
    
    return 
    
def posteriorDistribution(x,z,beta,sigma2):
    """
    Plot the contours of the posterior distribution p(a|x,z)
    
    Inputs:
    ------
    x: inputs from training set
    z: targets from traninng set
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    
    Outputs: 
    -----
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """

    Var_x_inverse = [[1/beta, 0], 
                     [0, 1/beta]]
    Var_z_inverse = 1/sigma2
    A = np.append(np.ones(shape=(len(x), 1)), x, axis=1)
    y = z
    
    # Determine parameters of the posterior distribution, p(a|x,z) ~ N(mu, Cov)
    Cov = np.linalg.inv(Var_x_inverse + Var_z_inverse * np.dot(A.T, A))  
    print("Cov:",Cov)
    mu = (np.matmul(np.matmul(Cov, A.T), y)*Var_z_inverse).flatten()
    print("mu:",mu)
    datalen = len(y)


    # plot the gausian contour
    X, Y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    X_flat = X.flatten().reshape(-1, 1)
    #print("X_flat",np.shape(X_flat))
    Y_flat = Y.flatten().reshape(-1, 1)
    X_set = np.concatenate((X_flat, Y_flat), axis = 1)
    #print("X_set",np.shape(X_set))
    Z = util.density_Gaussian(mu,Cov, X_set).reshape((100, 100))
    #print("Z",np.shape(Z))

    plt.contour(X, Y, Z, colors = 'blue')
    plt.xlabel('a0')
    plt.ylabel('a1')
    title = 'posterior'+ str(datalen)
    plt.title(title)
    plt.show()

    return (mu,Cov)

def predictionDistribution(x,beta,sigma2,mu,Cov,x_train,z_train):
    """
    Make predictions for the inputs in x, and plot the predicted results 
    
    Inputs:
    ------
    x: new inputs
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    mu: output of posteriorDistribution()
    Cov: output of posteriorDistribution()
    x_train,z_train: training samples, used for scatter plot
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here

    #===========================================================================================
    M_a = mu
    Var_a = Cov
    Var_w = sigma2
    
    
    # Plot the new input and standard deviation
    for i in x:    
        x_input = np.array([[1], [i]])
        
        # Determine parameters of the likelihood distribution, p(z|a,x,z) ~ N(mu, Cov)
        M_z = np.matmul(M_a.T, x_input)
        Var_z = np.matmul(x_input.T, np.matmul(Var_a, x_input)) + Var_w
        std_dev = np.sqrt(Var_z)
        
        plt.errorbar(i, M_z, yerr = std_dev, 
                    fmt = 'o', color = 'grey', 
                    ecolor = 'lightgrey')

    # Plot training sample
    plt.scatter(x_train, z_train, color = 'blue')
        
    # Clean up graph
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.xlabel('x')
    plt.ylabel('z')

    train_data_len = np.size(x_train, 0)
    plt.title('Prediction with' + str(train_data_len) + ' training Samples')  
    plt.show()
    #===========================================================================================
    
    return 

if __name__ == '__main__':
    
    # training data
    x_train, z_train = util.get_data_in_file('training.txt')
    # print("x_train",x_train)
    # print("z_train",z_train)

    # new inputs for prediction 
    x_test = [x for x in np.arange(-4,4.01,0.2)]
    
    # known parameters 
    sigma2 = 0.1
    beta = 1
    
    # prior distribution p(a)
    priorDistribution(beta)

    # number of training samples used to compute posterior
    L_ns = [1,5,100]

    for ns in L_ns:
        print("ns = ",ns,"==========================================")
    
        # used samples
        x = x_train[0:ns]
        z = z_train[0:ns]
        
        
        # posterior distribution p(a|x,z)
        mu, Cov = posteriorDistribution(x,z,beta,sigma2)
        
        # distribution of the prediction
        predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z)
    

   

    
    
    

    
