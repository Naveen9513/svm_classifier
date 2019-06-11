# Question 3
import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import cvxopt

class SVM_binary:
    """
    This is the class which perform support vector machine for binary classification
    """
    
    def __init__(self,kernel='linear',param=None,c=1.0):
        """
        Initiate SVM
        kernel:    kernel function we are using ;default=linear
                        linear
                        poly
                        rbf
        param:    gamma value use for rbf or d value use for polynomial
        c:        how much violation allowed; default=1.0   
        """
        # These are the parameters we want to find
        # w is not computed directly.
        self.b=None
        
        #user given parameters 
        self.c=c             #use to fuzzy separation
        self.kernel=kernel   # kernel using for kernel trick: default='linear'
        self.param=param
        
        #support vectors and lagrange multipliers store here
        self.sv_alphas=None
        self.sv_x=None
        self.sv_y=None
        
    def linear_kernel(self,x,y):
        """
        This function returns dot product of x and y vectors
        x:         number_of_examples x num_of_features matrix
        y:         number_of_examples x num_of_features matrix
        returns :  returns m x m vector where each element contains the x_i.y_i
        """
        return np.dot(x,y.T)
    
    def poly_kernel(self,x,y,d=3):
        """
        This function returns polynomial function value of degree d for x and y
        x:         number_of_examples x num_of_features matrix
        y:         number_of_examples x num_of_features matrix
        returns :  returns m x m vector where each element contais the (1+ x_i.y_i)**d
        """
        c=(1 + np.dot(x,y.T))**d
        print(c.shape)
        return (1 + np.dot(x, y.T)) ** d
    
    def rbf_kernel(self,x,y,gamma=0.1):
        """
        This function returns rbf function value for x and y 
        x:         number_of_examples x num_of_features matrix
        y:         number_of_examples x num_of_features matrix
        returns :  returns m x m vector where each element contais the (1+ x_i.y_i)**d
        
        """
        out=np.zeros((x.shape[0],y.shape[0]))   # m x m matrix
            
        # out[i,j]=exp(-gamma*( ||x[i]-y[i]||**2))
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                out[i,j]=np.exp(-(np.linalg.norm(x[i]-y[j])**2)*gamma)
        return out
        
    def fit(self,X,Y):
        """
        This function use to fit the SVM model  
        X:         number_of_examples x num_of_features matrix
        Y:         number_of_examples x 1 matrix
        """
        m,n=X.shape #dimensions of X
        Y=Y.reshape(m,1)
        Y.shape[1] == 1, 'Dimension of Y did not match.\n Expected (m,1) shape '
        
        #creating H
        Y_=np.dot(Y,Y.T)  # creating m x m matrix where Y_[i,j]=y_i*y_j
        
        #deciding kernel
        if (self.kernel=='linear'):
            H=self.linear_kernel(X,X)  #dot product of X_i and X_j's
        
        elif(self.kernel=='poly'):
            H=self.poly_kernel(X,X,self.param)    #polynomial function of X_i and X_j
        
        elif(self.kernel=='rbf'):
            H=self.rbf_kernel(X,X,self.param)     #rbf function of X_i and X_j
            
        else:
            print('Invalid kernel')
            return None
        
        H=H*Y_*1.  #multiply by 1. to make values to float
        
        #creating cvxopt matrices
        P=cvxopt_matrix(H)
        q =cvxopt_matrix(-np.ones(m))         # m x 1 matrix of -1's
        A = cvxopt_matrix(Y.T*1.)             # label vector of y x 1                  
        b = cvxopt_matrix(np.zeros(1)*1.)     # scaler 0
        
        #if c is none there is one less condition in lagrange multiplier
        #G and h computed for one condition
        if (self.c==None):
            G =cvxopt_matrix(-np.eye(m))          # a diagonal matrix of -1's
            h =cvxopt_matrix(np.zeros((m,1)))     # m x 1 zero matrix 
        
        #G and h computed for two conditions
        else:
            #computing G
            G1=-np.eye(m)
            G2=np.eye(m)
            G_=np.vstack((G1,G2))
            G=cvxopt_matrix(G_)

            #computing h
            h1=np.zeros((m,1))
            h2=np.ones((m,1))*self.c
            h_=np.vstack((h1,h2))
            h=cvxopt_matrix(h_)
            
            
        #Run solver
        cvxopt_solvers.options['show_progress'] = False
        sol = cvxopt_solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x'])  
        
        #extracting suppport vectors
        sv=alphas > 1e-5                    #boolean array contains true at the locations where condition satisfies 
        self.sv_alphas=alphas[sv[:,0],:]    # alpha values not equal to zero(correspondant to support vectors )
        
        # support vectors
        self.sv_x=X[sv[:,0],:]
        self.sv_y=Y[sv[:,0],:]
        
        #computing b
        
        #b1 is sum of sv_y's a 1 x 1 scalar
        b1=(np.sum(self.sv_y,axis=0) )
        
        tmp_b1=(self.sv_y*self.sv_alphas).T   # 1 x k matrix contains [alpha_1*y_1 alpha_2*y_2.......alpha_k*y_k]
    
        #tmp_b2 is a k x k vector which contains:
        #    tmp_b2[i,j]=kernel(x_sv[i],x_sv[j])
        
        if (self.kernel=='linear'):
            tmp_b2=self.linear_kernel(self.sv_x,self.sv_x)  
        elif(self.kernel=='poly'):
            tmp_b2=self.poly_kernel(self.sv_x,self.sv_x,self.param)
        elif(self.kernel=='rbf'):
            tmp_b2=self.rbf_kernel(self.sv_x,self.sv_x,self.param)
            
        
        tmp_b3=np.dot(tmp_b1,tmp_b2).T  # k x 1 vector which contains w.x_sv results
        
        #b2 is sum of tmp_b3
        b2=np.sum(tmp_b3,axis=0)
        
        b=b1-b2
        
        self.b=b/self.sv_alphas.shape[0]   #averaging b over support vectors         

        
    def predict(self,X):
        """
        This function use to predict the class of X using the SVM model  
        X:         number_of_examples x num_of_features matrix
        returns:
            predictions (contains prediction scores) : number_of_examples x 1
        """
        #sigma alpha_1*y_i matrix : 1 x k matrix
        tmp1=(self.sv_y*self.sv_alphas).T
        
        #deciding the kernel
        if (self.kernel=='linear'):
            tmp2=self.linear_kernel(self.sv_x,X)
        elif(self.kernel=='poly'):
            tmp2=self.poly_kernel(self.sv_x,X)
        elif(self.kernel=='rbf'):
            tmp2=self.rbf_kernel(self.sv_x,X)
        
        #final prediction score is matrix multification of tmp1 and tmp2
        # This contains prediction score for binary classification
        predictions=np.dot(tmp1,tmp2).T +self.b  

        
        return predictions
    
    
class SVM():
    """
    SVM class perform support vector machine for multiclass classification
    """
    
    def __init__(self,kernel='linear',param=None,c=None):
        """
        Initiate SVM
        kernel:    kernel function we are using ;default=linear
                        linear
                        poly
                        rbf
        param:    gamma value use for rbf or d value use for polynomial
        c:        how much violation allowed; default=1.0   
        """
        self.kernel=kernel
        self.c=c
        self.param=param
        
        #use to store classification models for all classes
        self.classifiers=[]
    
    def fit(self,X,Y):
        """
        This function use to fit the SVM model  
        X:         number_of_examples x num_of_features matrix
        Y:         number_of_examples x 1 matrix
        """
        #unique classes
        classes=np.unique(Y)
        
        for class_id in classes:
            y=(class_id==Y).astype(float) # indexes which contains class_id become 1 and all others are 0
            ind=(y==0) #find indexes of negative class
            y[ind]=-1  #convert 0 to -1
            
            clf=SVM_binary(self.kernel,param=self.param,c=self.c) #create clf model by taking one class as possiive and
                                                                  #all others as negative
            clf.fit(X,y) #fit the classifier
            self.classifiers.append(clf)#store classifier
            
    def predict(self,X):
        """
        This function use to predict the class of X using the SVM model  
        X:         number_of_examples x num_of_features matrix
        returns:
            predictions : number_of_examples x 1
        """
        #store scores here
        scores=np.empty(shape=(X.shape[0],0))
        i=0
        for clf in self.classifiers:
            pred=clf.predict(X)          # m x 1 array
            scores=np.append(scores,pred,axis=1)
            i+=1
        #class which have highest score considered as the predicted class
        predictions=np.argmax(scores,axis=1)
        
        return predictions.T