import numpy as np;
import sys;
sys.path.insert(0, '/Users/echo/Desktop/')
import Monotone_SS as myfun;
import matplotlib.pyplot as plt;
import random

if __name__=='__main__':
    def tempfn(x):
#        return((x+0.7)*(x+0.2)*(x-0.8))
        return(-2*(x-0.5)*(x)*(x+0.5))


    alpha_seq=10.**np.arange(-4., 1, 1)
    a=-1
    b=1
    n=100;
    mu=0;
    n_repeat=10
    errorSD=0.2

    #simulate data
    X=np.random.uniform(a, b, n);
    error=np.random.normal(mu, errorSD, n)
    Y=tempfn(X)+error

    train_sample=random.sample(range(0,n), n)
    
    #train monotone increasing constrained model
    model=myfun.Monotone_SS(samplesize=20, constrains_sign=[-1,0])
    model.CV(X, Y, alpha_seq=alpha_seq, n_folds=3, refit=True, n_cores=1)

    #train unconstrained model
    model1=myfun.Monotone_SS(samplesize=20, constrains_sign=[0,0])
    model1.CV(X, Y, alpha_seq=alpha_seq, n_folds=3, refit=True, n_cores=1)

    #for plotting true and estimated line
    interval=np.arange(a, b, 0.01)
    truey=tempfn(interval)
    estimatedy1=model1.predict(interval)
    estimatedy=model.predict(interval)
    
    #plot true line 
    plt.plot(interval, truey, linewidth=1)
    plt.plot(interval, estimatedy, linewidth=1, color='red')
    plt.plot(interval, estimatedy1, linewidth=1, color='green')
    plt.scatter(X, Y)
    
    #plt.suptitle("alpha="+str(model.alpha), fontsize=20)
    print(model.alpha)
    plt.suptitle("blue:true, green:no constrain, red:monotone increasing constrained")
    plt.show()
    plt.close()
        
