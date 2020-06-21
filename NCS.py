import numpy as np;

#define class of natural cublic spline;
class NCS(object):
    #initialization of NCS class requires knots and it corresponding function value.  As knots, it's assumed they are sorted.  Using same notation as Green and Silverman, we also calculates Q, R, R^(-1) and K using the knots value.  Also using the function value, gamma is also calculated.
    def __init__(self, knots, f):
        if(any(np.diff(np.sort(knots)))<=0):
	    print(knots)
	    print("knots must be unique and ordered increasingly");
	    stop;
        self.knots=knots;
	self.f=f;
	self.h=np.diff(knots);
	#calucate Q matrix
	n=knots.size;
	self._Q=np.zeros([n,(n-1)]);
	for j in range(1,n-1):
	    for i in range((j-1),(j+2)):
	        if(i==j): self._Q[i,j]=-(1./self.h[j-1])-(1./self.h[j]);
		if(i==(j-1)): self._Q[i,j]=1./self.h[i];
		if(i==(j+1)): self._Q[i,j]=1./self.h[j];
	self._Q=np.delete(self._Q, 0, axis=1);
	#calculate R Matrix
	self._R=np.zeros([n,n]);
        for i in range(1, (n-1)):
            for j in range(i, (i+2)):
                if(i==j):
                    self._R[i,j]=(self.h[i-1]+self.h[i])/3.;
                if(j==(i+1)):
                    self._R[i,j]=self.h[i]/6.;
                    self._R[j,i]=self.h[i]/6.;
        self._R=self._R[ 1:(n-1), 1:(n-1) ];
	#calculate K matrix
        self._inv_R=np.linalg.inv(self._R);
        self._K=np.dot(np.dot(self._Q, self._inv_R), self._Q.transpose());
        #checked above
        self.gamma=np.zeros(n);
        self.gamma[1:(n-1)]=np.dot(np.dot(self._inv_R, self._Q.transpose()),self.f);

    #Calculate the derivative of f at the knots.  This is written as a method because it is not always needed
    def prime(self):
        n=self.f.size;
        b=np.zeros(n);
        b[0]=((self.f[1]-self.f[0])/(self.h[0]))-(1./6.)*(self.h[0])*self.gamma[1];
        for j in range(1, n):
                b[j]=((self.f[j]-self.f[j-1])/self.h[j-1])+(1./6.)*self.h[j-1]*(self.gamma[j-1]+2*self.gamma[j]);
        return(b);

    #The method is not exposed.  It is used for calculating fuction value at a single point.  The plot() method is a wrapper of this function that allows both a scalar input or vector input. 
    def _plot(self,x):
        if(x<=self.knots[0]):
            g_prime_t1=((self.f[1]-self.f[0])/(self.h[0]))-(1./6.)*(self.h[0])*self.gamma[1];
            y=self.f[0]-(self.knots[0]-x)*g_prime_t1;
        elif(x>=self.knots[(self.knots.size-1)]):
            n=self.f.size;
            g_prime_tn=((self.f[n-1]-self.f[n-2])/(self.h[n-2]))+(1./6.)*(self.h[n-2])*self.gamma[n-2];
            y=self.f[self.f.size-1]+(x-self.knots[n-1])*g_prime_tn;
        else:
            i=np.max(np.where(self.knots<x));
            hi=self.h[i];
            y=( ( (x-self.knots[i])*self.f[i+1] + (self.knots[i+1]-x)*self.f[i] ) /hi)-(1./6.)*(x-self.knots[i])*(self.knots[i+1]-x)*( self.gamma[i+1]* (1.+((x-self.knots[i])/hi)) +self.gamma[i]*(1.+((self.knots[i+1]-x)/hi))  );
        return(y);

    #wrapper of _plot method
    def plot(self, x):
        x=np.array(x)       
        if(x.size==1):
            return(self._plot(x))
        else:
            return(np.array(map(self._plot, x)))

    #method to evaluate f'' at any point.
    def _primeprime_plot(self, x):
        i=np.max(np.where(self.knots<x));
        result=((x-self.knots[i])*self.gamma[i+1]+(self.knots[i+1]-x)*self.gamma[i])/(self.h[i])
        return(result)

    #wrapper of _plot method
    def primeprime_plot(self, x):
        x=np.array(x)
        if(x.size==1):
            return(self._primeprime_plot(x))
        else:
            return(np.array(map(self._primeprime_plot, x)))


    #Given the full function, calcuates piecewise polynomial coefficients
    def _poly_coefs(self):
        #to get f, f' and f''
        fp=self.prime()
        n=self.knots.size
        #column 0,1,2,3 represents coefficient a,b,c,d respectively
        #each row represents an interval.  e.g. row 0 represents (-infinity, z_1), row 1 represents (z_1, z_2)
        poly_coef=np.zeros((n+1,4))
        #c_0
        poly_coef[0,2]=fp[0]
        #c_n+1
        poly_coef[n,2]=fp[n-1]
        #a_i's
        poly_coef[1:n,0]=(1./6.)*(np.diff(self.gamma)/self.h)
        #b_i's
        poly_coef[1:n,1]=(1./2.)*(self.gamma[0:(n-1)]-6.*poly_coef[1:n,0]*self.knots[0:n-1])
        #c_1,...,c_n
        poly_coef[1:n,2]=fp[0:(n-1)]-3.*poly_coef[1:n,0]*(self.knots[0:n-1]**2)-2.*poly_coef[1:n,1]*self.knots[0:(n-1)]
        #d_0
        poly_coef[0,3]=self.f[0]-poly_coef[0,2]*self.knots[0]
        #d_{n+1}
        poly_coef[n,3]=self.f[n-1]-poly_coef[n,2]*self.knots[n-1]
        #d_1,...,d_n
        poly_coef[1:n,3]=self.f[0:(n-1)]-poly_coef[1:n,0]*(self.knots[0:n-1]**3)-poly_coef[1:n,1]*(self.knots[0:n-1]**2)-poly_coef[1:n,2]*(self.knots[0:n-1])
        self.poly_coef_=poly_coef  #tested I can reconstuct the NCS with these values

    #given a point, decide which piece of the piecewise polynomial the point belongs to
    def _which_poly(self, z):
        n=self.knots.size
        if(z<=self.knots[0]):
            return(0)
        elif (z>=self.knots[n-1]):
            return(n)
        else:
            return(np.max(np.where(self.knots<z)) + 1)

    #just used this to check piecewise polynomail representation gives the same output as plot function
    def _plot_alt(self, scalar): 
        X=np.array([scalar**3, scalar**2, scalar, 1])
        return(np.dot(X, self.poly_coef_[self._which_poly(scalar),:]))

    #For a scalar, evalue f' at a scalar point. (Note that this requires that polynomial coefficients are already calculated.  If not, then just run _poly_coefs() first.)
    def _plot_fprime(self, scalar):
        X=np.array([scalar**2, scalar, 1])
        which_interval=self._which_poly(scalar)
        coef= self.poly_coef_[which_interval,:]
        n=self.knots.size
        if ( (scalar<self.knots[0]) | (scalar>=self.knots[n-1])):
            return(coef[2])
        else:
            temp=np.array([3,2,1])*coef[0:3]
            return(np.dot(X, temp))

    #A wrapper for _plot_fprime, can accept either a scaler or vector input
    def prime_plot(self, x):
        x=np.array(x)
        #check if piecewise polynomial coefficients exist already.  if not, calculate it.
        try:
            self.poly_coef_            
        except AttributeError:
            self._poly_coefs()

        if(x.size==1):
            return(self._plot_fprime(x))
        else:
            return(np.array(map(self._plot_fprime, x)))


class BASIS:
    def __init__(self, X, num_knots):
        self.num_knots=num_knots
        self.X=np.sort(np.unique(X))
        #maximum number of available knots
        self.n=self.X.size
        if( (num_knots>=self.n) ):
            #if number of knots=sample size, the knots are each data point
            self.knots=self.X
            self.num_knots=self.n
        else:
            #if number of knots<sample size, we choose knots to be evenly spaced percentile
            increment=(100./(num_knots-1.))
            cutpoints=np.zeros(num_knots)
            cutpoints[num_knots-1]=100.
            cutpoints[1:(num_knots-1)]= np.arange(1, num_knots-1)*increment
            self.knots_percentile=cutpoints
            self.knots=np.zeros(num_knots)
            for i in range(0,num_knots):
                self.knots[i]=np.percentile(self.X, cutpoints[i])
            #the following line and if code is a quick and dirty way to due with non-uniqueness
            unique_knots=np.unique(self.knots)
            if(unique_knots.size<num_knots):
                print("reducing number of knots due to non-uniqueness")
                self.knots=np.sort(np.unique(self.knots))
                self.num_knots=self.knots.size

    #d_k(X) function f1rom Element of Statistical Learning eq (5.5)
    def _dk_X(self, X , k):
        a=X-self.knots[k-1]
        b=X-self.knots[self.num_knots-1]
        a[a<0]=0
        b[b<0]=0
        numerator=(a**3)-(b**3)
        denominator=self.knots[self.num_knots-1]-self.knots[k-1]
        return(numerator/denominator)
    
    #Basis using Element of Statistical Learning eq (5.4)
    def predict(self, X):
        basis=np.empty((X.size, self.num_knots))
        basis[:,0]=1
        basis[:,1]=X
        #the following k follows (5.4) in ESL
        d_K_minus_1_X=self._dk_X(X, self.num_knots-1)
        for k in range(1,(self.num_knots-1)):
            basis[:,k+1]=self._dk_X(X, k) - d_K_minus_1_X
        return(basis)



#same as bs() in splines package in R
class bs:
    def __init__(self, X, num_int_knots=15, interior_knots=None, boundary_knots=None, degree=3):
        sorted_x=np.sort(np.unique(X))
        #deal with boundary knots
        self.boundary_knots=np.zeros(2)
        if(boundary_knots is None):
            self.boundary_knots[0]=sorted_x[0]
            self.boundary_knots[1]=sorted_x[sorted_x.size-1]
        else:
#            if(boundary_knots[0]>sorted_x[0]):
#                print("Lower boundary knot must be smaller than min(X), min(X) used instead.")
                self.boundary_knots[0]=sorted_x[0]
#            if(boundary_knots[1]<sorted_x[0]):
#                print("Upper boundary knot must be larger than max(X), max(X) used instead.")
                self.boundary_knots[1]=sorted_x[sorted_x.size-1]
        #deal with interior knots
        if(interior_knots is not None):
            self.interior_knots=interior_knots
        else:
            if(num_int_knots>=sorted_x.size):
                self.interior_knots=sorted_x[1:(sorted_x.size-1)] 
            else:
                increment=(100./(num_int_knots-1.))
                cutpoints= np.arange(1, num_int_knots-1)*increment
                self.knots_percentile=cutpoints
                self.interior_knots=np.zeros(num_int_knots-2)
                for i in range(0,num_int_knots-2):
                    self.interior_knots[i]=np.percentile(sorted_x, cutpoints[i])
                #the following line and if code is a quick and dirty way to due with non-uniqueness
        unique_knots=np.unique(self.interior_knots)
        if(unique_knots.size<(num_int_knots-2)):
            print("reducing number of knots due to non-uniqueness")
            self.interior_knots=np.sort(np.unique(self.interior_knots))
        self.num_knots=self.interior_knots.size+2
        self.knots=np.hstack((self.boundary_knots[0], self.interior_knots, self.boundary_knots[1]))

        #excluding boundary knots
        self.K=self.interior_knots.size
        self.M=degree+1
        self.tau=np.empty(self.K+2*self.M)
        #In theory it is fine as long as tau 1 to M is smaller than the boundary knot (larger than for the upper end).  However, in computation we cannot just fix at a super small number like 10e-20 because it will lead to rounding error.  In order to make sure predict(min(X)) retruns basis (1,0,0,...,0) and predict(max(X)) returns (0,...,0,1), this small number need to depend on data.  I think it depends on the distance between boundary and first interior knots (and boundary and last interior knots for the upper end).  Below "a_small_num" seems to work for variety of data.
        #divided by 10000 was tested to work!!!!!!
        a_small_num1=(self.knots[1]-self.knots[0])/10000.
        a_small_num2=(self.knots[self.num_knots-1]-self.knots[self.num_knots-2])/10000.
        self.tau[0:(self.M)]=self.boundary_knots[0]-a_small_num1
        #self.tau[0:(self.M)]=self.tau[0:(self.M)]-np.array([1,1,1,0])        
        self.tau[(self.K+self.M) : (self.K+2*self.M)]=self.boundary_knots[1]+a_small_num2
#        self.tau[(self.K+self.M) : (self.K+2*self.M)]=self.tau[(self.K+self.M) : (self.K+2*self.M)]+np.array([0,1,1,1])

        self.tau[self.M:self.M+self.K]=self.interior_knots

    def _predict(self, x, M=None):
        if(M is None):
            M=self.M
        B=np.zeros(self.K+2*M-1)
        lowerbound=self.tau[0:self.tau.size-1]
        upperbound=self.tau[1:self.tau.size] 
        #i such that x>=tau_i and x<tau_{i+1}
        condition=(x>=lowerbound) & (x<upperbound)
        if(not all(condition==False)):
            i=np.max(np.where(condition)[0])
            B[i]=1
        for m in range(2,M+1):
            B_previous=B
            B=np.zeros(self.K+2*M-m)
            for i in range(1, self.K+2*M-m+1):
                #B_i_m=0 under if tau_i=tau_{i+1}=...=tau_{i+m}.  i here follow element of staistical learning.
                if(any((self.tau[i:(i+m)]-np.ones(m)*self.tau[i-1])!=0)):
                    front=0
                    back=0
                    if(B_previous[i-1]!=0):
                        front=((x-self.tau[i-1])/(self.tau[i+m-2]-self.tau[i-1]))*B_previous[i-1]
                    if(B_previous[i]!=0):
                        back=((self.tau[i+m-1]-x)/(self.tau[i-1+m]-self.tau[i]))*B_previous[i]
                    B[i-1]=front+back        
        return(B)
                


    def _predict_deriv(self, x, M=None):
        if(M is None):
            M=self.M
        B=np.zeros(self.K+2*M-1)
        lowerbound=self.tau[0:self.tau.size-1]
        upperbound=self.tau[1:self.tau.size] 
        #i such that x>=tau_i and x<tau_{i+1}
        condition=(x>=lowerbound) & (x<upperbound)
        if(not all(condition==False)):
            i=np.max(np.where(condition)[0])
            B[i]=1
        for m in range(2,M+1):
            B_previous=B
            B=np.zeros(self.K+2*M-m)
            for i in range(1, self.K+2*M-m+1):
                #B_i_m=0 under if tau_i=tau_{i+1}=...=tau_{i+m}.  i here follow element of staistical learning.
                if(any((self.tau[i:(i+m)]-np.ones(m)*self.tau[i-1])!=0)):
                    front=0
                    back=0
                    if(m==M):
                        if(B_previous[i-1]!=0):
                            front=(1/(self.tau[i+m-2]-self.tau[i-1]))*B_previous[i-1]
                        if(B_previous[i]!=0):
                            back=(1/(self.tau[i-1+m]-self.tau[i]))*B_previous[i]
                        B[i-1]=(front-back)*(m-1)
                    else:
                        if(B_previous[i-1]!=0):
                            front=((x-self.tau[i-1])/(self.tau[i+m-2]-self.tau[i-1]))*B_previous[i-1]
                        if(B_previous[i]!=0):
                            back=((self.tau[i+m-1]-x)/(self.tau[i-1+m]-self.tau[i]))*B_previous[i]
                        B[i-1]=front+back        
        return(B)

    def _predict_2nd_deriv(self, x, M=None):
        if(M is None):
            M=self.M
        B=np.zeros(self.K+2*M-1)
        lowerbound=self.tau[0:self.tau.size-1]
        upperbound=self.tau[1:self.tau.size] 
        #i such that x>=tau_i and x<tau_{i+1}
        condition=(x>=lowerbound) & (x<upperbound)
        if(not all(condition==False)):
            i=np.max(np.where(condition)[0])
            B[i]=1
        for m in range(2,M):
            B_previous=B
            B=np.zeros(self.K+2*M-m)
            for i in range(1, self.K+2*M-m+1):
                    #B_i_m=0 under if tau_i=tau_{i+1}=...=tau_{i+m}.  i here follow element of staistical learning.
                    if((m==(M-1)) & (i!=self.K+2*M-m)):
                        a=0
                        b=0
                        c=0
                        temp1=(self.tau[i+M-2]-self.tau[i-1])
                        temp2=(self.tau[i+M-3]-self.tau[i-1])
                        temp3=(self.tau[i+M-1]-self.tau[i])
                        temp4=(self.tau[i+M-2]-self.tau[i])
                        temp5=(self.tau[i+M-1]-self.tau[i+1])
                        if((temp1!=0) & (temp2!=0) & (B_previous[i-1]!=0)):
                            a=B_previous[i-1]/(temp1*temp2)
                        if(B_previous[i]!=0):
                            f1=0
                            f2=0
                            if((temp4!=0) & (temp3!=0)):
                                f1=(1/(temp3*temp4))
                            if((temp1!=0) & (temp4!=0)):
                                f2=(1/(temp1*temp4))
                            b=B_previous[i]*(f1+f2)
                        if((temp3!=0) & (temp5!=0) & (B_previous[i+1]!=0)):
                            c=B_previous[i+1]/(temp3*temp5)
                        B[i-1]=(c-b+a)*(M-1)*(M-2)
                    else:
                        if(any((self.tau[i:(i+m)]-np.ones(m)*self.tau[i-1])!=0)):
                            front=0
                            back=0
                            if(B_previous[i-1]!=0):
                                front=((x-self.tau[i-1])/(self.tau[i+m-2]-self.tau[i-1]))*B_previous[i-1]
                            if(B_previous[i]!=0):
                                back=((self.tau[i+m-1]-x)/(self.tau[i-1+m]-self.tau[i]))*B_previous[i]
                            B[i-1]=front+back
        B=np.delete(B, self.K+M)
        return(B)


    def predict(self, X, derivative=0, M=None):
        if(M is None):
            M=self.M
        else:
            print("Only M=4 is tested.  Use at other M at you own risk!")
        X=np.array(X)
        if(any(X<self.boundary_knots[0]) | any(X>self.boundary_knots[1])):
            print("ALL X must be within self.boundary_knots")
            import sys
            sys.exit()
        if(derivative==0):
            def tempfn(i):
                x=X[i]
                return(self._predict(x, M))
        elif (derivative==1):
            def tempfn(i):
                x=X[i]
                return(self._predict_deriv(x, M))
        elif (derivative==2):
            def tempfn(i):
                x=X[i]
                return(self._predict_2nd_deriv(x, M))
        if(X.size==1):
            basis=self._predict_deriv(X, M)
        else:
            basis=map(tempfn, range(0, X.size))
        return(np.array(basis))


    


def cleanDuplicate(X,Y):
    mydictionary={}
    needfix=0
    for i in X:
        if(mydictionary.has_key(i)==False):
            mydictionary[i]=1
        else:
            needfix+=1
            mydictionary[i]+=1
    if(needfix==0):
        return((X,Y))
    else:
        for key in mydictionary.keys():
            if mydictionary[key]>1:
                newy=np.mean(Y[np.where(X==key)])
                tobedelete=np.where(X==key)[0][1:] #delete all but the first appearence
                Y=np.delete(Y, tobedelete)
                X=np.delete(X, tobedelete)
                Y[np.where(X==key)]=newy #replace the y value to the mean of y
        return((X,Y))

