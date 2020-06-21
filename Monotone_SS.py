import numpy as np;
import sys
import NCS as NCScode;

class NullWriter(object):
    def write(self, arg):
        pass
    def flush(self):
        pass
#the above class and following 3 lines and the try finally are to disable annoying messange from NLP
nullwrite = NullWriter()
oldstdout = sys.stdout
sys.stdout = nullwrite # disable output
try:
    from openopt import NLP;
finally:
    sys.stdout = oldstdout # enable output




class Monotone_SS(NCScode.NCS):
    def __init__(self, constrains_sign=[0,0], alpha=[], samplesize=15, iprint=-1):
        self.samplesize=samplesize
        self.iprint=iprint
        self.alpha=np.array(alpha)
        self.constrains_sign=constrains_sign

    def get_params(self, deep=True):
        return({"alpha":self.alpha})

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

    def fit(self, X, Y):
        X,Y=NCScode.cleanDuplicate(X,Y)
        #order z
        zorder=X.ravel().argsort()
        z=X[zorder]
        y=Y[zorder]

        if(self.samplesize>=z.size):
            print("No approximation")
            #for ns in R, when df= num_data-1, it's the same as having knots at each data point.
            self.samplesize=z.size-1
#######################################################################
#NCS basis version
#######################################################################
        '''
        #create basis and knots.  Knots based on percentile.
        B=NCScode.BASIS(z, self.samplesize)
        '''
#########################################################################


#########################################################################
#basis spline version
########################################################################
        #create basis and knots.  Knots based on percentile.
        B=NCScode.bs(X=z, num_int_knots=self.samplesize)
##########################################################################

        #create the basis matrix
        BASIS=B.predict(z)
        
        #get K matrix given our data point z
        K=NCScode.NCS(z, np.zeros(z.size))._K

        #initial guess for g with a flat line of mean(Y)
        initial_guess=np.zeros(BASIS.shape[1])

        #optimization
        args=(z, y, BASIS, K, self.alpha, self.constrains_sign);
        if(self.constrains_sign==[0,0]):
            InversePart=np.linalg.inv(np.dot(BASIS.transpose(), BASIS)+self.alpha*BASIS.shape[0]*np.dot(np.dot(BASIS.transpose(), K), BASIS))
            theta = InversePart.dot(np.dot(BASIS.transpose(), y))
        else:
            problem=NLP(f=_obj,x0=initial_guess, args=args, df=_obj_prime, c=_constrains_gen, iprint = self.iprint);
            tt=problem.solve('ralg')
            theta = tt.xf
        #calculate function value at knots
        g=np.dot(BASIS, theta)
            
        #Because the smoothing spline is defined by knots and function value at knots, so all we need to initiate the NCS is knots and g. This is the resulted model.   Beware we should not initiate by z and its corresponding function value because we created the basis using based on knots.
        #Using of super helps make sure Monotone SS inherent superclass NCS class.         
        super(Monotone_SS, self).__init__(z, g)
        
        #print warning if derivative of f is not all of the correct sign
        constrain_check(NCScode.NCS(z, g), z, self.constrains_sign, rounding=5)
        return self;

    def predict(self, z):
        return(self.plot(z))
    
    def CV(self, X, Y, alpha_seq, n_folds=5, refit=True, n_cores=4, verbose=1):
        import time
        #It's important that X is not ordered.  For example, 2 fold CV will just that the first half of X as one fold and the remaining as second fold, if X is ordered, we just get the left half data.  To avoid this, I shuffle to data randomly.
        a=np.arange(0, Y.size)
        np.random.shuffle(a)
        z=X[a]
        y=Y[a]
        if(verbose>0):
            print(str(n_folds)+"-fold CV begins")
        start_time = time.time()

        #following 3 lines for for supressing annoying openopt message
        nullwrite = NullWriter()
        oldstdout = sys.stdout
        sys.stdout = nullwrite # disable output
        
        try:
            if(n_cores==1):
                self._CV_sklearn(z, y, alpha_seq, n_folds=n_folds, n_cores=n_cores, verbose=verbose)
            elif(n_cores>1):
                self._CV_pathos(z, y, alpha_seq, n_folds=n_folds, n_cores=n_cores, verbose=verbose)
        except:
            print("Error, trying 1 core version")
            a=np.arange(0, Y.size)
            np.random.shuffle(a)
            z=X[a]
            y=Y[a]
            self._CV_sklearn(z, y, alpha_seq, n_folds=n_folds, n_cores=1, verbose=verbose)
        finally:
            sys.stdout = oldstdout # enable output
        elapsed_time = (time.time() - start_time)/60.
        if(verbose>0):
            print("CV completed - time elapsed: "+str(round(elapsed_time, 2))+"Min")
        if(refit==True):
            if(verbose>0):
                print("Using optimal tuning parameter to refit full data")
            self.fit(X,Y)
    
    #hidden function for CV when only one core is wanted.
    def _CV_sklearn(self, z, y, alpha_seq, n_folds=5, n_cores=1, verbose=1):
        from sklearn import grid_search

        #During initialization, Monotone_SS is initialized with default parameter.  Passing the following "parameters" allow initialization with the following parameters.
        parameters={'alpha':alpha_seq, 'samplesize':[self.samplesize], 'iprint':[self.iprint], 'constrains_sign':[self.constrains_sign]}

        model=Monotone_SS()
        clf = grid_search.GridSearchCV(model, parameters, scoring="mean_squared_error", cv=n_folds, n_jobs=1, refit=False, verbose=verbose)
        clf.fit(z,y)

        self.alpha=clf.best_params_['alpha']
        self.CV_results_=clf

    #hidden function for CV when multiple core is wanted.
    def _CV_pathos(self, z, y, alpha_seq, n_folds=5, n_cores=2, verbose=1):
        from sklearn.cross_validation import KFold
        from multiprocessing import Pool

        kf = KFold(len(z), n_folds=n_folds, indices=False)
        
        result=np.empty((n_folds, len(alpha_seq)))
        i=0
        for train, test in kf:
            X_train, X_test, y_train, y_test = z[train], z[test], y[train], y[test]
            #second argument in mywrapper
            param=[alpha_seq, self.constrains_sign, self.samplesize, self.iprint, X_train, y_train, X_test, y_test]
            #create a list of input.
            myinput=[(a, b) for a in range(0,len(alpha_seq)) for b in [param]]
            p=Pool(n_cores)
            result[i]=list(p.map(my_wrapper,  myinput )) 
            i=i+1
        ave_score=np.mean(result, axis=0)
        print(ave_score)
        self.alpha=np.array(alpha_seq)[ave_score==np.min(ave_score)]
        self.CV_results_=result








#$f(x)=B(x)\theta$ where $B(x)$ is basis created using $x$, and $\theta$ is coefficients to be optimized
#let g1=(g1_1,...,g1_n)=(f(x_1), ..., f(x_n))
#so objective is \frac{1}{n} \sum \limits_{i=1}^N (y_i - g1_i)^2 + lambda_1 g1^T K g1 
def _obj(theta, *args):
    knots, y, BASIS, K, lambda1, sign = args;
    #g1 is for the error in loss function, g2 is for the penalty term.
    g1=np.dot(BASIS, theta)
    error=y-g1
    result=(np.dot(error, error)/y.size)+lambda1*np.dot(np.dot(g1.transpose(), K), g1)
    return(result);

#let objective be $L1(\theta)=\frac{1}{n} \sum \limits_{i=1}^N (y_i - g1_i)^2$ and $L2(\theta)=lambda_1 g1^T K g1$
#deriveative of objective w.r.t theta $\frac{dL}{d\theta}=
#\frac{dg1}{d\theta}\frac{dL1}{dg1}+\frac{dg2}{d\theta}\frac{dL2}{dg2}$
# where $\frac{dg1}{d\theta}=\frac{dg2}{d\theta}=B(x)^T$ and $frac{dL1}{dg1}=((-2y+2g1)/N)$ and frac{dL2}{dg2}=2lambda_1 Kg2$
def _obj_prime(theta, *args):
    knots, y, BASIS, K, lambda1, sign = args;
    #g1 is for the error in loss function, g2 is for the penalty term.
    g1=np.dot(BASIS, theta)
    result=np.dot(BASIS.transpose(), 2*((g1-y)/y.size))+np.dot(BASIS.transpose(), 2*lambda1*np.dot(K,g1) )
    return(result);

def _constrains_gen(theta, *args):
    knots, y, BASIS, K, lambda1, sign = args;
    n=knots.size
    
    g=np.dot(BASIS, theta) 
    #obtain NCS with knots and f(z)
    myncs=NCScode.NCS(knots, g);

    #read in constrains
    monotonicitySign=sign[0]
    convexitySignList=sign[1]
    #convert simple constraint to complex constraint.  Simply means it's a scaler of -1, 0, or 1 indicating the entire function is constrain to concave, no constraint or convex, respectively.  Or complex if it is a list indicating the sign of f'' value at each knot.
    if(isinstance(convexitySignList, list)==False):
        convexitySignList=[convexitySignList]*myncs.gamma.size
    noConvexityConstrain=all(e==0 for e in convexitySignList)

    #for the case of no constraint at all
    if ((monotonicitySign==0)&(noConvexityConstrain)):
        #no any kind of constraint, done
        return(-1);
    #if there is any sort of constrain, handle in below code
    else:
        #for convexity constrain
        if(noConvexityConstrain):
            convexConstrain=np.array([])
        else:
            convexitySignList=np.array(convexitySignList)
            gamma=myncs.gamma
            convexConstrain=-np.ones(gamma.size)

            #if constrain is satisfied, sign of gamma should be the same as sign of constrain, so product of their sign is positive.  The negative ones violate constrain
            violationLocation=((np.sign(convexitySignList)*np.sign(gamma))==-1)
            safeLocation=((np.sign(convexitySignList)*np.sign(gamma))==1)
            #for those violation position, make the gamma value positive
            if(np.sum(violationLocation)>0):
                convexConstrain[violationLocation]=np.abs(gamma[violationLocation])     
            #for those ok location, make the gamma value negative
            if(np.sum(safeLocation)>0):
                convexConstrain[safeLocation]=-np.abs(gamma[safeLocation])  
            convexConstrain=convexConstrain[1:(gamma.size-1)]

        #for monotone constrain
        if(monotonicitySign==0):
            monotoneConstrain=np.array([])
        else:
            #matrix of polynomial coefficients a and b in polynomial a x^3 + b x^2 + c x + d, first columns correponds to a. We need not worry about linear part beyond the boundary, so we only take row 1 to n-1, correponds to all the interior polynomial.
            myncs._poly_coefs()
            poly_coef_mat=myncs.poly_coef_[1:n,0:2]
            non_zero_denominator_loc=poly_coef_mat[:,0]!=0
            functional_min=np.ones(poly_coef_mat.shape[0])*float('inf')
            functional_min[non_zero_denominator_loc]=(-poly_coef_mat[non_zero_denominator_loc,1]/(3*poly_coef_mat[non_zero_denominator_loc,0]))
            
            #check if the minimum lies inside the interval (if not, don't bother to add that as constrain)
            lower=knots[0:n-1]
            upper=knots[1:n]
            #boolean of wheter stationary point in the corresponding interval
            in_interval=(functional_min>lower)&(functional_min<upper)
            #those points position
            stationary_pt=functional_min[in_interval]

            #calculate f' value at boundary
            fp_at_knots=myncs.prime_plot(knots)
            lower_fp=fp_at_knots[0:n-1]
            upper_fp=fp_at_knots[1:n]
            #create n-1 by 2 matrix, each row is the function value at left and round boundary
            fval_at_knots=np.vstack((lower_fp, upper_fp)).transpose()
            fval_at_stationary_pt=myncs.prime_plot(stationary_pt)

            #flipped sign for NLP
            if(monotonicitySign==1):
                #all_constrains is c(x), we need c(x)>=0, but optimization method only does c(x)<=0, so we flip sign to get c(x)<=0
                #n-1 vector contraining minimum of each row in fval_at_knots
                monotoneConstrain=np.min(fval_at_knots, axis=1)
                #for those stationary point in the interval, we also need to include them, because we already have minimum of the boundary, we only need to compare the f'(stationary point) and the all_constrains above (which constains the minimum at boundary)
                temp=np.vstack((monotoneConstrain[in_interval], fval_at_stationary_pt)).transpose()
                monotoneConstrain[in_interval]=np.min(temp, axis=1)
                monotoneConstrain=-monotoneConstrain;
            elif (monotonicitySign==-1):
                #all_constrains is c(x), optimization method does c(x)<=0
                #n-1 vector contraining maximum of each row in fval_at_knots
                monotoneConstrain=np.max(fval_at_knots, axis=1)
                #for those stationary point in the interval, we also need to include them, because we already have minimum of the boundary, we only need to compare the f'(stationary point) and the all_constrains above (which constains the minimum at boundary)
                temp=np.vstack((monotoneConstrain[in_interval], fval_at_stationary_pt)).transpose()
                monotoneConstrain[in_interval]=np.max(temp, axis=1)
        
        #no constrain
        all_constrains=np.hstack((monotoneConstrain, convexConstrain))
        return(all_constrains)


#a wrapper for multicore version of cross validation.  it need to be completely self contained, and only take one argument
def my_wrapper(args):
    import numpy as np
    def cv_score(truey, estimatedy):
        error=truey-estimatedy
        return(np.dot(error,error)/error.size)
    def tempfn(i, param):
        from Monotone_SS import Monotone_SS
        #load in all infomation need
        alpha_seq=param[0]
        constrains_sign=param[1]
        samplesize=param[2]
        iprint=param[3]
        X_train=param[4]
        y_train=param[5]
        X_test=param[6]
        y_test=param[7]
        #fit model
        model=Monotone_SS(alpha=alpha_seq[i],  constrains_sign=constrains_sign, samplesize=samplesize, iprint=iprint)
        model.fit(X_train, y_train)
        predictedy=model.predict(X_test)
        return(cv_score(y_test, predictedy))
    return(tempfn(*args))



def constrain_check(NCS_obj, knots, constrains_sign, rounding=5):
    monotonicitySign=constrains_sign[0]
    convexitySign=constrains_sign[1]
    check_points=np.random.uniform(knots[0], knots[knots.size-1], 1000)
    #print warning if derivative of f is not all of the correct sign
    if(monotonicitySign==1):
        if(np.any(np.round(NCS_obj.prime_plot(check_points), rounding)<0)):
            print("Problem: Function is not monotone increasing!");
    elif(monotonicitySign==-1):
        if(np.any(np.round(NCS_obj.prime_plot(check_points), rounding)>0)):
            print("Problem: Function is not monotone decreasing!");
    elif(convexitySign==2):
        if(np.any(np.round(NCS_obj.primeprime_plot(check_points), rounding)<0)):
            print("Problem: Function is not convex!");
    elif(convexitySign==-2):
        if(np.any(np.round(NCS_obj.primeprime_plot(check_points), rounding)>0)):
            print("Problem: Function is not concave!");


