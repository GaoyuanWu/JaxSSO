"""
A optimization module for JAX-SSO.
Available optimizers:
    1. Vanilla gradient descent for unconstrained problems
    2. SLSQP for constrained problems from Nlopt (Install Nlopt: pip install nlopt) [https://nlopt.readthedocs.io/en/latest/NLopt_Python_Reference]
"""


import numpy as np
#import nlopt

class Optimization():
    """
    A class for optimization
    """

    def __init__(self,n_x,method='GD'):
        """
        Initialize an Optimization() object.
        
        Parameters:
            method: str
                'GD': vanilla gradient descent, default
                'SLSQP': sequential least-squares quadratic programming:
                    Reference:Dieter Kraft, "A software package for sequential quadratic programming", Technical Report DFVLR-FB 88-28, Institut für Dynamik der Flugsysteme, Oberpfaffenhofen, July 1988.
               
            n_x: int
                number of parameters to optimize        
        """
        
        self.method = method      # Optimizer
        self.n_x = n_x   # Number of parameters

        self.if_eq_c = False # if there is equality constraint
        self.if_ineq_c = False #if there is inequality constraint
        self.if_bounds = False #if there are bounds for parameters
        self.if_maxiter = False # if there is stop criteria: Number of maximum function evaluation
        self.if_ftol_rel = False # if there is stop criteria: Relative tolerance
        self.if_ftol_abs = False # if there is stop criteria: Absolute tolerance
        self.GD_norm = False #if the gradients are normalized

    def set_objective(self,fun):
        """
        The objective function to minimize, and its gradient

        Parameters:
        -----
        fun: callable 
            fun(x, *args) -> tuple, (fun_value,grad_fun)
                fun_value: float
                    the objective value
        
                grad_fun: ndarry of shape (n_x,)
                    the gradient of the objective

        """   
        self.fun = fun #the objective function

    def set_maxiter(self,maxiter):
        """
        Set the maximum number of iterations

        Parameters:
        -----
        maxiter: int 
        """
        self.if_maxiter = True
        self.maxiter = maxiter #set maxiter

    def set_step_size(self,ln_rate):
        """
        Set the step size for gradient descent

        Parameters:
        -----
        ln_rate: float
        """
        
        if self.method == 'GD':
            self.ln_rate = ln_rate

    def set_GD_normalized(self,normalized=False):
        """
        If set to 'True', the gradient will be normalized by the max(abs) of the gradient.

        Parameters:
        -----
        ln_rate: float
        """
        
        if normalized:
            self.GD_norm = True
        else:
            self.GD_norm = False
    
    def set_ftol_rel(self,ftol_rel):
        """
        Stop criteria: relative function value

        Parameters:
        -----
        ftol_rel: float
        """
        self.if_ftol_rel = True
        self.ftol_rel = ftol_rel

    def set_ftol_abs(self,ftol_abs):
        """
        Stop criteria: relative function value

        Parameters:
        -----
        ftol_abs: float
        """
        self.if_ftol_abs = True
        self.ftol_abs = ftol_abs

    def set_bounds(self,x_lb,x_ub):
        """
        Setting bounds for parameters

        Parameters:
        -----
        x_lb: ndarray
            lower bounds
        
        x_ub: ndarray
            upper bounds
        """
        if self.method == 'GD':
            raise TypeError(" Gradient descent cannot deal with bounds, change \'method\'")
        else:
            self.if_bounds = True
            self.lb = x_lb
            self.ub = x_ub


    def set_inequality_constraints(self,fc,grad_fc,d_fc,tol=0):
        """
        Setting inequality constraints

        Parameters:
        -----
        fc: callable 
            fc(x, *args) -> float/ndarray of shape (d_fc,); function for constraints; fc(x) <= 0

        grad_fc: callable 
            grad_fc(x, *args) -> ndarray of shape (d_fc,n_x); gradients of constraints
        
        d_fc: int
            shape of the return of fc(x, *args); dimension of the constraints

        tol:float/ndarray of shape (d_fc,)
            tolerence for constraints; default is zero

        """
        if self.method == 'GD':
            raise TypeError(" Gradient descent cannot deal with constraints, change \'method\'")
        
        else:
            self.if_ineq_c = True
            self.ineq_fc = fc
            self.ineq_grad_fc = grad_fc
            self.d_ineq_c = d_fc
            if self.d_ineq_c > 1:
                if np.size(tol) == 1:
                    self.ineq_tol = tol * np.ones(self.d_ineq_c) 
                else:
                    self.ineq_tol = tol
            elif self.d_ineq_c == 1:
                self.ineq_tol = tol

    def set_equality_constraints(self,fc,grad_fc,d_fc,tol=0):
        """
        Setting equality constraints

        Parameters:
        -----
        fc: callable 
            fc(x, *args) -> float/ndarray of shape (d_fc,); function for constraints; fc(x) = 0

        grad_fc: callable 
            grad_fc(x, *args) -> ndarray of shape (d_fc,n_x); gradients of constraints
        
        d_fc: int
            shape of the return of fc(x, *args); dimension of the constraints

        tol:float/ndarray of shape (d_fc,)
            tolerence for constraints; default is zero

        """
        if method == 'GD':
            raise TypeError(" Gradient descent cannot deal with constraints, change \'method\'")
        else:
            self.if_eq_c = True 
            self.eq_fc = fc
            self.eq_grad_fc = grad_fc
            self.d_eq_c = d_fc
            if self.d_eq_c > 1:
                if np.size(tol) == 1:
                    self.eq_tol = tol * np.ones(self.d_eq_c) 
                else:
                    self.eq_tol = tol
            elif self.d_eq_c == 1:
                self.eq_tol = tol

    def Nlopt_objective(self):
        '''
        The objective function for Nlopt package.
        
        --------------------
        Form required by Nlopt
        --------------------
        def f(x, grad):
            if grad.size > 0:
               ...set grad to gradient, in-place...
               return ...value of f(x)...
        --------------------
        '''

        def f(x,grad):
            res_1,res_2 = self.fun(x)
            if grad.size>0:
                grad[:] = res_2

            return res_1


        return f

    def Nlopt_eq_constraints(self):
        '''
        The equality constraints for Nlopt package. Single-valued.
        
        --------------------
        Form required by Nlopt
        --------------------
        def f(x, grad):
            if grad.size > 0:
               ...set grad to gradient, in-place...
               return ...value of f(x)...
        --------------------
        '''

        #function for Nlopt
        def f(x,grad):
            if grad.size>0:
                grad[:] = self.eq_grad_fc(x)
            return self.eq_fc(x)

        return f

    def Nlopt_eq_mconstraints(self):
        '''
        The equality constraints for Nlopt package. Vector-valued.
        
        --------------------
        Form required by Nlopt
        --------------------
        def f(x, grad):
            if grad.size > 0:
               ...set grad to gradient, in-place...
               return ...value of f(x)...
        --------------------
        '''

        #function for Nlopt
        def c(result,x,grad):
            if grad.size>0:
                grad[:] = self.eq_grad_fc(x)
                result[:] = self.eq_fc(x)
            

        return c

    def Nlopt_in_constraints(self):
        '''
        The inequality constraints for Nlopt package.Single value.
        
        --------------------
        Form required by Nlopt
        --------------------
        def f(x, grad):
            if grad.size > 0:
               ...set grad to gradient, in-place...
               return ...value of f(x)...
        --------------------
        '''

        #function for Nlopt
        def f(x,grad):
            if grad.size>0:
                grad[:] = self.ineq_grad_fc(x)
            return self.ineq_fc(x)

        return f

    def Nlopt_in_mconstraints(self):
        '''
        The inequality constraints for Nlopt package. Vector-valued.
        
        --------------------
        Form required by Nlopt
        --------------------
        def f(x, grad):
            if grad.size > 0:
               ...set grad to gradient, in-place...
               return ...value of f(x)...
        --------------------
        '''

        #function for Nlopt
        def c(result,x,grad):
            if grad.size>0:
                grad[:] = self.ineq_grad_fc(x)
                result[:] = self.ineq_fc(x)
            

        return c

    #Conduct optimization
    def optimize(self,x_ini,log=True):
        '''
        Conduct optimization, given initial guess.
        
        Parameters:
        -----
        x_ini: ndarray
            initial guess of optimization parameters, shape of (n_x,)

        log: bool
            print log at each functional call

        Return:
        -----
        x_fin: ndarray
            optimized parameters
        --------------------
        '''

        #Vanilla gradient descent with/without normalized gradients
        if self.method == 'GD':
            i = 0 #initial iteration number
            current_x = x_ini #initial parameters
            f_store = [] # store the functional value
            while i < self.maxiter:
                obj,grad_obj = self.fun(current_x) #functional value and its gradient
                f_store.append (obj) #store the functional value
                normalized_grad = grad_obj #original gradient
                if self.GD_norm == True:
                    normalized_grad = normalized_grad/np.max(np.abs(normalized_grad)) #normalize the gradients
                if i>=1:
                    if self.if_ftol_rel==True:
                        if abs((f_store[i-1] - f_store[i])/f_store[i-1]) <= self.ftol_rel:
                            print('Stopping criteria \'ftol_rel\' met, end of optimization')
                            
                            break
                    if self.if_ftol_abs==True:
                        if abs(f_store[i-1] - f_store[i]) <= self.ftol_abs:
                            print('Stopping criteria \'ftol_abs\' met, end of optimization')
                            break

                
                current_x = current_x - self.ln_rate * normalized_grad #update parameters
                if log ==True:
                    print('Step {}, objective = {}'.format(i,f_store[i]))
                i += 1 #update iteration number

            x_fin = current_x #store the final value
            
            if i==self.maxiter:
                print('Stopping criteria \'maxiter\' met, end of optimization')

        #SLSQP, a SQP method
        elif self.method == 'SLSQP':
            opt = nlopt.opt(nlopt.LD_SLSQP,self.n_x) # opt objects

            #The objective function

            opt.set_min_objective(self.Nlopt_objective()) #objective function
            
            #bounds
            if self.if_bounds == True:
                opt.set_lower_bounds(self.lb)
                opt.set_upper_bounds(self.ub)

            #equality constraints
            if self.if_eq_c == True:
                if self.d_eq_c == 0:
                    opt.add_equality_constraint(self.Nlopt_eq_constraints(),self.eq_tol)
                else:
                    opt.add_equality_mconstraint(self.Nlopt_eq_mconstraints(),self.eq_tol)

            #inequality constraints
            if self.if_ineq_c == True:
                if self.d_ineq_c == 0:
                    opt.add_inequality_constraint(self.Nlopt_in_constraints(),self.ineq_tol)
                else:
                    opt.add_inequality_mconstraint(self.Nlopt_in_mconstraints(),self.ineq_tol)

            #stopping criteria
            if self.if_maxiter:
                opt.set_maxeval(self.maxiter)
            if self.if_ftol_abs:
                opt.set_ftol_abs(self.set_ftol_abs) 
            if self.if_ftol_rel:
                opt.set_ftol_rel(self.set_ftol_rel)

            #perform optimization
            x_fin = opt.optimize(x_ini)

        return x_fin

                




