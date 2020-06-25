import numpy as np
from scipy.stats import t

class MultinomialModel():
	"""Class for estimation of Multonimial Logit Models"""
	def __init__(self, random_state=None):
		"""Init Function
		
		Parameters
		----------
		random_state: an integer used as seed to generate random numbers with numpy
		"""
		self.coeff_names = None
		self.coeff_ = np.array([])
		self.stderr = None
		self.zvalues = None
		self.pvalues = None
		self.loglikelihood = None
		self.X = None #numpy array, shape [n_samples, n_alternatives, n_parameters], Design Matrix
		self.y = None #numpy array, shape [n_samples, n_alternatives], Matrix of choices by alternative
		self.convergence = False
		self.max_iterations = 0
		if(random_state != None):
			numpy.random.seed(random_state)


	def fit(self,X,y,alternatives = None,varnames = None,asvars = None,isvars = None,base_alt=None,fit_intercept = True, max_iterations = 2000):
		"""
		Fits multinomial model using the given data parameters. 
		
		Parameters
		----------
		X: numpy array, shape [n_samples, n_features], Data matrix in long format.
		y: numpy array, shape [n_samples, ], Vector of choices or discrete output.
		alternatives: list, List of alternatives names or codes.
		asvars: list, List of alternative specific variables
		isvars: list, List of individual specific variables
		base_alt: string, base alternative. When not specified, pymlogit uses the first alternative
			in alternatives vector by default.
		max_iterations: int, Maximum number of optimization iterations
		fit_intercept: bool
		"""
		self._validate_inputs(X,y,alternatives, varnames, asvars, isvars, base_alt, fit_intercept, max_iterations)
		
		asvars = [] if not asvars else asvars
		isvars = [] if not isvars else isvars
		
		data = X
		labels = y
		self.max_iterations = max_iterations

		J = len(alternatives)
		N = int(data.shape[0]/J)

		if base_alt==None:
			base_alt = alternatives[0]

		if fit_intercept:
		    isvars.insert(0,'_intercept')
		    varnames.insert(0,'_intercept')
		    data = np.hstack((np.ones(J*N)[:,np.newaxis],data)) #Prepend column of 1s to data

		ispos = np.array([varnames.index(i) for i in isvars]) #Position of IS vars
		aspos = np.array([varnames.index(i) for i in asvars]) #Position of AS vars

		#==== Create design matrix
		#For individual specific variables
		#Xis,Xas = np.array([]),np.array([])
		if isvars:
			dummy = np.tile(np.eye(J),reps=(N,1)) #Create a dummy individual specific variables for the alternatives
			dummy = np.delete(dummy, alternatives.index(base_alt), axis=1) #Remove base alternative
			Xis = data[:,ispos]
			Xis = np.einsum('ij,ik->ijk',Xis,dummy) #Multiply dummy representation by the individual specific data
			Xis = Xis.reshape(N,J,(J-1)*len(ispos))

		#For alternative specific variables
		if asvars:
			Xas = data[:,aspos] 
			Xas = Xas.reshape(N,J,-1)

		#Set design matrix based on existance of asvars and isvars
		if asvars and isvars:
			X = np.dstack((Xis,Xas))
		elif asvars:
			X = Xas
		elif isvars:
			X = Xis
		
		self.X = X #Design matrix   , shape [n_samples, n_alternatives, n_parameters]
		self.y = labels.reshape(N,J)  #Discrete output , shape [n_samples, n_alternatives]

		self.coeff_names = [isvar+"."+j for isvar in isvars for j in alternatives if j != base_alt ]
		self.coeff_names.extend(asvars)
		betas = np.zeros(len(self.coeff_names))
		
		#Call optimization routine
		convergence,betas,ll,g,Hinv,total_iter = self.bfgs_optimization(betas)

		#Save results
		self.convergence = convergence
		self.coeff_ = betas
		self.stderr = np.sqrt(np.diag(Hinv))
		self.zvalues = betas/self.stderr
		self.pvalues = 2*t.pdf(-np.abs(self.zvalues),df=N) #two tailed test
		self.loglikelihood = -ll
		if convergence:
			print("Optimization succesfully completed after "+str(total_iter)+" iterations. Use .summary() to see the estimated values")
		else:
			print("**** Maximum number of iterations reached. The optimization did not converge")

	def _validate_inputs(self,X,y,alternatives,varnames,asvars,isvars,base_alt,fit_intercept,max_iterations):
		if not varnames:
			raise ValueError('The parameter varnames is required')
		if not alternatives:
			raise ValueError('The parameter alternatives is required')
		if not asvars and not isvars:
			raise ValueError('Either isvars or asvars must be passed as parameter to the fit function')
		if X.ndim != 2:
			raise ValueError('X must be an array of two dimensions in long format')
		if y.ndim != 1:
			raise ValueError('y must be an array of one dimension in long format')
		if len(varnames) != X.shape[1]:
			raise ValueError('The length of varnames must match the number of columns in X')


	def log_lik(self,B):
		"""
		Computes the log likelihood of the parameters B with self.X and self.y data
		
		Parameters
		----------
			B: numpy array, shape [n_parameters, ], Vector of betas or parameters

		Returns
		----------
			ll: float, Optimal value of log likelihood (negative)
			g: numpy array, shape[n_parameters], Vector of individual gradients (negative) 
			Hinv: numpy array, shape [n_parameters, n_parameters] Inverse of the approx. hessian
		"""
		X = self.X
		y = self.y
		XB = X.dot(B)
		eXB =np.exp(XB)
		p = eXB/np.sum(eXB,axis=1, keepdims = True)
		#Log likelihood
		l =  np.sum(y*p,axis = 1)
		ll= np.sum(np.log(l))
		#Gradient
		g_i = np.einsum('nj,njk -> nk', (y-p) , X ) #Individual contribution to the gradient

		H = np.dot(g_i.T,g_i)
		Hinv = np.linalg.inv(H)
		g = np.sum(g_i,axis=0)
		return (-ll,-g, Hinv)

	def bfgs_optimization(self,B):
		"""
		Performs the BFGS optimization routine. For more information in this newton-based 
		optimization technique see: http://aria42.com/blog/2014/12/understanding-lbfgs
		
		Parameters
		----------
			B: numpy array, shape [n_parameters, ], Vector of betas or parameters

		Returns
		----------
			B: numpy array, shape [n_parameters, ], Optimized parameters
			res: float, Optimal value of optimization function (log likelihood in this case)
			g: numpy array, shape[n_parameters], Vector of individual gradients (negative) 
			Hinv: numpy array, shape [n_parameters, n_parameters] Inverse of the approx. hessian
			convergence: bool, True when optimization converges
			current_iteration: int, Iteration when convergence was reached
		"""
		res,g,Hinv = self.log_lik(B)
		current_iteration = 0
		convergence = False
		while True:
			old_g = g

			d = -Hinv.dot(g) 

			step = 2
			while True:
				step = step/2
				s = step*d 
				B = B + s
				resnew,gnew,Hinv2 = self.log_lik(B)
				if resnew <= res or step < 1e-10:
					break

			old_res = res
			res = resnew    
			g = gnew
			delta_g = g - old_g 

			Hinv = Hinv + ( ( ( s.dot(delta_g) + (delta_g[None,:].dot(Hinv)).dot(delta_g) )*np.outer(s,s) ) / (s.dot(delta_g))**2 ) - ( (np.outer(Hinv.dot(delta_g),s) + (np.outer(s,delta_g)).dot(Hinv)) / (s.dot(delta_g)) )
			current_iteration = current_iteration + 1
			if np.abs(res - old_res) < 0.00001:
				convergence = True
				break
			if current_iteration > self.max_iterations:
				convergence = False
				break

		return (convergence,B,res,g, Hinv,current_iteration)

	def summary(self):
		"""
		Prints in console the coefficients and additional estimation outputs
		"""
		if not np.any(self.coeff_):
			print('The current model has not been yet estimated')
			return
		if not self.convergence:
			print('***********************************************************************************************')
			print('WARNING: Convergence was not reached during estimation. The given estimates may not be reliable')
			print('***********************************************************************************************')
		print("-----------------------------------------------------------------------------------------")
		print("Coefficient          \tEstimate \tStd. Error \tz-value \tPr(>|z|)     ")
		print("-----------------------------------------------------------------------------------------")
		fmt = "{:16.22} \t{:0.10f} \t{:0.10f} \t{:0.10f} \t{:0.10f} {:5}"
		for i in range(len(self.coeff_)):
			signif = ""
			if self.pvalues[i] < 1e-15: 
				signif = '***'
			elif self.pvalues[i] < 0.001: 
				signif = '**'
			elif self.pvalues[i] < 0.01: 
				signif = '*'
			elif self.pvalues[i] < 0.05: 
				signif = '.'
			print(fmt.format(self.coeff_names[i],self.coeff_[i],self.stderr[i],self.zvalues[i],self.pvalues[i],signif))
		print("-----------------------------------------------------------------------------------------")
		print('Significance:  *** 0    ** 0.001    * 0.01    . 0.05')
		print('')
		print('Log-Likelihood= %.3f'%(self.loglikelihood))

