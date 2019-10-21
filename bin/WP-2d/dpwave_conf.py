def find(x):
	x = np.array(x)  # to ensure logical indexing works
	t = np.arange(len(x))
	return t[x != 0]


class conf:
	"""
	Port of bin/WP-2d/dpwave_conf.m
	"""

	def __init__(self, N, epsilon=1e-6, cplx_flag='complex'):
		self.N = N
		self.eps = epsilon
		self.cplx_flag = cplx_flag

		a0 = 8
		kend = np.log2(self.N/a0)

		N_nu = np.zeros(int(kend), dtype=int)
		N_nu[:2] = [1, 32]
		N_nu[2:] = np.floor(N_nu[1] *
			np.sqrt(2)**(np.arange(2, kend) - 1) / 4).astype(int)*4

		K1 = np.zeros(kend)
		K2 = np.zeros(kend)
		Theta_len = np.zeros(kend)

		for i in range(1, kend):
			K1[i] = a0*2**(k-1)
			##! Why is Theta_len calculated here if not used?
			# the line below is the original
			# Theta_len = 2*pi./N_nu;  # step in
			# but the way it is being used it should be
			Theta_len[k] = 2*pi./N_nu[k]
			K2[i] = np.floor(2*K1[k]*sin(2*pi/N_nu[k]));

		boxidx = np.zeros((sum(N_nu), 4))
		boxsize = np.zeros((kend, 2))
		oversamp = np.zeros((kend, 2))
		bc = 1

		self.xi1 = []; self.xi2 = []; xi_idx = []; w = []; K1low = np.zeros(kend);
		self.chi = [0, ]*kend

		for k in range(kend):
			scaletype = 'coarse' if k == 0 else 'detail'

			K1low[k] = ceil(K1[k]/4.*cos(8*pi/N_nu[k]))*2;
			if (k == 0) or (k == kend-1):
				# ! why is only K1[1] being used here?
    			[eta1, eta2] = np.meshgrid(
    			    np.linspace(-K1[1], K1[1], 128), np.linspace(-K1[1], K1[1], 128))
    			West = np.fftshift(np.ifft2(np.fftshift(
    			    dpwave_window(eta1, eta2, 2*K1[1], N_nu[k], scaletype))))
    			West = West/np.max(abs(West[:]))
    			WestBW = np.abs(West) > epsilon
			    box_size[k, 0] = 2*K1[1]
			    box_size[k, 1] = 2*K1[1]  # ! why are both K1[1]?

			    # find finds the indices of non-zero elements
    			oversamp[k, 0] = (box_size[k, 0]+2*np.floor((np.max(find(np.sum(WestBW, 0))
    			                  )-np.min(find(np.sum(WestBW, 0))))/2))/box_size[k, 0]

    			# ! why calculate oversamp[k,1] if it is not to be used here?
    			oversamp[k, 1] = (box_size[k, 1]+2*np.floor((np.max(find(np.sum(WestBW, 1))
    			                  )-np.min(find(np.sum(WestBW, 1))))/2))/box_size[k, 1]
			    [eta1, eta2] = np.meshgrid(-K1[1]: 1/oversamp[k, 0]: K1[1]-1, -K1[1]: 1/oversamp[k, 0]: K1[1]-1)
			    self.chi[k] = dpwave_window(
			        eta1, eta2, 2*K1[1], N_nu[k], scaletype)/sqrt(prod(oversamp[k, :]))

			    if strcmpi(cplx_flag, 'real'):
			    	conf.chi{k} = conf.chi{k}/sqrt(2)
		    else
	    		[eta1,eta2]=np.meshgrid(np.linspace(K1low[k],2*K1[k],16),np.linspace(-K2[k],K2[k],16));
	    		West=np.fftshift(np.ifft2(np.fftshift(dpwave_window(eta1,eta2,2*K1[k],N_nu[k],'detail'))));
	    		West=West/np.max(np.abs(West[:]));WestBW=np.abs(West)>epsilon;
			    box_size[k,0]=2*K1[k]-K1low[k];
			    box_size[k,1]=2*K2[k];
			    oversamp[k,0]=(box_size[k,0]+2*np.floor((np.max(find(np.sum(WestBW,0)))-np.min(find(np.sum(WestBW,0))))/2))/box_size[k,0];
			    oversamp[k,1]=(box_size[k,1]+2*np.floor((np.max(find(np.sum(WestBW,1)))-np.min(find(np.sum(WestBW,1))))/2))/box_size[k,1];
			    [eta1,eta2]=np.meshgrid(
			    	np.linspace(
			    		K1low[k]+1/oversamp[k,0],2*K1[k],np.floor(oversamp[k,1]*box_size[k,1])),
			    	np.linspace(
			    		-K2[k],K2[k]-1/oversamp[k,1],round(oversamp[k,1]*box_size[k,1]))
			    	)
			    conf.chi[k]=dpwave_window(eta1,eta2,2*K1[k],N_nu[k],scaletype)/np.sqrt(np.prod(oversamp[k,:]))

			u1, v1 = eta1.shape
			u2, v2 = eta2.shape

			# 1/((eta1(1,2)-eta1(1,1))*size(eta1,2))*[-size(eta1,2)/2:size(eta1,2)/2-1]
			# 1/((eta2(2,1)-eta2(1,1))*size(eta2,1))*[-size(eta2,1)/2:size(eta2,1)/2-1]

  			[y1,y2]=meshgrid(
  				1/((eta1[0,1]-eta1[0,0])*v1)*[-v1/2:v1/2-1], 
  				1/((eta2[1,0]-eta2[0,0])*u2)*[-u2/2:u2/2-1]
  				)

			if self.cplx_flag=='real': 
				nu_len=np.ceil(N_nu[k]/2)
			else:
				nu_len=N_nu[k]

			for nu in range(nu_len):
			    theta_l=(nu-1)*Theta_len[k];    
			    z1=eta1*np.cos(theta_l)+eta2*np.sin(theta_l);
			    z2=-eta1*np.sin(theta_l)+eta2*np.cos(theta_l);
			    # conf.xi1=[conf.xi1;z1(:)];
			    # conf.xi2=[conf.xi2;z2(:)];
			    conf.xi1.extend(z1)
			    conf.xi2.extend(z2)

			    x1=y1*np.cos(theta_l)+y2*np.sin(theta_l);
			    x2=-y1*np.sin(theta_l)+y2*np.cos(theta_l);  

			    ####! pending to look at lines from here

			    self.x1{k}{nu}=x1;
			    self.x2{k}{nu}=x2;    
			    self.theta{k}(nu)=theta_l;
			    xi_idx=[xi_idx;length(z1[:])];
			    if (k==1):
				    boxidx[bc,1]=1;
				    boxidx[bc,2]=length(self.chi{k}(:));
				    boxidx[bc,3]=size(self.chi{k},1);
				    boxidx[bc,4]=size(self.chi{k},2);
			    else:
			    	boxidx[bc,1]=boxidx[bc-1,2]+1;
			    	boxidx[bc,2]=boxidx[bc-1,2]+length(self.chi{k}(:));
			    	boxidx[bc,3]=size(self.chi{k},1);
			    	boxidx[bc,4]=size(self.chi{k},2);

			    w=[w;self.chi{k}(:)];
			    bc=bc+1;
			    self.xi1_0{k}{nu}=z1(ceil(size(z1,1)/2),ceil(size(z1,2)/2));
			    self.xi2_0{k}{nu}=z2(ceil(size(z1,1)/2),ceil(size(z1,2)/2));

				self.boxgrid[k,0]=size(self.chi{k},0)
				self.boxgrid[k,1]=size(self.chi{k},1)

				self.N=N;
				self.K1low=K1low;
				self.K1=K1;
				self.boxidx=boxidx;
				self.epsilon=epsilon;
				self.N_nu=N_nu;
				self.box_size=box_size;
				self.cplx_flag=cplx_flag;
				self.filt=Uop(self.xi1,self.xi2,w.^2,self.N,self.N,self.epsilon);
				self.oversamp=oversamp;
				if strcmpi(cplx_flag,'real'), self.filt=2*real(self.filt);end;
					self.filt=fftshift(fft2(fftshift(self.filt)));



