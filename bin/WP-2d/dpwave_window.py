def apol(t, deg=4):
    if deg>10:
        n=2*(deg+1)
        a = np.ones((1,n))
        A = np.zeros((n,n))

        for j in range(deg): #n/2 - 1
            A[j,0:n-j-1] = a
            a = np.polyder(a)

        for j in range(deg):
            A[j+n/2, n-j-1] = A[j, n-j-1]

        tar = np.zeros((n,1))
        tar[0, :] = 1

        A_sym = A[:, :deg+1]

        A_sym = (A_sym + A_sym.T)/2.0
        p = np.matmul(np.linalg.pinv(A_sym), tar)

    else:
        P = np.array([
            [-2, 6, -20, 70, -252, 924, -3432, 12870, -48620, 184756],
            [3, -15, 70, -315, 1386, -6006, 25740, -109395, 461890, -1939938],
            [0, 10, -84, 540, -3080, 16380, -83160, 408408, -1956240, 9189180],
            [0, 0, 35, -420, 3465, -24024, 150150, -875160, 4849845, -25865840],
            [0, 0, 0, 126, -1980, 20020, -163800, 1178100, -7759752, 47927880],
            [0, 0, 0, 0, 462, -9009, 108108, -1021020, 8314020, -61108047],
            [0, 0, 0, 0, 0, 1716, -40040, 556920, -5969040, 54318264],
            [0, 0, 0, 0, 0, 0, 6435, -175032, 2771340, -33256080],
            [0, 0, 0, 0, 0, 0, 0, 24310, -755820, 13430340],
            [0, 0, 0, 0, 0, 0, 0, 0, 92378, -3233230],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 352716]
            ])
        p = P[0:deg+1, deg-1]

def w_meyer(x, type_, deg):
    if type_.lower() == 'detail':
        L = np.sin(np.pi/2*apol(2*x-1, deg))
        R = np.cos(np.pi/2*apol(x-1, deg))
        y = (x>=0.5)*(x<1)*L + (x>=1)*(x<=2)*R

    elif type_.lower() == 'coarse':
        t = np.abs(x)
        y = np.cos(np.pi/2*apol(2*t-1, deg))

    else:
        L = np.sin(np.pi/2*apol(2*x-1, deg))
        y = (x>=1/2)*(x<1)*L+(x>=1)*(x<=2)+(x>2)

    return y


def dpwave_window(eta1, eta2, k1_high, N_nu, type_):
    r = np.sqrt(eta1**2 + eta2**2)
    theta = np.atan2(eta2, eta1)

    tlen = np.pi if N_nu > 2 else np.pi/8

    angle_neighbors = N_nu+1
    theta_j = r/k1_high*theta

    wt = np.cos(apol(np.abs(theta_j/(2*tlen)), 9)*np.pi/2)**2*\
         (np.abs(theta_j)<=tlen*2)

    wt_rep = 0*wt

    j_arr = np.linspace(-N_nu/2, N_nu/2, angle_neighbors)

    for j in j_arr:
        theta_j = r/k1_high*(theta+2*tlen*j)
        wt_rep = wt_rep + np.cos(apol(
            np.abs(theta_j/(2*tlen)), 
            9)*np.pi/2)**2*(np.abs(theta_j)<=tlen*2)

    wt = np.sqrt(wt/(wt_rep + 1e-32))
    wr = w_meyer(2*r/k1_high, type_, 9)

    if (N_nu>2):
        w = wt*wr
    else:
        w = wr

    return w

