    
import torch
import os
# os.environ['pRT_input_data_path'] = os.path.join('/home/mvasist/pRT/input_data_v2.4.9/input_data')
from petitRADTRANS.retrieval.util import *
import petitRADTRANS as prt
from added_scripts.corner_modified import *
from lampe.plots import nice_rc, corner, mark_point


def removing_columns(theta):
    theta_c = torch.unsqueeze(theta[:,columns[0]], 1) 
    labels = ( LABELS[columns[0]], ) 
    lower =  ( LOWER[columns[0]], )
    upper = ( UPPER[columns[0]], )

    for c in columns[1:]:
        theta_c = torch.hstack(( theta_c, torch.unsqueeze(theta[:,c], 1) )) 
        labels =  labels + (LABELS[c],) 
        lower = lower + (LOWER[c],) 
        upper = upper + (UPPER[c],) 
    return theta_c, labels, lower, upper
        
def appending_params(theta_old, tthh, fn, labels, label, lows, low, ups, up, i):
    theta_new = torch.cat((theta_old, torch.unsqueeze(fn(tthh), -1)), -1) 
    labels_new = labels + (label,)
    lower_new = lows + (low,)
    upper_new = ups + (up,)
    return theta_new, labels_new, lower_new, upper_new

def ratio(theta):
    N14 = 10**theta['$NH_3$']
    N15 = 10**theta['$^{15}NH_3$']
    ratio = (N14*18.02)/ (N15*17.027)
    return ratio

def computing_gravity(theta):
    gravity = nc.G * (theta['Mass']* prt.nat_cst.m_jup)/(theta['$R_{P}$']*prt.nat_cst.r_jup_mean)**2
    return torch.log10(gravity)

def computing_mass(theta):
    radius = theta['$R_{P}$'] * prt.nat_cst.r_jup_mean  # Convert to meters
    gravity = 10 ** theta['$\log g$']  # Convert log10(g) back to linear scale
    mass = (gravity * radius**2) / nc.G  # Compute mass
    return mass / prt.nat_cst.m_jup  # Convert to Jupiter masses

class plots(): 

    def __init__(self, runpath, ep, estimator, x_star):
        self.runpath = runpath
        self.ep = ep

        self.savepath_plots = self.runpath  / ('plots_' + str(ep)+ '_reprocessed231123')
        self.savepath_plots.mkdir(parents=True, exist_ok=True)

        self.estimator = estimator.cuda()
        states = torch.load(self.runpath / ('states_' + str(ep) + '.pth'), map_location='cpu')
        self.estimator.load_state_dict(states['estimator'])
        self.estimator.cuda().eval()

        posterior_cf10 = estimator_cf10.flow(torch.from_numpy(x_star_hg).unsqueeze(0).float().cuda())

        # log_p_cf10 = posterior_cf10.log_prob(thetacm_cf10.float().cuda())
        # log_p_cf10 = posterior_cf10.log_prob(thetacf10[:5000].float().cuda())
        # log_p_cf10 = log_p_cf10.cpu()
        # index_cf10 = np.where(log_p_cf10 == max(log_p_cf10))[0]




        self.x_star = x_star


    def sampling_from_post(self, x, name, only_returning = True):
        
            if not only_returning: 
                with torch.no_grad():
                    theta = torch.cat([
                        self.estimator.flow(x.cuda()).sample((2**14,)).cpu()
                        for _ in tqdm(range(2**6))
                    ])
                    theta = theta.squeeze()
                ##Saving to file
                theta_numpy = theta.double().numpy() #convert to Numpy array
                df_theta = pd.DataFrame(theta_numpy) #convert to a dataframe
                df_theta.to_csv( name ,index=False) #save to file
                return theta
            
            #Then, to reload:
            df_theta = pd.read_csv(name)
            theta = df_theta.values
            return torch.from_numpy(theta)

    def coverage(self, testsets, pipe, sim): 
        ranks = []
        with torch.no_grad():
            
            for data_tuple in islice(zip(*testsets), 128):
                instrument_data = {}
                idx = 0
                for type in config_script['simulator']["type"]:
                    instrument_data[type] = {}
                    for instrument in config_script['instruments']:
                        instrument_data[type][instrument] = data_tuple[idx]
                        idx += 1
                theta, x = pipe(instrument_data, sim[type][instrument], return_loss = False)
                theta, x = theta.cuda(), x.cuda()        
                
                posterior = self.estimator.flow(x)
                samples = posterior.sample((1024,))
                log_p = posterior.log_prob(theta)
                log_p_samples = posterior.log_prob(samples)

                ranks.append((log_p_samples < log_p).float().mean(dim=0).cpu())

        ranks = torch.cat(ranks)   
        ranks_numpy = ranks.double().numpy() #convert to Numpy array
        df_ranks = pd.DataFrame(ranks_numpy) #convert to a dataframe
        df_ranks.to_csv(self.savepath_plots /"ranks.csv",index=False) #save to file

        df_ranks = pd.read_csv(self.savepath_plots/"ranks.csv")
        ranks = df_ranks.values

        # Coverage
        a=[]
        r = np.sort(np.asarray(ranks))

        for alpha in np.linspace(0,1,100):
            a.append((r > (1-alpha)).mean())

        cov_fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlabel(r'Credibility level $1-\alpha$', fontsize = 12)
        ax.set_ylabel(r'Coverage probability', fontsize= 12)
        ax.plot(np.linspace(0,1,100),a, color='steelblue', label='upper right') #a[::-1]
        ax.plot([0, 1], [0, 1], color='k', linestyle='--')
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(fontsize=12)
        cov_fig.savefig(self.savepath_plots / 'coverage.pdf') 
        return cov_fig   

    def cornerplot(self):

        self.theta = self.sampling_from_post(torch.from_numpy(self.x_star).unsqueeze(0).float().cuda(), self.savepath_plots/'theta.csv', only_returning = False) #float()


        labels = rolling(LABELS)
        lower = rolling(LOWER)
        upper = rolling(UPPER)
        theta_rolled = rolling(self.theta)

        fs = self.theta.size()[-1]
        fig = corner_mod(theta= [theta_rolled[:20469,10:]], legend=['NPE'], \
                    color= ['steelblue'] , figsize=(fs-10,fs-10), \
                domain = (lower[10:], upper[10:]), labels= labels[10:], \
                    labelsize = 20, legend_fontsize = 22,\
                xtick_labelsize = 18 , ytick_labelsize = 18,)
        fig.savefig(self.savepath_plots / 'corner.pdf')
        return fig

    def ptprofile(self, theta):

        theta = self.sampling_from_post(torch.from_numpy(self.x_star).float().cuda(), self.savepath_plots/'theta.csv', only_returning = True)
        mask = torch.any(torch.isinf(self.theta), dim=-1)
        print('No of inf in theta (post) is ', mask.sum())
        self.theta = self.theta[~mask]

        # pt_paul=pd.read_csv('/home/mvasist/WISEJ1828/WISEJ1828/4/best_fit_PT.dat',sep=" ",header=0)
        fig, ax = plt.subplots(figsize=(5,5))

        pressures = simulator_miri_cloudfree.atmosphere.press / 1e6

        # ax.plot(pt_paul.iloc[:,1].values, pt_paul.iloc[:,0].values, color = 'orange')
        fig_pt = PT_plot(fig, ax, self.theta[:2**8], pressures, invert = True, \
                            legend_fontsize = 12, fontsize= 16, \
                            xtick_labelsize = 12 , ytick_labelsize = 12) #, self.theta_star)
        #         fig_pt = PT_plot(fig_pt, ax, self.theta_paul[:2**8], invert = True, color = 'orange') #, theta_star)
        fig_pt.savefig(self.savepath_plots / 'pt_profile.pdf')
        return fig_pt

    def consistencyplot_MIRI(self):
            # wlen = obs_wlen_hst
        self.theta = self.sampling_from_post(torch.from_numpy(self.x_star).float().cuda(), self.savepath_plots/'theta.csv', only_returning = True)

        fig = MIRI_consistency( self.theta[:512], 
                                simulator_miri_cloudy = None,
                                simulator_miri_cloudfree = simulator_miri_cloudfree,
                                savepath_plots = self.savepath_plots,
                                cloud = 'cloudfree', 
                                obs_miri = obs_miri, 
                                obs_wlen_miri = obs_wlen_miri, 
                                sigmaM = sigmaM,
                                only_returning = False,
                                p = None).fig
        return fig

    def consistencyplot_Gemini(self):
        # wlen = obs_wlen_hst
        self.theta = self.sampling_from_post(torch.from_numpy(self.x_star).float().cuda(), self.savepath_plots/'theta.csv', only_returning = True)

        fig = Gemini_consistency(  self.theta[:512], 
                                simulator_hst_cloudy = None,
                                simulator_hst_cloudfree = simulator_hst_cloudfree,
                                mode = 'MIRI + HST+ Gemini', 
                                savepath_plots = self.savepath_plots,
                                cloud = 'cloudfree',
                                obs_gemini = obs_gemini, 
                                obs_wlen_gemini = obs_wlen_gemini,
                                sigmaG = sigmaG,  
                                only_returning = False,
                                p = None).fig
        return fig
    
    def consistencyplot_HST(self):
        # wlen = obs_wlen_hst
        self.theta = self.sampling_from_post(torch.from_numpy(self.x_star).float().cuda(), self.savepath_plots/'theta.csv', only_returning = True)

        fig = HST_consistency(  self.theta[:512], 
                                simulator_hst_cloudy = simulator_hst_cloudfree,
                                simulator_hst_cloudfree = simulator_hst_cloudfree,
                                mode = 'MIRI + HST+ Gemini', 
                                savepath_plots = self.savepath_plots,
                                cloud = 'cloudfree',
                                obs_hst = obs_hst, 
                                obs_wlen_hst = obs_wlen_hst,
                                sigmaH = sigmaH, 
                                only_returning = False,
                                p = None, 
                                ).fig
        

                
        return fig
    

    def cornerWratio(self):
    
        self.theta = self.sampling_from_post(torch.from_numpy(self.x_star).float().cuda(), self.savepath_plots/'theta.csv', only_returning = True)
        
        def appending_params(theta_old, fn, labels, label, lows, low, ups, up):
            theta_new = torch.cat((theta_old, torch.unsqueeze(fn(theta_old), -1)), -1)
            labels_new = labels + (label,)
            lower_new = lows + (low,)
            upper_new = ups + (up,)
            return theta_new, labels_new, lower_new, upper_new
        
        def ratio(theta):
            N14 = 10**theta[:,16]
            N15 = 10**theta[:,19]
            ratio = (N14*18.02)/ (N15*17.027)
            return ratio
            
        def computing_gravity(theta):
            gravity = nc.G * (theta[:,1]* prt.nat_cst.m_jup)/(theta[:,0]*prt.nat_cst.r_jup_mean)**2
            return torch.log10(gravity)
        
        def roll_here(label, low, up, th):
            labels_rolled = rolling(label)
            lower_rolled = rolling(low)
            upper_rolled = rolling(up)
            thetar_rolled = rolling(th)
            return labels_rolled, lower_rolled, upper_rolled, thetar_rolled
        
        theta_new, labels_new, lower_new, upper_new = appending_params(self.theta, ratio, LABELS, r'$14N/15N$', LOWER, 0, UPPER, 1000)
        theta_new, labels_new, lower_new, upper_new = appending_params(theta_new, computing_gravity, labels_new, r'$log_g$', lower_new, 2, upper_new, 6)
        labels_rolled, lower_rolled, upper_rolled, thetar_rolled = roll_here(labels_new,  lower_new, upper_new, theta_new)
    
        fs = self.theta.size()[-1]
        fig = corner_mod(theta= [thetar_rolled[:20469,10:]], legend=['NPE'], \
                    color= ['steelblue'] , figsize=(fs-8,fs-8), \
                domain = (lower_rolled[10:], upper_rolled[10:]), labels= labels_rolled[10:], \
                    labelsize = 20, legend_fontsize = 22,\
                xtick_labelsize = 18 , ytick_labelsize = 18,)
        
        # # mark_point(fig, theta_star_rolled[10:], color='black')
        fig.savefig(self.savepath_plots / 'corner_withRatio.pdf')

        import corner
        figure = corner.corner(thetar_rolled[:100000,10:],
        #                         hist_bin_factor = 10,
                        labels= labels_rolled[10:],
                        range = [(lower_rolled[i+10], upper_rolled[i+10]) for i in range(len(thetar_rolled[0])-10)],
        #                         quantiles=[0.16, 0.5, 0.84],
                        show_titles=True,
                        title_kwargs={"fontsize": 12},
                    )
        figure.savefig(self.savepath_plots / 'corner_corner_withRatio.pdf')
        
        return fig


    def cornerWratio_notfull(self, LOWER, UPPER, LABELS, theta=None, columns = [0,1,2], appending_params_dict = {r'$^{14}N/^{15}N$' : {"limits": [0, 1000], "method" : ratio}}, legends = ['NPE'], colors = ['steelblue'], savepath= None, \
                                labelsize = 18, titlesize = 20, fontsize= 16, legend_fontsize = 20, xtick_labelsize = 28 , ytick_labelsize = 28,  \
                                theta_star= None, loc= 'center', bbox_to_anchor= (0.4,0.9), labl= True, alpha = [0, 0.9]):   


        '''
        appending_params_dict = {r'$^{14}N/^{15}N$' : {"limits": [0, 1000], "method" : ratio}, 
                                r'$log g$' : {"limits": [2, 6], "method" : computing_gravity},
                                r'$Mass$' : {"limits": [1, 50], "method" : computing_mass}}

        '''
        if theta is None:
            theta = self.sampling_from_post(torch.from_numpy(self.x_star).float().cuda(), self.savepath_plots/'theta.csv', only_returning = True)

        theta_new, labels_new, lower_new, upper_new = {}, {}, {}, {}
        
        for i, th in enumerate(theta):
            theta_dict = {label: theta[:, i] for i, label in enumerate(LABELS)}
            th_small, labels, lower, upper = removing_columns(th)
            theta_new[str(i)], labels_new[str(i)], lower_new[str(i)], upper_new[str(i)] = th_small, labels, lower, upper
            for param, settings in appending_params_dict.items():
                limits = settings["limits"]
                method = settings["method"]
                theta_new[str(i)], labels_new[str(i)], lower_new[str(i)], upper_new[str(i)] = appending_params(theta_new[str(i)], theta_dict, method, labels_new[str(i)], param, lower_new[str(i)], limits[0], upper_new[str(i)], limits[1], i)

        fs = len(th_small[0].size()[-1]) 
        fig = corner_mod(theta= [ theta_new[str(i)][:20469] for i in range(len(theta)) ], legend=[leg for leg in legends], 
                    color= [col for col in colors] , figsize=(fs,fs), \
                domain = (lower_new[str(0)], upper_new[str(0)]), labels= labels_new[str(0)], labelsize = labelsize, titlesize = titlesize, \
                fontsize= fontsize, legend_fontsize = legend_fontsize, xtick_labelsize = xtick_labelsize , ytick_labelsize = ytick_labelsize, \
                            loc= loc, bbox_to_anchor= bbox_to_anchor, labl = labl, alpha= alpha )
        
        ### adding a mark point
        theta_new, labels_new, lower_new, upper_new = {}, {}, {}, {}

        if theta_star is not None:
            for i, th in enumerate(theta_star):
                theta_star_dict = {label: th.unsqueeze(0)[:, i] for i, label in enumerate(LABELS)}
                th_small, labels , lower, upper = removing_columns(th.unsqueeze(0))
                theta_new[str(i)], labels_new[str(i)], lower_new[str(i)], upper_new[str(i)] = th_small, labels , lower, upper
                for param, settings in appending_params_dict.items():
                    limits = settings["limits"]
                    method = settings["method"]
                    theta_new[str(i)], labels_new[str(i)], lower_new[str(i)], upper_new[str(i)] = appending_params(theta_new[str(i)], theta_star_dict, method, labels_new[str(i)], param, lower_new[str(i)], limits[0], upper_new[str(i)], limits[1], i)
                mark_point(fig, theta_new[0], color='black')
    
        if savepath is not None:
            fig.savefig(savepath / 'corner_withRatio.pdf', bbox_inches='tight', pad_inches= 0.2)


        return fig


    def PT_profile(self) :
        wlen_miri_cf10, atmosphere_miri_cf10, temperatures_cf10, bf_contribution_miri_cf10, bf_wlen_miri_cf10, \
        bf_spectrum_miri_cf10, spectral_weights_miri_cf10, weights_miri_cf10, \
        contr_em0_miri_cf10, contr_em_miri_cf10, contr_em_weigh_miri_cf10, _ = plotting_contribution(
                                                                                            theta_cf10.numpy(),
                                                                                            4500,
                                                                                            index_cf10[0],
                                                                                            frac=0.8,
                                                                                            simulated_sample=False,
                                                                                            simulator=simulator_miri_cloudfree_cf10,
                                                                                            cloud=False,
        )

        wlen_inst_cf10, atmosphere_inst_cf10, _, bf_contribution_inst_cf10, bf_wlen_inst_cf10, bf_spectrum_inst_cf10, \
        spectral_weights_inst_cf10, weights_inst_cf10, \
        contr_em0_inst_cf10, contr_em_inst_cf10, contr_em_weigh_inst_cf10, _ = plotting_contribution(
                                                                                            theta_cf10.numpy(),
                                                                                            4500,
                                                                                            index_cf10[0],
                                                                                            frac=0.8,
                                                                                            simulated_sample=False,
                                                                                            simulator=simulator_hst_cloudfree_cf10,
                                                                                            cloud=False,
        )

