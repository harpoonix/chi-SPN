from pathlib import Path

import numpy as np
np.set_printoptions(edgeitems=30, linewidth=100000, suppress=True, formatter={'float_kind':"{:.2f}".format})
import matplotlib.pyplot as plt
import os
import pickle

"""
Structural Causal Model

    SocioEconomic Status = uniform distributed, discrete, 0-10
    Quality of Education = Gaussian Noise, continuous
    Motivation = Gaussian Noise, continuous
    Test Scores = Gaussian Noise, continuous
    Cultural Activities = Pareto noise, continuous
    Selection = 3 types, discrete (elite/regional/local)
    
    SocioEconomic Status -> Quality of Education
    Quality of Education -> Test Scores
    Motivation -> Test Scores
    Quality of Education -> Cultural Activities
    Motivation -> Cultural Activities
    Test Scores -> Selection
    Cultural Activities -> Selection
    
    
    
    A = N_A, N_A is Uniform distributed, A in N
    F = 1_X(A) OR N_F, N_F is Bernoulli distributed, F in {0,1}
    H = alpha * F + beta * A + gamma * N_H, N_H is Bernoulli distributed and alpha + beta + gamma = 1, H in (0, 1]
    M = delta * H + (1-delta) * N_M, N_M is Bernoulli distributed, M in (0, 1]
    
    Diagnose = (1/(1 + (A - 5)^2) * N(5, 5)) +
               (1/(1 + (A - 20)^2) * N(30, 10)) * 0.5 * H +
               (1/(1 + (A - 40)^2) * N(40, 20)) * 2.2 * F +
               (1/(1 + (A - 60)^2) * N(60, 10)) * M * H

    Age -> Food Habit
    Age -> Health
    Food Habit -> Health
    Health -> Mobility
    
    Age, Food Habit, Health, Mobility -> Diagnose
    
"""

"""
changes
1) make Age categorical
2) could try making diagnose variables more categorical, or more continuous
for now diagnose is argmax
"""

class SCM_Student():

    def __init__(self, seed = None):
        self.rng = np.random.default_rng()
        # self.rng = np.random.default_rng(seed=seed)
        
        socio = lambda size : self.rng.choice(5, size=size)
        quality = lambda socio : 10 - socio + self.rng.normal(2.5, 3, size=socio.shape)
        motivation = lambda size : self.rng.normal(10, 3, size=size)
        cultural = lambda quality, motivation : 0.8*quality + 0.2*motivation + self.rng.pareto(a = 3, size=quality.shape)
        testscore = lambda quality, motivation : 0.4*quality + 0.6*motivation + self.rng.normal(0, 1, size=quality.shape)
        selection = lambda testscore, cultural : (0.5*testscore + 0.5*cultural >= 14).astype(int) + (0.5*testscore + 0.5*cultural >= 11).astype(int)
        

        self.equations = {
            'Sc' : socio,
            'Q' : quality,
            'M' : motivation,
            'C' : cultural,
            'T' : testscore,
            'S' : selection
        }
        self.intervention = None
        self.intervention_range = None

    def create_data_sample(self, sample_size):
        Scs = np.array([self.equations['Sc'](1) for _ in range(sample_size)])
        Qs = np.array([self.equations['Q'](Sc) for Sc in Scs])
        Ms = np.array([self.equations['M'](1) for _ in range(sample_size)])
        Cs = np.array([self.equations['C'](Q, M) for Q, M in zip(Qs, Ms)])
        Ts = np.array([self.equations['T'](Q, M) for Q, M in zip(Qs, Ms)])
        Ss = np.array([self.equations['S'](T, C) for T, C in zip(Ts, Cs)])


        data = {'Sc': Scs, 'Q': Qs, 'M': Ms, 'C': Cs, 'T': Ts, 'S': Ss}

        return data

    def do(self, intervention, low = 0, high = 100, discrete = False):
        """
        perform a uniform intervention on a single node
        """
        if intervention is None or "None" in intervention:
            return

        if True:
            # low=0
            # high=100
            print(intervention)
            intervention0 = intervention[0].split("do(")[1].split(")")[0]
            intervention1 = intervention[1].split("do(")[1].split(")")[0]
            if discrete[0]:
                print(f'discrete {intervention0}, low is {low[0]}, high is {high[0]}')
                self.equations[intervention0] = lambda *args: np.round(self.rng.uniform(low[0],high[0]))
            else:
                self.equations[intervention0] = lambda *args: self.rng.uniform(low[0],high[0])
                
            if discrete[1]:
                print(f'discrete {intervention1}, low is {low[1]}, high is {high[1]}')
                self.equations[intervention1] = lambda *args: np.round(self.rng.uniform(low[1],high[1]))
            else:
                self.equations[intervention1] = lambda *args: self.rng.uniform(low[1],high[1])
            
            print("Performed Uniform Intervention do({}=U({},{}))".format(intervention,low,high))
            self.intervention = intervention
            self.intervention_range = (low, high)
        elif "N" in intervention:
            low=0
            high=100
            mu = int(intervention.split("N(")[1].split(",")[0])
            sigma = int(intervention.split("N(")[1].split(",")[1].split(")")[0])
            intervention = intervention.split("do(")[1].split(")")[0]
            self.equations[intervention] = lambda *args: self.rng.normal(mu, np.sqrt(sigma))
            print("Performed Normal Intervention do({}=N({},{}))".format(intervention,mu,sigma))
            self.intervention = intervention
            self.intervention_range = (low, high)
        elif "SBeta" in intervention:
            low=0
            high=100
            p = float(intervention.split("SBeta(")[1].split(",")[0])
            q = float(intervention.split("SBeta(")[1].split(",")[1].split(")")[0])
            intervention = intervention.split("do(")[1].split(")")[0]
            self.equations[intervention] = lambda *args: self.rng.beta(p, q) * (high - low) + low
            print("Performed Non-Standard Beta Intervention do({}=SBeta({},{}))".format(intervention,p,q))
            self.intervention = intervention
            self.intervention_range = (low, high)
        elif "Gamma" in intervention:
            low=0
            high=100
            p = float(intervention.split("Gamma(")[1].split(",")[0])
            q = float(intervention.split("Gamma(")[1].split(",")[1].split(")")[0])
            intervention = intervention.split("do(")[1].split(")")[0]
            self.equations[intervention] = lambda *args: self.rng.gamma(p,q)
            print("Performed Gamma Intervention do({}=Gamma({},{}))".format(intervention,p,q))
            self.intervention = intervention
            self.intervention_range = (low, high)
        elif "[" in intervention:
            low=0
            high=100
            a = int(intervention.split("[")[1].split(",")[0])
            b = int(intervention.split("[")[1].split(",")[1].split("]")[0])
            intervention = intervention.split("do(")[1].split(")")[0]
            self.equations[intervention] = lambda *args: self.rng.choice([a,b])
            print("Performed Choice Intervention do({}=[{},{}])".format(intervention,a,b))
            self.intervention = intervention
            self.intervention_range = (low, high)
        elif intervention is not None:
            low=0
            high=100
            scalar = int(intervention.split("=")[1])
            intervention = intervention.split("do(")[1].split(")")[0]
            self.equations[intervention] = lambda *args: scalar
            print("Performed perfect Intervention do({}={})".format(intervention,scalar))
            self.intervention = intervention
            self.intervention_range = (low, high)
        else:
            raise ValueError(f"Unknown intervention type ({intervention})")

# interventions = [
#     (None, "None", 0, 0, False),
#     ("Q", "do(Q)=U(Q)", 0, 20, False),
#     ("M", "do(M)=U(M)", 0, 20, False),
#     ("C", "do(C)=U(C)", 0, 20, False),
#     ("T", "do(T)=U(T)", 0, 20, False),
# ]

interventions = [
    (("C", "T"), ("do(C)=U(C)", "do(T)=U(T)"), (0, 0), (20, 20), (False, False)),
    (("C", "M"), ("do(C)=U(C)", "do(M)=U(M)"), (0, 0), (20, 20), (False, False)),
    (("C", "Q"), ("do(C)=U(C)", "do(Q)=U(Q)"), (0, 0), (20, 20), (False, False)),
    (("T", "M"), ("do(T)=U(T)", "do(M)=U(M)"), (0, 0), (20, 20), (False, False)),
    (("T", "Q"), ("do(T)=U(T)", "do(Q)=U(Q)"), (0, 0), (20, 20), (False, False)),
    (("M", "Q"), ("do(M)=U(M)", "do(Q)=U(Q)"), (0, 0), (20, 20), (False, False)),
]

# seed = 123
# np.random.seed(seed)
num_samples = 120000
test_split = 0.2

dir_save = Path(f"./StudentData")
dir_save.mkdir(exist_ok=True)
save = True
save_plot_and_info = True
plot_and_info_dir = dir_save / "info" / "double"
if save_plot_and_info:
    plot_and_info_dir.mkdir(exist_ok=True)


num_samples_train = int(num_samples * (1 - test_split))
num_samples_test = int(num_samples * test_split)

for j, (N, data_name) in enumerate([(num_samples_test, 'test')]):
    print(f"[{data_name}]")
    for i, interv in enumerate(interventions):
        interv, interv_desc, low, high, discrete = interv

        # create a dataset
        scm = SCM_Student()
        # scm = SCM_Student(seed+100*j+i)
        scm.do(interv_desc, low, high, discrete)
        data = scm.create_data_sample(N)


        if save_plot_and_info:
            with open(plot_and_info_dir / f"info_{interv_desc}_{data_name}.txt", "w+") as fi:
                for ind_d, (d, e) in enumerate(zip(["Socio", "Quality", "Motivation", "Cultural", "Test", "Selection"], ['Sc', 'Q', 'M', 'C', 'T', 'S'])):
                    print('Min {:.2f}\t Max {:.2f}\t Mean {:.2f}\t Median {:.2f}\t STD {:.2f}\t\t - {}'
                          .format(np.min(data[e]), np.max(data[e]), np.mean(data[e]), np.median(data[e]), np.std(data[e]), d),
                          file=fi)
                n=25
                print('(Continuous) First {} samples from a total of {} samples:\n'
                      '\tSocio       = {}\n'
                        '\tQuality     = {}\n'
                        '\tMotivation  = {}\n'
                        '\tCultural    = {}\n'
                        '\tTest        = {}\n'
                        '\tSelection   = {}\n'
                      '\n\n***********************************\n\n'.format(n, N,
                                                                           data['Sc'][:n],
                                                                            data['Q'][:n],
                                                                            data['M'][:n],
                                                                            data['C'][:n],
                                                                            data['T'][:n],
                                                                            data['S'][:n]),
                      file=fi)

                # plot the median health per age group
                plt.figure(figsize=(12,7))
                for v, dd in zip(['Socio', 'Quality', 'Motivation', 'Cultural', 'Test', 'Selection'], ['Sc', 'Q', 'M', 'C', 'T', 'S']):
                    #TODO CONTINUE FROM HERE
                    median_var_per_socio = []
                    mean_var_per_socio = []
                    std_var_per_socio = []
                    #age_intervals = [(0, 10), (10, 30), (30, 55), (55, 75), (75, 100)]
                    socio_intervals = [(n, n+1) for n in range(5)]
                    for a in socio_intervals:
                        indices = np.where(np.logical_and(data['Sc'] >= a[0],data['Sc'] < a[1]))[0]
                        corresponding_var_data = [data[dd][i] for i in indices]
                        median_var = np.median(corresponding_var_data)
                        mean_var = np.mean(corresponding_var_data)
                        std_var = np.std(corresponding_var_data)
                        median_var_per_socio.append(median_var)
                        mean_var_per_socio.append(mean_var)
                        std_var_per_socio.append(std_var)

                    factor = 1
                    # factor = 80 if dd == 'D' else (3 if dd == 'E' or dd == 'W' else 1)

                    e = dd
                    p = plt.plot(range(len(socio_intervals)), np.array(mean_var_per_socio)*factor, label='{} |All Data {:.1f}*scaled{:.1f}, {:.1f}, {:.1f}|'.format(v,np.mean(data[e]), factor, np.min(data[e]), np.max(data[e])))
                    plt.errorbar(range(len(socio_intervals)), np.array(mean_var_per_socio)*factor, yerr=std_var_per_socio, color=p[0].get_color())
                    plt.title('Intervention: {} {}\nContinuous Data Mean Values per Age intervals x<a<y (Sampled {} Persons via SCM)\nVariable Name |All Data Mean, Min, Max|'.format(interv_desc, scm.intervention_range,N))
                    plt.xlabel('SocioEconomic Condition $Sc$')
                    plt.ylabel('Mean for Variable in Interval')
                    plt.xticks(range(len(socio_intervals)), [str(x) for x in socio_intervals])
                # plt.ylim(-10,70)
                plt.legend(bbox_to_anchor=[0.5, -0.13], loc='center', ncol=3)
                plt.tight_layout()
                axes = plt.gca()
                #plt.show()
                plt.savefig(plot_and_info_dir / f"age_avg_{interv_desc}.jpg", dpi=300)
                
                fig, axs = plt.subplots(4,2,figsize=(12,10))
                # 8 colors
                colors = ['black', 'red', 'green', 'blue', 'cyan', 'magenta', 'orange', 'yellow']
                for ind_d, (d, e) in enumerate(zip(['Socio', 'Quality', 'Motivation', 'Cultural', 'Test', 'Selection'], ['Sc', 'Q', 'M', 'C', 'T', 'S'])):
                    axs.flatten()[ind_d].set_title('{}'.format(d))
                    if d == "Socio":
                        # limit range of diagnoses
                        axs.flatten()[ind_d].hist(data[e], bins=5, color=colors[ind_d])
                        axs.flatten()[ind_d].set_xlim(-0.1, 6.1)
                    elif d == "Selection":
                        # decision is binary
                        axs.flatten()[ind_d].hist(data[e], bins=20, color=colors[ind_d])
                        axs.flatten()[ind_d].set_xlim(-0.1, 2.1)
                    else:
                        axs.flatten()[ind_d].hist(data[e], bins=50, color=colors[ind_d])
                        #axs.flatten()[ind_d].set_title('{}'.format(d))
                        axs.flatten()[ind_d].set_xlim(-0.1,20)
                plt.suptitle('Intervention: {} {}\nHistograms for {} Samples (x: Value, y: Frequency)'.format(interv_desc, scm.intervention_range,N))
                plt.savefig(plot_and_info_dir / f"stats_{data_name}_{interv_desc}.jpg", dpi=300)
                
        if save:
            save_location = os.path.join(dir_save,
                                   f'DoubleIntervention_{"_".join(interv_desc)}_N{N}_{data_name}.pkl')
            #excludeData = ['1', '2', '3'] # exclude intermediate diagnoses class data
            # excludeData = ['D'] # exclude diagnoses class data
            # for i in excludeData:
                # del data[i]
            print("Saving data with keys:", data.keys())
            with open(save_location, 'wb') as f:
                pickle.dump(data, f)
                print("Saved Data @ {}".format(save_location))
