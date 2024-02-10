from pathlib import Path

import numpy as np
np.set_printoptions(edgeitems=30, linewidth=100000, suppress=True, formatter={'float_kind':"{:.2f}".format})
import matplotlib.pyplot as plt
import os
import pickle

"""
Structural Causal Model

    Ed = 5 categories, discrete
    W = Chi Squared distributed, continuous
    Skills and Qualification = Pareto noise, continuous
    Interview Performance = Gaussian noise, continuous
    Identity = Uniform distributed, discrete
    Biases of Interviewer = Gaussian noise, continuous
    Job Offer = Bernoulli distributed, discrete
    
    Education -> Skills and Qualification
    Work Experience -> Skills and Qualification
    Skills and Qualification -> Interview Performance
    Sociodemographic -> Biases of Interviewer
    Sociodemographic -> Education
    Bias of Interviewer -> Interview Performance
    Interview Performance -> Job Offer
    Skills and Qualification -> Job Offer
    
    
    
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

class SCM_Job_Classification():

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed=seed)
        
        socio = lambda size : self.rng.choice(10, size=size)
        work = lambda size : 0.5*self.rng.chisquare(df = 4, size=size)
        education = lambda socio : np.round(self.rng.choice([1,2,3,4,5], size=socio.size, p=[0.11, 0.25, 0.40, 0.2, 0.04]) + 0.9 - 0.2*socio)
        skills = lambda education, work : (0.8*education + 1.2*work + self.rng.pareto(a=2.75, size=education.size))/1.5
        bias = lambda socio : (5 + socio + self.rng.normal(loc=0, scale=1.5, size=socio.size))/2
        interview = lambda skills, bias : 0.25*(10 + 3*skills - 0.5*bias + self.rng.normal(loc=0, scale=4, size=skills.size))
        decision = lambda interview, skills : ((3*interview + skills) >= 23).astype(int)

        self.equations = {
            'E': education,
            'W': work,
            'Sk': skills,
            'Sc': socio,
            'B': bias,
            'I': interview,
            'D': decision
        }
        self.intervention = None
        self.intervention_range = None

    def create_data_sample(self, sample_size):
        Ws = np.array([self.equations['W'](1) for _ in range(sample_size)])
        Scs = np.array([self.equations['Sc'](1) for _ in range(sample_size)])
        Es = np.array([self.equations['E'](sc) for sc in Scs])
        Sks = np.array([self.equations['Sk'](e, w) for e, w in zip(Es, Ws)])
        Bs = np.array([self.equations['B'](sc) for sc in Scs])
        Is = np.array([self.equations['I'](sk, b) for sk, b in zip(Sks, Bs)])
        Ds = np.array([self.equations['D'](i, sk) for i, sk in zip(Is, Sks)])


        data = {'E' : Es, 'W' : Ws, 'Sk' : Sks, 'Sc' : Scs, 'B' : Bs, 'I' : Is, 'D' : Ds}

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
#     ("E", "do(E)=U(E)", 0, 6, True),
#     ("Sk", "do(Sk)=U(Sk)", 1, 8, False),
#     ("B", "do(B)=U(B)", 1, 8, False),
#     ("I", "do(I)=U(I)", 0, 10, False),
# ]

interventions = [
    (("E", "I"), ("do(E)=U(E)", "do(I)=U(I)"), (0, 0), (6, 10), (True, False)),
    
    (("Sk", "B"), ("do(Sk)=U(Sk)", "do(B)=U(B)"), (1, 1), (8, 8), (False, False)),
    (("Sk", "I"), ("do(Sk)=U(Sk)", "do(I)=U(I)"), (1, 0), (8, 10), (False, False)),
    (("B", "E"), ("do(B)=U(B)", "do(E)=U(E)"), (1, 0), (8, 6), (False, True)),
]

seed = 100
np.random.seed(seed)
num_samples = 120000
test_split = 0.2

dir_save = Path(f"./JobClassification")
dir_save.mkdir(exist_ok=True)
save = True
save_plot_and_info = True
plot_and_info_dir = dir_save / "info" / "double"
if save_plot_and_info:
    plot_and_info_dir.mkdir(exist_ok=True)


num_samples_train = int(num_samples * (1 - test_split))
num_samples_test = int(num_samples * test_split)

for j, (N, data_name) in enumerate([ (num_samples_test, 'test')]):
    print(f"[{data_name}]")
    for i, interv in enumerate(interventions):
        interv, interv_desc, low, high, discrete = interv

        # create a dataset
        scm = SCM_Job_Classification(seed+100*j+i)
        scm.do(interv_desc, low, high, discrete)
        data = scm.create_data_sample(N)


        if save_plot_and_info:
            with open(plot_and_info_dir / f"info_{interv_desc}_{data_name}.txt", "w+") as fi:
                for ind_d, (d, e) in enumerate(zip(["Education", "Work Exp.", "Skills", "Socio", "Bias", "Interview", "Decision"], ['E', 'W', 'Sk', 'Sc', 'B', 'I', 'D'])):
                    print('Min {:.2f}\t Max {:.2f}\t Mean {:.2f}\t Median {:.2f}\t STD {:.2f}\t\t - {}'
                          .format(np.min(data[e]), np.max(data[e]), np.mean(data[e]), np.median(data[e]), np.std(data[e]), d),
                          file=fi)
                n=25
                print('(Continuous) First {} samples from a total of {} samples:\n'
                      '\tEducation   = {}\n'
                    '\tWork Exp.   = {}\n'
                    '\tSkills      = {}\n'
                    '\tSocio       = {}\n'
                    '\tBias        = {}\n'
                    '\tInterview   = {}\n'
                    '\tDecision    = {}\n'
                      '\n\n***********************************\n\n'.format(n, N,
                                                                           data['E'][:n],
                                                                          data['W'][:n],
                                                                            data['Sk'][:n],
                                                                            data['Sc'][:n],
                                                                            data['B'][:n],
                                                                            data['I'][:n],
                                                                            data['D'][:n]),
                      file=fi)

                # plot the median health per age group
                plt.figure(figsize=(12,7))
                for v, dd in zip(['Education', 'Work Exp.', 'Skills', 'Bias', 'Interview', 'Decision'], ['E', 'W', 'Sk', 'B', 'I', 'D']):
                    #TODO CONTINUE FROM HERE
                    median_var_per_socio = []
                    mean_var_per_socio = []
                    std_var_per_socio = []
                    #age_intervals = [(0, 10), (10, 30), (30, 55), (55, 75), (75, 100)]
                    socio_intervals = [(2*n, 2*n+1) for n in range(5)]
                    for a in socio_intervals:
                        indices = np.where(np.logical_and(data['Sc'] >= a[0],data['Sc'] <= a[1]))[0]
                        corresponding_var_data = [data[dd][i] for i in indices]
                        median_var = np.median(corresponding_var_data)
                        mean_var = np.mean(corresponding_var_data)
                        std_var = np.std(corresponding_var_data)
                        median_var_per_socio.append(median_var)
                        mean_var_per_socio.append(mean_var)
                        std_var_per_socio.append(std_var)

                    factor = 80 if dd == 'D' else (3 if dd == 'E' or dd == 'W' else 1)

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
                for ind_d, (d, e) in enumerate(zip(['Education', 'Work Exp.', 'Skills', 'Socio', 'Bias', 'Interview', 'Decision'], ['E', 'W', 'Sk', 'Sc', 'B', 'I', 'D'])):
                    axs.flatten()[ind_d].set_title('{}'.format(d))
                    if d == "Socio":
                        # limit range of diagnoses
                        axs.flatten()[ind_d].hist(data[e], bins=10, color=colors[ind_d])
                        axs.flatten()[ind_d].set_xlim(-0.1, 10.1)
                    elif d == "Decision":
                        # decision is binary
                        axs.flatten()[ind_d].hist(data[e], bins=10, color=colors[ind_d])
                        axs.flatten()[ind_d].set_xlim(-0.1, 1.1)
                    elif d == "Education":
                        # 5 kinds of education
                        axs.flatten()[ind_d].hist(data[e], bins=6, color=colors[ind_d])
                        axs.flatten()[ind_d].set_xlim(-0.1, 8.1)
                    elif d == "Work Exp.":
                        # range from 0 to 20
                        axs.flatten()[ind_d].hist(data[e], bins=40, color=colors[ind_d])
                        axs.flatten()[ind_d].set_xlim(-0.1, 10.1)
                    elif d == "Bias":
                        # range from -4 to 15
                        axs.flatten()[ind_d].hist(data[e], bins=40, color=colors[ind_d])
                        axs.flatten()[ind_d].set_xlim(-0.1, 10.1)
                    else:
                        axs.flatten()[ind_d].hist(data[e], bins=100, color=colors[ind_d])
                        #axs.flatten()[ind_d].set_title('{}'.format(d))
                        axs.flatten()[ind_d].set_xlim(-1,11)
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
