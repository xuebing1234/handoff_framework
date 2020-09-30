import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


def main(dataset_path, shap_path, patientid, outcome):

    dataset = pd.read_csv(dataset_path)
    shap_values = pd.read_csv(shap_path)
    yticks = dataset.columns.values

    import numpy as np
    number_variables = 10
    A = np.abs(shap_values.values[patientid,1:])
    B=sorted(range(len(A)),key=lambda x:A[x],reverse=True)
    B=np.array(B[0:number_variables])[::-1]
    B+=1
    shap_pneu_group = shap_values.iloc[shap_values.index[dataset[outcome] == 1], :]

    fig = plt.figure()
    ax = plt.subplot(111)
    a = 0.5
    x_ticks = np.append(np.array([' ']), np.array(yticks)[B[::-1].tolist()])
    new_shap = np.append(np.array([0]),shap_values.values[patientid,B[::-1]])
    y_ticks =  (1/(1+np.exp(-1*np.cumsum(new_shap))) )[::-1][0:number_variables+1][::-1]
    ax.bar(x_ticks, new_shap,alpha=a)
    ax.plot(x_ticks, y_ticks )
    new_shap_pneu =  np.append(np.array([0]),np.mean(shap_pneu_group.values[:,B[::-1]],axis=0))
    new_shap_pneu_sd= np.append(np.array([0]),np.std(shap_pneu_group.values[:,B[::-1]],axis=0))
    ax.bar(x_ticks,new_shap_pneu,alpha=a,  color  = 'orange', )
    # ax.bar(x_ticks,new_shap_pneu,alpha=a, yerr=new_shap_pneu_sd, color  = 'orange', error_kw=dict(ecolor='gray', lw=1, capsize=5, capthick=1))
    ax.legend(['Cumulative Risk Score','Variable Contribution',  'Average of Patients with Complication'])
    plt.xticks(rotation=90)
    plt.ylabel('Variable Importance by Average SHAP Values')
    plt.xlabel('Clinical Variables')
    plt.tight_layout()
    plt.ylim((-0.6,1))
    fig.savefig(str(patientid)+ "1.pdf", bbox_inches='tight')

    shap_values_original = shap_values
    average_healthy = np.mean(shap_values_original.values,axis=0)
    indx = shap_values_original.columns.get_loc("MBP_NonInvasive_Energy")
    pid = patientid
    labelset = ["Energy", "Entropy", "Correlation", "Sknewness", "Kurtosis", "Trend", "Mean", "Minimum", "Maximum"]
    fig, ax = plt.subplots()
    ax.axis('equal')
    width = 0.3
    cm = plt.get_cmap("tab20c")
    color = cm(np.array([1,2,3,4,5,6,7,8,9,10]))
    shares = average_healthy[indx:indx+9]/np.sum(abs(average_healthy[indx:indx+9]))
    pcg= []
    shares = abs(shares)
    for i, s in enumerate(shares):
        pcg.append(str(labelset[i]) + ": " + "{0:.0%}".format(s))
    pie, _ = ax.pie(shares, radius=1, labels=pcg, colors=color)
    plt.setp( pie, width=width, edgecolor='white')
    shares_pid = shap_values_original.values[pid,indx:indx+9]/np.sum(abs(shap_values_original.values[pid,indx:indx+9]))
    pcg= []
    shares_pid = abs(shares_pid)
    for i, s in enumerate(shares_pid):
        pcg.append(str(labelset[i]) + ": " +  "{0:.0%}".format(s))
    pie2, _ = ax.pie(shares_pid, radius=0.9-width, labels=pcg,labeldistance=0.4, colors=color)
    plt.setp( pie2, width=0.9-width, edgecolor='white')
    plt.show()
    fig.savefig(str(patientid) + "2.pdf", bbox_inches='tight')

    measurement=dataset.values[pid,indx:indx+9]
    measurement_avg = np.mean(dataset.values,axis=0)[indx:indx+9]
    ax = plt.subplot(111)
    x=np.array([[1,2,3,4,5,6,7,8,9]])
    ax.bar(x[0], measurement, width=0.4, color='b', align='center')
    max=1
    min=-1
    for i, v in enumerate(measurement):
        if v>0:
            if v>max:
                max=v
            ax.text(x[0,i]-0.2 , v + 0.1, str(float("{:.3f}".format(v))),fontsize=8)
        else:
            if v<min:
                min=v
            ax.text(x[0, i] - 0.2, v - 0.1, str(float("{:.3f}".format(v))), fontsize=8)
    plt.xticks(x[0], labelset, rotation=45)
    plt.show()
    ax.legend(["Normalized Measurement of Patient"],bbox_to_anchor=(0.23,1.02),loc="lower left", borderaxespad=0.)
    plt.ylim((min-0.3, max+0.3))
    plt.tight_layout()
    fig.savefig(str(patientid) + "3.pdf", bbox_inches='tight')

    fig = plt.figure()
    ax = plt.subplot(111)
    a = 0.5
    x_ticks = np.append(np.array([' ']), np.array(yticks)[B[::-1].tolist()])
    new_shap = np.append(np.array([0]), shap_values.values[patientid, B[::-1]])
    y_ticks = (1 / (1 + np.exp(-1 * np.cumsum(new_shap))))[::-1][0:number_variables + 1][::-1]
    ax.bar(x_ticks, new_shap, alpha=a)
    ax.plot(x_ticks, y_ticks)
    shap_healthy_group = shap_values.iloc[shap_values.index[dataset[outcome]==0],:]
    new_healthy_pneu = np.append(np.array([0]), np.mean(shap_healthy_group.values[:, B[::-1]], axis=0))
    ax.bar(x_ticks, new_healthy_pneu, alpha=a, color='grey', )
    # ax.bar(x_ticks,new_shap_pneu,alpha=a, yerr=new_shap_pneu_sd, color  = 'orange', error_kw=dict(ecolor='gray', lw=1, capsize=5, capthick=1))
    ax.legend(['Cumulative Risk Score', 'Variable Contribution', 'Average of Patients without Complication'])
    plt.xticks(rotation=90)
    plt.ylabel('Variable Importance by Average SHAP Values')
    plt.xlabel('Clinical Variables')
    plt.tight_layout()
    plt.ylim((-0.6, 1))
    plt.title('PID=' + str(patientid))
    fig.savefig(str(patientid) + "4.pdf", bbox_inches='tight')

    fig = plt.figure()
    ax = plt.subplot(111)
    a = 0.5
    x_ticks = np.append(np.array([' ']), np.array(yticks)[B[::-1].tolist()])
    new_shap = np.append(np.array([0]), shap_values.values[patientid, B[::-1]])
    y_ticks = (1 / (1 + np.exp(-1 * np.cumsum(new_shap))))[::-1][0:number_variables + 1][::-1]
    ax.bar(x_ticks, new_shap, alpha=a)
    ax.plot(x_ticks, y_ticks)
    shap_healthy_group = shap_values.iloc[shap_values.index[dataset[outcome]==0],:]
    new_healthy_pneu = np.append(np.array([0]), np.mean(shap_healthy_group.values[:, B[::-1]], axis=0))
    ax.bar(x_ticks, new_healthy_pneu, alpha=a, color='grey', )
    shap_pneu_group = shap_values.iloc[shap_values.index[dataset[outcome]==1],:]
    new_shap_pneu = np.append(np.array([0]), np.mean(shap_pneu_group.values[:, B[::-1]], axis=0))
    ax.bar(x_ticks, new_shap_pneu, alpha=a, color='orange', )
    # ax.bar(x_ticks,new_shap_pneu,alpha=a, yerr=new_shap_pneu_sd, color  = 'orange', error_kw=dict(ecolor='gray', lw=1, capsize=5, capthick=1))
    ax.legend(['Cumulative Risk Score', 'Variable Contribution', 'Average of Patients without Complication', 'Average of Patients with Complication'])
    plt.xticks(rotation=90)
    plt.ylabel('Variable Importance by Average SHAP Values')
    plt.xlabel('Clinical Variables')
    plt.tight_layout()
    plt.ylim((-0.6, 1))
    fig.savefig(str(patientid) + ".pdf", bbox_inches='tight')
