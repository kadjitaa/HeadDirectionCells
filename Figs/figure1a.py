# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 00:29:49 2020

@author: kasum
"""

coefs=[]
figure()
for i in spikes.keys():
    ax=subplot(gs[i])
    idx=cond[1][i].values >= tuning_curves_1[i].max() * 0.70
    
    #hist(cond[1][i].values[idx])
    
    scatter(cond[0][i].index.values[idx], np.unwrap(cond[0][i][idx].values), label=str(i))
    #Regression Fit Line
    sns.set_style(style='white')
    sns.regplot( x=array(cond[0].index[idx]),y=array(np.unwrap(cond[0][i][idx].values)), line_kws={'color':'red'}) #scatter_kws can also be used to change the scatter color
    legend(['linear_fit','spikes'])
    #gca().set_ylabel('Unwrapped Head Direction (rad)')
    gca().set_xlabel('Time (s)')
    gca().spines['right'].set_visible(False)
    gca().spines['top'].set_visible(False)
    gca().set_ylim(0,2*np.pi)
    
    #Regression Stats
    x=array(cond[0].index[idx]).reshape(-1,1)
    y=array(np.unwrap(cond[0][i][idx].values)).reshape(-1,1)
    regr=linear_model.LinearRegression()
    
    regr.fit(x,y)
    reg_pred=regr.predict(y)
    coefs.append(regr.coef_)