

# massive helper function to create a list of datatables (one per sim) and a paramtable of parameter vectors
# The function "generateFrames" takes in two parameters, the "results" and "decisions" variables created in
# the code cell above it.  It outputs two variables:
#
#    - "paramtable" which is a dataframe of the parameter vectors for each sim (automatically limited as to only
#       contain the parameters that are actually different for different sims).
#    - "datatables" which is a list of data frames, one dataframe per sim with one row per trial.
#       It contains columns that indicate the times (absolute time within the sim run) of stimuli/decisions/rewards,
#       as well as the duration (decision-stimulus) with and without the reward delay. It contains the network's
#       decision, as well as the correct decision and the reward.
#       Those last two will require adjustments depending on what sort of sim we are running.
#       It also contains both the mean and integral firing rates for all populations for all 3 time intervals
#       (stimulus-decision, decision-reward, stimulus-reward).

import os, sys
import pandas as pd
import numpy as np
import random
import math

import cbgt.netgen as ng

def generateFrames(results,decisions):
    
    datatables = []
    
    paramtable = pd.DataFrame()
    index = 0
    for confignum in range(0,len(results)):
        for repnum in range(0,len(results[confignum])):
            paramvector = pd.DataFrame([[index,confignum,repnum]],columns=['index','confignum','repnum'])
            index += 1
            for key,value in results[confignum][repnum].items():
                if isinstance(value, (float,int,str)):
                    paramvector[key] = value
            paramvector.set_index('index',inplace=True)
            paramtable = paramtable.append(paramvector)
            
            datatable = pd.DataFrame()
            
            result = results[confignum][repnum]
            
            stagecount = len(decisions[confignum][repnum])
            
            for i in range(0, stagecount):
                
                row = pd.DataFrame([[i]],columns=['trial'])
                
                
                if decisions[confignum][repnum][i]['pathvals'] != None and len(decisions[confignum][repnum][i]['pathvals']) > 0:
                    row['decision'] = decisions[confignum][repnum][i]['pathvals'][0]
                else:
                    row['decision'] = None
                try:
                    row['stimulusstarttime'] = decisions[confignum][repnum][i]['time'] - decisions[confignum][repnum][i]['delay']
                except:
                    row['stimulusstarttime'] = None
                row['decisiontime'] = decisions[confignum][repnum][i]['time']
                row['decisionduration'] = decisions[confignum][repnum][i]['delay']
                try:
                    row['decisiondurationplusdelay'] = decisions[confignum][repnum][i]['delay'] + 300
                except:
                    row['decisiondurationplusdelay'] = None
                try:
                    row['rewardtime'] = decisions[confignum][repnum][i]['time'] + 300
                except:
                    row['rewardtime'] = None
                    
                
                #row['decisiontime50'] = decisions[confignum][repnum][i]['time'] - 50
                #row['decisiontime30'] = decisions[confignum][repnum][i]['time'] - 30
                
                # NOTE: has to match code in netgen
                row['correctdecision'] = int(i>=20)
                row['reward'] = row.apply(lambda x: int(x.decision == x.correctdecision), axis=1)
                
                
                columnsException = result['popfreqs'].drop(columns='Time (ms)').mean().to_frame().T*0
                
                # Save the mean and the integral values from stimulus to decision
                try:
                    msd = result['popfreqs'][result['popfreqs']['Time (ms)'].between(row.loc[0,'stimulusstarttime'], row.loc[0,'decisiontime'])].drop(columns='Time (ms)').mean().to_frame().T
                except:
                    msd=columnsException
                
                try:
                    isd = msd.copy().multiply(row.loc[0,'decisiontime'] - row.loc[0,'stimulusstarttime']).divide(1000)
                except:
                    isd=columnsException
                    
                msd.columns = ["msd_" + str(col) for col in msd.columns]
                row = row.join(msd)
                isd.columns = ["isd_" + str(col) for col in isd.columns]
                row = row.join(isd)

                
                
                # Save the mean and the integral values from decision to reward
                try:
                    mdr = result['popfreqs'][result['popfreqs']['Time (ms)'].between(row.loc[0,'decisiontime'], row.loc[0,'rewardtime'])].drop(columns='Time (ms)').mean().to_frame().T
                except:
                    mdr=columnsException
                
                try:
                    idr = mdr.copy().multiply(row.loc[0,'rewardtime'] - row.loc[0,'decisiontime']).divide(1000)
                except:
                    idr=columnsException
                    
                mdr.columns = ["mdr_" + str(col) for col in mdr.columns]
                row = row.join(mdr)
                idr.columns = ["idr_" + str(col) for col in idr.columns]
                row = row.join(idr)
                
                
                # Save the mean and the integral values from stimulus to reward
                try:
                    msr = result['popfreqs'][result['popfreqs']['Time (ms)'].between(row.loc[0,'stimulusstarttime'], row.loc[0,'rewardtime'])].drop(columns='Time (ms)').mean().to_frame().T
                except:
                    msr=columnsException
                
                try:
                    isr = msr.copy().multiply(row.loc[0,'rewardtime'] - row.loc[0,'stimulusstarttime']).divide(1000)
                except:
                    isr=columnsException
                    
                msr.columns = ["msr_" + str(col) for col in msr.columns]
                row = row.join(msr)
                isr.columns = ["isr_" + str(col) for col in isr.columns]
                row = row.join(isr)
                
                
                # Save the mean from decision-50ms to decision and
                # from decision-30ms to decision
                try:
                    md50 = result['popfreqs'][result['popfreqs']['Time (ms)'].between(row.loc[0,'decisiontime']-50, row.loc[0,'decisiontime'])].drop(columns='Time (ms)').mean().to_frame().T
                except:
                    md50=columnsException
                
                try:
                    md30 = result['popfreqs'][result['popfreqs']['Time (ms)'].between(row.loc[0,'decisiontime']-30, row.loc[0,'decisiontime'])].drop(columns='Time (ms)').mean().to_frame().T
                except:
                    md30=columnsException
                    
                md50.columns = ["md50_" + str(col) for col in md50.columns]
                row = row.join(md50)
                md30.columns = ["md30_" + str(col) for col in md30.columns]
                row = row.join(md30)
                
                
                # Save the max and the min values from stimulus to decision
                try:
                    maxsd = result['popfreqs'][result['popfreqs']['Time (ms)'].between(row.loc[0,'stimulusstarttime'], row.loc[0,'decisiontime'])].drop(columns='Time (ms)').max().to_frame().T
                except:
                    maxsd=columnsException
                

                try:
                    minsd = result['popfreqs'][result['popfreqs']['Time (ms)'].between(row.loc[0,'stimulusstarttime'], row.loc[0,'decisiontime'])].drop(columns='Time (ms)').min().to_frame().T
                except:
                    minsd=columnsException
                    
                maxsd.columns = ["maxsd_" + str(col) for col in maxsd.columns]
                row = row.join(maxsd)
                minsd.columns = ["minsd_" + str(col) for col in minsd.columns]
                row = row.join(minsd)
                
                # Save the max and the min values from stimulus to reward
                try:
                    maxsr = result['popfreqs'][result['popfreqs']['Time (ms)'].between(row.loc[0,'stimulusstarttime'], row.loc[0,'rewardtime'])].drop(columns='Time (ms)').max().to_frame().T
                except:
                    maxsr=columnsException
                
                try:
                    minsr = result['popfreqs'][result['popfreqs']['Time (ms)'].between(row.loc[0,'stimulusstarttime'], row.loc[0,'rewardtime'])].drop(columns='Time (ms)').min().to_frame().T
                except:
                    minsr=columnsException
                    
                maxsr.columns = ["maxsr_" + str(col) for col in maxsr.columns]
                row = row.join(maxsr)
                minsr.columns = ["minsr_" + str(col) for col in minsr.columns]
                row = row.join(minsr)
                
                
                datatable = datatable.append(row)
            
            
            datatable.set_index('trial',inplace=True)
            datatables.append(datatable)
    

    nunique = paramtable.apply(pd.Series.nunique)
    cols_to_keep = pd.Index(['confignum','repnum']).append(nunique[nunique > 1].index).unique()
    paramtable = paramtable[cols_to_keep]
    
    return (paramtable,datatables)
